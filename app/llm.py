"""大模型与重排的客户端。

两个设计决定：

1. LLM 走 OpenAI 兼容协议（/chat/completions），不绑定厂商。
   这样从 DeepSeek 换到通义、Kimi、Qwen、本地 vLLM / Ollama 都只是改环境变量。
   甲方私有化部署时这一点几乎是硬要求。

2. LLM 不可用时不报错，降级为「抽取式回答」。
   这让我们可以在没有任何 API key 的情况下单独评测检索质量——评测检索和评测生成
   必须能分开跑，否则检索坏了会被 LLM 的润色掩盖。
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass

from .config import Settings


class LLMClient:
    def __init__(self, base_url: str, api_key: str, model: str, temperature: float = 0.1, timeout: float = 90.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.timeout = timeout

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def chat(self, system: str, user: str) -> str:
        import httpx

        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    resp = client.post(f"{self.base_url}/chat/completions", json=payload, headers=headers)
                if resp.status_code in (429, 500, 502, 503, 504):
                    raise RuntimeError(f"LLM 返回 {resp.status_code}")
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"].strip()
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                time.sleep(2.0 * (attempt + 1))
        raise RuntimeError(f"LLM 调用失败：{last_exc}") from last_exc

    def chat_json(self, system: str, user: str) -> dict:
        text = self.chat(system, user)
        text = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start, end = text.find("{"), text.rfind("}")
            if start != -1 and end > start:
                return json.loads(text[start : end + 1])
            raise


class Reranker:
    """BGE-reranker 类的 cross-encoder 重排。

    为什么值得加：RRF 只是把两路名次合并，它不理解「问题与条款是否真的相关」。
    重排模型逐对打分，能显著纠正 RRF 的排序错误。经验上 Recall@5 能提升 8~15 个点。
    代价是延迟（每对一次前向），所以只对前 N 条做。
    """

    def __init__(self, base_url: str, api_key: str, model: str, timeout: float = 60.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    @property
    def available(self) -> bool:
        return bool(self.api_key and self.base_url)

    def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        if not self.available:
            return [0.0] * len(texts)
        import httpx

        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {"model": self.model, "query": query, "documents": texts, "top_n": len(texts)}
        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(f"{self.base_url}/rerank", json=payload, headers=headers)
            resp.raise_for_status()
            rows = resp.json()["results"]
        except Exception as exc:  # noqa: BLE001
            print(f"  [警告] 重排失败，跳过：{exc}")
            return [0.0] * len(texts)

        out = [0.0] * len(texts)
        for row in rows:
            idx = row.get("index", 0)
            if 0 <= idx < len(out):
                out[idx] = float(row.get("relevance_score", 0.0))
        return out


@dataclass
class ModelBundle:
    llm: LLMClient | None
    reranker: Reranker | None


def build_models(cfg: Settings) -> ModelBundle:
    llm = LLMClient(
        cfg.llm_base_url, cfg.llm_api_key, cfg.llm_model, cfg.llm_temperature, cfg.llm_timeout
    )
    reranker = None
    if cfg.rerank_enabled:
        reranker = Reranker(cfg.rerank_base_url, cfg.rerank_api_key, cfg.rerank_model)
        if not reranker.available:
            print("  [提示] RERANK_ENABLED=true 但未配置 RERANK_API_KEY，已跳过重排")
            reranker = None
    return ModelBundle(llm=llm if llm.available else None, reranker=reranker)
