"""Embedding 适配器。

三种实现，用环境变量 EMBED_PROVIDER 切换：

- hash   ：零依赖的哈希词袋向量。**不含语义**，只用于在没配 API key 时跑通全链路、
           以及做回归测试的基线。README 里明确标注了这一点，不能当效果数据用。
- openai ：任何 OpenAI 兼容的 /embeddings 接口。硅基流动、阿里百炼、DeepSeek、
           vLLM、TEI、Ollama 都是这个协议，所以一份代码通吃。
- local  ：本地 sentence-transformers 加载 BGE-M3，适合完全离线/数据不出内网的场景
           （政务和企业客户经常提这个要求，这是一个真实的卖点）。

把 Embedding 做成适配器而不是写死，是因为甲方环境差异极大：
有的不允许出网、有的只给 OpenAI 兼容网关、有的要国产模型。
"""
from __future__ import annotations

import hashlib
import time

import numpy as np

from .config import Settings
from .index import tokenize


class BaseEmbedder:
    name = "base"
    available = False

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]


class HashEmbedder(BaseEmbedder):
    """哈希词袋（类似 sklearn HashingVectorizer）。

    用 token 的哈希做特征位，带符号避免冲突累积。语义能力为零，
    但因为与 BM25 共用同一套分词，它能作为一条可用的词面召回通道做兜底。
    """

    name = "hash"

    def __init__(self, dim: int = 2048) -> None:
        self.dim = dim
        self.available = True

    def _one(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        for tok in tokenize(text):
            digest = hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest()
            idx = int.from_bytes(digest[:4], "little") % self.dim
            sign = 1.0 if digest[4] & 1 else -1.0
            vec[idx] += sign
        return vec

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.vstack([self._one(t) for t in texts])


class OpenAICompatEmbedder(BaseEmbedder):
    name = "openai"

    def __init__(self, base_url: str, api_key: str, model: str, batch: int = 16, timeout: float = 60.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.batch = max(1, batch)
        self.timeout = timeout
        self.available = bool(api_key)

    def _call(self, inputs: list[str]) -> list[list[float]]:
        import httpx

        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {"model": self.model, "input": inputs, "encoding_format": "float"}
        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    resp = client.post(f"{self.base_url}/embeddings", json=payload, headers=headers)
                if resp.status_code in (429, 500, 502, 503, 504):
                    raise RuntimeError(f"embedding 服务返回 {resp.status_code}")
                resp.raise_for_status()
                data = resp.json()["data"]
                data.sort(key=lambda d: d.get("index", 0))
                return [d["embedding"] for d in data]
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"embedding 调用失败：{last_exc}") from last_exc

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch):
            out.extend(self._call(texts[i : i + self.batch]))
        return np.asarray(out, dtype=np.float32)


class LocalEmbedder(BaseEmbedder):
    name = "local"

    def __init__(self, model: str = "BAAI/bge-m3", device: str | None = None) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            self.available = False
            self._model = None
            return
        self._model = SentenceTransformer(model, device=device)
        self.available = True

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)
        return np.asarray(
            self._model.encode(texts, batch_size=16, show_progress_bar=False, normalize_embeddings=False),
            dtype=np.float32,
        )


def build_embedder(cfg: Settings) -> BaseEmbedder:
    provider = (cfg.embed_provider or "hash").lower()
    if provider == "openai":
        if not cfg.embed_api_key:
            print("  [提示] EMBED_PROVIDER=openai 但未配置 EMBED_API_KEY，降级为 hash")
            return HashEmbedder()
        return OpenAICompatEmbedder(cfg.embed_base_url, cfg.embed_api_key, cfg.embed_model, cfg.embed_batch)
    if provider == "local":
        emb = LocalEmbedder(cfg.embed_model)
        if not emb.available:
            print("  [提示] 未安装 sentence-transformers，降级为 hash")
            return HashEmbedder()
        return emb
    return HashEmbedder()
