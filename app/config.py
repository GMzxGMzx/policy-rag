"""全局配置。

设计约束：本模块不依赖任何三方库，保证在最小环境下也能被导入和单元测试。
所有参数均可通过环境变量或 .env 覆盖。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> None:
    """极简 .env 解析，避免为一行配置引入 python-dotenv。"""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


_load_dotenv(ROOT / ".env")


def _str(name: str, default: str = "") -> str:
    val = os.environ.get(name, "")
    return val if val.strip() else default


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "") or default)
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "") or default)
    except ValueError:
        return default


def _bool(name: str, default: bool) -> bool:
    val = _str(name).lower()
    return default if not val else val in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    # 路径
    data_dir: Path = field(default_factory=lambda: ROOT / "data")
    raw_dir: Path = field(default_factory=lambda: ROOT / "data" / "raw")
    index_dir: Path = field(default_factory=lambda: ROOT / "data" / "index")
    eval_dir: Path = field(default_factory=lambda: ROOT / "data" / "eval")

    # 切分
    max_clause_chars: int = _int("MAX_CLAUSE_CHARS", 900)
    min_body_chars: int = _int("MIN_BODY_CHARS", 20)

    # 检索
    bm25_k1: float = _float("BM25_K1", 1.5)
    bm25_b: float = _float("BM25_B", 0.75)
    retrieve_top_k: int = _int("RETRIEVE_TOP_K", 20)
    rerank_top_n: int = _int("RERANK_TOP_N", 8)
    rrf_k: int = _int("RRF_K", 60)
    context_top_n: int = _int("CONTEXT_TOP_N", 5)

    # 向量
    embed_provider: str = _str("EMBED_PROVIDER", "hash")
    embed_base_url: str = _str("EMBED_BASE_URL", "https://api.siliconflow.cn/v1")
    embed_api_key: str = _str("EMBED_API_KEY")
    embed_model: str = _str("EMBED_MODEL", "BAAI/bge-m3")
    embed_batch: int = _int("EMBED_BATCH", 16)

    # 重排
    rerank_enabled: bool = _bool("RERANK_ENABLED", False)
    rerank_base_url: str = _str("RERANK_BASE_URL")
    rerank_api_key: str = _str("RERANK_API_KEY")
    rerank_model: str = _str("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")

    # LLM
    llm_base_url: str = _str("LLM_BASE_URL", "https://api.deepseek.com/v1")
    llm_api_key: str = _str("LLM_API_KEY")
    llm_model: str = _str("LLM_MODEL", "deepseek-chat")
    llm_temperature: float = _float("LLM_TEMPERATURE", 0.1)
    llm_timeout: float = _float("LLM_TIMEOUT", 90.0)


settings = Settings()


def ensure_dirs() -> None:
    for path in (settings.raw_dir, settings.index_dir, settings.eval_dir):
        path.mkdir(parents=True, exist_ok=True)
