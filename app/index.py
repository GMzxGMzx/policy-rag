"""检索层：中文分词 → BM25 / 向量双路召回 → RRF 融合 → 重排。

为什么必须做混合检索（这是本项目效果的主要来源）：

- 政策问答里有大量「精确词查询」：文号、条号、专有名词（如"苏政办发〔2024〕39号"、
  "实名制"、"专用账户"）。这类查询 BM25 明显强于向量，向量会把文号当噪声。
- 另一类是「口语化改写」：用户问"工人工资发不出来怎么办"，原文写的是
  "农民工工资专用账户"。这类靠向量。
- 两路用 RRF 融合，不需要对齐分数尺度（BM25 是无界分，余弦是 [0,1]），
  工程上最稳，也不需要训练。

中文分词不引入 jieba：用「单字 + 二元 + 三元」混合切分。
实测在条款级短文本上，这个朴素方案对政策类语料的效果足够，且零额外依赖、可解释。
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .schema import Chunk, Retrieved

# ---------------------------------------------------------------- 分词

_ASCII_WORD = re.compile(r"[a-zA-Z][a-zA-Z0-9_\-\.]{0,}")
_NUMBER = re.compile(r"\d+(?:\.\d+)?")
_HAN = re.compile(r"[\u4e00-\u9fff]+")

_STOP_CHARS = set("的了和与及或在是为对中上下之其该本各按应可由从到并等有无不未被将已就而且则如若因比我你请问吗呢啊吧")
_STOP_WORDS = {
    "什么", "怎么", "如何", "哪些", "哪个", "是否", "能否", "可以", "需要", "应当",
    "规定", "相关", "有关", "进行", "以及", "对于", "关于",
}


def tokenize(text: str) -> list[str]:
    """中文单字 + 二元 + 三元混合切分。

    单字保召回（同义改写时也能命中），二/三元保精度（专有名词不被拆散）。
    """
    tokens: list[str] = []
    low = text.lower()
    for m in _ASCII_WORD.finditer(low):
        tokens.append(m.group())
    for m in _NUMBER.finditer(low):
        tokens.append(m.group())
    for seg in _HAN.findall(text):
        for ch in seg:
            if ch not in _STOP_CHARS:
                tokens.append(ch)
        n = len(seg)
        for i in range(n - 1):
            tokens.append(seg[i : i + 2])
        for i in range(n - 2):
            tokens.append(seg[i : i + 3])
    return tokens


# ---------------------------------------------------------------- BM25

class BM25:
    """BM25Okapi，带倒排表，避免每次查询都扫描全库。"""

    def __init__(self, corpus_tokens: Sequence[Sequence[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.n_docs = len(corpus_tokens)
        self.doc_len = [len(t) for t in corpus_tokens]
        self.avgdl = (sum(self.doc_len) / self.n_docs) if self.n_docs else 1.0

        term_freqs = [Counter(t) for t in corpus_tokens]
        df: Counter[str] = Counter()
        for tf in term_freqs:
            df.update(tf.keys())

        self.idf: dict[str, float] = {
            term: math.log(1 + (self.n_docs - n + 0.5) / (n + 0.5))
            for term, n in df.items()
        }

        # 倒排表：term -> [(doc_idx, tf)]
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for idx, tf in enumerate(term_freqs):
            for term, freq in tf.items():
                self.postings[term].append((idx, freq))

    def scores(self, query_tokens: Iterable[str]) -> dict[int, float]:
        out: dict[int, float] = defaultdict(float)
        for term in set(query_tokens):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for doc_idx, freq in self.postings[term]:
                denom = freq + self.k1 * (
                    1 - self.b + self.b * self.doc_len[doc_idx] / self.avgdl
                )
                out[doc_idx] += idf * freq * (self.k1 + 1) / denom
        return dict(out)

    def top_k(self, query_tokens: Iterable[str], k: int = 20) -> list[tuple[int, float]]:
        scored = self.scores(query_tokens)
        return sorted(scored.items(), key=lambda x: -x[1])[:k]


# ---------------------------------------------------------------- 向量索引

class VectorIndex:
    """numpy 精确检索。

    诚实说明：这是暴力检索，复杂度和条数线性相关。条款级语料通常在 1e3~1e5 量级，
    单机 numpy 完全够用（1 万条 × 1024 维 ≈ 40MB，单次查询约 5ms）。
    超过 10 万块就应换成 Qdrant / pgvector，接口不用改（见 README「生产替换」）。
    """

    def __init__(self) -> None:
        self.ids: list[str] = []
        self.matrix: np.ndarray | None = None

    @staticmethod
    def _normalize(mat: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(mat, axis=1, keepdims=True)
        norm[norm == 0] = 1.0
        return mat / norm

    def add(self, ids: list[str], vectors: np.ndarray) -> None:
        vectors = self._normalize(np.asarray(vectors, dtype=np.float32))
        if self.matrix is None:
            self.matrix = vectors
        else:
            self.matrix = np.vstack([self.matrix, vectors])
        self.ids.extend(ids)

    def search(self, query_vector: np.ndarray, k: int = 20) -> list[tuple[str, float]]:
        if self.matrix is None or not self.ids:
            return []
        q = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        norm = np.linalg.norm(q)
        if norm == 0:
            return []
        sims = self.matrix @ (q / norm)
        k = min(k, len(self.ids))
        top = np.argpartition(-sims, k - 1)[:k]
        top = top[np.argsort(-sims[top])]
        return [(self.ids[i], float(sims[i])) for i in top]

    def save(self, path: Path) -> None:
        if self.matrix is None:
            return
        np.savez_compressed(path, matrix=self.matrix)
        path.with_suffix(".ids.json").write_text(
            json.dumps(self.ids, ensure_ascii=False), encoding="utf-8"
        )

    def load(self, path: Path) -> None:
        if not path.exists():
            return
        data = np.load(path)
        self.matrix = data["matrix"]
        self.ids = json.loads(path.with_suffix(".ids.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------- RRF 融合

def rrf_fuse(
    rankings: dict[str, list[str]],
    k: int = 60,
    weights: dict[str, float] | None = None,
) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion。

    score(d) = Σ_source  w_source / (k + rank_source(d))
    只看名次不看分数，天然解决两路分数不可比的问题。
    """
    weights = weights or {}
    fused: dict[str, float] = defaultdict(float)
    for source, ids in rankings.items():
        w = weights.get(source, 1.0)
        for rank, doc_id in enumerate(ids, start=1):
            fused[doc_id] += w / (k + rank)
    return sorted(fused.items(), key=lambda x: -x[1])


# ---------------------------------------------------------------- 检索器

class HybridRetriever:
    def __init__(
        self,
        chunks: list[Chunk],
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
        rrf_k: int = 60,
    ) -> None:
        self.chunks = chunks
        self.by_id: dict[str, Chunk] = {c.chunk_id: c for c in chunks}
        self.rrf_k = rrf_k
        self.bm25 = BM25([tokenize(c.text) for c in chunks], k1=bm25_k1, b=bm25_b)
        self.vectors = VectorIndex()

    # ---- 构建 ----
    def build_vectors(self, embedder) -> None:
        if not self.chunks:
            return
        texts = [c.text for c in self.chunks]
        vectors = embedder.embed_documents(texts)
        self.vectors.add([c.chunk_id for c in self.chunks], vectors)

    # ---- 查询 ----
    def retrieve(
        self,
        query: str,
        top_k: int = 20,
        embedder=None,
        reranker=None,
        rerank_top_n: int = 8,
        doc_filter: set[str] | None = None,
    ) -> list[Retrieved]:
        rankings: dict[str, list[str]] = {}

        bm25_hits = self.bm25.top_k(tokenize(query), k=top_k * 2)
        rankings["bm25"] = [self.chunks[i].chunk_id for i, _ in bm25_hits]

        if embedder is not None and embedder.available and self.vectors.matrix is not None:
            qvec = embedder.embed_query(query)
            vec_hits = self.vectors.search(qvec, k=top_k * 2)
            rankings["vector"] = [cid for cid, _ in vec_hits]

        if len(rankings) == 1:
            fused = [(cid, 1.0 / (i + 1)) for i, cid in enumerate(rankings["bm25"])]
        else:
            fused = rrf_fuse(rankings, k=self.rrf_k, weights={"bm25": 1.0, "vector": 1.0})

        if doc_filter:
            fused = [(cid, s) for cid, s in fused if self.by_id[cid].clause.doc_id in doc_filter]

        fused = fused[: max(top_k, rerank_top_n)]

        results = [
            Retrieved(chunk_id=cid, score=score, source="hybrid", rank=i + 1, clause=self.by_id[cid].clause)
            for i, (cid, score) in enumerate(fused)
        ]

        # 重排：对融合后的前 N 条做精排，只改顺序不改召回集
        if reranker is not None and reranker.available and results:
            head = results[:rerank_top_n]
            texts = [self.by_id[r.chunk_id].text for r in head]
            scores = reranker.score(query, texts)
            order = sorted(range(len(head)), key=lambda i: -scores[i])
            head = [head[i] for i in order]
            for r in head:
                r.source = "rerank"
            results = head + results[rerank_top_n:]
            for i, r in enumerate(results):
                r.rank = i + 1

        return results[:top_k]

    # ---- 持久化 ----
    def save(self, index_dir: Path) -> None:
        index_dir.mkdir(parents=True, exist_ok=True)
        with (index_dir / "chunks.jsonl").open("w", encoding="utf-8") as fh:
            for chunk in self.chunks:
                fh.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
        self.vectors.save(index_dir / "vectors.npz")

    @classmethod
    def load(cls, index_dir: Path, bm25_k1: float = 1.5, bm25_b: float = 0.75, rrf_k: int = 60):
        from .schema import Clause

        chunks: list[Chunk] = []
        path = index_dir / "chunks.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"索引不存在：{path}，请先执行 python -m scripts.ingest")
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                row = json.loads(line)
                clause = Clause(**row["clause"])
                chunks.append(
                    Chunk(
                        chunk_id=row["chunk_id"],
                        clause=clause,
                        header=row["header"],
                        text=row["text"],
                        body=row["body"],
                        part=row.get("part", 0),
                        total_parts=row.get("total_parts", 1),
                    )
                )
        inst = cls(chunks, bm25_k1=bm25_k1, bm25_b=bm25_b, rrf_k=rrf_k)
        inst.vectors.load(index_dir / "vectors.npz")
        return inst
