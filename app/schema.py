"""领域数据模型。

为什么单独定义 Clause 和 Chunk：
- Clause 是“政策条款”这一业务对象，带完整的溯源元数据（发文机关/文号/时效性）；
- Chunk 是“检索单元”，是 Clause 的一个切片，可能一条拆成多块，但引用必须回到 Clause。
这个分离是政策问答能给出准确引用的前提，也是本项目区别于通用 RAG demo 的地方。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Clause:
    """一条政策条款（可独立引用的最小单位）。"""

    doc_id: str                     # 稳定标识：文件指纹
    doc_title: str                  # 文件标题，如《苏州市建筑劳务企业管理办法》
    article_no: str                 # 第X条 / 前言 / 附件
    body: str                       # 条文正文（不含标题）
    chapter: str = ""               # 所属章
    section: str = ""               # 所属节
    doc_no: str = ""                # 文号，如 苏政办发〔2024〕39号
    issuer: str = ""                # 发文机关
    region: str = ""                # 适用地区
    publish_date: str = ""          # 发布日期 YYYY-MM-DD
    effective_date: str = ""        # 施行日期
    is_repealed: bool = False       # 是否已废止/失效
    repealed_hint: str = ""         # 废止依据或提示
    source_path: str = ""

    @property
    def citation(self) -> str:
        """人类可读引用，直接用于回答中的出处标注。"""
        parts = [f"《{self.doc_title}》"]
        if self.chapter:
            parts.append(self.chapter)
        if self.section:
            parts.append(self.section)
        parts.append(self.article_no)
        return " ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Chunk:
    """检索单元。text 会带上上下文头，header 单独保留用于还原显示。"""

    chunk_id: str
    clause: Clause
    header: str                     # 【《文件名》第X条｜机关｜日期】
    text: str                       # header + 正文，用于 BM25 / embedding
    body: str                       # 本块的正文
    part: int = 0                   # 第几块（0 起）
    total_parts: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "header": self.header,
            "text": self.text,
            "body": self.body,
            "part": self.part,
            "total_parts": self.total_parts,
            "clause": self.clause.to_dict(),
        }


@dataclass
class Retrieved:
    """一路召回的中间结果，用于调试与评测归因。"""

    chunk_id: str
    score: float
    source: str                     # bm25 | vector | hybrid | rerank
    rank: int = 0
    clause: Clause | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "score": round(self.score, 6),
            "source": self.source,
            "rank": self.rank,
            "citation": self.clause.citation if self.clause else "",
        }


@dataclass
class Citation:
    """回答中实际引用的条款。"""

    citation: str
    chunk_id: str
    doc_title: str
    article_no: str
    is_repealed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Answer:
    question: str
    answer: str
    citations: list[Citation] = field(default_factory=list)
    refused: bool = False
    retrieved: list[Retrieved] = field(default_factory=list)
    used_chunks: list[str] = field(default_factory=list)
    latency_ms: int = 0
    debug: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "citations": [c.to_dict() for c in self.citations],
            "refused": self.refused,
            "retrieved": [r.to_dict() for r in self.retrieved],
            "used_chunks": self.used_chunks,
            "latency_ms": self.latency_ms,
            "debug": self.debug,
        }


@dataclass
class EvalCase:
    """评测样本。

    must_refuse=True 表示该问题在现有文档中无答案，用于检验系统会不会编。
    gold_article 用于计算条款级 Recall@K 和 MRR。
    """

    id: str
    question: str
    gold_doc: str = ""
    gold_article: str = ""
    must_refuse: bool = False
    category: str = "general"       # 时效性 / 跨文件 / 长条款 / 否定题 ...
    gold_keywords: list[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class EvalResult:
    case: EvalCase
    answer: Answer
    hit_doc: bool = False
    hit_article: bool = False
    hit_rank: int = 0
    refusal_correct: bool = False
    citation_valid: bool = False
    keyword_hit: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.case.id,
            "question": self.case.question,
            "category": self.case.category,
            "must_refuse": self.case.must_refuse,
            "hit_doc": self.hit_doc,
            "hit_article": self.hit_article,
            "hit_rank": self.hit_rank,
            "refusal_correct": self.refusal_correct,
            "citation_valid": self.citation_valid,
            "keyword_hit": round(self.keyword_hit, 4),
            "refused": self.answer.refused,
            "citations": [c.citation for c in self.answer.citations],
            "answer": self.answer.answer,
            "latency_ms": self.answer.latency_ms,
        }
