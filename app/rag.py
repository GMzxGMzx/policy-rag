"""RAG 编排：检索 → 重排 → 组装上下文 → 生成 → 引用校验。

这个模块的核心不是"调模型"，而是三件事，它们决定了系统能不能在合规场景落地：

1. **引用可校验**：回答里的每个【《文件》第X条】都会被反查，确认它真的来自当次召回的
   上下文。模型编造的条号会被识别出来（hallucinated citation），这是评测里的硬指标。
2. **可拒答**：证据不足时必须回答"根据现有资料无法确定"，而不是给一个看起来合理的答案。
   在政策场景里，错误答案的代价远高于没有答案。
3. **时效性提示**：引用到已废止条款时必须先提示失效。这是最容易出事的地方——
   新旧文件并存时，模型会自信地引用旧文件。
"""
from __future__ import annotations

import re
import time

from .config import Settings
from .index import HybridRetriever
from .schema import Answer, Chunk, Citation, Retrieved

REFUSAL_MARKER = "根据现有资料无法确定"

SYSTEM_PROMPT = """你是建筑与工程领域的政策合规问答助手。必须严格遵守以下规则：

【证据规则】
1. 只能使用【参考资料】中的条款作答，不得使用任何外部知识，不得推测和补全。
2. 每条结论后必须标注出处，格式固定为：【《文件标题》第X条】。一句话有多个依据时全部标注。
3. 严禁编造条号、文件名、文号、日期。资料里没有的内容一律不写。

【拒答规则】
4. 如果参考资料不能直接回答该问题，必须原样输出这一行：根据现有资料无法确定
   然后用一句话说明缺少哪类信息。不要给出任何猜测性结论。

【时效与适用规则】
5. 若引用的条款标注为「已废止」，必须先明确提示该条款已失效，再判断是否存在现行替代条款。
6. 注意条款的适用范围（地区）和施行日期。若问题涉及其他地区，或文件尚未施行，必须明确指出。

【表达规则】
7. 先给结论，再列依据，用分点作答。语言简洁。
8. 不使用法律意见式措辞（如"建议您""本律师认为""请咨询专业机构"）。"""

_CITE_BLOCK = re.compile(r"【([^】]{2,120})】")
_TITLE_IN_CITE = re.compile(r"《([^》]{2,60})》")


def _article_present(article_no: str, text: str) -> bool:
    """判断条号是否出现在文本里。

    必须处理「附件1」误匹配「附件10」这种前缀问题，否则引用会算到错误的条款上。
    """
    if not article_no:
        return False
    start = text.find(article_no)
    while start != -1:
        end = start + len(article_no)
        tail = text[end : end + 1]
        if not (article_no[-1].isdigit() and tail.isdigit()):
            return True
        start = text.find(article_no, start + 1)
    return False


def build_context(hits: list[Retrieved], chunks_by_id: dict[str, Chunk]) -> str:
    blocks: list[str] = []
    for i, hit in enumerate(hits, start=1):
        chunk = chunks_by_id[hit.chunk_id]
        clause = chunk.clause
        title = f"【{chunk.header}】"
        if clause.is_repealed:
            title += "〔注意：该条款标注为已废止〕"
        blocks.append(f"[{i}] {title}\n{chunk.body}")
    return "\n\n".join(blocks)


def parse_citations(text: str, hits: list[Retrieved], chunks_by_id: dict[str, Chunk]) -> tuple[list[Citation], list[str]]:
    """从回答里抽引用，区分「有效引用」和「编造引用」。

    早期实现只认「第X条」，结果通知体文件（引用单位是「一、核查对象」）和附件
    （引用单位是「附件1」）的引用全部被判为无效 —— 而这两类恰好是真实语料的主体。
    现在改成「拿上下文里真实存在的条号去回答里找」，对三种命名方式一视同仁。

    返回 (有效引用列表, 无法在上下文中找到的引用字符串列表)。
    """
    candidates: list[Chunk] = []
    for hit in hits:
        chunk = chunks_by_id.get(hit.chunk_id)
        if chunk is not None:
            candidates.append(chunk)
    # 长条号优先，避免「附件1」抢在「附件10」前面
    candidates.sort(key=lambda c: -len(c.clause.article_no))

    valid: list[Citation] = []
    invalid: list[str] = []
    seen: set[tuple[str, str]] = set()

    for raw in _CITE_BLOCK.findall(text):
        block = raw.strip()
        # 正文里也可能出现【】标记（例如文件头部的【虚构示例】【附件】），
        # 那不是引用。只接受两种形态：含书名号（即《文件》第X条），或与某个条号完全相同。
        if "《" not in block and not any(block == c.clause.article_no for c in candidates):
            continue

        title_match = _TITLE_IN_CITE.search(block)
        title = title_match.group(1) if title_match else ""

        matched: Chunk | None = None
        for chunk in candidates:
            if not _article_present(chunk.clause.article_no, block):
                continue
            if title and title not in chunk.clause.doc_title and chunk.clause.doc_title not in title:
                continue
            matched = chunk
            break

        if matched is None:
            invalid.append(block)
            continue

        key = (matched.clause.doc_title, matched.clause.article_no)
        if key in seen:
            continue
        seen.add(key)
        valid.append(
            Citation(
                citation=matched.clause.citation,
                chunk_id=matched.chunk_id,
                doc_title=matched.clause.doc_title,
                article_no=matched.clause.article_no,
                is_repealed=matched.clause.is_repealed,
            )
        )

    return valid, invalid


class RAGPipeline:
    def __init__(self, retriever: HybridRetriever, embedder, llm=None, reranker=None, cfg: Settings | None = None) -> None:
        self.retriever = retriever
        self.embedder = embedder
        self.llm = llm
        self.reranker = reranker
        self.cfg = cfg or Settings()
        self.chunks_by_id = retriever.by_id

    # ------------------------------------------------------------------ 检索
    def search(self, question: str, top_k: int | None = None, doc_filter: set[str] | None = None) -> list[Retrieved]:
        top_k = top_k or self.cfg.context_top_n
        return self.retriever.retrieve(
            question,
            top_k=top_k,
            embedder=self.embedder,
            reranker=self.reranker,
            rerank_top_n=self.cfg.rerank_top_n,
            doc_filter=doc_filter,
        )

    # ------------------------------------------------------------------ 回答
    def answer(self, question: str, doc_filter: set[str] | None = None, top_k: int | None = None) -> Answer:
        started = time.perf_counter()
        hits = self.search(question, top_k=top_k or self.cfg.context_top_n, doc_filter=doc_filter)

        if not hits:
            return Answer(
                question=question,
                answer=f"{REFUSAL_MARKER}：检索结果为空，知识库中没有相关条款。",
                refused=True,
                latency_ms=int((time.perf_counter() - started) * 1000),
                debug={"mode": "empty"},
            )

        used = hits[: self.cfg.context_top_n]
        context = build_context(used, self.chunks_by_id)

        if self.llm is None:
            answer_text, mode = self._extractive(used), "extractive"
        else:
            user = f"【参考资料】\n{context}\n\n【问题】\n{question}"
            try:
                answer_text, mode = self.llm.chat(SYSTEM_PROMPT, user), "llm"
            except Exception as exc:  # noqa: BLE001
                answer_text = self._extractive(used)
                mode = f"extractive(LLM失败:{exc})"

        citations, invalid = parse_citations(answer_text, used, self.chunks_by_id)
        refused = REFUSAL_MARKER in answer_text

        return Answer(
            question=question,
            answer=answer_text,
            citations=citations,
            refused=refused,
            retrieved=used,
            used_chunks=[h.chunk_id for h in used],
            latency_ms=int((time.perf_counter() - started) * 1000),
            debug={
                "mode": mode,
                "invalid_citations": invalid,
                "top_scores": [round(h.score, 4) for h in used],
            },
        )

    # ------------------------------------------------------------------ 抽取式降级
    def _extractive(self, hits: list[Retrieved]) -> str:
        """无大模型时的降级输出：直接给最强证据原文，不做归纳。

        这样做的价值是：检索质量可以被单独评测，不会被模型的润色掩盖。
        注意这里**不能**输出 REFUSAL_MARKER —— 否则会被误判为拒答，
        导致「应拒答正确率」虚高而「可答误拒答率」虚高，指标全废。
        """
        lines = ["（未启用大模型，以下为检索命中的条款原文摘录，需人工核对）"]
        for hit in hits[:3]:
            chunk = self.chunks_by_id[hit.chunk_id]
            note = "（已废止）" if chunk.clause.is_repealed else ""
            lines.append(f"\n【{chunk.clause.citation}】{note}\n{chunk.body}")
        return "\n".join(lines)
