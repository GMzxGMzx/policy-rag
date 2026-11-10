"""核心逻辑测试。

不依赖 data/ 下的任何文件 —— 全部用内联夹具。
理由：你会不断替换语料，测试不能因为"示例文件被删了"而失败。

覆盖的每一条用例，都对应一次在真实语料上踩到的坑：
- 章/节继承、条款级切分
- 「一、二、三」通知体（真实语料里超过一半是这种）
- 附件边界（否则最后一条会吞掉整份附件）
- PDF 硬换行回流
- 标题不能从「根据《XXX》」里取
- 引用不只认「第X条」，还要认「一、核查对象」「附件1」

运行：python tests/test_core.py    或    pytest tests/
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.embedder import HashEmbedder  # noqa: E402
from app.evaluate import classify_badcase, compute_metrics  # noqa: E402
from app.index import BM25, HybridRetriever, rrf_fuse, tokenize  # noqa: E402
from app.ingest import (  # noqa: E402
    chunk_clauses,
    extract_meta,
    find_effective_date,
    find_repealed_docs,
    normalize_text,
    reflow,
    split_clauses,
    strip_noise,
)
from app.rag import REFUSAL_MARKER, RAGPipeline, parse_citations  # noqa: E402
from app.schema import Answer, Citation, EvalCase, EvalResult  # noqa: E402

FIXTURE_ARTICLE = """本文件为测试夹具，不是真实政策。
测试用建筑劳务企业信用管理办法
测建规〔2024〕7 号
苏州市测试局
2024 年 11 月 20 日

第一章 总则

第一条 为规范本市测试行为，制定本办法。

第二条 本办法适用于苏州市行政区域内的测试活动。

第二章 信用评价

第三条 信用等级分为 AA、A、B、C、D 五级。AA 级为 90 分及以上，A 级为 80 至 89 分，B 级为 70 至 79 分，C 级为 60 至 69 分，D 级为 60 分以下。

第四条 有下列情形之一的，予以扣分：
（一）未按时报送信息的，每次扣 2 分；
（二）拖欠工资未在 30 日内解决的，每次扣 10 分；
（三）被通报批评的，每次扣 5 分；
（四）提供虚假材料的，一次性扣 20 分。

第三章 附则

第五条 本办法自 2025 年 1 月 1 日起施行。

第六条 《测试用暂行办法》同时废止。

附件1：测试评分表
基本信息分 30 分，能力分 20 分。
"""

FIXTURE_NOTICE = """本文件为测试夹具，不是真实政策。
测试用动态核查工作通知
测建〔2024〕9 号
苏州市测试局
2024 年 3 月 1 日

各有关单位：

一、核查对象
在本市从事测试活动的企业。

二、核查周期
每季度更新一次。

三、结果公布
在苏州市测试局网站公布。
"""

FIXTURE_REFERENCE_ONLY = """各有关单位：
根据《省测试厅关于开展测试工作的通知》（测厅〔2024〕1 号）要求，现就有关事项通知如下：
一、对象
测试企业。
二、周期
每季度一次。
"""


def _parse_text(name: str, text: str):
    clean = strip_noise(normalize_text(text))
    path = Path(name)
    meta = extract_meta(clean, path)
    meta["effective_date"] = find_effective_date(clean)
    meta["repealed_docs"] = find_repealed_docs(clean)
    return split_clauses(clean, meta, path), meta


def _corpus():
    a, ma = _parse_text("fixture_article.txt", FIXTURE_ARTICLE)
    b, mb = _parse_text("fixture_notice.txt", FIXTURE_NOTICE)
    return a + b, [ma, mb]


def _retriever(max_chars: int = 900):
    clauses, _ = _corpus()
    chunks = chunk_clauses(clauses, max_chars)
    retriever = HybridRetriever(chunks)
    emb = HashEmbedder(dim=1024)
    retriever.build_vectors(emb)
    return retriever, emb, chunks


# ---------------------------------------------------------------- 解析

def test_clause_count():
    doc1, _ = _parse_text("fixture_article.txt", FIXTURE_ARTICLE)
    articles = [c for c in doc1 if c.article_no.startswith("第")]
    attaches = [c for c in doc1 if c.article_no.startswith("附件")]
    assert len(articles) == 6, [c.article_no for c in doc1]
    assert len(attaches) == 1, [c.article_no for c in doc1]
    assert doc1[0].article_no == "前言"


def test_meta_extraction():
    _, meta = _parse_text("fixture_article.txt", FIXTURE_ARTICLE)
    assert meta["title"] == "测试用建筑劳务企业信用管理办法", meta["title"]
    assert meta["doc_no"] == "测建规〔2024〕7号", meta["doc_no"]
    assert meta["issuer"] == "苏州市测试局", meta["issuer"]
    assert meta["publish_date"] == "2024-11-20", meta["publish_date"]
    assert meta["effective_date"] == "2025-01-01", meta["effective_date"]
    assert meta["repealed_docs"] == ["测试用暂行办法"], meta["repealed_docs"]


def test_chapter_inheritance():
    doc1, _ = _parse_text("fixture_article.txt", FIXTURE_ARTICLE)
    by_no = {c.article_no: c for c in doc1 if c.article_no.startswith("第")}
    assert "总则" in by_no["第一条"].chapter
    assert "信用评价" in by_no["第三条"].chapter
    assert "附则" in by_no["第五条"].chapter


def test_title_not_taken_from_reference():
    """标题不能从「根据《XXX》要求」里取 —— 真实语料里踩过这个坑。"""
    _, meta = _parse_text("fixture_ref.txt", FIXTURE_REFERENCE_ONLY)
    assert meta["title"] != "省测试厅关于开展测试工作的通知", meta["title"]


def test_attach_boundary_not_swallowed():
    """附件必须切断最后一条，否则会出现几千字的畸形条款。"""
    doc1, _ = _parse_text("fixture_article.txt", FIXTURE_ARTICLE)
    by_no = {c.article_no: c for c in doc1 if c.article_no.startswith("第")}
    assert "30 分" not in by_no["第六条"].body, by_no["第六条"].body[:80]
    assert max(len(c.body) for c in doc1) < 300


def test_reflow_joins_broken_lines():
    out = reflow("满分 30\n分）、劳务能力\n分（满分 20 分）")
    assert "\n" not in out, out
    assert "劳务能力分" in out, out


# ---------------------------------------------------------------- 切分

def test_long_clause_split():
    clauses, _ = _corpus()
    chunks = chunk_clauses(clauses, max_chars=80)
    multi = [c for c in chunks if c.total_parts > 1]
    assert multi, "max_chars=80 时应出现拆分块"
    for chunk in multi:
        assert len(chunk.body) <= 110, f"{chunk.chunk_id} 过长：{len(chunk.body)}"


def test_chunk_header_contains_context():
    clauses, _ = _corpus()
    chunks = chunk_clauses(clauses, 900)
    target = next(c for c in chunks if c.clause.article_no == "第三条")
    assert target.text.startswith("【")
    assert "《" in target.header


def test_no_split_when_short():
    clauses, _ = _corpus()
    chunks = chunk_clauses(clauses, 5000)
    assert all(c.total_parts == 1 for c in chunks)


def test_notice_uses_item_structure():
    """通知体（一、二、三）必须被切成小节，而不是一整段。"""
    doc2, _ = _parse_text("fixture_notice.txt", FIXTURE_NOTICE)
    items = [c.article_no for c in doc2 if c.article_no.startswith(("一、", "二、", "三、"))]
    assert len(items) == 3, items
    assert any("核查对象" in x for x in items), items


# ---------------------------------------------------------------- 检索

def test_tokenize_mixed():
    toks = tokenize("实名制 90% 上传率 ABC-123")
    assert "实名" in toks, toks
    assert "90" in toks, toks
    assert any(t.startswith("abc") for t in toks), toks


def test_bm25_finds_correct_clause():
    retriever, _, chunks = _retriever()
    hits = retriever.bm25.top_k(tokenize("拖欠工资扣多少分"), k=3)
    assert chunks[hits[0][0]].clause.article_no == "第四条", chunks[hits[0][0]].clause.article_no


def test_rrf_ordering():
    fused = rrf_fuse({"bm25": ["a", "b", "c"], "vector": ["b", "a", "d"]}, k=60)
    ids = [i for i, _ in fused]
    assert set(ids[:2]) == {"a", "b"}, ids
    assert ids.index("c") > ids.index("a")


def test_hybrid_end_to_end():
    retriever, emb, _ = _retriever()
    pipe = RAGPipeline(retriever, emb, None, None)
    hits = pipe.search("信用等级一共分几级", top_k=3)
    assert hits and hits[0].clause is not None
    assert hits[0].score > 0


# ---------------------------------------------------------------- 引用

def test_citation_valid_and_invalid():
    retriever, emb, _ = _retriever()
    pipe = RAGPipeline(retriever, emb, None, None)
    hits = pipe.search("信用等级一共分几级", top_k=5)
    text = (
        "信用等级分为五级。"
        "【《测试用建筑劳务企业信用管理办法》第三条】"
        "另有【《不存在的文件》第九十九条】的说法。"
    )
    valid, invalid = parse_citations(text, hits, pipe.chunks_by_id)
    assert len(valid) == 1, valid
    assert valid[0].article_no == "第三条"
    assert len(invalid) == 1, invalid


def test_citation_supports_item_and_attach():
    """引用不能只认「第X条」——通知体的「一、核查周期」和「附件1」同样要能校验。"""
    retriever, emb, _ = _retriever()
    pipe = RAGPipeline(retriever, emb, None, None)

    hits = pipe.search("核查周期", top_k=10)
    valid, _ = parse_citations(
        "【《测试用动态核查工作通知》二、核查周期】", hits, pipe.chunks_by_id
    )
    assert valid and valid[0].article_no == "二、核查周期", valid

    hits2 = pipe.search("基本信息分", top_k=10)
    valid2, _ = parse_citations(
        "【《测试用建筑劳务企业信用管理办法》附件1】", hits2, pipe.chunks_by_id
    )
    assert valid2 and valid2[0].article_no == "附件1", valid2


def test_extractive_does_not_trigger_refusal():
    """回归：抽取式降级文本不能包含拒答标记，否则全部指标失真。"""
    retriever, emb, _ = _retriever()
    pipe = RAGPipeline(retriever, emb, None, None)
    ans = pipe.answer("信用等级一共分几级")
    assert REFUSAL_MARKER not in ans.answer
    assert ans.refused is False
    assert ans.citations, "抽取式也必须产出可校验引用"


# ---------------------------------------------------------------- 评测

def test_metrics_math():
    case_ok = EvalCase(id="1", question="q", gold_doc="D", gold_article="第一条")
    case_miss = EvalCase(id="2", question="q", gold_doc="D", gold_article="第二条")
    case_refuse = EvalCase(id="3", question="q", must_refuse=True, category="拒答")

    def mk(case, hit_rank, refused):
        ans = Answer(
            question=case.question,
            answer=REFUSAL_MARKER if refused else "答",
            citations=[] if refused else [Citation("c", "x", "D", "第一条")],
            refused=refused,
        )
        return EvalResult(
            case=case,
            answer=ans,
            hit_doc=hit_rank > 0,
            hit_article=hit_rank > 0,
            hit_rank=hit_rank,
            refusal_correct=(refused if case.must_refuse else not refused),
            citation_valid=not refused,
            keyword_hit=1.0 if not refused else 0.0,
        )

    results = [mk(case_ok, 1, False), mk(case_miss, 2, False), mk(case_refuse, 0, True)]
    metrics = compute_metrics(results, k=5)
    assert metrics["Recall@k_条款"] == 1.0
    assert metrics["MRR@k"] == round((1.0 + 0.5) / 2, 4)
    assert metrics["应拒答_正确拒答率"] == 1.0
    assert metrics["可答_误拒答率"] == 0.0


def test_badcase_classification():
    case = EvalCase(id="1", question="q", gold_doc="D", gold_article="第一条")
    ans = Answer(question="q", answer="答", refused=False, citations=[])
    res = EvalResult(case=case, answer=ans, hit_article=True, hit_rank=1, citation_valid=False)
    assert classify_badcase(res).startswith("D_")

    res2 = EvalResult(
        case=EvalCase(id="2", question="q", must_refuse=True),
        answer=Answer(question="q", answer="编造答案", refused=False),
    )
    assert classify_badcase(res2).startswith("A_")

    res3 = EvalResult(
        case=EvalCase(id="3", question="q", gold_doc="D", gold_article="第一条"),
        answer=Answer(question="q", answer=REFUSAL_MARKER, refused=True),
        hit_article=False,
        hit_rank=0,
        refusal_correct=False,
    )
    assert classify_badcase(res3).startswith("B_")


# ---------------------------------------------------------------- runner

def _run_all() -> int:
    tests = [
        (name, fn)
        for name, fn in sorted(globals().items())
        if name.startswith("test_") and callable(fn)
    ]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as exc:
            failed += 1
            print(f"  FAIL  {name}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  ERROR {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} 通过")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
