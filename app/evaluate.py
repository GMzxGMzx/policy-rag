"""评测与回归。

为什么这个模块比检索代码更值钱：
产品经理岗的 JD 写的是「搭建效果评测体系，设计评测集、回归测试、Badcase 闭环」。
这三件事都在这一个文件里，而且它们能被量化、能写进简历、能在面试里讲 20 分钟。

指标设计的原则是**分层归因**：不能只有一个"准确率"。
必须能回答"这次改动到底改善了哪一环、恶化哪一环"：
- 检索没召回到  → 切分/分词/embedding 的问题
- 召回到了但拒答 → 提示词或阈值的问题
- 没拒答却编了  → 拒答规则失效，最严重
- 引用了不存在的条 → 幻觉，最严重

所以指标分成 检索 / 决策 / 引用 三层，Badcase 也按层归因。
"""
from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .rag import RAGPipeline
from .schema import EvalCase, EvalResult

# ---------------------------------------------------------------- 指标

def _mean(values: list[float]) -> float:
    return round(statistics.fmean(values), 4) if values else 0.0


def compute_metrics(results: list[EvalResult], k: int) -> dict[str, Any]:
    answerable = [r for r in results if not r.case.must_refuse]
    refusal_cases = [r for r in results if r.case.must_refuse]
    answered = [r for r in answerable if not r.answer.refused]

    metrics: dict[str, Any] = {
        "样本数": len(results),
        "可答样本数": len(answerable),
        "应拒答样本数": len(refusal_cases),
        f"Recall@k_文档": _mean([1.0 if r.hit_doc else 0.0 for r in answerable]),
        f"Recall@k_条款": _mean([1.0 if r.hit_article else 0.0 for r in answerable]),
        f"MRR@k": _mean([1.0 / r.hit_rank if r.hit_rank else 0.0 for r in answerable]),
        "拒答准确率": _mean([1.0 if r.refusal_correct else 0.0 for r in results]),
        "应拒答_正确拒答率": _mean([1.0 if r.answer.refused else 0.0 for r in refusal_cases]),
        "可答_误拒答率": _mean([1.0 if r.answer.refused else 0.0 for r in answerable]),
        "平均延迟ms": int(_mean([float(r.answer.latency_ms) for r in results])),
        "k": k,
    }

    # 这两项只在「可答且实际作答」的样本上有意义。
    # 分母为 0 时直接不给值，避免把一个不适用的问题报成 0 分。
    if answered:
        metrics["引用有效率"] = _mean([1.0 if r.citation_valid else 0.0 for r in answered])
        metrics["要点命中率"] = _mean([r.keyword_hit for r in answered])

    return metrics


def by_category(results: list[EvalResult], k: int) -> dict[str, dict[str, Any]]:
    """分层指标：这是 Badcase 归因的入口。"""
    groups: dict[str, list[EvalResult]] = {}
    for r in results:
        groups.setdefault(r.case.category or "general", []).append(r)
    return {cat: compute_metrics(rows, k) for cat, rows in sorted(groups.items())}


def classify_badcase(result: EvalResult) -> str:
    """把失败样本归到一个可行动的类别。空字符串表示通过。"""
    if result.case.must_refuse:
        if not result.answer.refused:
            return "A_该拒答却作答（幻觉风险，最高优先级）"
        return ""
    if result.answer.refused:
        if not result.hit_article:
            return "B_检索未命中导致误拒答（切分/分词/embedding）"
        return "C_召回到了却拒答（提示词或阈值）"
    if not result.citation_valid:
        return "D_引用了不存在的条款（引用校验失败）"
    if not result.hit_article:
        return "E_答案对但条款没召回（运气命中，不可靠）"
    if result.keyword_hit < 0.5:
        return "F_要点覆盖不足（生成content问题）"
    return ""


# ---------------------------------------------------------------- 单条执行

def run_case(pipeline: RAGPipeline, case: EvalCase, k: int) -> EvalResult:
    answer = pipeline.answer(case.question, top_k=k)

    hit_doc = False
    hit_article = False
    hit_rank = 0
    for pos, hit in enumerate(answer.retrieved, start=1):
        clause = hit.clause
        if clause is None:
            continue
        if case.gold_doc and clause.doc_title == case.gold_doc:
            hit_doc = True
            if case.gold_article and clause.article_no == case.gold_article:
                if hit_rank == 0:
                    hit_rank = pos
                hit_article = True
                break
        elif not case.gold_doc and case.gold_article and clause.article_no == case.gold_article:
            hit_doc = True
            hit_article = True
            hit_rank = hit_rank or pos
            break

    if case.must_refuse:
        refusal_correct = answer.refused
    else:
        refusal_correct = not answer.refused

    invalid = answer.debug.get("invalid_citations") or []
    citation_valid = bool(answer.citations) and not invalid

    if case.gold_keywords:
        joined = answer.answer
        found = sum(1 for kw in case.gold_keywords if kw in joined)
        keyword_hit = found / len(case.gold_keywords)
    else:
        keyword_hit = 1.0 if answer.answer.strip() else 0.0

    return EvalResult(
        case=case,
        answer=answer,
        hit_doc=hit_doc,
        hit_article=hit_article,
        hit_rank=hit_rank,
        refusal_correct=refusal_correct,
        citation_valid=citation_valid,
        keyword_hit=keyword_hit,
    )


def run_eval(pipeline: RAGPipeline, cases: list[EvalCase], k: int = 5, verbose: bool = True) -> list[EvalResult]:
    results: list[EvalResult] = []
    for i, case in enumerate(cases, start=1):
        try:
            result = run_case(pipeline, case, k)
        except Exception as exc:  # noqa: BLE001
            if verbose:
                print(f"  [{i}/{len(cases)}] 失败 {case.id}: {exc}")
            continue
        results.append(result)
        if verbose:
            flag = "OK " if not classify_badcase(result) else "BAD"
            print(f"  [{i}/{len(cases)}] {flag} {case.id} rank={result.hit_rank} refused={result.answer.refused}")
    return results


# ---------------------------------------------------------------- 数据集

def load_cases(path: Path) -> list[EvalCase]:
    cases: list[EvalCase] = []
    if not path.exists():
        return cases
    # utf-8-sig：评测集经常被人用记事本/Excel 改过，带上 BOM 会让 json 解析直接失败
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        row = json.loads(line)
        cases.append(
            EvalCase(
                id=str(row.get("id") or f"q{len(cases) + 1}"),
                question=row["question"],
                gold_doc=row.get("gold_doc", ""),
                gold_article=row.get("gold_article", ""),
                must_refuse=bool(row.get("must_refuse", False)),
                category=row.get("category", "general"),
                gold_keywords=list(row.get("gold_keywords") or []),
                notes=row.get("notes", ""),
            )
        )
    return cases


# ---------------------------------------------------------------- 报告

def render_report(
    metrics: dict[str, Any],
    groups: dict[str, dict[str, Any]],
    results: list[EvalResult],
    config: dict[str, Any],
    baseline: dict[str, Any] | None = None,
) -> str:
    lines: list[str] = []
    lines.append("# 政策问答系统 · 评测报告\n")
    lines.append(f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    lines.append("## 1. 运行配置\n")
    lines.append("| 配置项 | 值 |")
    lines.append("|---|---|")
    for key, val in config.items():
        lines.append(f"| {key} | {val} |")
    lines.append("")

    lines.append("## 2. 总体指标\n")
    lines.append("| 指标 | 本次 | 基线 | 变化 |")
    lines.append("|---|---|---|---|")
    for key, val in metrics.items():
        base = (baseline or {}).get(key)
        if isinstance(val, float) and isinstance(base, (int, float)):
            delta = val - base
            arrow = "↑" if delta > 0 else ("↓" if delta < 0 else "→")
            lines.append(f"| {key} | {val} | {base} | {arrow} {delta:+.4f} |")
        elif base is not None:
            lines.append(f"| {key} | {val} | {base} | |")
        else:
            lines.append(f"| {key} | {val} | — | — |")
    lines.append("")

    lines.append("## 3. 分层指标\n")
    key_cols = ["样本数", "Recall@k_条款", "MRR@k", "拒答准确率", "引用有效率"]
    lines.append("| 类别 | " + " | ".join(key_cols) + " |")
    lines.append("|---|" + "---|" * len(key_cols))
    for cat, row in groups.items():
        cells = [str(row.get(col, "")) for col in key_cols]
        lines.append(f"| {cat} | " + " | ".join(cells) + " |")
    lines.append("")

    bad = [(r, classify_badcase(r)) for r in results]
    bad = [(r, cls) for r, cls in bad if cls]
    lines.append(f"## 4. Badcase 归因（{len(bad)} / {len(results)}）\n")
    if not bad:
        lines.append("本次无失败样本。\n")
    else:
        buckets: dict[str, list[EvalResult]] = {}
        for r, cls in bad:
            buckets.setdefault(cls, []).append(r)
        lines.append("| 归因类别 | 数量 | 说明 |")
        lines.append("|---|---|---|")
        for cls, rows in sorted(buckets.items()):
            lines.append(f"| {cls} | {len(rows)} | 见下方明细 |")
        lines.append("")
        for cls, rows in sorted(buckets.items()):
            lines.append(f"### {cls}\n")
            for r in rows[:10]:
                lines.append(f"- **{r.case.id}**：{r.case.question}")
                lines.append(f"  - 命中条款：{r.hit_rank or '未召回'}｜拒答：{r.answer.refused}")
                lines.append(f"  - 回答：{r.answer.answer[:180].replace(chr(10), ' ')}")
                if r.case.notes:
                    lines.append(f"  - 备注：{r.case.notes}")
            lines.append("")

    lines.append("## 5. 结论与下一步\n")
    lines.append(_advice(metrics))
    lines.append("")
    return "\n".join(lines)


def _advice(metrics: dict[str, Any]) -> str:
    tips: list[str] = []
    recall = metrics.get("Recall@k_条款", 0.0)
    mrr = metrics.get("MRR@k", 0.0)
    refused = metrics.get("可答_误拒答率", 0.0)
    cite = metrics.get("引用有效率", 0.0)
    missed = metrics.get("应拒答_正确拒答率", 1.0)

    if recall < 0.8:
        tips.append(f"- 条款召回率 {recall} 偏低：优先查切分粒度（是否把一条切成多块后丢失上下文）、"
                    "分词是否切坏了专有名词、embedding 是否用了无语义的 hash 模式。")
    if mrr < 0.6:
        tips.append(f"- MRR {mrr} 偏低说明正确答案排在后面：开启 RERANK_ENABLED 做重排，"
                    "或提高 CONTEXT_TOP_N。")
    if refused > 0.15:
        tips.append(f"- 可答问题误拒答率 {refused}：检查 SYSTEM_PROMPT 的拒答条件是否过严。")
    if cite < 0.9:
        tips.append(f"- 引用有效率 {cite}：模型在编条号。需要在提示词里强化引用格式，"
                    "并把编造引用直接判为失败（当前已计入 D 类）。")
    if missed < 0.8:
        tips.append(f"- 该拒答却没拒答的比例偏高：这是最危险的一类错误，"
                    "建议在生成前加一道检索分数阈值，低于阈值直接拒答。")
    if not tips:
        tips.append("- 各项指标达标。下一步可扩充评测集到 150+ 条，并补充跨文件冲突、"
                    "新旧版本并存、地区差异三类难样本。")
    return "\n".join(tips)


# ---------------------------------------------------------------- 落盘

def save_run(out_dir: Path, metrics: dict[str, Any], results: list[EvalResult], report: str, tag: str = "") -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    name = f"{stamp}{('-' + tag) if tag else ''}"
    (out_dir / f"{name}.metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (out_dir / f"{name}.details.jsonl").open("w", encoding="utf-8") as fh:
        for r in results:
            fh.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")
    report_path = out_dir / f"{name}.report.md"
    report_path.write_text(report, encoding="utf-8")
    return report_path


def load_baseline(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
