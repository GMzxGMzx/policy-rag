"""从已建好的索引里自动生成评测集骨架。

这一步解决的是最耗时的环节：写评测集。做法是反向生成 ——
先从条款出发，让大模型"就这条款出一道题"，gold 答案天然已知。
没有配 LLM 时退化为骨架（只填 gold，问题留 TODO），你手工补问题的成本也远低于从零写。

关键约束：**评测集必须人工复核**。
大模型生成的题目常有歧义、把条款原文当问题、或存在多个正确答案。
自动生成只负责把 150 条的工作量压到 30 条。
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings  # noqa: E402
from app.index import HybridRetriever  # noqa: E402
from app.llm import build_models  # noqa: E402

QUESTION_PROMPT = """你在为「工程政策问答系统」编写评测集。给你一条政策条款，请出一道**真实用户会问**的问题。

要求：
1. 问题必须能且只能由这条条款回答，不要引入条款外的信息。
2. 用口语化提问，模拟基层监管人员或企业经办人的真实问法，不要照抄条款原文措辞。
3. 不要出现"根据第X条"这类提示，否则等于泄露答案。
4. 只输出 JSON：{"question": "...", "keywords": ["要点1", "要点2", "要点3"]}
   keywords 是标准答案里必须出现的关键词或短语，用于衡量回答要点覆盖率，3 到 5 个。

条款：
【{header}】
{body}
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="生成评测集骨架")
    parser.add_argument("--index", default=str(settings.index_dir))
    parser.add_argument("--out", default=str(settings.eval_dir / "eval_set.jsonl"))
    parser.add_argument("--n", type=int, default=30, help="生成多少条")
    parser.add_argument("--refuse", type=int, default=5, help="额外插入多少条应拒答样本")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--append", action="store_true", help="追加到已有评测集")
    args = parser.parse_args()

    retriever = HybridRetriever.load(Path(args.index))
    chunks = retriever.chunks
    if not chunks:
        print("索引为空，请先执行 python -m scripts.ingest")
        return 1

    # 前言通常是发文信息（标题/文号/机关），不适合出题；过短的条款也没有出题价值
    chunks = [
        c for c in chunks
        if c.clause.article_no != "前言" and len(c.body) >= 25 and not c.body.startswith("【")
    ]
    if not chunks:
        print("没有适合出题的条款（已过滤前言与过短条款）")
        return 1

    random.seed(args.seed)
    # 按文件分层抽样，避免全来自同一份文件
    by_doc: dict[str, list] = {}
    for chunk in chunks:
        by_doc.setdefault(chunk.clause.doc_title, []).append(chunk)
    per_doc = max(1, args.n // max(1, len(by_doc)))
    sampled = []
    for doc, rows in by_doc.items():
        sampled.extend(random.sample(rows, min(per_doc, len(rows))))
    sampled = sampled[: args.n]

    models = build_models(settings)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.append else "w"

    rows: list[dict] = []
    for i, chunk in enumerate(sampled, start=1):
        question = ""
        keywords: list[str] = []
        if models.llm is not None:
            try:
                data = models.llm.chat_json(
                    "你是一个严谨的评测集编写助手，只输出 JSON。",
                    QUESTION_PROMPT.format(header=chunk.header, body=chunk.body[:900]),
                )
                question = str(data.get("question", "")).strip()
                keywords = [str(k).strip() for k in (data.get("keywords") or []) if str(k).strip()]
            except Exception as exc:  # noqa: BLE001
                print(f"  [警告] 第 {i} 条生成失败：{exc}")
        if not question:
            question = f"TODO：针对《{chunk.clause.doc_title}》{chunk.clause.article_no} 写一个真实提问"

        rows.append(
            {
                "id": f"auto-{i:03d}",
                "question": question,
                "gold_doc": chunk.clause.doc_title,
                "gold_article": chunk.clause.article_no,
                "must_refuse": False,
                "category": "auto",
                "gold_keywords": keywords,
                "notes": "",
            }
        )
        print(f"  [{i}/{len(sampled)}] {question[:60]}")

    # 应拒答样本：题库里没有答案的问题，用来检验会不会编
    refuse_samples = [
        ("在苏州做建筑劳务分包，需要给工人缴纳住房公积金吗？", "题库无公积金条款"),
        ("2027 年江苏省建筑工人最低工资标准是多少？", "题库无工资标准"),
        ("外资企业申请建筑施工总承包一级资质需要多少净资产？", "题库未收录资质标准"),
        ("施工现场发生死亡事故，项目经理会被追究刑事责任吗？", "涉及刑法，题库不覆盖"),
        ("雄安新区的建筑劳务企业备案流程是什么？", "跨地区，题库仅覆盖苏州"),
    ]
    for j in range(min(args.refuse, len(refuse_samples))):
        q, note = refuse_samples[j]
        rows.append(
            {
                "id": f"refuse-{j + 1:03d}",
                "question": q,
                "gold_doc": "",
                "gold_article": "",
                "must_refuse": True,
                "category": "拒答",
                "gold_keywords": [],
                "notes": note,
            }
        )

    with out_path.open(mode, encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    todo = sum(1 for r in rows if r["question"].startswith("TODO"))
    print(f"\n已写入 {out_path}（{'追加' if args.append else '覆盖'}）共 {len(rows)} 条")
    if todo:
        print(f"其中 {todo} 条为骨架，需要你手工补问题。")
    print("下一步：人工复核每一题（是否可答、是否只有一个正确答案），然后执行")
    print("  python -m scripts.evaluate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
