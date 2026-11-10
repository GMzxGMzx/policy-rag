"""评测 CLI（含基线对比，用于回归测试）。

用法：
    python -m scripts.evaluate                       # 跑一次评测
    python -m scripts.evaluate --k 8                 # 换 Recall@K 的 K
    python -m scripts.evaluate --baseline data/eval/runs/xxx.metrics.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings  # noqa: E402
from app.embedder import build_embedder  # noqa: E402
from app.evaluate import (  # noqa: E402
    by_category,
    compute_metrics,
    load_baseline,
    load_cases,
    render_report,
    run_eval,
    save_run,
)
from app.index import HybridRetriever  # noqa: E402
from app.llm import build_models  # noqa: E402
from app.rag import RAGPipeline  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="跑评测集并生成报告")
    parser.add_argument("--cases", default=str(settings.eval_dir / "eval_set.jsonl"))
    parser.add_argument("--index", default=str(settings.index_dir))
    parser.add_argument("--out", default=str(settings.eval_dir / "runs"))
    parser.add_argument("--k", type=int, default=settings.context_top_n)
    parser.add_argument("--baseline", default="", help="基线 metrics.json 路径，用于回归对比")
    parser.add_argument("--tag", default="", help="本次运行标签，如 bm25-only / with-rerank")
    args = parser.parse_args()

    cases = load_cases(Path(args.cases))
    if not cases:
        print(f"评测集为空：{args.cases}")
        return 1

    retriever = HybridRetriever.load(
        Path(args.index), settings.bm25_k1, settings.bm25_b, settings.rrf_k
    )
    embedder = build_embedder(settings)
    models = build_models(settings)
    pipeline = RAGPipeline(retriever, embedder, models.llm, models.reranker, settings)

    print(f"样本 {len(cases)} 条｜embedding={embedder.name}｜rerank={'on' if models.reranker else 'off'}"
          f"｜llm={'on' if models.llm else 'off(抽取式)'}")
    results = run_eval(pipeline, cases, k=args.k)

    metrics = compute_metrics(results, args.k)
    groups = by_category(results, args.k)

    config = {
        "embedding": embedder.name,
        "rerank": "on" if models.reranker else "off",
        "llm": settings.llm_model if models.llm else "抽取式降级",
        "BM25": f"k1={settings.bm25_k1}, b={settings.bm25_b}",
        "RRF_K": settings.rrf_k,
        "K": args.k,
        "检索块数": len(retriever.chunks),
    }

    baseline = load_baseline(Path(args.baseline)) if args.baseline else None
    report = render_report(metrics, groups, results, config, baseline)
    path = save_run(Path(args.out), metrics, results, report, tag=args.tag)

    print("\n===== 总体指标 =====")
    for key, val in metrics.items():
        print(f"  {key}: {val}")
    print(f"\n报告已生成：{path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
