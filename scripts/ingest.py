"""建索引 CLI。

用法：
    python -m scripts.ingest
    python -m scripts.ingest --raw data/raw --out data/index
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import ensure_dirs, settings  # noqa: E402
from app.embedder import build_embedder  # noqa: E402
from app.index import HybridRetriever  # noqa: E402
from app.ingest import chunk_clauses, load_corpus  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="解析政策文件并构建检索索引")
    parser.add_argument("--raw", default=str(settings.raw_dir), help="政策原文目录")
    parser.add_argument("--out", default=str(settings.index_dir), help="索引输出目录")
    args = parser.parse_args()

    ensure_dirs()
    raw_dir = Path(args.raw)
    out_dir = Path(args.out)
    if not raw_dir.exists():
        print(f"目录不存在：{raw_dir}")
        return 1

    print(f"[1/4] 读取政策文件：{raw_dir}")
    clauses, metas = load_corpus(raw_dir)
    if not clauses:
        print("没有解析到条款。请把 .txt/.md/.pdf/.docx 放进 data/raw/")
        return 1

    print(f"[2/4] 条款级切分：{len(clauses)} 条")
    chunks = chunk_clauses(clauses, settings.max_clause_chars)
    long_clauses = sum(1 for c in chunks if c.total_parts > 1)
    print(f"      检索块：{len(chunks)} 个（其中 {long_clauses} 块来自超长条款拆分）")

    print("[3/4] 构建 BM25 倒排 + 向量索引")
    retriever = HybridRetriever(chunks, settings.bm25_k1, settings.bm25_b, settings.rrf_k)
    embedder = build_embedder(settings)
    print(f"      embedding provider = {embedder.name}")
    retriever.build_vectors(embedder)

    print(f"[4/4] 写入 {out_dir}")
    retriever.save(out_dir)
    (out_dir / "docs.json").write_text(
        json.dumps(metas, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n完成。文件清单：")
    for meta in metas:
        print(f"  《{meta['title']}》 {meta.get('doc_no', '')} {meta.get('clause_count', 0)} 条  "
              f"机关={meta.get('issuer') or '未识别'}  日期={meta.get('publish_date') or '未识别'}")
        if meta.get("repealed_docs"):
            print(f"    该文件声明废止：{'、'.join(meta['repealed_docs'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
