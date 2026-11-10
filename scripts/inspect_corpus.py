"""语料体检：在花时间调检索之前，先搞清楚语料本身健不健康。

真实项目里，检索效果差有一半原因不在算法，而在语料：
扫描件抽不出字、文件名乱导致标题识别错、同一份文件有多个版本、报告类文档没有条款结构。
这些必须在建索引之前看清楚，否则后面所有指标都建在沙子上。

用法：
    python -m scripts.inspect_corpus
    python -m scripts.inspect_corpus --raw data/raw --out data/reports/corpus_report.md
"""
from __future__ import annotations

import argparse
import hashlib
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings  # noqa: E402
from app.ingest import iter_source_files, parse_file  # noqa: E402


def _page_count(path: Path) -> int:
    if path.suffix.lower() != ".pdf":
        return 0
    try:
        from pypdf import PdfReader

        return len(PdfReader(str(path)).pages)
    except Exception:  # noqa: BLE001
        return 0


def _verdict(meta: dict, clauses: list) -> str:
    """判定文体。只认「第X条」会把通知体误判成"无结构"，而通知体恰恰是最常见的一类。"""
    if meta.get("needs_ocr"):
        return "需OCR（扫描件）"
    if not clauses:
        return "无内容"

    n_article = sum(
        1 for c in clauses if c.article_no.startswith("第") and c.article_no.endswith("条")
    )
    n_item = sum(1 for c in clauses if re.match(r"^[一二三四五六七八九十]{1,3}、", c.article_no))
    n_num = sum(1 for c in clauses if re.fullmatch(r"\d{1,2}", c.article_no))
    n_attach = sum(1 for c in clauses if c.article_no.startswith("附件"))

    if n_article >= 5:
        return f"条例体（{n_article} 条）"
    if n_item >= 3:
        return f"通知体（{n_item} 节）"
    if n_num >= 3:
        return f"数字编号体（{n_num} 项）"
    if n_attach >= 2:
        return f"附件体（{n_attach} 个附件）"
    if len(clauses) >= 3:
        return "段落型（无结构标题）"
    return "内容过少"


def main() -> int:
    parser = argparse.ArgumentParser(description="语料健康检查")
    parser.add_argument("--raw", default=str(settings.raw_dir))
    parser.add_argument("--out", default=str(Path(settings.data_dir) / "reports" / "corpus_report.md"))
    args = parser.parse_args()

    raw_dir = Path(args.raw)
    files = iter_source_files(raw_dir)
    if not files:
        print(f"没有可解析文件：{raw_dir}")
        return 1

    rows: list[dict] = []
    for path in files:
        try:
            clauses, meta = parse_file(path)
            err = ""
        except Exception as exc:  # noqa: BLE001
            clauses, meta, err = [], {}, str(exc)

        article_count = sum(1 for c in clauses if c.article_no.startswith("第"))
        digest = ""
        if clauses:
            digest = hashlib.sha1(
                "".join(c.body for c in clauses).encode("utf-8")
            ).hexdigest()[:10]

        rows.append(
            {
                "file": path.name,
                "pages": _page_count(path),
                "chars": meta.get("chars", 0),
                "clauses": len(clauses),
                "articles": article_count,
                "title": meta.get("title", ""),
                "doc_no": meta.get("doc_no", ""),
                "issuer": meta.get("issuer", ""),
                "date": meta.get("publish_date", ""),
                "eff": meta.get("effective_date", ""),
                "repealed": len(meta.get("repealed_docs") or []),
                "verdict": _verdict(meta, clauses) if not err else f"读取失败：{err}",
                "digest": digest,
            }
        )

    # ---- 控制台输出 ----
    print(f"\n语料体检：{raw_dir}  共 {len(rows)} 个文件\n")
    print(f"{'文件':40s} {'页':>3} {'字符':>7} {'条款':>4} {'条式':>4}  状态")
    print("-" * 96)
    for r in rows:
        name = r["file"] if len(r["file"]) <= 38 else r["file"][:36] + ".."
        print(f"{name:40s} {r['pages']:>3} {r['chars']:>7} {r['clauses']:>4} {r['articles']:>4}  {r['verdict']}")

    # ---- 重复检测 ----
    groups: dict[str, list[str]] = defaultdict(list)
    for r in rows:
        if r["digest"]:
            groups[r["digest"]].append(r["file"])
    dups = {k: v for k, v in groups.items() if len(v) > 1}

    # ---- Markdown 报告 ----
    lines = ["# 语料体检报告\n", f"目录：`{raw_dir}`　文件数：{len(rows)}\n"]
    ok = [r for r in rows if r["verdict"].startswith(("条例体", "通知体", "数字编号体"))]
    attach = [r for r in rows if r["verdict"].startswith("附件体")]
    ocr = [r for r in rows if "需OCR" in r["verdict"]]
    noclause = [r for r in rows if r["verdict"].startswith(("段落型", "内容过少"))]
    lines.append(f"- 可结构化问答（条例体/通知体）：{len(ok)}")
    lines.append(f"- 仅附件体：{len(attach)}")
    lines.append(f"- 需 OCR（扫描件）：{len(ocr)}")
    lines.append(f"- 无结构标题（报告/名单）：{len(noclause)}")
    lines.append("")

    lines.append("## 文件明细\n")
    lines.append("| 文件 | 页 | 字符 | 条款 | 条式 | 标题识别 | 文号 | 状态 |")
    lines.append("|---|---:|---:|---:|---:|---|---|---|")
    for r in rows:
        lines.append(
            f"| {r['file'][:44]} | {r['pages']} | {r['chars']} | {r['clauses']} | {r['articles']} "
            f"| {r['title'][:22]} | {r['doc_no']} | {r['verdict']} |"
        )
    lines.append("")

    if dups:
        lines.append("## 疑似重复文件\n")
        for files_ in dups.values():
            lines.append("- " + " ／ ".join(files_))
        lines.append("")
        lines.append("> 内容指纹相同，通常是同一份文件既收了通知又收了附件，或收了印刷稿和解读稿。"
                     "重复入库会让检索结果被同一内容占满，建议只保留一份。\n")

    lines.append("## 建议动作\n")
    if ocr:
        lines.append(f"1. **{len(ocr)} 份扫描件需 OCR**，否则这部分政策完全不可用：")
        for r in ocr:
            lines.append(f"   - {r['file']}")
        lines.append("   可选：本地 PaddleOCR / Tesseract 批量转文本后另存为 .txt 放进 raw；"
                     "或先用 PDF 转图片工具导出 JPG，再走视觉模型识别。")
    if noclause:
        lines.append(f"\n2. **{len(noclause)} 份无条款结构**，命中它们时引用会落到“段N”而不是第X条：")
        for r in noclause:
            lines.append(f"   - {r['file']}")
        lines.append("   如果它们只提供统计数据，可以保留；如果只是名单，建议移出 raw。")
    if not ocr and not noclause:
        lines.append("- 语料结构良好，可以直接建索引。")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"\n可用条款文件 {len(ok)} / {len(rows)}")
    if ocr:
        print(f"需 OCR：{len(ocr)} 份")
    if dups:
        print(f"疑似重复：{len(dups)} 组")
    print(f"\n报告已写入：{out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
