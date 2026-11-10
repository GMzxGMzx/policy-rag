"""把政府网站的网页正文转换成规范的语料 txt。

为什么需要这一步：从官网复制正文，能绕开扫描件 OCR，是补齐语料最快的路径。
但直接粘贴会带上大量噪声（导航、索引号表格、分享按钮），而且**标题和文号经常丢失**——
这两个字段一旦丢了，下游的引用和元数据全乱。

本工具做三件事：
1. 剥掉 HTML 结构与页面噪声
2. 复用 ingest 的元数据抽取逻辑，保证与解析器口径一致
3. 输出带规范头部的 txt（标题／文号／机关／日期 + 空行 + 正文）

用法：
    # 保存的网页（浏览器 Ctrl+S）
    python -m scripts.import_web_text --file page.html

    # 已经复制好的纯文本
    python -m scripts.import_web_text --file body.txt

    # 手工指定字段（自动识别不准时）
    python -m scripts.import_web_text --file body.txt --title "关于…的通知" --doc-no "苏建函建〔2024〕63号"
"""
from __future__ import annotations

import argparse
import re
import sys
from html.parser import HTMLParser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings  # noqa: E402
from app.ingest import extract_meta, find_effective_date, normalize_text  # noqa: E402

# 页面噪声：整行匹配即丢弃。注意不要误伤「附件：」这类有信息量的行。
_NOISE_PATTERNS = [
    r"^当前位置", r"^索引号", r"^发布机构", r"^成文日期", r"^信息名称", r"^主题分类",
    r"^文号[：:]\s*$", r"^浏览次数", r"^分享到", r"^打印$", r"^关闭窗口$",
    r"^【我要打印】", r"^相关文件", r"^相关专题", r"^相关报道", r"^图解", r"^解读$",
    r"^视频[：:]?$", r"^附件下载", r"^扫一扫", r"^上一篇", r"^下一篇",
    r"^来源[：:]", r"^字号", r"^大\s*中\s*小", r"^字体[：:]", r"^[Aa]\+?$",
    r"^[—\-－]{2,}$", r"^\d{1,4}$",
]
_NOISE = [re.compile(p) for p in _NOISE_PATTERNS]


class _HTMLText(HTMLParser):
    BLOCK = {
        "p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6",
        "section", "article", "table", "ul", "ol", "blockquote",
    }
    SKIP = {"script", "style", "noscript", "head", "title"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):  # noqa: ANN001
        if tag in self.SKIP:
            self._skip += 1
        elif tag in self.BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag):  # noqa: ANN001
        if tag in self.SKIP and self._skip:
            self._skip -= 1
        elif tag in self.BLOCK:
            self.parts.append("\n")

    def handle_data(self, data):  # noqa: ANN001
        if not self._skip:
            self.parts.append(data)

    def text(self) -> str:
        raw = "".join(self.parts)
        lines = []
        for line in raw.split("\n"):
            line = re.sub(r"[ \t\u3000\xa0]+", " ", line).strip()
            lines.append(line)
        return "\n".join(lines)


def html_to_text(html: str) -> str:
    parser = _HTMLText()
    parser.feed(html)
    parser.close()
    return parser.text()


_META_LABEL = re.compile(r"^(索引号|发布机构|成文日期|信息名称|文号|主题分类|来源|发文机关)\s*[：:]\s*(.*)$")
_DROP_LABELS = {"索引号", "主题分类"}


def strip_noise_lines(text: str) -> str:
    kept: list[str] = []
    for line in text.split("\n"):
        if not line:
            kept.append("")
            continue
        m = _META_LABEL.match(line)
        if m:
            label, value = m.group(1), m.group(2).strip()
            if label in _DROP_LABELS or not value:
                continue
            # 「发布机构:苏州市住房和城乡建设局」→ 只留值，
            # 否则机关字段会识别成「发布机构:苏州市住房和城乡建设局」
            line = value
        if any(p.search(line) for p in _NOISE):
            continue
        kept.append(line)
    out = "\n".join(kept)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


def _safe_filename(name: str) -> str:
    name = re.sub(r'[\\/:*?"<>|\r\n\t]+', "-", name).strip(" -")
    return name[:80] or "未命名文件"


def main() -> int:
    parser = argparse.ArgumentParser(description="网页/纯文本 → 规范语料 txt")
    parser.add_argument("--file", required=True, help="HTML 或纯文本文件")
    parser.add_argument("--out", default=str(settings.raw_dir), help="输出目录")
    parser.add_argument("--title", default="")
    parser.add_argument("--doc-no", dest="doc_no", default="")
    parser.add_argument("--issuer", default="")
    parser.add_argument("--date", default="")
    parser.add_argument("--dry-run", action="store_true", help="只打印识别结果，不落盘")
    args = parser.parse_args()

    src = Path(args.file)
    if not src.exists():
        print(f"文件不存在：{src}")
        return 1

    raw = src.read_text(encoding="utf-8-sig", errors="replace")
    text = html_to_text(raw) if src.suffix.lower() in {".html", ".htm"} else raw
    text = strip_noise_lines(normalize_text(text))
    if len(text) < 100:
        print(f"正文过短（{len(text)} 字），可能是动态加载页面。请在浏览器里选中正文后复制，另存为 .txt 再试。")
        return 1

    # 复用解析器的元数据逻辑，保证与 ingest 口径一致
    out_name = _safe_filename(args.title or src.stem)
    meta = extract_meta(text, Path(out_name))
    title = args.title.strip() or meta["title"]
    doc_no = args.doc_no.strip() or meta["doc_no"]
    issuer = args.issuer.strip() or meta["issuer"]
    publish = args.date.strip() or meta["publish_date"]
    effective = find_effective_date(text)

    if not args.title and meta["title"] in {out_name, Path(out_name).name}:
        print("  [提示] 未能从正文识别出标题（可能页面的标题被拆成多行），建议用 --title 指定")

    header = [title]
    if doc_no:
        header.append(doc_no)
    if issuer:
        header.append(issuer)
    if publish:
        header.append(publish)
    final = "\n".join(header) + "\n\n" + text

    print("=" * 70)
    print(f"来源文件：{src.name}")
    print(f"识别标题：{title}")
    print(f"识别文号：{doc_no or '（未识别）'}")
    print(f"发文机关：{issuer or '（未识别）'}")
    print(f"发布日期：{publish or '（未识别）'}｜施行日期：{effective or '（未识别）'}")
    print(f"正文字数：{len(text)}")
    print("-" * 70)
    print(text[:300].replace("\n", " / "))
    print("=" * 70)

    if args.dry_run:
        print("dry-run，未写入文件")
        return 0

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{_safe_filename(title)}.txt"
    if target.exists():
        target = out_dir / f"{_safe_filename(title)}-imported.txt"
    target.write_text(final, encoding="utf-8")
    print(f"已写入：{target}")
    print("下一步：python -m scripts.ingest  然后  python -m scripts.evaluate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
