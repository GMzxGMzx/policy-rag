"""文档接入：解析 → 清洗 → 条款级切分。

为什么这层值得单独做（也是面试中最能讲的部分）：

1. 政策文件的结构是 章-节-条-款-项 五级，而「条」是法律上可独立引用的最小单位。
   通用 RAG 的固定长度切分会把一条完整规则切断，导致引用错位 —— 在合规场景里这是致命错误。
2. 条款编号是中文数字，还有「第X条之一」这种修订增补形态。
3. PDF 转文本会丢掉换行、把页眉页脚页码混进正文。
4. 存在已废止/被替代的条款，必须识别出来，否则系统会自信地引用一份失效文件。

切分策略的选择直接决定检索质量，这里的默认值是「条款级 + 超长按项再切」，
并且每块都带上上下文头（文件名/条号/机关/日期），因为大量提问会带时间和发文机关线索。
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from .schema import Chunk, Clause

# ---------------------------------------------------------------- 常量与正则

_CN = "零〇一二三四五六七八九十百千两0-9０-９"
_CN_NO_DIGIT = "零〇一二三四五六七八九十百千两"

ARTICLE_PAT = re.compile(rf"第[{_CN}]{{1,8}}条(?:之[{_CN_NO_DIGIT}]{{1,2}})?")
CHAPTER_PAT = re.compile(rf"第[{_CN}]{{1,6}}章")
SECTION_PAT = re.compile(rf"第[{_CN}]{{1,6}}节")

DOC_NO_PAT = re.compile(r"[\u4e00-\u9fa5]{2,12}[〔\[（(]\s*20\d{2}\s*[〕\]）)]\s*第?\s*\d{1,4}\s*号")
DATE_PAT = re.compile(r"(20\d{2})\s*[年\-\.]\s*(\d{1,2})\s*[月\-\.]\s*(\d{1,2})\s*日?")
TITLE_PAT = re.compile(r"《([^》]{4,60})》")
PAGE_NO_PAT = re.compile(r"^\s*[-—－]?\s*\d{1,4}\s*[-—－]?\s*$")
HEADER_FOOTER_PAT = re.compile(r"^\s*(第\s*\d+\s*页\s*[／/]?\s*共?\s*\d*\s*页?|共\s*\d+\s*页)\s*$")

ITEM_SPLIT_PAT = re.compile(r"(?=[（(][一二三四五六七八九十0-9]{1,4}[）)])")

REGION_KEYWORDS = [
    "江苏省", "苏州市", "南京", "无锡", "常州", "南通", "徐州", "扬州", "盐城", "泰州",
    "镇江", "淮安", "连云港", "宿迁", "常熟", "张家港", "昆山", "太仓", "吴江",
    "相城区", "高新区", "工业园区", "吴中区", "姑苏区",
]

_ACCEPT_BEFORE = set("\n\r。；;：::　 、，,")

# 中文标点也按 CJK 处理，避免 reflow 时在「〔」和数字之间插入空格
_CJK_PUNCT = set("〔〕（）()、，。；：！“”‘’《》【】…—－·％／／%")
_CJK_RANGE = ("\u4e00", "\u9fff")


def _is_cjk(ch: str) -> bool:
    return _CJK_RANGE[0] <= ch <= _CJK_RANGE[1] or ch in _CJK_PUNCT


# ---------------------------------------------------------------- 文本清洗


def reflow(text: str) -> str:
    """把 PDF 抽取产生的硬换行合并成自然段落。

    为什么必须做：PDF 每行都带换行，抽出来是
        "满分 30 分）、劳务能力分（满分 20 分）…"
    这种被切碎的行会污染切块、降低 BM25 词频统计质量，
    也会让引用返回一堆破碎的换行。
    中文行尾直接拼接，英文行尾补空格；空行保留为段落分隔。
    """
    merged: list[str] = []
    for raw in text.split("\n"):
        line = raw.strip()
        if not line:
            if merged and merged[-1] != "":
                merged.append("")
            continue
        if merged and merged[-1] != "":
            prev = merged[-1][-1]
            joiner = "" if (_is_cjk(prev) or _is_cjk(line[0])) else " "
            merged[-1] = merged[-1] + joiner + line
        else:
            merged.append(line)
    return "\n".join(merged).strip()

def normalize_text(raw: str) -> str:
    """统一换行与空白，做最小必要规整。不做全半角折叠，避免破坏条款编号。"""
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u3000", " ")          # 全角空格
    text = text.replace("\ufeff", "").replace("\u200b", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def strip_noise(text: str) -> str:
    """去掉页码行、页眉页脚行。政策 PDF 里这两类噪声最容易污染条款正文。"""
    keep: list[str] = []
    for line in text.split("\n"):
        if PAGE_NO_PAT.match(line):
            continue
        if HEADER_FOOTER_PAT.match(line):
            continue
        keep.append(line)
    return "\n".join(keep)


def _is_marker_boundary(text: str, pos: int) -> bool:
    """marker 前面必须是行首或标点，防止把“依据第三章规定”当成章标题。"""
    if pos == 0:
        return True
    prev = text[pos - 1]
    return prev in _ACCEPT_BEFORE


# ---------------------------------------------------------------- 元数据

_TITLE_SUFFIXES = ("办法", "规定", "细则", "通知", "意见", "方案", "条例", "规范", "指南", "决定", "公告", "规则")

# 标题里应该出现的词。用来挡掉「《营业执照》」这类正文里的普通书名号。
_DOC_TITLE_KEYWORDS = (
    "办法", "规定", "细则", "通知", "意见", "方案", "报告", "公告", "名单",
    "规划", "指南", "规范", "条例", "决定", "规则", "标准", "要求", "批复",
)
# 这些词出现在书名号前面，说明它引用的是**别的**文件，不是本文标题
_REFERENCE_MARKERS = ("根据", "依据", "按照", "印发", "对照", "参照", "执行", "落实", "贯彻", "遵守", "符合")


def _looks_like_title(text: str) -> bool:
    return 6 <= len(text) <= 80 and any(k in text for k in _DOC_TITLE_KEYWORDS)


def _clean_filename_title(stem: str) -> str:
    """从文件名里抠标题。

    文件名往往形如「16 苏人社发〔2022〕149 号-关于公布…的通知」，
    需要剥掉序号前缀和文号前缀，但**不能把年份当序号剥掉** ——
    早期版本把「2021年度江苏建筑业发展报告」剥成了「年度江苏建筑业发展报告」。
    """
    s = stem
    s = re.sub(r"^\d{1,2}[\.\、\s]\s*", "", s)  # 只剥离「16 」「1. 」这类短序号
    # 「（印刷稿_(5)」这类后缀会一路带进标题，直接从该标记切掉
    s = re.sub(r"[（(]\s*(印刷稿|印刷版|修改稿|送审稿|征求意见稿)[^）)]*[）)]?.*$", "", s)
    s = re.sub(r"^(修订版|正式版|终版|最新版)[—\-－:：]\s*", "", s)
    m = re.search(r"号\s*[-—－]\s*(.+)$", s)
    if m:
        s = m.group(1)
    else:
        m = re.search(r"号\s+(.+)$", s)
        if m and _looks_like_title(m.group(1)):
            s = m.group(1)
    s = re.sub(r"[（(]?(正式稿|征求意见稿|印发稿|定稿)[）)]?$", "", s)
    return s.strip(" -—_　")


def _pick_title_ref(head: str) -> str:
    """从书名号里挑「本文自己」的标题。

    通知正文里会引用一堆别的文件（「根据《XXX》要求」），直接取第一个书名号必然踩坑：
    实测把《省人力资源社会保障厅关于开展…评定工作的通知》当成了本文标题，
    而这实际上是它引用的依据文件。
    """
    scope = _header_scope(head)
    for m in TITLE_PAT.finditer(scope):
        before = scope[max(0, m.start() - 14) : m.start()]
        if any(k in before for k in _REFERENCE_MARKERS):
            continue
        candidate = m.group(1).strip()
        if _looks_like_title(candidate):
            return candidate
    return ""


def _header_scope(head: str) -> str:
    """标题只可能出现在正文之前，用第一个结构标记当右边界。

    这条约束是必要的：正文里会引用大量别的文件名（「《测试用暂行办法》同时废止」），
    如果不限定范围，就会把被废止的那份文件当成本文标题。
    """
    cut = len(head)
    for pat in (CHAPTER_PAT, ARTICLE_PAT):
        m = pat.search(head)
        if m and m.start() < cut:
            cut = m.start()
    m = _ITEM_PAT.search(head)
    if m and m.start() < cut:
        cut = m.start()
    return head[:cut]


def _title_from_lines(head: str) -> str:
    """很多政策 txt 的标题行不带书名号，靠后缀特征识别。

    这是真实语料里的高频情况：从网站复制正文时书名号经常丢失。
    含书名号的行一律跳过 —— 那种行通常是「关于印发《X》的通知」的碎片。
    """
    for line in head.split("\n")[:15]:
        line = line.strip()
        if not line or len(line) < 6 or len(line) > 80:
            continue
        if any(ch in line for ch in "【】〔〕《》"):
            continue
        if re.search(r"号\s*$", line):
            continue
        # 只排除「整行就是一个日期」的情况。不能因为标题里含年份就跳过 ——
        # 「关于做好2024年第一季度…的通知」这类标题非常常见。
        if re.fullmatch(r"\s*20\d{2}\s*[年\-\./]\s*\d{1,2}\s*([月\-\./]\s*\d{1,2}\s*日?)?\s*", line):
            continue
        if line.endswith(_TITLE_SUFFIXES):
            return line
    return ""


def extract_meta(text: str, path: Path) -> dict[str, Any]:
    """从正文头部和文件名推断文件级元数据。启发式，不追求 100% 准确，但要求可解释。"""
    head = text[:1200]

    title = _pick_title_ref(head)
    if not title:
        title = _title_from_lines(_header_scope(head))
    # 文件名是人工命名的，往往比从碎片化 PDF 文本里拼出来的更完整。
    # 判定条件：文件名版本是「更完整」的那个（行内标题是它的子串），才采信文件名。
    from_name = _clean_filename_title(path.stem)
    if from_name and _looks_like_title(from_name):
        if not title or (title and title in from_name and len(from_name) > len(title)):
            title = from_name
    title = title or path.name
    # PDF 里标题常常跨行，抽出来会带换行，必须先并成一行再入库。
    # 否则引用里会出现《建筑工人职业技能等级评价 \n实施细则》，也会让 gold_doc 匹配失败。
    if any("\u4e00" <= ch <= "\u9fff" for ch in title):
        title = re.sub(r"\s+", "", title)

    doc_no = ""
    found = DOC_NO_PAT.search(head)
    if found:
        doc_no = re.sub(r"\s+", "", found.group(0))

    issuer = ""
    # 取文号所在行前面那段作为发文机关；退化为在头部找常见机关后缀
    if found:
        line_start = head.rfind("\n", 0, found.start()) + 1
        prefix = head[line_start:found.start()].strip()
        if 2 <= len(prefix) <= 30:
            issuer = prefix
    if not issuer:
        for line in head.split("\n")[:8]:
            line = line.strip()
            if re.search(r"(局|厅|委|政府|办公室|办公厅|管委会|中心|集团|协会|联合会|委员会|部|总局)$", line) and len(line) <= 30:
                issuer = line
                break

    publish_date = ""
    found = DATE_PAT.search(head)
    if found:
        y, m, d = found.groups()
        publish_date = f"{int(y):04d}-{int(m):02d}-{int(d):02d}"

    haystack = f"{issuer} {title} {path.name}"
    region = ""
    for kw in REGION_KEYWORDS:
        if kw in haystack:
            region = kw
            break

    return {
        "title": title,
        "doc_no": doc_no,
        "issuer": issuer,
        "publish_date": publish_date,
        "region": region,
    }


def find_effective_date(text: str) -> str:
    """找施行日期，用于回答“什么时候生效”类问题。"""
    for pat in (r"自(20\d{2})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日起(?:施行|执行|实施)",
                r"(20\d{2})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日起(?:施行|执行|实施)"):
        found = re.search(pat, text)
        if found:
            y, m, d = found.groups()
            return f"{int(y):04d}-{int(m):02d}-{int(d):02d}"
    return ""


def find_repealed_docs(text: str) -> list[str]:
    """抽取「下列文件同时废止」清单里的文件名。

    这是一条真实的合规痛点：新旧文件并存时，系统必须知道哪份已经失效。
    """
    names: list[str] = []
    for seg in re.split(r"[。\n]", text):
        if re.search(r"(废止|失效|停止执行)", seg) and "《" in seg:
            names.extend(m.strip() for m in TITLE_PAT.findall(seg) if m.strip())
    # 去重保序
    seen: set[str] = set()
    out = []
    for n in names:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def mark_repealed(body: str) -> tuple[bool, str]:
    """判断单条是否自述失效。只认显式表述，宁可漏判也不误判 —— 误判会让有效条款消失。"""
    if len(body) > 400:
        return False, ""
    m = re.search(r"(本条|本款|本项).{0,12}(废止|失效|停止执行)", body)
    if m:
        return True, body[:120]
    m = re.search(r"(同时|一并)(废止|停止执行)", body)
    if m:
        return True, body[:120]
    return False, ""


# ---------------------------------------------------------------- 结构切分

_ATTACH_PAT = re.compile(r"^[ \t]*附件\s*[0-9０-９]{0,2}\s*[：:、]?[^\n]*", re.M)
_ITEM_PAT = re.compile(r"^[ \t]*[一二三四五六七八九十]{1,3}\s*、[^\n]*", re.M)
# 「1.」「2、」这类数字编号，只在既无条号、又无中文序号时才启用（见 split_clauses）
_NUM_ITEM_PAT = re.compile(r"^[ \t]*\d{1,2}\s*[\.．、][^\n]*", re.M)


_NUM_ITEM_PREFIX = re.compile(r"^[ \t]*(\d{1,2})\s*[\.．、]", re.M)


def _num_item_markers(text: str) -> list[tuple[int, int, str]]:
    """数字编号（1. 2. 3.）的小节。

    与「一、二、三」不同：数字编号后面往往直接跟正文而不是短标题，
    所以 marker 只覆盖编号本身，绝不覆盖整行 —— 否则正文会被切进条号字段丢掉。
    """
    return [(m.start(), m.end(), "numitem") for m in _NUM_ITEM_PREFIX.finditer(text)]


def _heading_markers(
    text: str, pat: "re.Pattern", kind: str, max_head: int = 40
) -> list[tuple[int, int, str]]:
    """标题型 marker：整行作为标题，行尾即正文起点。

    只接受「短行」当标题。若整行很长，说明是正文里恰好以编号开头的句子，
    不能当标题 —— 否则会把一大段正文塞进条号字段，而正文反而丢了。
    """
    out: list[tuple[int, int, str]] = []
    for m in pat.finditer(text):
        nl = text.find("\n", m.start())
        line_end = nl + 1 if nl != -1 else len(text)
        heading = text[m.start() : line_end].strip()
        if len(heading) > max_head:
            continue
        out.append((m.start(), line_end, kind))
    return out


def _heading_no(raw: str) -> str:
    """把标题行压成条号：「附件1：建筑劳务企业基本信息分…」->「附件1」。"""
    head = re.split(r"[：:]", raw.strip(), maxsplit=1)[0]
    return re.sub(r"\s+", "", head)[:40]


def _find_markers(text: str) -> list[tuple[int, int, str]]:
    """返回 [(start, end, kind)]，kind ∈ {chapter, section, article, attach}。"""
    markers: list[tuple[int, int, str]] = []
    for kind, pat in (("chapter", CHAPTER_PAT), ("section", SECTION_PAT), ("article", ARTICLE_PAT)):
        for m in pat.finditer(text):
            if _is_marker_boundary(text, m.start()):
                markers.append((m.start(), m.end(), kind))

    # 附件是硬边界。不加这一条，最后一个条款会把后面整整二十页附件全部吞掉，
    # 实测出现过单条 6417 字的畸形条款，检索和引用都会失效。
    # 附件标题可能很长（「附件：苏建函建管〔2023〕371号《…》」），所以放宽行长限制。
    markers.extend(_heading_markers(text, _ATTACH_PAT, "attach", max_head=150))

    markers.sort(key=lambda x: x[0])
    # 丢弃重叠（例如“第X条”被“第X章”覆盖的极端情况）
    cleaned: list[tuple[int, int, str]] = []
    for mk in markers:
        if cleaned and mk[0] < cleaned[-1][1]:
            continue
        cleaned.append(mk)
    return cleaned


def _heading_text(text: str, start: int, end: int) -> str:
    """章/节标题取到行尾为止。"""
    nl = text.find("\n", end)
    stop = nl if nl != -1 and nl - end < 40 else min(end + 30, len(text))
    return text[start:stop].strip()


def split_clauses(text: str, meta: dict[str, Any], path: Path) -> list[Clause]:
    """按标记把全文切成条款序列，并继承所属章节。

    真实政策文件有三类结构，必须分别处理：
    - 条例/办法/细则型：章-节-条，以「第X条」为引用单位（最理想）
    - 通知/意见型：多用「一、二、三」编号，没有条号，以带标题的小节为引用单位
    - 报告/名单/表格型：无结构，按段落切，引用退化到「段N」

    只认「第X条」不够用：实测 5 份可解析文本里只有 1 份是条例体，
    其余 4 份是通知体，如果不支持「一、二、三」，它们会各自退化成一个大段落。
    """
    doc_id = hashlib.sha1(f"{path.name}::{meta['title']}".encode("utf-8")).hexdigest()[:12]
    base = _find_markers(text)
    has_article = any(kind == "article" for _, _, kind in base)

    if has_article:
        markers = base
    else:
        item_marks = _heading_markers(text, _ITEM_PAT, "item")
        if not item_marks:
            # 既没有条号、也没有「一、二、三」时，才退到「1. 2. 3.」数字编号。
            # 分层是刻意的：混用会把本来分节正常的文档过度切碎。
            item_marks = _num_item_markers(text)
        markers = sorted(base + item_marks, key=lambda x: x[0])
        deduped: list[tuple[int, int, str]] = []
        for mk in markers:
            if deduped and mk[0] < deduped[-1][1]:
                continue
            deduped.append(mk)
        markers = deduped

    clauses: list[Clause] = []
    chapter = ""
    section = ""

    def build(article_no: str, body: str) -> Clause:
        body = reflow(body)
        is_rep, hint = mark_repealed(body)
        return Clause(
            doc_id=doc_id,
            doc_title=meta["title"],
            article_no=article_no,
            body=body.strip(),
            chapter=chapter,
            section=section,
            doc_no=meta["doc_no"],
            issuer=meta["issuer"],
            region=meta["region"],
            publish_date=meta["publish_date"],
            effective_date=meta.get("effective_date", ""),
            is_repealed=is_rep,
            repealed_hint=hint,
            source_path=str(path),
        )

    if not markers:
        # 兜底：没有条款编号的文件，按空行分段
        for i, seg in enumerate(re.split(r"\n\s*\n", text)):
            seg = seg.strip()
            if len(seg) >= 20:
                clauses.append(build("前言" if i == 0 else f"段{i}", seg))
        return clauses

    # 第一个 marker 之前的内容作为前言
    preamble = text[: markers[0][0]].strip()
    if len(preamble) >= 20:
        clauses.append(build("前言", preamble))

    for idx, (start, end, kind) in enumerate(markers):
        nxt = markers[idx + 1][0] if idx + 1 < len(markers) else len(text)
        if kind == "chapter":
            chapter = _heading_text(text, start, end)
            section = ""
            continue
        if kind == "section":
            section = _heading_text(text, start, end)
            continue
        if kind in ("item", "attach"):
            article_no = _heading_no(text[start:end])
            body = text[end:nxt]
            if kind == "attach":
                # 附件不属于任何一章，清掉章节归属，避免引用成「第八章 附件1」
                chapter = ""
                section = ""
        elif kind == "numitem":
            # 只保留编号本身，正文从编号之后开始
            article_no = text[start:end].strip().rstrip(".．、 ")
            body = text[end:nxt]
        else:
            article_no = text[start:end].strip()
            body = text[end:nxt]
            body = re.sub(r"^\s*[、\.．，,:：]\s*", "", body)
        if len(body.strip()) < 2:
            # 只有编号没有正文，跳过
            continue
        clauses.append(build(article_no, body))

    return clauses


# ---------------------------------------------------------------- 切块

def _split_long_body(body: str, max_chars: int) -> list[str]:
    """超长条款按（一）（二）项切；仍超长则按句子滑窗切，并保留少量重叠。"""
    if len(body) <= max_chars:
        return [body]

    items = [seg for seg in ITEM_SPLIT_PAT.split(body) if seg.strip()]
    if len(items) > 1:
        parts: list[str] = []
        buf = ""
        for seg in items:
            if len(buf) + len(seg) > max_chars and buf:
                parts.append(buf)
                buf = seg
            else:
                buf += seg
        if buf:
            parts.append(buf)
        if all(len(p) <= max_chars * 1.2 for p in parts):
            return parts

    # 句子级滑窗
    sentences = re.split(r"(?<=[。；！？])", body)
    parts = []
    buf = ""
    overlap = max(40, max_chars // 8)
    for sent in sentences:
        if not sent:
            continue
        if len(buf) + len(sent) > max_chars and buf:
            parts.append(buf)
            buf = buf[-overlap:] + sent
        else:
            buf += sent
    if buf.strip():
        parts.append(buf)
    return parts or [body[:max_chars]]


def chunk_clauses(clauses: list[Clause], max_chars: int = 900) -> list[Chunk]:
    """把条款变成检索单元。上下文头会被一起送入 BM25 和 embedding。"""
    chunks: list[Chunk] = []
    for clause in clauses:
        parts = _split_long_body(clause.body, max_chars)
        total = len(parts)
        for i, part in enumerate(parts):
            header_bits = [f"《{clause.doc_title}》"]
            if clause.chapter:
                header_bits.append(clause.chapter)
            if clause.section:
                header_bits.append(clause.section)
            header_bits.append(clause.article_no)
            if clause.issuer:
                header_bits.append(clause.issuer)
            if clause.publish_date:
                header_bits.append(clause.publish_date)
            if total > 1:
                header_bits.append(f"{i + 1}/{total}")
            header = "【" + "｜".join(header_bits) + "】"

            suffix = f"-p{i}" if total > 1 else ""
            chunks.append(
                Chunk(
                    chunk_id=f"{clause.doc_id}-{clause.article_no}{suffix}",
                    clause=clause,
                    header=header,
                    text=f"{header}\n{part.strip()}",
                    body=part.strip(),
                    part=i,
                    total_parts=total,
                )
            )
    return chunks


# ---------------------------------------------------------------- 文件读取

def _read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("解析 PDF 需要 pypdf，请执行 pip install pypdf") from exc
    reader = PdfReader(str(path))
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def _read_docx(path: Path) -> str:
    try:
        import docx
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("解析 docx 需要 python-docx，请执行 pip install python-docx") from exc
    doc = docx.Document(str(path))
    return "\n".join(p.text for p in doc.paragraphs)


def read_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _read_pdf(path)
    if suffix == ".docx":
        return _read_docx(path)
    for enc in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"无法识别文件编码：{path}")


def parse_file(path: Path, overrides: dict[str, Any] | None = None) -> tuple[list[Clause], dict[str, Any]]:
    raw = read_text(path)
    text = strip_noise(normalize_text(raw))
    meta = extract_meta(text, path)
    meta["effective_date"] = find_effective_date(text)
    meta["repealed_docs"] = find_repealed_docs(text)

    # 人工校准优先于自动抽取。红头文件的标题在 PDF 里没有任何语义标记，
    # 自动识别必然有错，生产做法是「自动抽取 + 人工校准」，见 data/raw/_meta.json
    if overrides:
        for key in ("title", "doc_no", "issuer", "region", "publish_date", "effective_date"):
            val = overrides.get(key)
            if isinstance(val, str) and val.strip():
                meta[key] = val.strip()
        meta["overridden"] = True

    # 扫描件检测：政务 PDF 有相当比例是图片型，取出全文后字符极少。
    # 这种情况必须显式报出来，绝不能安静地产生 0 条 —— 静默丢失语料是最难排查的故障。
    meta["chars"] = len(text)
    meta["needs_ocr"] = len(text) < 50
    if meta["needs_ocr"]:
        return [], meta

    clauses = split_clauses(text, meta, path)
    return clauses, meta


def load_manifest(raw_dir: Path) -> dict[str, dict[str, Any]]:
    """读取人工校准清单 data/raw/_meta.json（可选）。

    格式：{"文件名.pdf": {"title": "...", "doc_no": "...", "publish_date": "2024-03-01"}}
    存在的理由：红头文件的标题、发文机关在 PDF 里没有语义标记，纯正则必然出错。
    与其追求 100% 自动，不如把「人工校准」做成一个正式环节。
    """
    path = raw_dir / "_meta.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception as exc:  # noqa: BLE001
        print(f"  [警告] _meta.json 解析失败，已忽略：{exc}")
        return {}
    return data if isinstance(data, dict) else {}


# 这些目录里放的是生成产物，绝对不能当作语料读进来。
# 踩过的坑：把评测报告写到 data/samples/runs/ 后，下次 ingest 会把 *.report.md 当成政策文档，
# 语料被污染、指标无故下降，而且很难看出原因。
_SKIP_DIRS = {"runs", "index", "index_sample", "reports", "ocr_pages", "__pycache__", ".git"}


def iter_source_files(raw_dir: Path) -> list[Path]:
    """列出目录下真正待解析的语料文件，自动跳过生成产物目录。"""
    exts = {".txt", ".md", ".pdf", ".docx"}
    out: list[Path] = []
    for path in raw_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in exts:
            continue
        if path.name.startswith((".", "~$")):
            continue
        parts = path.relative_to(raw_dir).parts[:-1]
        if any(part in _SKIP_DIRS for part in parts):
            continue
        out.append(path)
    return sorted(out)


def load_corpus(raw_dir: Path) -> tuple[list[Clause], list[dict[str, Any]]]:
    """读取目录下所有支持的文件，返回条款与文件级元数据。"""
    files = iter_source_files(raw_dir)
    manifest = load_manifest(raw_dir)
    clauses: list[Clause] = []
    metas: list[dict[str, Any]] = []
    for path in files:
        try:
            file_clauses, meta = parse_file(path, manifest.get(path.name))
        except Exception as exc:
            print(f"  [跳过] {path.name}: {exc}")
            continue
        meta["source_path"] = str(path)
        meta["clause_count"] = len(file_clauses)
        clauses.extend(file_clauses)
        metas.append(meta)
        if meta.get("needs_ocr"):
            print(f"  [需OCR] {path.name}: 未抽到文本（字符数 {meta.get('chars', 0)}），疑似扫描件")
        else:
            tag = "（人工校准）" if meta.get("overridden") else ""
            print(f"  [解析] {path.name}: {len(file_clauses)} 条 -> 《{meta['title']}》{tag}")
    return clauses, metas
