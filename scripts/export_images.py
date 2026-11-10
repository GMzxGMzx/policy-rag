"""把扫描版 PDF 拆成逐页图片，供 OCR 使用。

为什么必须自己做：政务 PDF 里大量是纸质文件扫描件，页面里没有任何字体资源，
文本抽取库只能拿到空字符串。实测这批 12 份里 7 份是纯图片。

pypdf 自带的 `page.images` 需要 Pillow，而环境里常常只有 pypdf 没有 Pillow，
所以这里不依赖 Pillow，直接处理 PDF 图像流的两种真实编码：

- DCTDecode / JPXDecode：流本身就是 JPEG / JPEG2000 字节，原样落盘即可
- FlateDecode：zlib 压缩的原始像素，解压后用纯 Python 写 PNG（zlib + struct 足够）

还有第三类情况会失败：**文字被转成矢量曲线的 PDF**（印刷稿常见）。
它既没有字体也没有图片，只有几十万条绘图指令，必须用渲染器（PyMuPDF / poppler）才能出图。
遇到这种会明确报出来，并给出替代方案。

导出后有三条 OCR 路径：
1. 白描 / 微信「提取文字」/ WPS —— 最快，几十页够用
2. 云 OCR（阿里/腾讯/百度）—— 中文公文准确率最高
3. 本地 PaddleOCR —— 可离线，适合数据不出内网

用法：
    python -m scripts.export_images --only-scanned
    python -m scripts.export_images --file "data/raw/xxx.pdf"
"""
from __future__ import annotations

import argparse
import struct
import sys
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings  # noqa: E402
from app.ingest import parse_file  # noqa: E402

_PDF_IMAGE_FILTERS = {
    "/DCTDecode": ".jpg",
    "/JPXDecode": ".jp2",
}


def _write_png(path: Path, width: int, height: int, pixels: bytes, channels: int) -> None:
    """纯 Python 写 PNG：IHDR + IDAT(zlib) + IEND。不依赖 Pillow。"""
    color_type = 2 if channels == 3 else 0
    stride = width * channels
    expected = stride * height
    if len(pixels) < expected:
        raise ValueError(f"像素数据不足：{len(pixels)} < {expected}")
    rows = bytearray()
    for y in range(height):
        rows.append(0)  # 每行的 filter type = None
        rows += pixels[y * stride : (y + 1) * stride]

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(bytes(rows), 6))
        + chunk(b"IEND", b"")
    )


def _channels(obj) -> int:
    cs = obj.get("/ColorSpace")
    try:
        cs = cs.get_object() if hasattr(cs, "get_object") else cs
    except Exception:  # noqa: BLE001
        pass
    name = str(cs)
    if "DeviceRGB" in name:
        return 3
    if "DeviceGray" in name:
        return 1
    if "ICCBased" in name or isinstance(cs, list):
        try:
            icc = cs[1].get_object() if isinstance(cs, list) else cs
            return int(icc.get("/N", 3))
        except Exception:  # noqa: BLE001
            return 3
    return 3


def _has_predictor(obj) -> bool:
    parms = obj.get("/DecodeParms")
    try:
        parms = parms.get_object() if hasattr(parms, "get_object") else parms
    except Exception:  # noqa: BLE001
        return False
    if not parms:
        return False
    try:
        return int(parms.get("/Predictor", 1)) > 1
    except Exception:  # noqa: BLE001
        return False


def _extract_one(obj, target_stem: Path) -> tuple[Path | None, str]:
    """返回 (落盘路径, 说明)。失败时路径为 None。"""
    filt = obj.get("/Filter")
    filters = [str(f) for f in filt] if isinstance(filt, list) else [str(filt)]

    width = int(obj.get("/Width", 0) or 0)
    height = int(obj.get("/Height", 0) or 0)
    if not width or not height:
        return None, "缺少宽高"

    try:
        from pypdf import PdfReader  # noqa: F401
    except ImportError:
        pass

    # 1) JPEG / JPEG2000：原始流就是图片文件
    for f in filters:
        if f in _PDF_IMAGE_FILTERS:
            suffix = _PDF_IMAGE_FILTERS[f]
            raw = getattr(obj, "_data", None)
            if not raw:
                try:
                    raw = obj.get_data()
                except Exception:  # noqa: BLE001
                    return None, f"{f} 流读取失败"
            target = target_stem.with_suffix(suffix)
            target.write_bytes(raw)
            return target, f"{f} 直出"

    # 2) FlateDecode：解压出像素，自己写 PNG
    if "/FlateDecode" in filters:
        if _has_predictor(obj):
            return None, "FlateDecode+Predictor，需先做 PNG 反预测，暂不支持"
        try:
            pixels = obj.get_data()
        except Exception as exc:  # noqa: BLE001
            return None, f"Flate 解压失败：{exc}"
        channels = _channels(obj)
        if channels not in (1, 3):
            return None, f"不支持的通道数 {channels}（CMYK 等需额外转换）"
        target = target_stem.with_suffix(".png")
        try:
            _write_png(target, width, height, pixels, channels)
        except Exception as exc:  # noqa: BLE001
            return None, f"PNG 写入失败：{exc}"
        return target, f"Flate -> PNG（{width}x{height}, {channels}ch）"

    return None, f"未支持的编码 {filters}"


def export_pdf_images(pdf: Path, out_dir: Path) -> tuple[int, list[str]]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("需要 pypdf：pip install pypdf") from exc

    reader = PdfReader(str(pdf))
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = pdf.stem[:40].replace(" ", "_")

    count = 0
    notes: list[str] = []

    for page_idx, page in enumerate(reader.pages, start=1):
        try:
            resources = page.get("/Resources") or {}
            xobj = resources.get("/XObject")
        except Exception:  # noqa: BLE001
            continue
        if not xobj:
            continue
        try:
            keys = list(xobj.keys())
        except Exception:  # noqa: BLE001
            continue

        img_idx = 0
        for key in keys:
            try:
                obj = xobj[key].get_object()
                if obj.get("/Subtype") != "/Image":
                    continue
            except Exception:  # noqa: BLE001
                continue
            img_idx += 1
            target = out_dir / f"{stem}_p{page_idx:03d}_{img_idx}"
            path, note = _extract_one(obj, target)
            if path:
                count += 1
                notes.append(f"  p{page_idx} {key}: {note}")
            else:
                notes.append(f"  p{page_idx} {key}: 跳过 —— {note}")
    return count, notes


def _is_scanned(path: Path) -> bool:
    try:
        _, meta = parse_file(path)
        return bool(meta.get("needs_ocr"))
    except Exception:  # noqa: BLE001
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="导出扫描 PDF 的页面图片，供 OCR")
    parser.add_argument("--raw", default=str(settings.raw_dir))
    parser.add_argument("--out", default=str(Path(settings.data_dir) / "ocr_pages"))
    parser.add_argument("--only-scanned", action="store_true", help="只处理抽不出文本的扫描件")
    parser.add_argument("--file", default="", help="只处理指定文件")
    parser.add_argument("--verbose", action="store_true", help="打印每个图像对象的处理结果")
    args = parser.parse_args()

    out_dir = Path(args.out)
    files = [Path(args.file)] if args.file else sorted(p for p in Path(args.raw).rglob("*.pdf") if p.is_file())
    if not files:
        print("没有找到 PDF 文件")
        return 1

    total = 0
    failed: list[str] = []
    for pdf in files:
        if args.only_scanned and not _is_scanned(pdf):
            continue
        try:
            n, notes = export_pdf_images(pdf, out_dir)
        except Exception as exc:  # noqa: BLE001
            print(f"[失败] {pdf.name}: {exc}")
            failed.append(pdf.name)
            continue
        total += n
        status = f"{n} 张图" if n else "未导出图片"
        print(f"[{'导出' if n else '无法导出'}] {pdf.name}  ->  {status}")
        if args.verbose and notes:
            for line in notes:
                print(line)
        if n == 0:
            failed.append(pdf.name)

    print(f"\n共导出 {total} 张图片到：{out_dir}")
    if failed:
        print(f"\n以下 {len(failed)} 份无法直接导出图片：")
        for name in failed:
            print(f"  - {name}")
        print("\n常见原因与处理：")
        print("  1) 文字被转成矢量曲线（印刷稿常见）——没有字体也没有图片，需要渲染器：")
        print("     用 WPS / Adobe 打开后「导出为图片」，或装 PyMuPDF 后 pip install pymupdf 再渲染")
        print("  2) 加密或异常结构 —— 用 WPS 另存一次再试")
        print("  3) 更省事的办法：去发文单位网站找同一份文件的网页版/Word 版，直接复制成 txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
