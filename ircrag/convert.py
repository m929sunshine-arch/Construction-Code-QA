"""把 PDF / HTML / 纯文本整理成解析器能读的格式。

目标格式（与 data/sample/irc_sample.md 一致）：
    CHAPTER 6 WALL CONSTRUCTION
    SECTION R602 WOOD WALL FRAMING
    R602.3.1 Stud size, height and spacing. 正文……
    Exception: ……
    TABLE R602.3(5) SIZE, HEIGHT AND SPACING OF WOOD STUDS
    | Stud size | ... |
    | --- | ... |

用法：
    python -m ircrag.convert path/to/irc_ch6.pdf -o data/raw/irc_ch6.md
"""
from __future__ import annotations

import argparse
import re
import unicodedata
from collections import Counter

SECTION_ONLY_RE = re.compile(r"^SECTION\s+[RNMGPE]\d{3,4}$")
CHAPTER_ONLY_RE = re.compile(r"^CHAPTER\s+\d+$")
TABLE_ONLY_RE = re.compile(r"^TABLE\s+[RNMGPE]\d{3,4}(?:\.\d+)*(?:\(\d+\))?(?:\s*[—-]\s*continued)?$", re.I)
PAGE_NO_RE = re.compile(r"^(?:\d{1,4}|[ivxlc]{1,6}|page \d+)$", re.I)


def _cell(v) -> str:
    v = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(v or "").translate(SYMBOL_MAP))).strip().replace("|", "/")
    return re.sub(r"(\d)\s*×\s*(\d)", r"\1x\2", v)


def rows_to_markdown(rows: list[list]) -> str:
    rows = [[_cell(c) for c in r] for r in rows if any(_cell(c) for c in r)]
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    lines = ["| " + " | ".join(rows[0]) + " |", "|" + " --- |" * width]
    lines += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(lines)


def _mostly_upper(s: str) -> bool:
    letters = [c for c in s if c.isalpha()]
    return bool(letters) and sum(c.isupper() for c in letters) / len(letters) > 0.8  # 容忍脚注标记，如 STUDSa


def normalize_lines(lines: list[str], drop: set[str] | None = None) -> str:
    """清理页眉页脚、连字符断行，并把“SECTION R602”与下一行的标题合并。"""
    drop = drop or set()
    out: list[str] = []
    for raw in lines:
        s = unicodedata.normalize("NFKC", raw.translate(SYMBOL_MAP)).strip()  # 连字 ﬀ → ff 等
        s = TIMES_RE.sub(r"\1x\2", s)                 # 2 × 4 → 2x4，和用户输入一致
        if not s or s in drop or PAGE_NO_RE.match(s):
            if not s and out and out[-1] != "":
                out.append("")
            continue
        if out and out[-1].endswith("-") and s[:1].islower():
            out[-1] = out[-1][:-1] + s            # 连字符断行：build-\ning → building
            continue
        prev = out[-1] if out else ""
        if s == prev and s.startswith("TABLE "):
            continue                              # 表格标题既在正文又在表格首行时只保留一次
        if (SECTION_ONLY_RE.match(prev) or CHAPTER_ONLY_RE.match(prev) or TABLE_ONLY_RE.match(prev)) and _mostly_upper(s):
            out[-1] = f"{prev} {s}"               # 标题被拆到下一行
            continue
        out.append(s)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip() + "\n"


def _repeated_lines(pages: list[list[str]], ratio: float = 0.5) -> set[str]:
    """出现在一半以上页面的行视为页眉页脚（版权声明、书名等）。"""
    if len(pages) < 3:
        return set()
    cnt = Counter(line.strip() for p in pages for line in set(p) if line.strip())
    return {line for line, n in cnt.items() if n >= ratio * len(pages)}


FRACTION_WHOLE_RE = re.compile(r"(\d)\x01(\d+)\x02/\x01(\d+)\x02")
FRACTION_RE = re.compile(r"\x01(\d+)\x02/\x01(\d+)\x02")
SMALL_RE = re.compile(r"\x01([^\x02]*)\x02")
TIMES_RE = re.compile(r"(\d)\s*×\s*(\d)")
# Symbol 字体的私有区字符：PDF 里的 ×、≥、≤ 等常被提取成这些码位
SYMBOL_MAP = str.maketrans({"\uf0b4": "×", "\uf0b3": "≥", "\uf0a3": "≤", "\uf0b0": "°", "\uf0b1": "±", "\uf0a2": "′", "\uf0b2": "″"})


def _line_text(line: dict) -> str:
    """拼接一行的 span。字号明显偏小的数字是上下标：7 + 小号3 + / + 小号4 → 7-3/4（否则会变成“73/4”）。"""
    spans = [sp for sp in line["spans"] if sp["text"]]
    if not spans:
        return ""
    base = max(sp["size"] for sp in spans)
    parts = []
    for sp in spans:
        t = sp["text"]
        small = sp["size"] < 0.8 * base and t.strip().isdigit()
        parts.append(f"\x01{t.strip()}\x02" if small else t)
    text = "".join(parts)
    text = FRACTION_WHOLE_RE.sub(r"\1-\2/\3", text)
    text = FRACTION_RE.sub(r"\1/\2", text)
    return SMALL_RE.sub(r"\1", text)


def _is_figure_label(text: str, size: float, body: float) -> bool:
    """插图里的标注：字号小、几乎全大写（如“WALL STUDS SEE SECTION R602.3”）。表格标题除外。"""
    if re.match(r"\s*TABLE\s+[RNMGPE]\d{3}", text) or size >= 0.85 * body:
        return False
    letters = [c for c in text if c.isalpha()]
    return bool(letters) and sum(c.isupper() for c in letters) / len(letters) > 0.8


def _table_rows(table) -> list[list[str]]:
    """用 PyMuPDF 的 to_markdown 取单元格：它对多行表头的文字顺序处理得比 extract() 好。"""
    try:
        md = table.to_markdown()
    except Exception:
        return table.extract()
    rows = []
    for line in md.splitlines():
        line = line.strip()
        if not line.startswith("|") or re.fullmatch(r"\|(\s*:?-+:?\s*\|)+", line):
            continue
        cells = [c.replace("**", "").replace("<br>", " ").strip() for c in line.strip("|").split("|")]
        rows.append(["" if re.fullmatch(r"Col\d+", c) else c for c in cells])
    return rows


def _table_rows_from_spans(table, lines: list[tuple]) -> list[list[str]]:
    """按单元格坐标收集文字行（用 _line_text，表格里的分数也能正确还原，如 1-3/4"）。"""
    rows = []
    for row in table.rows:
        cells = []
        for bbox in row.cells:
            if bbox is None:
                cells.append("")
                continue
            x0, y0, x1, y1 = bbox
            inside = [(ly, lx, t) for (lx, ly, cx, cy, t) in lines if x0 <= cx <= x1 and y0 <= cy <= y1]
            cells.append(" ".join(t for _, _, t in sorted(inside)))
        rows.append(cells)
    # 合并单元格：PyMuPDF 只给第一格坐标，其余为 None；表头里沿用左侧文字，和 to_markdown 一致
    return rows


def _table_markdown(table, lines: list[tuple] | None = None) -> tuple[str, str]:
    """返回（表格标题行, Markdown 表格）。PyMuPDF 常把标题识别成表格第一行，这里拆出来。"""
    rows = _table_rows_from_spans(table, lines) if lines else []
    if not any(c for r in rows for c in r):
        rows = _table_rows(table)
    caption = ""
    if rows:
        first = " ".join(_cell(c) for c in rows[0] if _cell(c))
        if re.match(r"^TABLE\s+[RNMGPE]\d{3}", first):
            caption, rows = first, rows[1:]
    return caption, rows_to_markdown(rows)


def _page_items(page) -> list[str]:
    import pymupdf

    W, H = page.rect.width, page.rect.height
    try:
        tables = list(page.find_tables().tables)
    except Exception:
        tables = []
    boxes = [pymupdf.Rect(t.bbox) for t in tables]
    blocks = [b for b in page.get_text("dict")["blocks"] if b.get("type") == 0 and b.get("lines")]
    sizes = [sp["size"] for b in blocks for ln in b["lines"] for sp in ln["spans"] if sp["text"].strip()]
    body = max(set(round(s) for s in sizes), key=lambda s: sum(round(x) == s for x in sizes)) if sizes else 10

    # 页面上每一行文字的中心点，供表格按单元格取字
    lines = []
    for b in blocks:
        for ln in b["lines"]:
            x0, y0, x1, y1 = ln["bbox"]
            t = _line_text(ln).strip()
            if t:
                lines.append((x0, y0, (x0 + x1) / 2, (y0 + y1) / 2, t))

    items = []  # (y0, x0, 是否通栏, 文本)
    for b in blocks:
        x0, y0, x1, y1 = b["bbox"]
        if y1 < 0.07 * H or y0 > 0.945 * H:
            continue  # 页眉、页脚、页码
        r = pymupdf.Rect(b["bbox"])
        if any(bx.intersects(r) and (bx & r).get_area() > 0.5 * r.get_area() for bx in boxes):
            continue  # 表格区域的文字由表格提取代替
        text = "\n".join(_line_text(ln) for ln in b["lines"])
        size = max(sp["size"] for ln in b["lines"] for sp in ln["spans"])
        if _is_figure_label(text, size, body):
            continue
        items.append((y0, x0, (x1 - x0) > 0.55 * W, text))
    for t, bx in zip(tables, boxes):
        caption, md = _table_markdown(t, lines)
        if md:
            items.append((bx.y0, bx.x0, True, (caption + "\n" if caption else "") + md))

    # 双栏排版：通栏元素（表格、表注）把页面切成若干横条，每条内先左栏、再右栏
    items.sort(key=lambda it: it[0])
    out, band = [], []

    def flush():
        band.sort(key=lambda it: (0 if it[1] < W / 2 else 1, it[0]))
        out.extend(it[3] for it in band)
        band.clear()

    for it in items:
        if it[2]:
            flush()
            out.append(it[3])
        else:
            band.append(it)
    flush()
    return "\n".join(out).splitlines()


def chapter_pages(path: str, chapters: list[int]) -> list[int]:
    """根据 PDF 书签找到指定章节的页码（从 0 开始）。"""
    import pymupdf

    with pymupdf.open(path) as doc:
        marks = [(int(m.group(1)), page - 1) for _, title, page in doc.get_toc()
                 if (m := re.match(r"^CHAPTER\s+(\d+)\b", title))]
        n = doc.page_count
    pages = []
    for i, (ch, start) in enumerate(marks):
        if ch in chapters:
            end = marks[i + 1][1] if i + 1 < len(marks) else n
            pages.extend(range(start, end))
    if not pages:
        raise SystemExit(f"书签里没找到第 {chapters} 章，请改用 --pages 指定页码")
    return pages


def pdf_to_text(path: str, pages: list[int] | None = None) -> str:
    import pymupdf

    out: list[list[str]] = []
    with pymupdf.open(path) as doc:
        for i in pages if pages is not None else range(doc.page_count):
            out.append(_page_items(doc[i]))
    drop = _repeated_lines(out)
    return normalize_lines([line for p in out for line in p], drop)


def html_to_text(path: str) -> str:
    from bs4 import BeautifulSoup

    with open(path, encoding="utf-8", errors="ignore") as f:
        soup = BeautifulSoup(f.read(), "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer"]):
        tag.decompose()
    for table in soup.find_all("table"):
        rows = [[c.get_text(" ") for c in tr.find_all(["th", "td"])] for tr in table.find_all("tr")]
        table.replace_with("\n" + rows_to_markdown(rows) + "\n")
    for br in soup.find_all("br"):
        br.replace_with("\n")
    text = soup.get_text("\n")
    return normalize_lines(text.splitlines())


def convert(path: str, chapters: list[int] | None = None, pages: list[int] | None = None) -> str:
    low = path.lower()
    if low.endswith(".pdf"):
        if chapters:
            pages = chapter_pages(path, chapters)
        return pdf_to_text(path, pages)
    if low.endswith((".html", ".htm")):
        return html_to_text(path)
    with open(path, encoding="utf-8") as f:
        return normalize_lines(f.read().splitlines())


def main():
    ap = argparse.ArgumentParser(description="把 IRC 的 PDF/HTML/TXT 转成解析器格式")
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--chapters", nargs="+", help="只转换这些章（按 PDF 书签定位），如 --chapters 3 5 6 8 或 --chapters 1-43")
    ap.add_argument("--pages", help="页码范围（从 1 开始），如 54-113,160-197")
    args = ap.parse_args()
    pages = None
    if args.pages:
        pages = [i - 1 for part in args.pages.split(",") for a, _, b in [part.partition("-")]
                 for i in range(int(a), int(b or a) + 1)]
    chapters = None
    if args.chapters:
        chapters = [n for part in args.chapters for a, _, b in [part.partition("-")]
                    for n in range(int(a), int(b or a) + 1)]
    text = "\n\n".join(convert(p, chapters, pages) for p in args.inputs)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(text)
    from .ingest import parse_clauses
    clauses, _ = parse_clauses(text)
    tables = sum(c.kind == "table" for c in clauses)
    defs = sum(c.kind == "definition" for c in clauses)
    print(f"已写入 {args.output}：识别 {len(clauses)} 个条款（其中表格 {tables} 个、术语定义 {defs} 条）")
    if clauses:
        print("前 10 个：", ", ".join(c.section_id for c in clauses[:10]))


if __name__ == "__main__":
    main()
