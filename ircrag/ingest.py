"""解析 IRC 文本并分块。

两种分块方式：
- section：条款级分块。每个条款（如 R602.3.2）一块，Exception 与主条文放在同一块，
  块首拼接“章 > 节 > 条款”标题路径；超长条款按句子再切。
- naive：固定长度滑窗，作为消融实验的基线。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

CHAPTER_RE = re.compile(r"^CHAPTER\s+(\d+)\s+(.+)$")
# IRC 各部分的编号前缀：R 建筑、N 节能、M 暖通、G 燃气、P 给排水、E 电气；节号 3～4 位（R301、R1001、M1301）
PREFIX = r"[RNMGPE]\d{3,4}"
SECTION_RE = re.compile(rf"^SECTION\s+({PREFIX})\s+(.+)$")
# 条款行：R602.3.2 Top plate. 正文……
# 标题须以大写字母开头，避免把行首的交叉引用（“R302.1 and R302.2 shall…”）误认为新条款
# 节能、燃气章节在条款号后带有对应 IECC / IFGC 的编号，如“N1102.1.2 (R402.1.2) Insulation …”
CLAUSE_RE = re.compile(rf"^({PREFIX}(?:\.\d+)+)\s+(?:\([^)]{{1,20}}\)\s+)?([A-Z][^.]{{0,120}})\.\s*(.*)$")
# 第 2 章术语定义：“[RB] ACCESS (TO). That which enables …”
DEFINITION_RE = re.compile(r"^(?:\[[A-Z]{2}\]\s*)?([A-Z][A-Z0-9 ,'’()/&\-–—]{1,80}?)\.\s+(.+)$")
EXCEPTION_RE = re.compile(r"^Exceptions?\s*:", re.I)
# 表格：TABLE R602.3(5) SIZE, HEIGHT AND SPACING OF WOOD STUDS
TABLE_RE = re.compile(rf"^TABLE\s+({PREFIX}(?:\.\d+)*(?:\(\d+\))?)\s*(.*)$")
# 条款号，含表格编号：R311.7.2、R602.3(5)
SECTION_ID_RE = re.compile(rf"\b{PREFIX}(?:\.\d+)*(?:\(\d+\))?")
NOISE_RE = re.compile(r"^[^A-Za-z0-9]{1,3}$")  # PDF 里的修订标记等单独符号，如“¬”


@dataclass
class Clause:
    section_id: str
    title: str
    chapter: str
    section: str
    text: str
    start: int = 0  # 在全文中的字符偏移，用于给 naive 分块标注覆盖的条款
    end: int = 0
    kind: str = "clause"  # clause | table | definition

    @property
    def parent(self) -> str:
        if self.kind == "table":
            return re.sub(r"\(\d+\)$", "", self.section_id)
        if self.kind == "definition":
            return self.section_id
        return self.section_id.rsplit(".", 1)[0]

    @property
    def label(self) -> str:
        if self.kind == "table":
            return f"Table {self.section_id}"
        if self.kind == "definition":
            return f"{self.section_id} {self.title}"
        return self.section_id

    @property
    def path(self) -> str:
        return f"{self.chapter} > {self.section} > {self.label} {self.title}"


@dataclass
class Chunk:
    chunk_id: str
    text: str                      # 送去检索和生成的文本
    section_ids: list[str]         # 该块覆盖的条款号（section 模式通常只有 1 个）
    title: str = ""
    path: str = ""
    meta: dict = field(default_factory=dict)


def _strip_comments(raw: str) -> str:
    return re.sub(r"<!--.*?-->", "", raw, flags=re.S)


def parse_clauses(raw: str) -> tuple[list[Clause], str]:
    """返回条款列表，以及去掉注释后的全文（用于 naive 分块）。"""
    text = _strip_comments(raw)
    clauses: list[Clause] = []
    chapter = section = ""
    offset = 0
    current: Clause | None = None

    def close(pos: int):
        nonlocal current
        if current:
            current.text = current.text.strip()
            current.end = pos
            clauses.append(current)
            current = None

    for line in text.splitlines(keepends=True):
        s = line.strip()
        if not s or NOISE_RE.match(s):
            offset += len(line)
            continue
        in_definitions = section.startswith("R202 ")
        if m := CHAPTER_RE.match(s):
            close(offset)
            chapter = f"Chapter {m.group(1)} {m.group(2).title()}"
        elif m := SECTION_RE.match(s):
            close(offset)
            section = f"{m.group(1)} {m.group(2).title()}"
        elif (m := TABLE_RE.match(s)) and _caption_ok(m.group(2)):
            close(offset)
            title = re.sub(r"^[—-]\s*continued\s*", "", m.group(2).strip(), flags=re.I)
            title = re.sub(r"^\([A-Z]?[\d.()]+\)\s*", "", title)  # 去掉对应 IECC 的编号，如 (R402.1.2)
            current = Clause(
                section_id=m.group(1), title=title.capitalize() or "Table", chapter=chapter,
                section=section, text="", start=offset, kind="table",
            )
        elif in_definitions and (m := DEFINITION_RE.match(s)) and _caption_ok(m.group(1)):
            close(offset)
            current = Clause(
                section_id="R202", title=m.group(1).strip().rstrip(",").capitalize(), chapter=chapter,
                section=section, text=m.group(2), start=offset, kind="definition",
            )
        elif m := CLAUSE_RE.match(s):
            close(offset)
            current = Clause(
                section_id=m.group(1), title=m.group(2).strip(), chapter=chapter,
                section=section, text=m.group(3), start=offset,
            )
        elif current is not None:
            # 表格逐行保留；Exception 换行；其余续行并入当前条款
            if current.kind == "table" or EXCEPTION_RE.match(s) or s.startswith("|"):
                current.text += "\n" + s
            else:
                current.text += " " + s
        offset += len(line)
    close(offset)
    return _merge_continued(clauses), text


def _caption_ok(title: str) -> bool:
    """真正的表格标题是全大写（允许脚注标记）；插图标注里的“TABLE R602.10.6.4 (on …”不算。"""
    letters = [c for c in title if c.isalpha()]
    return not letters or sum(c.isupper() for c in letters) / len(letters) > 0.8


def _merge_continued(clauses: list[Clause]) -> list[Clause]:
    """跨页表格（TABLE R602.3(1)—continued）在 PDF 里会出现多次，合并成一张。"""
    merged: list[Clause] = []
    seen: dict[tuple[str, str], Clause] = {}
    for c in clauses:
        key = (c.kind, c.section_id)
        if c.kind == "table" and key in seen:
            first = seen[key]
            rows = [ln for ln in c.text.splitlines() if ln not in first.text.splitlines()]
            first.text += "\n" + "\n".join(rows)
            first.end = c.end
            continue
        seen[key] = c
        merged.append(c)
    return merged


def _split_sentences(text: str, max_chars: int) -> list[str]:
    sents = re.split(r"(?<=[.;])\s+", text)
    parts, buf = [], ""
    for s in sents:
        if buf and len(buf) + len(s) + 1 > max_chars:
            parts.append(buf)
            buf = s
        else:
            buf = f"{buf} {s}".strip()
    if buf:
        parts.append(buf)
    return parts


def _split_table(text: str, max_chars: int) -> list[str]:
    """超长表格按行切开，每一块都重复表头，保证单独拿出来也能读懂。"""
    lines = text.splitlines()
    idx = [i for i, ln in enumerate(lines) if ln.startswith("|")]
    if len(idx) < 3:
        return [text]
    pre, header = lines[: idx[0]], lines[idx[0]: idx[0] + 2]
    body, post = lines[idx[0] + 2: idx[-1] + 1], lines[idx[-1] + 1:]
    # 表格前的说明文字只放在第一块；后续块只重复一行标题 + 表头，避免长说明被复制成上千份
    lead = pre[:1] if pre and len(pre[0]) <= 200 else []
    parts, buf, first = [], [], True
    for row in body:
        head = (pre if first else lead) + header
        if buf and len("\n".join(head + buf + [row])) > max_chars:
            parts.append("\n".join(head + buf))
            buf, first = [], False
        buf.append(row)
    parts.append("\n".join(((pre if first else lead) + header) + buf + post))  # 表注放在最后一块
    return parts


def _hard_split(text: str, max_chars: int) -> list[str]:
    """兜底：仍然超长（单行超大的表格、提取异常）时按行、再按空格硬切，避免超出向量模型长度。"""
    if len(text) <= 2 * max_chars:
        return [text]
    parts, buf = [], ""
    for tok in re.split(r"(\n|\s+)", text):
        if len(buf) + len(tok) > max_chars and buf.strip():
            parts.append(buf.strip())
            buf = ""
        buf += tok
    if buf.strip():
        parts.append(buf.strip())
    return parts


def section_chunks(clauses: list[Clause], max_chars: int = 1200) -> list[Chunk]:
    chunks = []
    for c in clauses:
        if len(c.text) <= max_chars:
            pieces = [c.text]
        elif c.kind == "table":
            pieces = _split_table(c.text, max_chars)
        else:
            pieces = _split_sentences(c.text, max_chars)
        pieces = [q for p in pieces for q in _hard_split(p, max_chars)]
        for i, piece in enumerate(pieces):
            cid = c.label if len(pieces) == 1 else f"{c.label}#{i + 1}"
            if c.kind == "definition":
                head = f"Definition ({c.section_id}): {c.title}. "
            else:
                head = f"{c.label} {c.title}." + ("\n" if c.kind == "table" else " ")
            chunks.append(Chunk(
                chunk_id=cid,
                text=f"[{c.path}]\n{head}{piece.strip()}",
                section_ids=[c.section_id],
                title=c.title,
                path=c.path,
                meta={"chapter": c.chapter, "section": c.section, "parent": c.parent, "kind": c.kind},
            ))
    return chunks


def naive_chunks(clauses: list[Clause], full_text: str, size: int = 500, overlap: int = 50) -> list[Chunk]:
    """固定长度切块。覆盖的条款号按字符区间重叠来判定，只用于评测，不写进文本。"""
    chunks = []
    step = max(1, size - overlap)
    for i, start in enumerate(range(0, len(full_text), step)):
        end = min(len(full_text), start + size)
        piece = full_text[start:end].strip()
        if not piece:
            continue
        covered = [c.section_id for c in clauses if c.start < end and c.end > start]
        chunks.append(Chunk(chunk_id=f"naive-{i}", text=piece, section_ids=covered))
        if end >= len(full_text):
            break
    return chunks


def load_chunks(path: str, mode: str = "section", max_chars: int = 1200,
                naive_size: int = 500, naive_overlap: int = 50) -> tuple[list[Chunk], list[Clause]]:
    with open(path, encoding="utf-8") as f:
        raw = f.read()
    clauses, full_text = parse_clauses(raw)
    if mode == "naive":
        return naive_chunks(clauses, full_text, naive_size, naive_overlap), clauses
    return section_chunks(clauses, max_chars), clauses
