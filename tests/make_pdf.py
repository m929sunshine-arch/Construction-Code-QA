"""把示例文本渲染成多页 PDF（带页眉页脚和画线表格），用于测试 PDF 转换。"""
import html
import re
import sys

import pymupdf


def md_to_html(md: str) -> list[str]:
    md = re.sub(r"<!--.*?-->", "", md, flags=re.S)
    blocks, cur, table = [], [], []
    for line in md.splitlines():
        if line.startswith("|"):
            if not set(line) <= set("|- "):
                table.append([c.strip() for c in line.strip("|").split("|")])
            continue
        if table:
            rows = "".join("<tr>" + "".join(f"<td>{html.escape(c)}</td>" for c in r) + "</tr>" for r in table)
            cur.append(f"<table border='1' style='border-collapse:collapse;font-size:7px'>{rows}</table>")
            table = []
        if line.strip():
            cur.append(f"<p>{html.escape(line)}</p>")
        if line.startswith(("CHAPTER", "SECTION")) and len(cur) > 12:
            blocks.append("".join(cur[:-1])); cur = cur[-1:]
    blocks.append("".join(cur))
    return blocks


def build(src: str, out: str):
    doc = pymupdf.open()
    for i, body in enumerate(md_to_html(open(src, encoding="utf-8").read())):
        page = doc.new_page()
        page.insert_text((72, 40), "2021 INTERNATIONAL RESIDENTIAL CODE (DEMO)", fontsize=8)
        page.insert_htmlbox(pymupdf.Rect(72, 60, 540, 760), f"<div style='font-size:9px'>{body}</div>")
        page.insert_text((300, 790), str(i + 1), fontsize=8)
    doc.save(out)


if __name__ == "__main__":
    build(sys.argv[1], sys.argv[2])
