"""生成与引用校验。"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

from .ingest import SECTION_ID_RE
from .retrieval import CJK_RE, Hit, tokenize

SYSTEM_PROMPT = """You are a building-code assistant for the International Residential Code (IRC).
Rules:
1. Answer ONLY from the provided code sections. Never use outside knowledge or invent numbers.
2. End every factual sentence with the section number it comes from, in square brackets, e.g. [R311.7.2].
   For tables cite the table number, e.g. [Table R602.3(5)].
   Only cite section numbers that appear in the provided sections.
3. If the provided sections do not answer the question, reply exactly with the refusal sentence
   (in the user's language) and nothing else:
   EN: "The knowledge base does not contain a provision that answers this question."
   ZH: "知识库中没有找到能回答该问题的条文。"
4. Format: first a one-sentence conclusion with the key value(s), then a short list of supporting points.
   Mention exceptions if a provided section has one.
5. Reply in the same language as the question (Chinese question -> Chinese answer, keep section numbers and units).
"""

REFUSAL = {"zh": "知识库中没有找到能回答该问题的条文。",
           "en": "The knowledge base does not contain a provision that answers this question."}
DISCLAIMER = {"zh": "具体适用以当地采纳的规范版本及注册工程师复核为准。",
              "en": "Verify against the locally adopted code edition and with a licensed professional."}


def lang_of(text: str) -> str:
    return "zh" if CJK_RE.search(text) else "en"


@dataclass
class Citation:
    section_id: str
    valid: bool
    source: str = ""  # retrieved = 出自召回的条文；cross_ref = 召回条文里提到的交叉引用；"" = 无依据


@dataclass
class Answer:
    text: str
    citations: list[Citation] = field(default_factory=list)
    refused: bool = False
    mode: str = "llm"           # llm | extractive | refused
    rewritten_query: str = ""
    glossary_hits: list[str] = field(default_factory=list)
    hits: list[Hit] = field(default_factory=list)

    @property
    def hallucinated(self) -> list[str]:
        return [c.section_id for c in self.citations if not c.valid]


def verify_citations(answer: str, hits: list[Hit]) -> list[Citation]:
    """检查答案里的每个条款号有没有依据。

    - retrieved：就是召回的条文（引用其上级条款号也算）
    - cross_ref：不是召回的条文，但召回的原文里明确提到了它（如“按 Section R310.2.3 设置”）。
      真实规范里交叉引用很多，模型照着原文转述不算编造，单独标出来供核对
    - 两者都不是：无依据，视为幻觉引用
    """
    retrieved = {sid for h in hits for sid in h.chunk.section_ids}
    context = " ".join(h.chunk.text for h in hits)
    seen, out = set(), []
    for sid in SECTION_ID_RE.findall(answer):
        if sid in seen:
            continue
        seen.add(sid)
        if any(r == sid or r.startswith(sid + ".") or r.startswith(sid + "(") for r in retrieved):
            out.append(Citation(sid, True, "retrieved"))
        elif re.search(re.escape(sid) + r"(?![\d(]|\.\d)", context):
            out.append(Citation(sid, True, "cross_ref"))
        else:
            out.append(Citation(sid, False))
    return out


def build_context(hits: list[Hit]) -> str:
    return "\n\n".join(f"<section id=\"{h.chunk.section_ids[0] if h.chunk.section_ids else h.chunk.chunk_id}\">\n"
                       f"{h.chunk.text}\n</section>" for h in hits)


class LLM:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        key = os.environ.get(cfg.get("api_key_env", "DEEPSEEK_API_KEY"))
        self.client = None
        if key:
            from openai import OpenAI
            self.client = OpenAI(base_url=cfg["base_url"], api_key=key)

    @property
    def available(self) -> bool:
        return self.client is not None

    def chat(self, messages: list[dict], retries: int = 3) -> str:
        """调用大模型；接口偶尔返回空响应或超时，最多重试 3 次（间隔 2s、4s、8s）。"""
        import time
        last_err = None
        for attempt in range(retries + 1):
            try:
                resp = self.client.chat.completions.create(
                    model=self.cfg["model"], messages=messages,
                    temperature=self.cfg.get("temperature", 0.1))
                content = resp.choices[0].message.content if resp and resp.choices else None
                if content:
                    return content.strip()
                last_err = RuntimeError("大模型返回了空响应")
            except Exception as e:  # 网络错误、限流、服务端 5xx
                last_err = e
            if attempt < retries:
                time.sleep(2 ** (attempt + 1))
        raise RuntimeError(f"大模型调用失败（已重试 {retries} 次）：{last_err}")

    def answer(self, question: str, hits: list[Hit]) -> str:
        return self.chat([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Code sections:\n{build_context(hits)}\n\nQuestion: {question}"},
        ])

    def rewrite_followup(self, question: str, history: list[str]) -> str:
        prompt = ("Rewrite the last question into a standalone question about the building code, "
                  "using the earlier questions for context. Keep the original language. "
                  "Output only the rewritten question.\n\n"
                  + "\n".join(f"Earlier: {q}" for q in history[-3:]) + f"\nLast: {question}")
        return self.chat([{"role": "user", "content": prompt}])


def _table_excerpt(h: Hit, q: set[str], max_rows: int = 3) -> str:
    """表格：保留表头，只摘与问题最相关的几行。"""
    rows = [ln for ln in h.chunk.text.splitlines() if ln.startswith("|")]
    header, body = rows[:2], rows[2:]
    first = lambda r: set(tokenize(r.strip("|").split("|")[0]))  # noqa: E731 行首单元格（如 2x6）
    keyed = [r for r in body if q & first(r)]
    best = keyed[:max_rows] if keyed else sorted(body, key=lambda r: -len(q & set(tokenize(r))))[:max_rows]
    best = sorted(best, key=body.index)
    return f"- Table {h.chunk.section_ids[0]}：\n\n" + "\n".join(header + best) + "\n"


def extractive_answer(question: str, expanded_query: str, hits: list[Hit], max_sections: int = 2) -> str:
    """离线摘录模式：不调用 LLM，从最相关的条款中挑出与问题重合度最高的句子，原句摘录并标注条款号。"""
    q = set(tokenize(expanded_query))
    lines = []
    top = hits[0].dense or 0.0
    # 第二条以后只收录与第一条相关度接近的条款，避免把无关条文摘进来
    chosen = [hits[0]] + [h for h in hits[1:max_sections] if (h.dense or 0.0) >= 0.6 * top]
    for h in chosen:
        sid = h.chunk.section_ids[0]
        if h.chunk.meta.get("kind") == "table":
            lines.append(_table_excerpt(h, q))
            continue
        body = h.chunk.text.split("\n", 1)[-1]
        body = re.sub(r"^[RNMGPE]\d{3,4}(?:\.\d+)*\s+[^.]+\.\s*", "", body)  # 去掉“条款号 + 标题”
        sents = [s.strip() for s in re.split(r"(?<=[.])\s+(?=[A-Z])", body) if s.strip()]
        scored = sorted(sents, key=lambda s: -len(q & set(tokenize(s))))
        best = [s for s in scored[:2] if q & set(tokenize(s))] or scored[:1]
        for s in sorted(best, key=sents.index):  # 保持原文顺序
            lines.append(f"- {s} [{sid}]")
    head = ("（离线摘录模式：未配置 LLM，以下为最相关条文的原句摘录）" if lang_of(question) == "zh"
            else "(Extractive mode: no LLM configured; the most relevant provisions are quoted below.)")
    return head + "\n" + "\n".join(lines)
