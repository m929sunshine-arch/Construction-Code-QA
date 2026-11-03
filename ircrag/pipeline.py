"""把解析、检索、生成、校验串起来。"""
from __future__ import annotations

import logging
import os
import re
from copy import deepcopy

import yaml

try:  # 自动读取 .env
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from .generation import (DISCLAIMER, LLM, REFUSAL, Answer, extractive_answer, lang_of,
                         verify_citations)
from .ingest import load_chunks
from .retrieval import HybridRetriever, QueryRewriter, Reranker, make_embedder, make_vector_store

log = logging.getLogger(__name__)

FOLLOWUP_RE = re.compile(r"^(那|那么|还有|另外|and |what about |how about )", re.I)


def load_config(path: str | None = None, **overrides) -> dict:
    """读取配置。未指定路径时用环境变量 IRC_RAG_CONFIG（可写在 .env 里），默认 config.yaml。"""
    path = path or os.environ.get("IRC_RAG_CONFIG") or "config.yaml"
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for dotted, value in overrides.items():  # 例：chunking__mode="naive"
        node = cfg
        *keys, last = dotted.split("__")
        for k in keys:
            node = node[k]
        node[last] = value
    return cfg


class RAGPipeline:
    def __init__(self, cfg: dict, embedder=None):
        self.cfg = deepcopy(cfg)
        c = self.cfg
        self.chunks, self.clauses = load_chunks(
            c["data"]["source"], c["chunking"]["mode"], c["chunking"]["max_chars"],
            c["chunking"]["naive_size"], c["chunking"]["naive_overlap"])
        self.embedder = embedder or make_embedder(c["embedding"])
        reranker = Reranker(c["rerank"]["model"]) if c["rerank"].get("enabled") else None
        self.store = make_vector_store(c.get("vector_store"))
        self.retriever = HybridRetriever(self.chunks, self.embedder, c["retrieval"], reranker, self.store)
        self.rewriter = QueryRewriter(c["data"].get("glossary"))
        self.llm = LLM(c["llm"])

    # 对外接口 ------------------------------------------------------------
    def ask(self, question: str, history: list[str] | None = None, use_llm: bool = True) -> Answer:
        history = history or []
        lang = lang_of(question)

        standalone = question
        if history and FOLLOWUP_RE.match(question.strip()):
            if self.llm.available and use_llm:
                standalone = self.llm.rewrite_followup(question, history)
            else:  # 离线：把上一问拼进来补全上下文
                standalone = f"{history[-1]} {question}"

        expanded, glossary_hits = self.rewriter.expand(standalone)
        hits = self.retriever.search(expanded)
        ans = Answer(text="", rewritten_query=expanded, glossary_hits=glossary_hits, hits=hits)

        if self._should_refuse(hits):
            ans.text, ans.refused, ans.mode = REFUSAL[lang], True, "refused"
            return ans

        if self.llm.available and use_llm:
            ans.text, ans.mode = self.llm.answer(question, hits), "llm"
            if ans.text.strip() in REFUSAL.values():
                ans.refused, ans.mode = True, "refused"
                return ans
        else:
            ans.text, ans.mode = extractive_answer(question, expanded, hits), "extractive"

        ans.citations = verify_citations(ans.text, hits)
        ans.text += f"\n\n{DISCLAIMER[lang]}"
        return ans

    def _should_refuse(self, hits) -> bool:
        """向量相似度过低，或开了重排且重排分过低，任一条件满足就拒答。

        校准发现重排模型对口语化的中文提问打分偏低（如“阳台栏杆缝隙”只有 0.02），
        单靠重排分做阈值会误拒，所以以向量相似度为主，重排阈值只拦极低分。
        """
        if not hits:
            return True
        ref = self.cfg["refusal"]
        dense = [h.dense for h in hits if h.dense is not None]
        if (max(dense) if dense else 0.0) < ref["min_dense_score"]:
            return True
        top_rerank = hits[0].rerank
        return bool(self.cfg["rerank"].get("enabled") and top_rerank is not None
                    and top_rerank < ref["min_rerank_score"])
