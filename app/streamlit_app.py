"""IRC building code Q&A demo. Run: streamlit run app/streamlit_app.py"""
from __future__ import annotations

import os
import sys

import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from ircrag.pipeline import RAGPipeline, load_config  # noqa: E402

st.set_page_config(page_title="Construction Code", page_icon="📐", layout="wide")

EXAMPLES = [
    "What is the maximum stud spacing for exterior walls?",
    "What is the minimum headroom in stairways?",
    "What are the minimum opening area and sill height of a bedroom escape window?",
    "What is the maximum riser height for stairs?",
    "How many exit stairs does a commercial office building need?",
]


@st.cache_resource(show_spinner="Building the index…")
def get_pipeline(mode: str, hybrid: bool, top_k: int) -> RAGPipeline:
    cfg = load_config(None, chunking__mode=mode, retrieval__hybrid=hybrid, retrieval__top_k=top_k)
    return RAGPipeline(cfg)


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Retrieval settings")
    mode = st.radio("Chunking", ["section", "naive"],
                    format_func=lambda m: "Clause-level" if m == "section" else "Fixed 500 chars (baseline)")
    hybrid = st.toggle("Hybrid retrieval (dense + BM25)", value=True)
    top_k = st.slider("Top-K chunks", 3, 10, 5)
    pipe = get_pipeline(mode, hybrid, top_k)
    use_llm = st.toggle("Generate with LLM", value=pipe.llm.available, disabled=not pipe.llm.available,
                        help="Available once DEEPSEEK_API_KEY is set; otherwise answers are extractive quotes")
    st.divider()
    st.caption(f"Knowledge base: {len(pipe.clauses)} provisions / {len(pipe.chunks)} chunks")
    st.caption(f"Embedding model: {pipe.embedder.name}")
    st.caption(f"Reranker: {'on' if pipe.cfg['rerank'].get('enabled') else 'off'}")
    st.caption(f"Generation: {'DeepSeek' if (pipe.llm.available and use_llm) else 'extractive (no LLM)'}")
    if st.button("Clear chat"):
        st.session_state.history = []
        st.rerun()

is_sample = "sample" in pipe.cfg["data"]["source"]
st.title("📐 Construction Code")
st.caption(
    "Ask in English or Chinese. Every statement cites its IRC section, and each citation is checked "
    "against the retrieved text. "
    + ("Demo data is a paraphrased sample, not the ICC text." if is_sample
       else "Source: International Residential Code (IRC) 2018.")
)

if "history" not in st.session_state:
    st.session_state.history = []  # [(question, Answer)]

cols = st.columns(len(EXAMPLES))
picked = None
for c, q in zip(cols, EXAMPLES):
    if c.button(q, width="stretch"):
        picked = q


def render_answer(ans):
    if ans.refused:
        st.warning(ans.text)
    else:
        st.markdown(ans.text)
    if ans.citations:
        def badge(c):
            if c.source == "retrieved":
                return f":green-badge[✓ {c.section_id}]"
            if c.source == "cross_ref":
                return f":orange-badge[↪ {c.section_id} cross-reference in source]"
            return f":red-badge[✗ {c.section_id} unsupported]"
        badges = " ".join(badge(c) for c in ans.citations)
        st.markdown(f"**Citation check** {badges}")
    with st.expander(f"Retrieval details ({len(ans.hits)} chunks)", expanded=False):
        if ans.glossary_hits:
            st.caption("Term expansion: " + ", ".join(ans.glossary_hits) + f" → `{ans.rewritten_query}`")
        st.dataframe(
            [{"Chunk": h.chunk.chunk_id,
              "Sections": ", ".join(h.chunk.section_ids),
              "Dense score": round(h.dense, 3) if h.dense is not None else None,
              "Dense rank": h.dense_rank, "BM25 rank": h.bm25_rank,
              "Rerank score": round(h.rerank, 3) if h.rerank is not None else None,
              "Found by": " + ".join(h.sources)} for h in ans.hits],
            hide_index=True, width="stretch")
        for h in ans.hits:
            st.markdown(f"**{h.chunk.chunk_id}**　{h.chunk.path}")
            st.code(h.chunk.text, language=None, wrap_lines=True)


for q, a in st.session_state.history:
    with st.chat_message("user"):
        st.write(q)
    with st.chat_message("assistant"):
        render_answer(a)

question = st.chat_input("e.g. How high should a stair handrail be?") or picked
if question:
    prior = [q for q, _ in st.session_state.history]
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        with st.spinner("Searching…"):
            ans = pipe.ask(question, prior, use_llm=use_llm)
        render_answer(ans)
    st.session_state.history.append((question, ans))
