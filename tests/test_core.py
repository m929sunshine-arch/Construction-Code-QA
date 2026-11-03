import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ircrag.generation import verify_citations  # noqa: E402
from ircrag.ingest import Chunk, load_chunks  # noqa: E402
from ircrag.pipeline import RAGPipeline, load_config  # noqa: E402
from ircrag.retrieval import Hit, tokenize  # noqa: E402

SAMPLE = "data/sample/irc_sample.md"


def test_section_chunking_keeps_exception_with_clause():
    chunks, clauses = load_chunks(SAMPLE, "section")
    assert len(chunks) == len(clauses) == 32
    top_plate = next(c for c in chunks if c.chunk_id == "R602.3.2")
    assert "Exception" in top_plate.text and top_plate.section_ids == ["R602.3.2"]


def test_naive_chunks_cover_multiple_sections():
    chunks, _ = load_chunks(SAMPLE, "naive")
    assert max(len(c.section_ids) for c in chunks) > 1


def test_section_id_prefix_tokens():
    assert {"r602", "r602.3", "r602.3.2"} <= set(tokenize("R602.3.2", expand_ids=True))


def test_verify_citations_flags_unretrieved():
    hits = [Hit(Chunk("R311.7.2", "Headroom ... spiral stairways per Section R311.7.10.1.", ["R311.7.2"]))]
    ans = "Headroom 6 ft 8 in [R311.7.2]. Spiral [R311.7.10.1]. Width [R311.7.1]."
    cites = {c.section_id: c.source for c in verify_citations(ans, hits)}
    assert cites == {"R311.7.2": "retrieved", "R311.7.10.1": "cross_ref", "R311.7.1": ""}


def test_pipeline_answers_and_refuses(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    cfg = load_config("config.yaml", embedding__backend="tfidf", rerank__enabled=False)
    p = RAGPipeline(cfg)
    ans = p.ask("外墙立柱间距最大可以做到多少？")
    assert not ans.refused and ans.hits[0].chunk.chunk_id == "R602.3.1"
    assert all(c.valid for c in ans.citations)
    assert p.ask("商业建筑的自动喷淋系统有什么要求？").refused


def test_milvus_store_matches_memory(tmp_path, monkeypatch):
    import pytest
    pytest.importorskip("pymilvus")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    q = "楼梯上方至少要留多高的净空？"
    mem = RAGPipeline(load_config("config.yaml", embedding__backend="tfidf", rerank__enabled=False)).ask(q)
    uri = str(tmp_path / "m.db")
    mil = RAGPipeline(load_config("config.yaml", embedding__backend="tfidf", rerank__enabled=False, vector_store__backend="milvus", vector_store__uri=uri))
    assert [h.chunk.chunk_id for h in mil.ask(q).hits] == [h.chunk.chunk_id for h in mem.hits]
    again = RAGPipeline(load_config("config.yaml", embedding__backend="tfidf", rerank__enabled=False, vector_store__backend="milvus", vector_store__uri=uri))
    from ircrag.retrieval import corpus_fingerprint
    assert again.store.has(corpus_fingerprint(again.chunks, again.embedder.name))  # 集合已建好，可复用
