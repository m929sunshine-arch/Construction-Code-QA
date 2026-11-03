"""用假的 OpenAI 客户端测试 LLM 路径：生成、引用校验、拒答透传、追问改写。"""
import os
import sys
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from ircrag.generation import REFUSAL  # noqa: E402
from ircrag.pipeline import RAGPipeline, load_config  # noqa: E402


class FakeClient:
    def __init__(self, reply):
        self.reply, self.calls = reply, []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, model, messages, temperature):
        self.calls.append(messages)
        text = self.reply(messages) if callable(self.reply) else self.reply
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


def make_pipe(reply):
    os.chdir(ROOT)
    p = RAGPipeline(load_config("config.yaml", embedding__backend="tfidf", rerank__enabled=False))
    p.llm.client = FakeClient(reply)
    return p


def test_llm_answer_flags_hallucinated_citation():
    p = make_pipe("楼梯净空不小于 6 英尺 8 英寸 [R311.7.2]。宽度不小于 36 英寸 [R399.9]。")
    ans = p.ask("楼梯上方至少要留多高的净空？")
    assert ans.mode == "llm"
    assert {c.section_id: c.valid for c in ans.citations} == {"R311.7.2": True, "R399.9": False}
    assert ans.hallucinated == ["R399.9"]
    prompt = p.llm.client.calls[0][1]["content"]
    assert '<section id="R311.7.2">' in prompt  # 条文带条款号送给模型


def test_llm_refusal_passes_through():
    p = make_pipe(REFUSAL["zh"])
    ans = p.ask("外墙立柱间距最大可以做到多少？")
    assert ans.refused and ans.mode == "refused" and not ans.citations


def test_threshold_refusal_skips_llm():
    p = make_pipe("should not be called")
    assert p.ask("商业建筑的自动喷淋系统有什么要求？").refused
    assert p.llm.client.calls == []


def test_followup_is_rewritten_by_llm():
    def reply(messages):
        if "Rewrite" in messages[-1]["content"]:
            return "楼梯踏面最小深度是多少？"
        return "踏面不小于 10 英寸 [R311.7.5.2]。"
    p = make_pipe(reply)
    ans = p.ask("那踏面呢？", history=["楼梯踏步高最大多少？"])
    assert "踏面最小深度" in ans.rewritten_query
    assert ans.hits[0].chunk.chunk_id == "R311.7.5.2" and all(c.valid for c in ans.citations)


def test_llm_retries_empty_response(monkeypatch):
    import ircrag.generation as g
    monkeypatch.setattr("time.sleep", lambda s: None)
    calls = {"n": 0}

    class Msg:  content = "OK [R311.7.2]"
    class Choice: message = Msg()
    class Resp: choices = [Choice()]

    class Completions:
        def create(self, **kw):
            calls["n"] += 1
            return None if calls["n"] < 3 else Resp()

    class Client:
        class chat:
            completions = Completions()

    llm = g.LLM({"model": "x", "base_url": "http://x"})
    llm.client = Client()
    assert llm.chat([{"role": "user", "content": "hi"}]) == "OK [R311.7.2]"
    assert calls["n"] == 3
