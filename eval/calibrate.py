"""校准拒答阈值：看库内问题和库外问题的检索分数分布，取中间值作为阈值。

换了向量模型或开了重排之后都要重跑：
    python -m eval.calibrate
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval.run_eval import HERE, load_questions  # noqa: E402
from ircrag.pipeline import RAGPipeline, load_config  # noqa: E402


_LINES: list[str] = []


def say(*args):
    """同时打印到屏幕和保存到文件，窗口关掉也能找回结果。"""
    line = " ".join(str(a) for a in args)
    print(line, flush=True)
    _LINES.append(line)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="默认读环境变量 IRC_RAG_CONFIG，再默认 config.yaml")
    ap.add_argument("--questions", default=os.path.join(HERE, "questions.jsonl"))
    args = ap.parse_args()
    cfg = load_config(args.config)
    pipe = RAGPipeline(cfg)
    rerank_on = bool(cfg["rerank"].get("enabled"))
    rows = []
    for q in load_questions(args.questions):
        query = q["question"]
        if q.get("history"):
            query = f"{q['history'][-1]} {query}"  # 与离线追问处理一致，不调用 LLM
        expanded, _ = pipe.rewriter.expand(query)
        hits = pipe.retriever.search(expanded)
        dense = max((h.dense for h in hits if h.dense is not None), default=0.0)
        rr = hits[0].rerank if hits and hits[0].rerank is not None else None
        rows.append((q["id"], bool(q["gold_sections"]), dense, rr, q["question"]))

    def report(name, idx):
        inscope = sorted(r[idx] for r in rows if r[1] and r[idx] is not None)
        ood = sorted(r[idx] for r in rows if not r[1] and r[idx] is not None)
        if not inscope or not ood:
            return
        lo, hi = inscope[0], ood[-1]
        say(f"\n[{name}] 库内问题最低分 {lo:.3f}（中位数 {inscope[len(inscope) // 2]:.3f}），库外问题最高分 {hi:.3f}")
        if lo > hi:
            say(f"  两组完全分开，建议阈值 ≈ {(lo + hi) / 2:.3f}")
        else:
            say("  两组有重叠，无论取多少都会有误拒或漏拒；重叠的问题如下：")
            for r in rows:
                if r[idx] is not None and lo <= r[idx] <= hi:
                    say(f"    {r[0]} {'库内' if r[1] else '库外'} {r[idx]:.3f}  {r[4]}")

    say(f"向量模型：{pipe.embedder.name}；重排：{'开' if rerank_on else '关'}")
    report("向量相似度 → refusal.min_dense_score", 2)
    if rerank_on:
        report("重排分 → refusal.min_rerank_score", 3)
    out = os.path.join(HERE, "results", "calibration.txt")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(_LINES) + "\n")
    say(f"\n结果已保存到 {out}")


if __name__ == "__main__":
    main()
