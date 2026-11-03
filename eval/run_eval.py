"""消融实验：同一评测集、同一生成方式，每组只改一个变量。

用法：
    python -m eval.run_eval                 # 跑 A/B/C（D 需要重排模型）
    python -m eval.run_eval --groups A B C D

输出：eval/results/ablation.csv、eval/results/ablation.md、eval/results/details.jsonl
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from statistics import mean

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ircrag.pipeline import RAGPipeline, load_config  # noqa: E402

GROUPS = {
    "A": {"desc": "固定 500 字分块 + 向量", "chunking__mode": "naive", "retrieval__hybrid": False, "rerank__enabled": False},
    "B": {"desc": "条款级分块 + 向量", "chunking__mode": "section", "retrieval__hybrid": False, "rerank__enabled": False},
    "C": {"desc": "条款级分块 + 向量 + BM25", "chunking__mode": "section", "retrieval__hybrid": True, "rerank__enabled": False},
    "D": {"desc": "条款级分块 + 混合检索 + 重排", "chunking__mode": "section", "retrieval__hybrid": True, "rerank__enabled": True},
}
HERE = os.path.dirname(os.path.abspath(__file__))


def load_questions(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def covers(sid: str, gold: str) -> bool:
    return sid == gold or sid.startswith(gold + ".")


def evaluate(pipe: RAGPipeline, questions: list[dict], use_llm: bool):
    rows = []
    for q in questions:
        ans = pipe.ask(q["question"], q.get("history"), use_llm=use_llm)
        retrieved = [h.chunk.section_ids for h in ans.hits]
        flat = [s for ids in retrieved for s in ids]
        gold = q["gold_sections"]
        cited = [c.section_id for c in ans.citations]
        row = {
            "id": q["id"], "type": q["type"], "lang": q["lang"], "refused": ans.refused,
            "retrieved": [h.chunk.chunk_id for h in ans.hits], "cited": cited,
            "hallucinated": ans.hallucinated,
            "cross_refs": [c.section_id for c in ans.citations if c.source == "cross_ref"],
            "sections_per_chunk": mean(len(ids) for ids in retrieved) if retrieved else 0.0,
        }
        if gold:
            row["recall@5"] = mean(any(covers(s, g) for s in flat) for g in gold)
            row["hit@1"] = float(bool(retrieved) and any(covers(s, g) for s in retrieved[0] for g in gold))
            rank = next((i + 1 for i, ids in enumerate(retrieved) if any(covers(s, g) for s in ids for g in gold)), None)
            row["mrr"] = 1 / rank if rank else 0.0
            row["cite_ok"] = float(any(covers(c, g) or covers(g, c) for c in cited for g in gold)) if not ans.refused else 0.0
        rows.append(row)
    return rows


def summarize(rows):
    inscope = [r for r in rows if "recall@5" in r]
    ood = [r for r in rows if "recall@5" not in r]
    zh = [r for r in inscope if r["lang"] == "zh"]
    en = [r for r in inscope if r["lang"] == "en"]
    n_cites = sum(len(r["cited"]) for r in rows)
    return {
        "条款召回率@5": mean(r["recall@5"] for r in inscope),
        "Top-1 命中率": mean(r["hit@1"] for r in inscope),
        "MRR": mean(r["mrr"] for r in inscope),
        "引用准确率": mean(r["cite_ok"] for r in inscope),
        "幻觉引用率": (sum(len(r["hallucinated"]) for r in rows) / n_cites) if n_cites else 0.0,
        "交叉引用占比": (sum(len(r["cross_refs"]) for r in rows) / n_cites) if n_cites else 0.0,
        "拒答准确率": mean(r["refused"] for r in ood) if ood else 0.0,
        "误拒率": mean(r["refused"] for r in inscope),
        "中文召回率@5": mean(r["recall@5"] for r in zh) if zh else 0.0,
        "英文召回率@5": mean(r["recall@5"] for r in en) if en else 0.0,
        # 固定长度分块一块常跨多个条款，召回率会被“放大”，需结合该列和引用准确率一起看
        "每块平均条款数": mean(r["sections_per_chunk"] for r in rows if r["retrieved"]),
    }


def main():
    logging.basicConfig(level=logging.WARNING)
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="默认读环境变量 IRC_RAG_CONFIG，再默认 config.yaml")
    ap.add_argument("--questions", default=os.path.join(HERE, "questions.jsonl"))
    ap.add_argument("--groups", nargs="+", default=["A", "B", "C"])
    ap.add_argument("--no-llm", action="store_true", help="强制用离线摘录模式，排除 LLM 的影响")
    args = ap.parse_args()

    questions = load_questions(args.questions)
    out_dir = os.path.join(HERE, "results")
    os.makedirs(out_dir, exist_ok=True)
    table, details = [], []
    for g in args.groups:
        spec = dict(GROUPS[g])
        desc = spec.pop("desc")
        try:
            pipe = RAGPipeline(load_config(args.config, **spec))
        except Exception as e:
            print(f"[{g}] 跳过：{desc}（{type(e).__name__}: {e}）")
            continue
        rows = evaluate(pipe, questions, use_llm=not args.no_llm)
        s = summarize(rows)
        mode = "LLM" if (pipe.llm.available and not args.no_llm) else "离线摘录"
        table.append({"组别": g, "配置": desc, "向量模型": pipe.embedder.name, "生成": mode,
                      **{k: round(v, 3) for k, v in s.items()}})
        details += [{"group": g, **r} for r in rows]
        print(f"[{g}] {desc}: " + ", ".join(f"{k} {v:.2f}" for k, v in s.items()))

    if not table:
        return
    with open(os.path.join(out_dir, "ablation.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)
    with open(os.path.join(out_dir, "details.jsonl"), "w", encoding="utf-8") as f:
        for d in details:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    cols = ["组别", "配置", "条款召回率@5", "Top-1 命中率", "引用准确率", "幻觉引用率", "拒答准确率", "误拒率", "中文召回率@5", "每块平均条款数"]
    fmt = {"每块平均条款数": lambda v: f"{v:.1f}"}
    pct = lambda v: f"{float(v):.0%}" if isinstance(v, (int, float)) else str(v)  # noqa: E731
    md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    md += ["| " + " | ".join(fmt.get(c, pct)(r[c]) for c in cols) + " |" for r in table]
    meta = f"\n\n评测集 {len(questions)} 题；向量模型 {table[0]['向量模型']}；生成方式 {table[0]['生成']}。\n"
    with open(os.path.join(out_dir, "ablation.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + meta)
    print("\n" + "\n".join(md) + meta)


if __name__ == "__main__":
    main()
