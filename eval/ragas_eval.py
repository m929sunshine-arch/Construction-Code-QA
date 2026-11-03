"""RAGAS 评测：忠实度（答案能否被检索文本支持）+ 上下文精确度（召回的条文是否有用）。

需要 LLM 生成答案，并用同一个 LLM 当评审，所以必须配置 DEEPSEEK_API_KEY。
    uv pip install -r requirements-eval.txt
    python -m eval.ragas_eval                      # 跑评测，结果写入 eval/results/ragas.csv
    python -m eval.ragas_eval --dry-run            # 只生成样本，不调用评审（用于检查数据）
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval.run_eval import HERE, load_questions  # noqa: E402
from ircrag.generation import DISCLAIMER  # noqa: E402
from ircrag.pipeline import RAGPipeline, load_config  # noqa: E402


def build_samples(pipe: RAGPipeline, questions: list[dict], use_llm: bool = True) -> list[dict]:
    """库外问题和被拒答的问题不参与：RAGAS 衡量的是“回答了的那部分”质量好不好。"""
    samples = []
    for q in questions:
        if not q["gold_sections"]:
            continue
        ans = pipe.ask(q["question"], q.get("history"), use_llm=use_llm)
        if ans.refused:
            continue
        response = ans.text
        for d in DISCLAIMER.values():
            response = response.replace(d, "").strip()
        samples.append({
            "id": q["id"],
            "user_input": q["question"],
            "retrieved_contexts": [h.chunk.text for h in ans.hits],
            "response": response,
        })
    return samples


def run_ragas(samples: list[dict], llm_cfg: dict):
    from langchain_openai import ChatOpenAI
    from ragas import EvaluationDataset, evaluate
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import Faithfulness, LLMContextPrecisionWithoutReference

    judge = LangchainLLMWrapper(ChatOpenAI(
        model=llm_cfg["model"], base_url=llm_cfg["base_url"],
        api_key=os.environ[llm_cfg["api_key_env"]], temperature=0))
    data = EvaluationDataset.from_list([{k: v for k, v in s.items() if k != "id"} for s in samples])
    result = evaluate(data, metrics=[Faithfulness(llm=judge), LLMContextPrecisionWithoutReference(llm=judge)])
    df = result.to_pandas()
    df.insert(0, "id", [s["id"] for s in samples])
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="默认读环境变量 IRC_RAG_CONFIG，再默认 config.yaml")
    ap.add_argument("--questions", default=os.path.join(HERE, "questions.jsonl"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    pipe = RAGPipeline(cfg)
    if not pipe.llm.available and not args.dry_run:
        sys.exit(f"未检测到 {cfg['llm']['api_key_env']}：RAGAS 需要 LLM 生成和评审。可先用 --dry-run 检查样本。")

    samples = build_samples(pipe, load_questions(args.questions), use_llm=pipe.llm.available)
    out_dir = os.path.join(HERE, "results")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "ragas_samples.jsonl"), "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"样本 {len(samples)} 条 → eval/results/ragas_samples.jsonl")
    if args.dry_run:
        return

    df = run_ragas(samples, cfg["llm"])
    df.to_csv(os.path.join(out_dir, "ragas.csv"), index=False, encoding="utf-8-sig")
    metrics = [c for c in df.columns if c not in ("id", "user_input", "retrieved_contexts", "response")]
    print(df[metrics].mean(numeric_only=True).round(3).to_string())


if __name__ == "__main__":
    main()
