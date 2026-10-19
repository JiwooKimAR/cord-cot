"""Step 1 - baseline CoT and answer Y_0 for one (model, task) cell (vLLM).

The CoT M is generated greedily from the question X; Y_0 is then re-elicited
behind the fixed answer cue. Writes ``<results>/<model>/<task>/baseline.jsonl``.

    python scripts/01_generate_baseline.py --model Qwen/Qwen2.5-Math-7B-Instruct --task gsm8k
"""

import argparse
import os

import common
from datasets import load_dataset

from crr.answers import evaluate_correctness, extract_gold_answer
from crr.models.chat_template import render_chat, uses_user_only_chat
from crr.models.generation import vllm_batch_generate
from crr.models.loader import load_vllm_engine, unload_vllm_engine
from crr.prompts import ANSWER_CUE, SYSTEM_PROMPT

ANSWER_TOKENS = 64


def format_query(tokenizer, question: str, model: str) -> str:
    if uses_user_only_chat(model):
        messages = [{"role": "user", "content": f"{question}\n{SYSTEM_PROMPT}"}]
    else:
        messages = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": question}]
    return render_chat(tokenizer, messages)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True, choices=common.MODELS)
    ap.add_argument("--task", required=True, choices=common.TASKS)
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--limit", type=int, default=None, help="first N questions only")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    out_path = os.path.join(common.cell_dir(args.results_dir, args.model, args.task),
                            "baseline.jsonl")
    if os.path.exists(out_path) and not args.overwrite:
        print(f"[skip] {out_path} exists (use --overwrite)")
        return

    ds = common.DATASETS[args.task]
    data = load_dataset(ds["name"], ds["subset"], split=ds["split"]) if ds["subset"] \
        else load_dataset(ds["name"], split=ds["split"])
    n = len(data) if args.limit is None else min(args.limit, len(data))
    questions = [data[i][ds["question_field"]] for i in range(n)]
    gold_raws = [data[i][ds["answer_field"]] for i in range(n)]

    engine, tok = load_vllm_engine(common.engine_config(args.model))
    prompts = [format_query(tok, q, args.model) for q in questions]

    # Phase 1: free CoT.
    gens = vllm_batch_generate(engine, tok, prompts,
                               max_new_tokens=common.resolve_max_new_tokens(args.model, args.task))
    cot_ids = [g["generated_ids"] for g in gens]

    # Keep prompt + CoT + answer cue + answer within the model's position limit.
    max_pos = common.native_max_positions(args.model)
    if max_pos:
        cue_len = len(tok.encode(ANSWER_CUE, add_special_tokens=False))
        for i, p in enumerate(prompts):
            budget = max_pos - len(tok.encode(p)) - cue_len - ANSWER_TOKENS
            if len(cot_ids[i]) > budget:
                cot_ids[i] = cot_ids[i][:max(1, budget)]
    cot_texts = [tok.decode(ids, skip_special_tokens=True) for ids in cot_ids]

    # Phase 2: Y_0 behind the answer cue.
    answers = vllm_batch_generate(engine, tok,
                                  [p + c + ANSWER_CUE for p, c in zip(prompts, cot_texts)],
                                  max_new_tokens=ANSWER_TOKENS)
    unload_vllm_engine(engine)

    rows = []
    for i in range(n):
        gold = extract_gold_answer(str(gold_raws[i]), args.task)
        ev = evaluate_correctness("\\boxed{" + answers[i]["generated_text"], gold)
        rows.append({
            "query_id": i,
            "dataset_idx": i,
            "dataset_type": args.task,
            "question": questions[i],
            "prompt": prompts[i],
            "gold_answer": gold,
            "baseline_answer": ev["predicted_answer"],
            "baseline_correct": ev["correct"],
            "baseline_cot": cot_texts[i],
            "baseline_generated_ids": cot_ids[i],
            "baseline_n_tokens": len(cot_ids[i]),
            "model": args.model,
        })
    common.write_jsonl(out_path, rows)
    acc = sum(r["baseline_correct"] for r in rows) / max(1, len(rows))
    print(f"[done] {len(rows)} rows, acc={acc:.3f} -> {out_path}")


if __name__ == "__main__":
    main()
