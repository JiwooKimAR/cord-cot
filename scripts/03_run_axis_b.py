"""Step 3 - B-axis (sufficiency): X/M-mask recovery over the position grid.

The question X and all but the first p% of the CoT are attention-masked and the
answer is regenerated (HF). All (p, unit) conditions of an instance run as one
stacked batch. Rows are appended per instance, so an interrupted run resumes.
Writes ``<results>/<model>/<task>/experiment_b.jsonl``.

    python scripts/03_run_axis_b.py --model Qwen/Qwen2.5-Math-7B-Instruct --task gsm8k
"""

import argparse
import os

import common

import crr.config as cfg
from crr.axes.sufficiency import recover_under_xmask, recover_under_xmask_batched
from crr.judge import judge_shortcut
from crr.models.loader import load_model_and_tokenizer

ANSWER_TOKENS = 64
# Both units were measured for the paper (tables use "token"); keeping both also
# keeps the batch composition, which can matter for bf16 greedy decoding.
MASK_UNITS = ["token", "sentence"]


def result_row(args, row, p, unit, out=None, error=None):
    base = {"query_id": row["query_id"], "dataset_type": args.task, "model": args.model,
            "condition_type": "main", "p_visible": p, "mask_unit": unit,
            "baseline_answer": row["baseline_answer"]}
    if error is not None:
        return {**base, "predicted_answer": None, "extracted": False, "is_shortcut": None,
                "skipped": True, "skip_reason": str(error)[:200]}
    j = judge_shortcut(out["full_cot"], row["baseline_answer"])
    return {**base, "predicted_answer": j["predicted_answer"], "extracted": j["extracted"],
            "is_shortcut": j["is_shortcut"], "no_answer_reason": j.get("no_answer_reason"),
            "continuation": out["continuation"], "n_tokens": out["n_tokens"],
            "n_question_masked": out.get("n_question_masked"),
            "n_cot_masked": out.get("n_cot_masked"), "q_start": out.get("q_start"),
            "cot_start": out.get("cot_start"), "span_method": out.get("span_method")}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True, choices=common.MODELS)
    ap.add_argument("--task", required=True, choices=common.TASKS)
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--limit", type=int, default=None, help="first N instances only")
    ap.add_argument("--batch-tokens", type=int, default=60000,
                    help="rows x prompt length per stacked batch (0 = per-condition calls)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    d = common.cell_dir(args.results_dir, args.model, args.task)
    out_path = os.path.join(d, "experiment_b.jsonl")
    if args.overwrite and os.path.exists(out_path):
        os.remove(out_path)
    rows = common.read_jsonl(os.path.join(d, "baseline.jsonl"))[:args.limit]
    if not rows:
        raise SystemExit(f"no baseline.jsonl in {d}; run 01_generate_baseline.py first")
    conditions = [(p, u) for p in cfg.POSITION_GRID for u in MASK_UNITS]
    done = {}
    for r in common.read_jsonl(out_path):
        done[r["query_id"]] = done.get(r["query_id"], 0) + 1
    todo = [r for r in rows if done.get(r["query_id"], 0) < len(conditions)]
    if any(0 < done.get(r["query_id"], 0) < len(conditions) for r in todo):
        raise SystemExit(f"{out_path} has partially written instances; rerun with --overwrite")
    print(f"[axis-b] {len(rows) - len(todo)} instances done, {len(todo)} to run")
    if not todo:
        return

    model, tok = load_model_and_tokenizer(common.engine_config(args.model), gpu_id=0)
    kw = dict(question=None, model_name=args.model, max_new_tokens=ANSWER_TOKENS)
    with open(out_path, "a") as f:
        for i, row in enumerate(todo):
            kw.update(question=row["question"], cot_text=row["baseline_cot"],
                      cot_token_ids=row["baseline_generated_ids"])
            outs = None
            if args.batch_tokens:
                try:
                    outs = recover_under_xmask_batched(model, tok, conditions=conditions,
                                                       max_batch_tokens=args.batch_tokens, **kw)
                except RuntimeError as e:
                    print(f"[qid {row['query_id']}] batched path failed ({e}); per-condition")
            recs = []
            for ci, (p, u) in enumerate(conditions):
                if outs is not None:
                    recs.append(result_row(args, row, p, u, out=outs[ci]))
                    continue
                try:
                    out = recover_under_xmask(model, tok, p_visible=p, mask_unit=u, **kw)
                    recs.append(result_row(args, row, p, u, out=out))
                except RuntimeError as e:  # question span not located reliably
                    recs.append(result_row(args, row, p, u, error=e))
            common.append_jsonl(f, recs)
            if (i + 1) % 50 == 0:
                print(f"[axis-b] {i + 1}/{len(todo)}")
    print(f"[done] -> {out_path}")


if __name__ == "__main__":
    main()
