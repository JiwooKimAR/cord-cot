"""Step 2 - A-axis (necessity): inverse decoding over the full P x R grid.

Pass 1 (HF): one forward pass over prompt + baseline CoT per instance gives the
logits; for every (rank r, position p) the CoT suffix from p% on is replaced
with rank-r tokens and the edited prompt is built. Prompts are streamed to a
state file, so an interrupted run resumes.
Pass 2 (vLLM): the answer is generated for every edited prompt and judged
against Y_0. Writes ``<results>/<model>/<task>/experiment_a.jsonl``.

    python scripts/02_run_axis_a.py --model Qwen/Qwen2.5-Math-7B-Instruct --task gsm8k
"""

import argparse
import gc
import hashlib
import json
import os

import common

import crr.config as cfg
from crr.axes.necessity import (
    build_edited_prompt, compute_baseline_logits_batched, select_inverse_tokens_batched_view,
    set_special_tokens,
)
from crr.judge import judge_shortcut
from crr.models.generation import vllm_batch_generate
from crr.models.loader import load_model_and_tokenizer, load_vllm_engine, unload_vllm_engine

ANSWER_TOKENS = 256


def baseline_sig(r) -> str:
    """Fingerprint of the baseline an instance's edited prompts were built from."""
    return hashlib.sha1(json.dumps([r["prompt"], r["baseline_generated_ids"]]).encode()).hexdigest()


def load_state(state_path, rows):
    """Pass-1 entries whose baseline signature still matches (stale entries are dropped)."""
    sigs = {r["query_id"]: baseline_sig(r) for r in rows}
    state = {}
    for s in common.read_jsonl(state_path):
        if sigs.get(s["query_id"]) == s.get("baseline_sig"):
            state[s["query_id"]] = s
    return state


def pass1(args, rows, state_path):
    done = load_state(state_path, rows)
    todo = [r for r in rows if r["query_id"] not in done]
    print(f"[pass1] {len(done)} instances done, {len(todo)} to process")
    if not todo:
        return
    model, tok = load_model_and_tokenizer(common.engine_config(args.model), gpu_id=0)
    set_special_tokens(tok)
    # Edited prompts must fit the pass-2 vLLM context (minus answer budget + margin).
    max_pos = common.native_max_positions(args.model)
    ctx = common.vllm_context_len(args.model)
    if max_pos:
        ctx = min(ctx, max_pos)
    max_prompt_tokens = ctx - 320
    if max_pos:
        max_prompt_tokens = min(max_prompt_tokens, max_pos - 256)

    import torch
    with open(state_path, "a") as f:
        for i, r in enumerate(todo):
            logits, _, prompt_lens, pad_lefts = compute_baseline_logits_batched(
                model, tok, [r["prompt"]], [r["baseline_generated_ids"]])
            prompts = []
            for rank in cfg.RANK_GRID:
                for p in cfg.POSITION_GRID:
                    inv = select_inverse_tokens_batched_view(
                        logits[0:1], pad_lefts[0], prompt_lens[0],
                        r["baseline_generated_ids"], rank, p, temperature=1.0)
                    prompts.append({
                        "rank_position": rank, "intervention_point": p,
                        "replacement_start": inv["replacement_start"],
                        "n_replaced": inv["n_replaced"],
                        "n_cot_original": inv["n_cot_original"],
                        "prompt": build_edited_prompt(tok, r["prompt"], inv["edited_cot_ids"],
                                                      max_total_tokens=max_prompt_tokens),
                    })
            common.append_jsonl(f, [{"query_id": r["query_id"], "baseline_sig": baseline_sig(r),
                                     "conditions": prompts}])
            del logits
            torch.cuda.empty_cache()
            if (i + 1) % 50 == 0:
                print(f"[pass1] {i + 1}/{len(todo)}")
    del model
    gc.collect()
    torch.cuda.empty_cache()


def pass2(args, rows, state_path, out_path):
    base = {r["query_id"]: r for r in rows}
    state = load_state(state_path, rows)
    missing = [q for q in base if q not in state]
    if missing:
        raise SystemExit(f"pass-1 state missing {len(missing)} instances (e.g. {missing[:5]})")
    entries = [(r["query_id"], c) for r in rows for c in state[r["query_id"]]["conditions"]]
    print(f"[pass2] {len(entries)} prompts")
    engine, tok = load_vllm_engine(common.engine_config(args.model, gpu_memory_utilization=0.6))
    outs = vllm_batch_generate(engine, tok, [c["prompt"] for _, c in entries],
                               max_new_tokens=ANSWER_TOKENS)
    unload_vllm_engine(engine)

    results = []
    for (qid, c), o in zip(entries, outs):
        j = judge_shortcut("\\boxed{" + o["generated_text"], base[qid]["baseline_answer"])
        results.append({
            "query_id": qid, "dataset_type": args.task, "model": args.model,
            "condition_type": "main",
            "rank_position": c["rank_position"], "intervention_point": c["intervention_point"],
            "baseline_answer": base[qid]["baseline_answer"],
            "predicted_answer": j["predicted_answer"], "extracted": j["extracted"],
            "is_shortcut": j["is_shortcut"], "no_answer_reason": j.get("no_answer_reason"),
            "replacement_start": c["replacement_start"], "n_replaced": c["n_replaced"],
            "n_cot_original": c["n_cot_original"],
        })
    common.write_jsonl(out_path, results)
    print(f"[done] {len(results)} rows -> {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True, choices=common.MODELS)
    ap.add_argument("--task", required=True, choices=common.TASKS)
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--limit", type=int, default=None, help="first N instances only")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    d = common.cell_dir(args.results_dir, args.model, args.task)
    out_path = os.path.join(d, "experiment_a.jsonl")
    state_path = os.path.join(d, "experiment_a_pass1_state.jsonl")
    if os.path.exists(out_path) and not args.overwrite:
        print(f"[skip] {out_path} exists (use --overwrite)")
        return
    if args.overwrite and os.path.exists(state_path):
        os.remove(state_path)
    rows = common.read_jsonl(os.path.join(d, "baseline.jsonl"))[:args.limit]
    if not rows:
        raise SystemExit(f"no baseline.jsonl in {d}; run 01_generate_baseline.py first")
    pass1(args, rows, state_path)
    pass2(args, rows, state_path, out_path)


if __name__ == "__main__":
    main()
