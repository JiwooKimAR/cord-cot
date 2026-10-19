"""Appendix Table 8 - A-axis under alternative necessity-side disruptions (GPU).

Same two-pass protocol as ``scripts/02_run_axis_a.py`` (HF logits, then vLLM
answers), but the CoT suffix from p% on is replaced by one of four operators
instead of inverse-decoding ranks:

  random           uniform over valid tokens         (seeded per query/position)
  kl               sampled from Q(t) ~ 1/p(t)         (seeded per query/position)
  filler           "..." tokens                       (Lanham et al., 2023)
  early_answering  suffix removed, no replacement     (Lanham et al., 2023)

Writes ``<results>/disruption/<model>/<task>/experiment_a_ops.jsonl`` with the
same schema as ``experiment_a.jsonl`` (``rank_position`` = operator name). The
five inverse-decoding rows of Table 8 come from the main ``experiment_a.jsonl``.

    python appendix_experiments/disruption/run_operators.py \
        --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --task gsm8k
"""

import argparse
import gc
import hashlib
import importlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "scripts"))
import common  # noqa: E402

import crr.config as cfg  # noqa: E402
from crr.axes.necessity import (  # noqa: E402
    build_edited_prompt, compute_baseline_logits_batched, select_inverse_tokens_batched_view,
    set_special_tokens,
)
from crr.judge import judge_shortcut  # noqa: E402
from crr.models.generation import vllm_batch_generate  # noqa: E402
from crr.models.loader import load_model_and_tokenizer, load_vllm_engine, unload_vllm_engine  # noqa: E402

axis_a = importlib.import_module("02_run_axis_a")  # baseline_sig / load_state / answer budget

OPERATORS = ["random", "kl", "filler", "early_answering"]
SEED = 42


def stochastic_seed(query_id, operator: str, position: int, base_seed: int = SEED) -> int:
    """Stable 32-bit seed per (query, operator, position) for the random / kl operators."""
    key = f"{base_seed}|{query_id}|{operator}|{position}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "big") & 0xFFFFFFFF


def filler_ids(tok):
    ids = tok.encode("...", add_special_tokens=False) or tok.encode("…", add_special_tokens=False)
    if not ids:
        raise RuntimeError("cannot tokenize the ellipsis for the filler operator")
    return ids


def pass1(args, rows, state_path):
    done = axis_a.load_state(state_path, rows)
    todo = [r for r in rows if r["query_id"] not in done]
    print(f"[pass1] {len(done)} instances done, {len(todo)} to process")
    if not todo:
        return
    model, tok = load_model_and_tokenizer(common.engine_config(args.model), gpu_id=0)
    set_special_tokens(tok)
    fill = filler_ids(tok)
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
            for op in args.operators:
                for p in cfg.POSITION_GRID:
                    inv = select_inverse_tokens_batched_view(
                        logits[0:1], pad_lefts[0], prompt_lens[0], r["baseline_generated_ids"], op, p,
                        temperature=1.0,
                        rng_seed=stochastic_seed(r["query_id"], op, p) if op in ("random", "kl") else None,
                        filler_token_ids=fill if op == "filler" else None)
                    prompts.append({
                        "rank_position": op, "intervention_point": p,
                        "replacement_start": inv["replacement_start"],
                        "n_replaced": inv["n_replaced"],
                        "n_cot_original": inv["n_cot_original"],
                        "prompt": build_edited_prompt(tok, r["prompt"], inv["edited_cot_ids"],
                                                      max_total_tokens=max_prompt_tokens),
                    })
            common.append_jsonl(f, [{"query_id": r["query_id"], "baseline_sig": axis_a.baseline_sig(r),
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
    state = axis_a.load_state(state_path, rows)
    missing = [q for q in base if q not in state]
    if missing:
        raise SystemExit(f"pass-1 state missing {len(missing)} instances (e.g. {missing[:5]})")
    entries = [(r["query_id"], c) for r in rows for c in state[r["query_id"]]["conditions"]]
    print(f"[pass2] {len(entries)} prompts")
    engine, tok = load_vllm_engine(common.engine_config(args.model, gpu_memory_utilization=0.6))
    outs = vllm_batch_generate(engine, tok, [c["prompt"] for _, c in entries],
                               max_new_tokens=axis_a.ANSWER_TOKENS)
    unload_vllm_engine(engine)
    results = []
    for (qid, c), o in zip(entries, outs):
        j = judge_shortcut("\\boxed{" + o["generated_text"], base[qid]["baseline_answer"])
        results.append({
            "query_id": qid, "dataset_type": args.task, "model": args.model,
            "condition_type": "operator",
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
    ap.add_argument("--task", default="gsm8k", choices=common.TASKS)
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--operators", nargs="+", default=OPERATORS, choices=OPERATORS)
    ap.add_argument("--limit", type=int, default=None, help="first N instances only")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    rows = common.read_jsonl(os.path.join(common.cell_dir(args.results_dir, args.model, args.task),
                                          "baseline.jsonl"))[:args.limit]
    if not rows:
        raise SystemExit("no baseline.jsonl; run scripts/01_generate_baseline.py first")
    d = os.path.join(args.results_dir, "disruption", common.short_name(args.model), args.task)
    os.makedirs(d, exist_ok=True)
    out_path = os.path.join(d, "experiment_a_ops.jsonl")
    state_path = os.path.join(d, "experiment_a_ops_pass1_state.jsonl")
    if os.path.exists(out_path) and not args.overwrite:
        print(f"[skip] {out_path} exists (use --overwrite)")
        return
    if args.overwrite and os.path.exists(state_path):
        os.remove(state_path)
    pass1(args, rows, state_path)
    pass2(args, rows, state_path, out_path)


if __name__ == "__main__":
    main()
