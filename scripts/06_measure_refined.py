"""Step 6 - re-measure the refined CoT on both axes for Table 4 (HF).

The refined solution from step 5 replaces the baseline CoT (its extracted answer
becomes the reference), then:
  inverse - A-axis bottom-1 inverse decoding         -> inverse.jsonl  (a_match_baseline)
  xmmask  - B-axis X/M-mask recovery (token unit)    -> xmmask.jsonl   (b_match_baseline)
  mmask   - M-mask only, question visible (token)    -> mmask.jsonl    (m_match_baseline)
Writes to ``<results>/refine/<model>/<task>/``.

    python scripts/06_measure_refined.py --model Qwen/Qwen2.5-Math-7B-Instruct --task math500

Answer budgets match the base measurements: inverse 256 tokens (as the base
A-axis, step 2), X/M-mask and M-mask 64 (as the base B-axis, step 3).
"""

import argparse
import os

import common

import crr.config as cfg
from crr.axes.necessity import build_edited_prompt, compute_baseline_logits, select_inverse_tokens, set_special_tokens
from crr.axes.sufficiency import recover_under_mmask, recover_under_xmask
from crr.judge import judge_shortcut
from crr.models.generation import greedy_generate
from crr.models.loader import load_model_and_tokenizer

AXES = {"inverse": ("a_match_baseline", 256),
        "xmmask": ("b_match_baseline", 64),
        "mmask": ("m_match_baseline", 64)}


def build_refined_baseline(rows, refine, tok):
    """Baseline rows with the refined solution as the CoT (skipped / empty refines dropped)."""
    base = {r["query_id"]: r for r in rows}
    out = []
    for f in refine:
        if f["query_id"] not in base or f.get("skipped_too_long"):
            continue
        text = f.get("refine_text", "")
        if not text or len(text) < 10:
            continue
        r = dict(base[f["query_id"]])
        r["baseline_cot"] = text
        if f.get("y_refined") is not None:
            r["baseline_answer"] = str(f["y_refined"])
        r["baseline_generated_ids"] = tok.encode(text, add_special_tokens=False)
        r["baseline_n_tokens"] = len(r["baseline_generated_ids"])
        out.append(r)
    return out


def measure(axis, model, tok, row, model_name, max_new_tokens):
    """Per-p judgments (1 unchanged / 0 changed / None no answer) for one refined instance."""
    out = {}
    if axis == "inverse":
        import torch
        prompt_ids = torch.tensor([tok.encode(row["prompt"])], device=model.device)
        logits, plen = compute_baseline_logits(model, prompt_ids, row["baseline_generated_ids"])
    for p in cfg.POSITION_GRID:
        if axis == "inverse":
            sel = select_inverse_tokens(logits, plen, row["baseline_generated_ids"], "bottom-1", p)
            enc = tok(build_edited_prompt(tok, row["prompt"], sel["edited_cot_ids"]),
                      return_tensors="pt").to(model.device)
            gen = greedy_generate(model, enc["input_ids"], enc["attention_mask"],
                                  max_new_tokens=max_new_tokens, eos_token_id=tok.eos_token_id)
            full = "\\boxed{" + tok.decode(gen["generated_ids"], skip_special_tokens=True)
        else:
            fn = recover_under_xmask if axis == "xmmask" else recover_under_mmask
            full = fn(model, tok, question=row["question"], cot_text=row["baseline_cot"],
                      cot_token_ids=row["baseline_generated_ids"], p_visible=p, mask_unit="token",
                      model_name=model_name, max_new_tokens=max_new_tokens)["full_cot"]
        j = judge_shortcut(full, row["baseline_answer"])
        out[p] = None if j["is_shortcut"] is None else int(j["is_shortcut"])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True, choices=common.MODELS)
    ap.add_argument("--task", required=True, choices=common.TASKS)
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--axes", nargs="+", default=list(AXES), choices=list(AXES))
    for ax, (_, n) in AXES.items():
        ap.add_argument(f"--{ax}-tokens", type=int, default=n, help=f"answer budget for {ax}")
    ap.add_argument("--limit", type=int, default=None, help="first N instances only")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    d = os.path.join(args.results_dir, "refine", common.short_name(args.model), args.task)
    rows = common.read_jsonl(os.path.join(common.cell_dir(args.results_dir, args.model, args.task),
                                          "baseline.jsonl"))
    refine = common.read_jsonl(os.path.join(d, "self_refine.jsonl"))
    if not rows or not refine:
        raise SystemExit("need baseline.jsonl and self_refine.jsonl (steps 1 and 5)")

    model, tok = load_model_and_tokenizer(common.engine_config(args.model), gpu_id=0)
    set_special_tokens(tok)
    refined = build_refined_baseline(rows, refine, tok)[:args.limit]
    print(f"[refined] {len(refined)} instances")
    for axis in args.axes:
        key = AXES[axis][0]
        n_tok = getattr(args, f"{axis}_tokens")
        out_path = os.path.join(d, f"{axis}.jsonl")
        if os.path.exists(out_path) and not args.overwrite:
            print(f"[skip] {out_path} exists (use --overwrite)")
            continue
        with open(out_path + ".tmp", "w") as f:
            for i, row in enumerate(refined):
                try:
                    res = measure(axis, model, tok, row, args.model, n_tok)
                    recs = [{"qid": row["query_id"], "p": p, "baseline_answer": row["baseline_answer"],
                             key: v} for p, v in res.items()]
                except RuntimeError as e:  # e.g. question span not located
                    recs = [{"qid": row["query_id"], "p": p, "error": str(e)[:200]}
                            for p in cfg.POSITION_GRID]
                common.append_jsonl(f, recs)
                if (i + 1) % 50 == 0:
                    print(f"[{axis}] {i + 1}/{len(refined)}")
        os.replace(out_path + ".tmp", out_path)
        print(f"[done] {axis} -> {out_path}")


if __name__ == "__main__":
    main()
