"""Appendix Table 26 - M-mask (X visible) on the baseline CoT (HF).

``scripts/06_measure_refined.py`` measures the M-mask only on the refined CoT;
this measures the same protocol (21 positions, token unit, question visible)
on the original baseline CoT so that Table 26 has a baseline column for it.
Writes ``<results>/refine/<model>/<task>/base_mmask.jsonl``.

    python appendix_experiments/refine/run_base_mmask.py \
        --model Qwen/Qwen2.5-Math-7B-Instruct --task math500
"""

import argparse
import importlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "scripts"))
import common  # noqa: E402

import crr.config as cfg  # noqa: E402
from crr.axes.necessity import set_special_tokens  # noqa: E402
from crr.models.loader import load_model_and_tokenizer  # noqa: E402

measure_refined = importlib.import_module("06_measure_refined")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True, choices=common.MODELS)
    ap.add_argument("--task", default="math500", choices=common.TASKS)
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--tokens", type=int, default=measure_refined.AXES["mmask"][1],
                    help="answer budget (default: the same as step 6)")
    ap.add_argument("--output-name", default="base_mmask.jsonl")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    d = os.path.join(args.results_dir, "refine", common.short_name(args.model), args.task)
    out_path = os.path.join(d, args.output_name)
    if os.path.exists(out_path) and not args.overwrite:
        print(f"[skip] {out_path} exists (use --overwrite)")
        return
    # Same instances as step 6: those with a usable refined solution.
    refine_qids = {f["query_id"] for f in common.read_jsonl(os.path.join(d, "self_refine.jsonl"))}
    rows = [r for r in common.read_jsonl(os.path.join(common.cell_dir(args.results_dir, args.model, args.task),
                                                      "baseline.jsonl")) if r["query_id"] in refine_qids]
    model, tok = load_model_and_tokenizer(common.engine_config(args.model), gpu_id=0)
    set_special_tokens(tok)
    os.makedirs(d, exist_ok=True)
    with open(out_path + ".tmp", "w") as f:
        for i, row in enumerate(rows):
            try:
                res = measure_refined.measure("mmask", model, tok, row, args.model, args.tokens)
                recs = [{"qid": row["query_id"], "p": p, "baseline_answer": row["baseline_answer"],
                         "m_match_baseline": v} for p, v in res.items()]
            except RuntimeError as e:  # e.g. question span not located
                recs = [{"qid": row["query_id"], "p": p, "error": str(e)[:200]} for p in cfg.POSITION_GRID]
            common.append_jsonl(f, recs)
            if (i + 1) % 50 == 0:
                print(f"[mmask] {i + 1}/{len(rows)}")
    os.replace(out_path + ".tmp", out_path)
    print(f"[done] -> {out_path}")


if __name__ == "__main__":
    main()
