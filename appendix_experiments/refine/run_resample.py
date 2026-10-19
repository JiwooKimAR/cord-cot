"""Appendix Table 25 - X-only resampling (vLLM).

Samples K answers from the original input X alone (the baseline prompt; no
rationale, critique or refinement context) with temperature 0.7, top-p 1.0
and a fixed seed, and compares each extracted answer with the greedy baseline
answer Y_0. Writes ``<results>/refine/<model>/<task>/resample.jsonl``.

    python appendix_experiments/refine/run_resample.py \
        --model Qwen/Qwen2.5-Math-7B-Instruct --task math500
"""

import argparse
import importlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "scripts"))
import common  # noqa: E402

from crr.models.loader import load_vllm_engine, unload_vllm_engine  # noqa: E402

refine = importlib.import_module("05_run_self_refine")  # answer extraction


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True, choices=common.MODELS)
    ap.add_argument("--task", default="math500", choices=common.TASKS)
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--k", type=int, default=10, help="samples per instance")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    d = os.path.join(args.results_dir, "refine", common.short_name(args.model), args.task)
    out_path = os.path.join(d, "resample.jsonl")
    if os.path.exists(out_path) and not args.overwrite:
        print(f"[skip] {out_path} exists (use --overwrite)")
        return
    rows = common.read_jsonl(os.path.join(common.cell_dir(args.results_dir, args.model, args.task),
                                          "baseline.jsonl"))
    if not rows:
        raise SystemExit("no baseline.jsonl; run scripts/01_generate_baseline.py first")

    engine, _ = load_vllm_engine({"models": [{"name": args.model, "dtype": "bfloat16",
                                              "gpu_memory_utilization": 0.85,
                                              "max_model_len": args.max_model_len}],
                                  "generation": {"max_new_tokens": args.max_new_tokens}})
    from vllm import SamplingParams
    sampling = SamplingParams(temperature=args.temperature, top_p=args.top_p, n=args.k,
                              max_tokens=args.max_new_tokens, seed=args.seed)
    outs = engine.generate([r["prompt"] for r in rows], sampling)
    unload_vllm_engine(engine)

    out = []
    for r, o in zip(rows, outs):
        y_0 = str(r.get("baseline_answer", "")).strip()
        answers = [refine.extract_refined_answer(s.text, args.task) for s in o.outputs]
        changed = [None if a is None else int(str(a).strip() != y_0) for a in answers]
        valid = [c for c in changed if c is not None]
        out.append({"query_id": r["query_id"], "model": args.model, "y_0": y_0,
                    "answers": answers, "changed": changed, "n_valid": len(valid),
                    "instance_change_rate": (sum(valid) / len(valid)) if valid else None})
    os.makedirs(d, exist_ok=True)
    common.write_jsonl(out_path, out)
    print(f"[done] {len(out)} rows -> {out_path}")


if __name__ == "__main__":
    main()
