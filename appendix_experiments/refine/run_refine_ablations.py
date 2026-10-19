"""Appendix Table 24 - refinement-input ablations of self-refine (vLLM).

Re-runs only the refine round of ``scripts/05_run_self_refine.py`` with one
part of the refine prompt changed, reusing the cached critique of step 5
(``critique_cache.jsonl``) so that only the tested input differs:

  trig_anti      trigger "The solution is likely correct; restate unless certain."
  trig_strong    trigger "The solution has clear errors that must be fixed."
  m_omit         Original solution -> [omitted]
  y0_omit        "Original answer: {y_0}" line removed
  critique_omit  Critique -> [omitted]

Writes ``<results>/refine/<model>/<task>/self_refine_<ablation>.jsonl``.

    python appendix_experiments/refine/run_refine_ablations.py \
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
from crr.selfrefine import REFINE_TRIGGER_VARIANTS, make_refine_prompt  # noqa: E402

refine = importlib.import_module("05_run_self_refine")  # wrap_chat / answer extraction / budgets

ABLATIONS = {
    "trig_anti": {"trigger_variant": "anti_trigger"},
    "trig_strong": {"trigger_variant": "strong_critique"},
    "m_omit": {"m_ablate": True},
    "y0_omit": {"y0_ablate": True},
    "critique_omit": {"critique_ablate": True},
}
TRIGGER_TOKENS = (10, 12)  # length-matched around the 11-token real trigger


def check_trigger_lengths(tok, ablations):
    for name in ["real"] + [ABLATIONS[a]["trigger_variant"] for a in ablations if "trigger_variant" in ABLATIONS[a]]:
        n = len(tok(REFINE_TRIGGER_VARIANTS[name], add_special_tokens=False)["input_ids"])
        if not TRIGGER_TOKENS[0] <= n <= TRIGGER_TOKENS[1]:
            raise SystemExit(f"trigger {name!r} has {n} tokens, outside {TRIGGER_TOKENS}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True, choices=common.MODELS)
    ap.add_argument("--task", default="math500", choices=common.TASKS)
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--ablations", nargs="+", default=list(ABLATIONS), choices=list(ABLATIONS))
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    d = os.path.join(args.results_dir, "refine", common.short_name(args.model), args.task)
    cache = {r["query_id"]: r for r in common.read_jsonl(os.path.join(d, "critique_cache.jsonl"))}
    if not cache:
        raise SystemExit(f"no critique_cache.jsonl in {d}; run scripts/05_run_self_refine.py first")
    rows = common.read_jsonl(os.path.join(common.cell_dir(args.results_dir, args.model, args.task),
                                          "baseline.jsonl"))
    todo = [a for a in args.ablations
            if args.overwrite or not os.path.exists(os.path.join(d, f"self_refine_{a}.jsonl"))]
    if not todo:
        print("[skip] all ablation outputs exist (use --overwrite)")
        return

    cfg = {"models": [{"name": args.model, "dtype": "bfloat16", "gpu_memory_utilization": 0.85,
                       "max_model_len": refine.MAX_MODEL_LEN}],
           "generation": {"max_new_tokens": refine.MAX_NEW_TOKENS * 2 + 1024}}
    engine, tok = load_vllm_engine(cfg)
    check_trigger_lengths(tok, todo)
    max_len = int(engine.llm_engine.model_config.max_model_len)
    from vllm import SamplingParams
    sampling = SamplingParams(temperature=0.0, max_tokens=refine.MAX_NEW_TOKENS)

    for name in todo:
        prompts = []
        for r in rows:
            c = cache.get(r["query_id"])
            if c is None or c.get("skipped_too_long") or c.get("critique_text") is None:
                prompts.append(None)
                continue
            p = refine.wrap_chat(tok, args.model, make_refine_prompt(
                query=r["question"], baseline_cot=r.get("baseline_cot", ""),
                y_0=str(r.get("baseline_answer", "")), critique=c["critique_text"], **ABLATIONS[name]))
            prompts.append(p if len(tok(p, add_special_tokens=False)["input_ids"]) <= int(max_len * 0.9) else None)
        idx = [i for i, p in enumerate(prompts) if p is not None]
        outs = engine.generate([prompts[i] for i in idx], sampling) if idx else []
        texts = [None] * len(prompts)
        for i, o in zip(idx, outs):
            texts[i] = o.outputs[0].text if o.outputs else ""
        out = []
        for r, t in zip(rows, texts):
            y_0 = str(r.get("baseline_answer", ""))
            y_ref = refine.extract_refined_answer(t or "", args.task)
            out.append({"query_id": r["query_id"], "model": args.model, "ablation": name,
                        "y_0": y_0, "y_refined": y_ref,
                        "change_flag": None if y_ref is None else int(str(y_ref).strip() != y_0.strip()),
                        "skipped_too_long": t is None, "refine_text": (t or "")[:10000]})
        path = os.path.join(d, f"self_refine_{name}.jsonl")
        common.write_jsonl(path, out)
        print(f"[done] {name}: {len(out)} rows -> {path}")
    unload_vllm_engine(engine)


if __name__ == "__main__":
    main()
