"""Step 5 - one round of self-refine (Madaan et al. 2023) for Table 4 (vLLM).

The model critiques its own baseline solution, then writes a refined solution
given the critique. Writes ``<results>/refine/<model>/<task>/self_refine.jsonl``
with Y_0, Y_refined and the change flag, and the full critiques to
``critique_cache.jsonl`` next to it.

    python scripts/05_run_self_refine.py --model Qwen/Qwen2.5-Math-7B-Instruct --task math500
"""

import argparse
import os
import re

import common

from crr.models.chat_template import render_chat, uses_user_only_chat
from crr.models.loader import load_vllm_engine, unload_vllm_engine
from crr.selfrefine import make_critique_prompt, make_refine_prompt

MAX_NEW_TOKENS = 2048
MAX_MODEL_LEN = 24576  # clamped by vLLM to the model's native limit
_ANS_HASH = re.compile(r"####\s*(-?\d+(?:\.\d+)?)")


def wrap_chat(tok, model: str, content: str) -> str:
    if uses_user_only_chat(model):
        messages = [{"role": "user", "content": content}]
    else:
        messages = [{"role": "system", "content": "You are a careful math tutor."},
                    {"role": "user", "content": content}]
    return render_chat(tok, messages)


def extract_refined_answer(text: str, task: str) -> str | None:
    """'#### N' (GSM8K), else the last complete \\boxed{...} (nested braces),
    else the last number (GSM8K)."""
    if not text:
        return None
    if task == "gsm8k":
        m = _ANS_HASH.search(text)
        if m:
            return m.group(1)
    out, i = None, 0
    while True:
        idx = text.find("\\boxed{", i)
        if idx == -1:
            break
        start, depth, end = idx + len("\\boxed{"), 0, None
        for j in range(start, len(text)):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                if depth == 0:
                    end = j
                    break
                depth -= 1
        if end is None:
            break
        out, i = text[start:end].strip(), end + 1
    if out is not None:
        return out
    if task == "gsm8k":
        nums = re.findall(r"-?\d+(?:\.\d+)?", text)
        if nums:
            return nums[-1]
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True, choices=common.MODELS)
    ap.add_argument("--task", required=True, choices=common.TASKS)
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--limit", type=int, default=None, help="first N instances only")
    ap.add_argument("--max-model-len", type=int, default=MAX_MODEL_LEN,
                    help="vLLM context; prompt + output beyond it is truncated")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    out_dir = os.path.join(args.results_dir, "refine", common.short_name(args.model), args.task)
    out_path = os.path.join(out_dir, "self_refine.jsonl")
    if os.path.exists(out_path) and not args.overwrite:
        print(f"[skip] {out_path} exists (use --overwrite)")
        return
    rows = common.read_jsonl(os.path.join(common.cell_dir(args.results_dir, args.model, args.task),
                                          "baseline.jsonl"))[:args.limit]
    if not rows:
        raise SystemExit("no baseline.jsonl; run 01_generate_baseline.py first")

    cfg = {"models": [{"name": args.model, "dtype": "bfloat16", "gpu_memory_utilization": 0.85,
                       "max_model_len": args.max_model_len}],
           "generation": {"max_new_tokens": MAX_NEW_TOKENS * 2 + 1024}}
    engine, tok = load_vllm_engine(cfg)
    max_len = int(engine.llm_engine.model_config.max_model_len)
    from vllm import SamplingParams
    sampling = SamplingParams(temperature=0.0, max_tokens=MAX_NEW_TOKENS)

    def fits(prompt):
        return len(tok(prompt, add_special_tokens=False)["input_ids"]) <= int(max_len * 0.9)

    def generate(prompts):
        idx = [i for i, p in enumerate(prompts) if p is not None]
        outs = engine.generate([prompts[i] for i in idx], sampling) if idx else []
        texts = [None] * len(prompts)
        for i, o in zip(idx, outs):
            texts[i] = o.outputs[0].text if o.outputs else ""
        return texts

    # Round 1: critique.
    crit_prompts = []
    for r in rows:
        p = wrap_chat(tok, args.model, make_critique_prompt(
            query=r["question"], baseline_cot=r.get("baseline_cot", ""),
            y_0=str(r.get("baseline_answer", ""))))
        crit_prompts.append(p if fits(p) else None)
    critiques = generate(crit_prompts)

    # Round 2: refine.
    ref_prompts = []
    for r, c in zip(rows, critiques):
        if c is None:
            ref_prompts.append(None)
            continue
        p = wrap_chat(tok, args.model, make_refine_prompt(
            query=r["question"], baseline_cot=r.get("baseline_cot", ""),
            y_0=str(r.get("baseline_answer", "")), critique=c))
        ref_prompts.append(p if fits(p) else None)
    refines = generate(ref_prompts)
    unload_vllm_engine(engine)

    out = []
    for r, c, t in zip(rows, critiques, refines):
        y_0 = str(r.get("baseline_answer", ""))
        y_ref = extract_refined_answer(t or "", args.task)
        out.append({
            "query_id": r["query_id"], "model": args.model,
            "y_0": y_0, "y_refined": y_ref,
            "change_flag": None if y_ref is None else int(str(y_ref).strip() != y_0.strip()),
            "skipped_too_long": t is None,
            "critique_text": (c or "")[:5000],
            "refine_text": (t or "")[:10000],
        })
    common.write_jsonl(out_path, out)
    # Full critiques, reused unchanged by the refinement-input ablations (appendix).
    common.write_jsonl(os.path.join(out_dir, "critique_cache.jsonl"),
                       [{"query_id": r["query_id"], "critique_text": c, "skipped_too_long": c is None}
                        for r, c in zip(rows, critiques)])
    print(f"[done] {len(out)} rows -> {out_path}")


if __name__ == "__main__":
    main()
