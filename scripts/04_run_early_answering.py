"""Step 4 - Early Answering (Lanham et al. 2023) for Table 6 (vLLM).

The baseline CoT is truncated at p% (snapped to a sentence boundary), the
question stays visible, and the answer is regenerated behind the answer cue.
Writes ``<results>/ea/<model>__<task>.jsonl``.

    python scripts/04_run_early_answering.py --model Qwen/Qwen2.5-Math-7B-Instruct --task gsm8k
"""

import argparse
import os
import re

import common

import crr.config as cfg
from crr.judge import judge_shortcut
from crr.models.loader import unload_vllm_engine
from crr.prompts import ANSWER_CUE

ANSWER_TOKENS = 64


def truncate_cot_sentence(cot_text: str, p_visible: int) -> str:
    """CoT prefix at the last sentence boundary (". " or newline) at or before p%."""
    if not cot_text or p_visible <= 0:
        return ""
    if p_visible >= 100:
        return cot_text
    boundaries = [0] + [m.end() for m in re.finditer(r"(?:\.\s|\n)", cot_text)] + [len(cot_text)]
    target = int(len(cot_text) * p_visible / 100)
    best = 0
    for b in boundaries:
        if b <= target:
            best = b
        else:
            break
    return cot_text[:best]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True, choices=common.MODELS)
    ap.add_argument("--task", required=True, choices=common.TASKS)
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--limit", type=int, default=None, help="first N instances only")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    out_path = os.path.join(args.results_dir, "ea", f"{common.short_name(args.model)}__{args.task}.jsonl")
    if os.path.exists(out_path) and not args.overwrite:
        print(f"[skip] {out_path} exists (use --overwrite)")
        return
    rows = common.read_jsonl(os.path.join(common.cell_dir(args.results_dir, args.model, args.task),
                                          "baseline.jsonl"))[:args.limit]
    if not rows:
        raise SystemExit("no baseline.jsonl; run 01_generate_baseline.py first")

    from vllm import LLM, SamplingParams
    native = common.native_max_positions(args.model) or 8192
    max_cot = max((r.get("baseline_n_tokens") or 0) for r in rows)
    max_len = int(min(int(native), max(8192, max_cot + 4096)))
    llm = LLM(model=args.model, dtype="bfloat16", trust_remote_code=True,
              max_model_len=max_len, gpu_memory_utilization=0.90)
    tok = llm.get_tokenizer()

    jobs, metas = [], []
    for r in rows:
        for p in cfg.POSITION_GRID:
            prompt = r["prompt"] + truncate_cot_sentence(r["baseline_cot"], p) + ANSWER_CUE
            if len(tok(prompt, add_special_tokens=False)["input_ids"]) > max_len - 70:
                metas.append((r, p, None))  # too long for the context
                continue
            metas.append((r, p, len(jobs)))
            jobs.append(prompt)
    outs = llm.generate(jobs, SamplingParams(temperature=0.0, max_tokens=ANSWER_TOKENS))
    texts = [o.outputs[0].text for o in outs]
    unload_vllm_engine(llm)

    recs = []
    for r, p, ji in metas:
        rec = {"query_id": r["query_id"], "p": p, "baseline_answer": r["baseline_answer"]}
        if ji is None:
            rec.update(m_pred=None, m_match_baseline=None, skipped_len=True)
        else:
            j = judge_shortcut("\\boxed{" + texts[ji], r["baseline_answer"])
            rec.update(m_pred=j["predicted_answer"],
                       m_match_baseline=None if j["is_shortcut"] is None else int(j["is_shortcut"]),
                       skipped_len=False)
        recs.append(rec)
    common.write_jsonl(out_path, recs)
    print(f"[done] {len(recs)} rows -> {out_path}")


if __name__ == "__main__":
    main()
