"""Appendix Table 8 - necessity-side disruption comparison on GSM8K (CPU only).

    python appendix_experiments/disruption/make_table8.py --results-dir results

Inverse-decoding rows come from the main ``experiment_a.jsonl`` (step 2); the
random / KL / filler / Early Answering rows from ``run_operators.py``. Each
operator's per-instance a is its mean answer preservation over the 21
positions; CORD is recomputed with that a and the main B-axis p*_B (A1).

Left panel  mean a per model and their ratio.
Right panel paired CORD difference |delta|, Cohen's d and -log10 p (Wilcoxon).
Both panels use the instances where both models answer the baseline correctly;
the left panel is also reported on the full set.
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
from scipy.stats import wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "scripts"))
import common  # noqa: E402
import make_tables as mt  # noqa: E402

import crr.config as cfg  # noqa: E402
from crr import cord  # noqa: E402

OPERATORS = ["random", "kl", "filler", "early_answering"]  # as in run_operators.py
PAIR = ("DeepSeek-R1-Distill-Qwen-7B", "Qwen2.5-Math-7B-Instruct")
LABELS = {"bottom-1": "Inverse, bottom-1", "bottom-5": "Inverse, bottom-5", "bottom-10": "Inverse, bottom-10",
          "top-5": "Inverse, top-5", "top-10": "Inverse, top-10", "random": "Random uniform",
          "kl": "KL-divergent", "filler": "Filler (\"...\")", "early_answering": "Early Answering"}


def per_operator_a(sources):
    """{operator: {qid: mean a over positions}} from (path, operators) pairs of experiment_a-style rows."""
    acc = defaultdict(lambda: defaultdict(list))
    for path, ops in sources:
        for r in common.read_jsonl(path):
            op = r.get("rank_position")
            if op not in ops or r.get("skipped") or r.get("is_shortcut") is None:
                continue
            acc[op][r["query_id"]].append(1.0 if r["is_shortcut"] else 0.0)
    return {op: {q: float(np.mean(v)) for q, v in d.items()} for op, d in acc.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--task", default="gsm8k", choices=common.TASKS)
    args = ap.parse_args()

    A, PI, BASE = {}, {}, {}
    for s in PAIR:
        cell = os.path.join(args.results_dir, s, args.task)
        A[s] = per_operator_a([
            (os.path.join(cell, "experiment_a.jsonl"), cfg.RANK_GRID),
            (os.path.join(args.results_dir, "disruption", s, args.task, "experiment_a_ops.jsonl"), OPERATORS)])
        PI[s], BASE[s] = mt.per_instance(args.results_dir, s, args.task)
    s1, s2 = PAIR
    full = sorted(set(PI[s1]) & set(PI[s2]))
    correct = [q for q in full if BASE[s1][q]["baseline_correct"] and BASE[s2][q]["baseline_correct"]]

    rows = []
    for op in cfg.RANK_GRID + OPERATORS:
        if op not in A[s1] or op not in A[s2]:
            print(f"[warn] no rows for {op}; skipped")
            continue
        rec = {"operator": LABELS[op]}
        for subset, qs in (("correct", correct), ("full", full)):
            qs = [q for q in qs if q in A[s1][op] and q in A[s2][op]]
            m1 = float(np.mean([A[s1][op][q] for q in qs]))
            m2 = float(np.mean([A[s2][op][q] for q in qs]))
            rec[subset] = {"n": len(qs), "mean_a_r1d": round(m1, 3), "mean_a_qwen": round(m2, 3),
                           "ratio": round(round(m1, 3) / round(m2, 3), 2)}
        qs = [q for q in correct if q in A[s1][op] and q in A[s2][op]]
        x = np.array([cord(A[s1][op][q], PI[s1][q]["p_star"]) for q in qs])
        y = np.array([cord(A[s2][op][q], PI[s2][q]["p_star"]) for q in qs])
        d = x - y
        p = float(wilcoxon(x, y, zero_method="wilcox")[1])
        rec["separation"] = {"n": len(qs), "abs_delta": round(abs(float(d.mean())), 3),
                             "cohens_d": round(float(d.mean() / d.std(ddof=1)), 2),
                             "neg_log10_p": round(float(-np.log10(p)), 2) if p > 0 else None}
        rows.append(rec)

    out = os.path.join(args.results_dir, "tables")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "table8.json"), "w") as f:
        json.dump(rows, f, indent=1)
    print(f"Table 8 - {args.task}, R1D-Qwen vs Qwen-Math "
          f"(baseline-correct n={len(correct)}, full n={len(full)})\n")
    print("| operator | R1D-Qwen a | Qwen-Math a | ratio | |delta| | Cohen's d | -log10 p | full-set a (R1D / Qwen) |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        c, s, f = r["correct"], r["separation"], r["full"]
        print(f"| {r['operator']} | {c['mean_a_r1d']:.3f} | {c['mean_a_qwen']:.3f} | {c['ratio']:.2f} | "
              f"{s['abs_delta']:.3f} | {s['cohens_d']:.2f} | {s['neg_log10_p']:.2f} | "
              f"{f['mean_a_r1d']:.3f} / {f['mean_a_qwen']:.3f} |")


if __name__ == "__main__":
    main()
