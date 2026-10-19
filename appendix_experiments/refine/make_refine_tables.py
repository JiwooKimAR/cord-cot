"""Appendix Tables 24-27 - self-refine analyses (CPU only).

    python appendix_experiments/refine/make_refine_tables.py --results-dir results

Inputs under ``<results>/refine/<model>/<task>/``:
  self_refine.jsonl            scripts/05_run_self_refine.py   (all tables)
  self_refine_<ablation>.jsonl run_refine_ablations.py         (Table 24)
  resample.jsonl               run_resample.py                 (Table 25)
  base_mmask.jsonl             run_base_mmask.py               (Table 26)
  inverse/xmmask/mmask.jsonl   scripts/06_measure_refined.py   (Table 26)
plus the main per-instance CORD (Tables 26-27). Missing inputs skip their table.

  Table 24  change rate of each refinement-input ablation (Qwen-Math x MATH-500)
  Table 25  refinement change (R) vs X-only resampling instability (U), MATH-500
  Table 26  baseline vs refined CoT, Table 4 plus the baseline M-mask column
  Table 27  CORD threshold sweep for skipping refinement (+ AUROC per cell)
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "scripts"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "scripts", "appendix"))
import common  # noqa: E402
import make_tables as mt  # noqa: E402
from make_appendix_tables import auroc  # noqa: E402

from crr.answers import check_answer  # noqa: E402

NON_R1 = ["Qwen2.5-Math-7B-Instruct", "Qwen2.5-7B-Instruct", "deepseek-math-7b-instruct", "Llama-3.1-8B-Instruct"]
MAIN = mt.REFINE_MODEL
ABLATIONS = [("real", "self_refine.jsonl"), ("trigger anti", "self_refine_trig_anti.jsonl"),
             ("trigger strong", "self_refine_trig_strong.jsonl"), ("M = [omitted]", "self_refine_m_omit.jsonl"),
             ("y0 removed", "self_refine_y0_omit.jsonl"), ("critique = [omitted]", "self_refine_critique_omit.jsonl")]
TAUS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
N_RANDOM, SEED = 100, 42


def rdir(results_dir, model, task):
    return os.path.join(results_dir, "refine", model, task)


def valid_refine(path):
    """{qid: row} for refine rows with an extracted refined answer."""
    return {r["query_id"]: r for r in common.read_jsonl(path)
            if not r.get("skipped_too_long") and r.get("change_flag") is not None}


def ablation_rates(results_dir):
    """Table 24."""
    out, real = [], None
    for label, fname in ABLATIONS:
        rows = valid_refine(os.path.join(rdir(results_dir, MAIN, "math500"), fname))
        if not rows:
            continue
        cr = float(np.mean([r["change_flag"] for r in rows.values()]))
        real = cr if label == "real" else real
        out.append({"ablation": label, "n": len(rows), "change_rate": round(cr, 4),
                    "delta_vs_real": None if label == "real" or real is None else round(cr - real, 4)})
    return out


def resample_coupling(results_dir):
    """Table 25 - 2x2 of R (refine changes the answer) and U (X-only unstable)."""
    out = []
    for m in NON_R1:
        d = rdir(results_dir, m, "math500")
        ref = valid_refine(os.path.join(d, "self_refine.jsonl"))
        res = {r["query_id"]: r for r in common.read_jsonl(os.path.join(d, "resample.jsonl"))
               if r.get("instance_change_rate") is not None}
        qs = sorted(set(ref) & set(res))
        if not qs:
            continue
        R = np.array([ref[q]["change_flag"] for q in qs])
        U = np.array([int(res[q]["instance_change_rate"] > 0) for q in qs])
        a, b = int(((R == 1) & (U == 1)).sum()), int(((R == 1) & (U == 0)).sum())
        c, d_ = int(((R == 0) & (U == 1)).sum()), int(((R == 0) & (U == 0)).sum())
        out.append({"model": mt.SHORT[m], "n": len(qs), "abcd": [a, b, c, d_],
                    "P_U": round(100 * (a + c) / len(qs), 1),
                    "P_R_given_U": round(100 * a / max(1, a + c), 1),
                    "P_R_given_notU": round(100 * b / max(1, b + d_), 1),
                    "P_U_given_R": round(100 * a / max(1, a + b), 1)})
    return out


def refined_vs_base(results_dir):
    """Table 26 - Table 4 (math500) plus the M-mask measured on the baseline CoT."""
    t4 = mt.refine_cord(results_dir, "math500")
    if t4 is None:
        return None
    d = rdir(results_dir, MAIN, "math500")

    def per_q(fname, key):
        out = defaultdict(dict)
        for r in common.read_jsonl(os.path.join(d, fname)):
            if r.get(key) is not None:
                out[r["qid"]][r["p"]] = float(r[key])
        return {q: float(np.mean(list(v.values()))) for q, v in out.items()}

    base, ref_m = per_q("base_mmask.jsonl", "m_match_baseline"), per_q("mmask.jsonl", "m_match_baseline")
    ref = valid_refine(os.path.join(d, "self_refine.jsonl"))
    if base and ref_m:
        qs = sorted(set(base) & set(ref_m))
        unch = [q for q in qs if q in ref and ref[q]["change_flag"] == 0]
        rec = {"axis": "M-mask (X visible)", "n": len(qs),
               "base_mean": round(float(np.mean([base[q] for q in qs])), 3),
               "refine_mean": round(float(np.mean([ref_m[q] for q in qs])), 3),
               "delta": round(float(np.mean([ref_m[q] - base[q] for q in qs])), 4),
               "spearman": round(float(spearmanr([base[q] for q in qs], [ref_m[q] for q in qs])[0]), 3),
               "n_unchanged": len(unch),
               "base_mean_unchanged": round(float(np.mean([base[q] for q in unch])), 3),
               "refine_mean_unchanged": round(float(np.mean([ref_m[q] for q in unch])), 3),
               "delta_unchanged": round(float(np.mean([ref_m[q] - base[q] for q in unch])), 4)}
        t4["axes"][1] = rec  # Table 4 has the refined M-mask only
    return t4


def threshold_sweep(results_dir, PI, BASE):
    """Table 27 - skip refinement when CORD > tau; enrichment and gain over same-rate random skips."""
    rng = np.random.default_rng(SEED)
    cells, rows = [], []
    for t in common.TASKS:
        for m in NON_R1:
            ref = valid_refine(os.path.join(rdir(results_dir, m, t), "self_refine.jsonl"))
            pi, gold = PI[m, t], {q: r.get("gold_answer") for q, r in BASE[m, t].items()}
            qs = sorted(q for q in set(ref) & set(pi) if gold.get(q) is not None)
            if not qs:
                continue
            C = np.array([pi[q]["cord"] for q in qs])
            unch = np.array([1 - ref[q]["change_flag"] for q in qs])
            y0_ok = np.array([int(check_answer(str(ref[q]["y_0"]), str(gold[q]))) for q in qs])
            yr_ok = np.array([int(ref[q]["y_refined"] is not None
                                  and check_answer(str(ref[q]["y_refined"]), str(gold[q]))) for q in qs])
            n, base_nc = len(qs), float(unch.mean())
            cells.append({"task": t, "model": mt.SHORT[m], "n": n, "base_nonchange": round(base_nc, 3),
                          "auroc": round(auroc(C, unch), 3)})
            for tau in TAUS:
                skip = C > tau
                ns = int(skip.sum())
                rec = {"task": t, "model": mt.SHORT[m], "tau": tau, "n_skip": ns, "skip_pct": round(100 * ns / n, 1)}
                if ns:
                    policy = float(np.where(skip, y0_ok, yr_ok).mean())
                    rand = []
                    for _ in range(N_RANDOM):
                        idx = np.zeros(n, bool)
                        idx[rng.permutation(n)[:ns]] = True
                        rand.append(float(np.where(idx, y0_ok, yr_ok).mean()))
                    nc = float(unch[skip].mean())
                    rec.update(nonchange_pct=round(100 * nc, 1), delta_pp=round(100 * (nc - base_nc), 1),
                               policy_acc=round(policy, 4), random_acc=round(float(np.mean(rand)), 4),
                               gain=round(policy - float(np.mean(rand)), 4))
                rows.append(rec)
    return {"cells": cells, "rows": rows}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results-dir", default="results")
    args = ap.parse_args()
    PI, BASE = {}, {}
    for m in NON_R1:
        for t in common.TASKS:
            PI[m, t], BASE[m, t] = mt.per_instance(args.results_dir, m, t)
    T = {"table24": ablation_rates(args.results_dir), "table25": resample_coupling(args.results_dir),
         "table26": refined_vs_base(args.results_dir), "table27": threshold_sweep(args.results_dir, PI, BASE)}
    out = os.path.join(args.results_dir, "tables")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "refine_tables.json"), "w") as f:
        json.dump(T, f, indent=1)

    print("## Table 24 - refinement-input ablations (Qwen-Math x MATH-500)\n")
    print("| ablation | n | change rate | delta vs real |\n|---|---|---|---|")
    for r in T["table24"]:
        dv = "-" if r["delta_vs_real"] is None else f"{r['delta_vs_real']:+.4f}"
        print(f"| {r['ablation']} | {r['n']} | {r['change_rate']:.4f} | {dv} |")
    print("\n## Table 25 - refinement change vs X-only instability (MATH-500)\n")
    print("| model | n | P(U) | P(R|U) | P(R|not U) | P(U|R) |\n|---|---|---|---|---|---|")
    for r in T["table25"]:
        print(f"| {r['model']} | {r['n']} | {r['P_U']}% | {r['P_R_given_U']}% | {r['P_R_given_notU']}% | {r['P_U_given_R']}% |")
    if T["table26"]:
        t = T["table26"]
        print(f"\n## Table 26 - baseline vs refined CoT (full n={t['n_full']}, unchanged n={t['n_unchanged']})\n")
        print("| measurement | base | refine | delta | rho | base (unch.) | refine (unch.) | delta (unch.) |")
        print("|---|---|---|---|---|---|---|---|")
        for r in t["axes"]:
            g = lambda k, f="{:.3f}": f.format(r[k]) if r.get(k) is not None else "-"  # noqa: E731
            print(f"| {r['axis']} | {g('base_mean')} | {g('refine_mean')} | {g('delta', '{:+.4f}')} | "
                  f"{g('spearman', '{:+.3f}')} | {g('base_mean_unchanged')} | {g('refine_mean_unchanged')} | "
                  f"{g('delta_unchanged', '{:+.4f}')} |")
    print("\n## Table 27 - CORD threshold sweep\n")
    print("AUROC (refine leaves the answer unchanged): " + ", ".join(
        f"{c['task']} {c['model']} {c['auroc']:.3f}" for c in T["table27"]["cells"]))
    print("\n| task | model | tau | skip (n) | non-change (delta) | gain vs random |\n|---|---|---|---|---|---|")
    for r in T["table27"]["rows"]:
        if "gain" in r:
            print(f"| {r['task']} | {r['model']} | {r['tau']:.1f} | {r['skip_pct']}% ({r['n_skip']}) | "
                  f"{r['nonchange_pct']}% ({r['delta_pp']:+.1f}p) | {r['gain']:+.4f} |")


if __name__ == "__main__":
    main()
