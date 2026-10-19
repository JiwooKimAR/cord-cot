"""Appendix tables computed from the main measurement outputs (CPU only).

    python scripts/appendix/make_appendix_tables.py --results-dir results

Uses only what steps 01-05 write (baseline, Experiment A/B, Early Answering and
the Qwen-Math x MATH-500 self-refine output); no new model generation. Writes
``<results>/tables/appendix_tables.json`` and ``appendix_tables.md``.

  Table  9  rank-wise CORD, R1D-Qwen vs Qwen-Math on GSM8K (+ cross-rank Spearman)
  Table 11  full vs adaptively truncated A-component (EA rho, refine AUROC)
  Table 12  per-cell Pearson r / Spearman rho / NMI between a_bar and q*_B
  Table 13  family means of Table 12
  Table 15  cross-task Kendall tau of the six-model ranking (S_A, S_B, CORD)
  Table 16  Spearman rho between CORD and Early Answering (= main Table 6)
  Table 17  Early Answering recovery rates at selected prefix positions
  Table 18  pair-mean CoT-length quartile upper bounds
  Table 19  first appearance of the answer string in the CoT
  Table 20  early-answer stress test
  Table 21  all 30 paired Wilcoxon comparisons

Table 20 needs the model tokenizers (offset mapping); the others are pure numpy.
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from itertools import combinations

import numpy as np
from scipy.stats import kendalltau, pearsonr, spearmanr, wilcoxon

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common  # noqa: E402
import make_tables as mt  # noqa: E402

import crr.config as cfg  # noqa: E402
from crr import cord  # noqa: E402
from crr.answers.leak import _build_direct_pattern, first_answer_pos_frac, normalize  # noqa: E402
from crr.metric.early_answering import ea_shortcut_score_integral  # noqa: E402
from crr.metric.grid import build_a_matrix, build_b_curve, compute_p_star  # noqa: E402

P, R = cfg.POSITION_GRID, cfg.RANK_GRID
SHORT, FAMILY, PAIRS = mt.SHORT, mt.FAMILY, mt.PAIRS
INV = {v: k for k, v in SHORT.items()}
NON_R1 = ["Qwen-Math", "Qwen-7B", "DeepSeek-Math", "Llama-3.1"]
RANK_PAIR = ("R1D-Qwen", "Qwen-Math", "gsm8k")      # Table 9
ANSWER_POS_MODELS = ["R1D-Llama", "R1D-Qwen", "Llama-3.1", "Qwen-Math"]  # Table 19
EA_POSITIONS = {"gsm8k": [0, 25, 50, 75, 100], "math500": [0, 50, 75, 100]}  # Table 17


def load_cell(results_dir, model, task):
    """{qid: {a, b, a_mean, p_star, cord}} and baseline rows; same instance set as make_tables."""
    d = common.cell_dir(results_dir, model, task)
    base = {r["query_id"]: r for r in common.read_jsonl(f"{d}/baseline.jsonl")}
    a_rows = mt.by_query(common.read_jsonl(f"{d}/experiment_a.jsonl"))
    b_rows = mt.by_query(common.read_jsonl(f"{d}/experiment_b.jsonl"))
    cells = {}
    for q in sorted(set(a_rows) & set(b_rows) & set(base)):
        a = build_a_matrix(a_rows[q], q, P, R)
        b = build_b_curve(b_rows[q], q, P, "token")
        if a is None or b is None:
            continue
        p_star = int(compute_p_star(base[q].get("baseline_cot") or "",
                                    str(base[q].get("baseline_answer") or ""), b, P, mode="A1"))
        a_mean = float(a.mean())
        cells[q] = {"a": a, "b": b, "a_mean": a_mean, "p_star": p_star, "cord": cord(a_mean, p_star),
                    "a_rank": rank_means(a_rows[q])}
    # Instances with an A-matrix regardless of B (used for the A-only summary of Table 15).
    a_only = {q: m for q in a_rows if (m := build_a_matrix(a_rows[q], q, P, R)) is not None}
    return cells, base, a_only


def rank_means(rows):
    """{rank: mean answer preservation over the positions that produced an answer} (Table 9)."""
    acc = defaultdict(list)
    for r in rows:
        if r.get("condition_type") == "main" and not r.get("skipped") and r.get("is_shortcut") is not None:
            acc[r["rank_position"]].append(1.0 if r["is_shortcut"] else 0.0)
    return {rk: float(np.mean(v)) for rk, v in acc.items()}


def load_ea(results_dir, s, t):
    ea = defaultdict(dict)
    for r in common.read_jsonl(os.path.join(results_dir, "ea", f"{s}__{t}.jsonl")):
        if r.get("m_match_baseline") is not None:
            ea[r["query_id"]][r["p"]] = float(r["m_match_baseline"])
    return ea


def q_star_b(p_star):
    return max(0.0, 1 - p_star / 100)


def a_mean_adaptive(a, p_star):
    """Mean of a(p, r) over p <= p*_B only (Appendix B.3.1)."""
    keep = np.asarray(P) <= p_star
    return float(a[keep].mean()) if keep.any() else 0.0


def p_star_a(a):
    """Rank-averaged first p with a(p, r) = 0 (100 if never), Appendix B.5.2."""
    per = [P[np.flatnonzero(a[:, j] == 0)[0]] if (a[:, j] == 0).any() else max(P) for j in range(a.shape[1])]
    return float(np.mean(per))


def auroc(scores, labels):
    """Rank-based AUROC with average ranks for ties; labels 1 = positive."""
    s, y = np.asarray(scores, float), np.asarray(labels, int)
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s))
    sv = s[order]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    n_p, n_n = int(y.sum()), int(len(y) - y.sum())
    if n_p == 0 or n_n == 0:
        return None
    return float((ranks[y == 1].sum() - n_p * (n_p + 1) / 2) / (n_p * n_n))


def wilcoxon_full(x, y):
    x, y = np.asarray(x), np.asarray(y)
    if np.all(x == y):
        return {"n": len(x), "delta_mean": 0.0, "delta_median": 0.0, "W": None, "p": 1.0}
    W, p = wilcoxon(x, y, zero_method="wilcox")
    return {"n": len(x), "delta_mean": round(float(np.mean(x - y)), 3),
            "delta_median": round(float(np.median(x - y)), 3), "W": float(W), "p": float(p)}


def paired_cells(C, m1, m2, t):
    d1, d2 = C[INV[m1], t], C[INV[m2], t]
    return d1, d2, sorted(set(d1) & set(d2))


# ---------------------------------------------------------------- tables

def rank_robustness(C, BASE):
    """Table 9 - CORD with each rank's necessity response, B-axis fixed."""
    m1, m2, t = RANK_PAIR
    d1, d2, qs_all = paired_cells(C, m1, m2, t)
    qs_bc = [q for q in qs_all if BASE[INV[m1], t][q]["baseline_correct"] and BASE[INV[m2], t][q]["baseline_correct"]]
    rows = []
    for subset, qs in (("full", qs_all), ("baseline_correct", qs_bc)):
        for j, rk in enumerate(R + ["5-rank average"]):
            def score(c):
                return c["cord"] if j == len(R) else cord(c["a_rank"].get(rk, 0.0), c["p_star"])
            x = np.array([score(d1[q]) for q in qs])
            y = np.array([score(d2[q]) for q in qs])
            rows.append({"subset": subset, "rank": rk, "n": len(qs), m1: round(float(x.mean()), 3),
                         m2: round(float(y.mean()), 3), "delta": round(float(np.mean(x - y)), 3),
                         "p": float(wilcoxon(x, y, zero_method="wilcox")[1])})
    corr = {}
    for i in range(len(R)):
        for j in range(i + 1, len(R)):
            qs = [q for q in qs_all if R[i] in d1[q]["a_rank"] and R[j] in d1[q]["a_rank"]]
            corr[f"{R[i]}|{R[j]}"] = round(float(spearmanr([d1[q]["a_rank"][R[i]] for q in qs],
                                                           [d1[q]["a_rank"][R[j]] for q in qs])[0]), 3)
    return {"rows": rows, "cross_rank_spearman": corr, "cross_rank_model": m1}


def adaptive_a(C, EA, refine):
    """Table 11 - full-grid a_bar vs a_bar_adp: EA rho (4 non-R1 MATH cells) and refine AUROC."""
    out = []
    for name, f in (("a_bar", lambda c: c["cord"]),
                    ("a_bar_adp", lambda c: cord(a_mean_adaptive(c["a"], c["p_star"]), c["p_star"]))):
        rhos = []
        for m in NON_R1:
            cells, ea = C[INV[m], "math500"], EA[INV[m], "math500"]
            qs = [q for q in sorted(set(cells) & set(ea)) if len(ea[q]) == len(P)]
            rhos.append(float(spearmanr([f(cells[q]) for q in qs],
                                        [ea_shortcut_score_integral(ea[q], P) for q in qs])[0]))
        row = {"a_component": name, "ea_rho": round(float(np.mean(rhos)), 3), "ea_rho_cells": rhos}
        if refine:
            cells = C[mt.REFINE_MODEL, "math500"]
            qs = sorted(set(refine) & set(cells))
            # Positive = refinement leaves the answer unchanged (higher CORD predicts no change).
            auc = auroc([f(cells[q]) for q in qs], [1 - refine[q] for q in qs])
            row.update(refine_auroc=round(auc, 3) if auc is not None else None, refine_n=len(qs))
        out.append(row)
    return out


def component_association(C):
    """Tables 12 and 13 - Pearson r, Spearman rho and NMI between a_bar and q*_B."""
    cells, pool = [], defaultdict(lambda: defaultdict(list))
    for m in common.MODELS:
        s = common.short_name(m)
        for t in common.TASKS:
            v = list(C[s, t].values())
            a = np.array([c["a_mean"] for c in v])
            b = np.array([q_star_b(c["p_star"]) for c in v])
            rec = {"model": SHORT[s], "task": t, "n": len(v), "pearson": round(float(pearsonr(a, b)[0]), 3),
                   "spearman": round(float(spearmanr(a, b)[0]), 3), "nmi": round(mt.nmi(a, b), 3)}
            cells.append(rec)
            for k in ("pearson", "spearman", "nmi"):
                pool[FAMILY[SHORT[s]]][k].append(rec[k])
    mean12 = {k: mt._mean3([c[k] for c in cells]) for k in ("pearson", "spearman", "nmi")}
    fam = [{"family": f, **{k: mt._mean3(d[k]) for k in ("pearson", "spearman", "nmi")}} for f, d in pool.items()]
    return {"per_cell": cells, "mean_12": mean12}, fam


def cross_task_tau(C, A_ONLY):
    """Table 15 - Kendall tau between the GSM8K and MATH-500 six-model rankings."""
    metrics = {
        "S_A": lambda s, t: np.mean([p_star_a(a) / 100 for a in A_ONLY[s, t].values()]),
        "S_B": lambda s, t: np.mean([q_star_b(c["p_star"]) for c in C[s, t].values()]),
        "CORD": lambda s, t: np.mean([c["cord"] for c in C[s, t].values()]),
    }
    out = []
    for name, f in metrics.items():
        means = {t: [float(f(common.short_name(m), t)) for m in common.MODELS] for t in common.TASKS}
        tau = kendalltau(means["gsm8k"], means["math500"])[0]
        out.append({"metric": name, "tau": round(float(tau), 3),
                    **{t: [round(v, 3) for v in means[t]] for t in common.TASKS}})
    return out


def ea_recovery(EA):
    """Table 17 - fraction of instances whose EA answer matches the baseline answer at p."""
    out = {}
    for t, ps in EA_POSITIONS.items():
        rows = []
        for m in common.MODELS:
            s = common.short_name(m)
            ea = EA[s, t]
            rec = {"model": SHORT[s], "n": len(ea)}
            for p in ps:
                vals = [ea[q][p] for q in ea if p in ea[q]]
                rec[f"p{p}"] = round(float(np.mean(vals)), 3) if vals else None
            rows.append(rec)
        out[t] = rows
    return out


def length_cuts(table2):
    """Table 18 - quartile upper bounds of pair-mean CoT length (from Table 2)."""
    return [{"task": r["task"], "contrast": f"{r['m1']} vs {r['m2']}", "cuts": r["cuts"],
             "q4_over_q1": round(r["cuts"][3] / r["cuts"][0], 1)} for r in table2]


def answer_first_position(BASE):
    """Table 19 - share of CoTs containing the answer string and its first normalized position."""
    rows = []
    for m in ANSWER_POS_MODELS:
        for t in common.TASKS:
            base = BASE[INV[m], t]
            fr = [f for r in base.values()
                  if (f := first_answer_pos_frac(r.get("baseline_cot") or "", str(r.get("baseline_answer") or "")))
                  is not None]
            rows.append({"model": m, "task": t, "n": len(base),
                         "in_cot_pct": round(100 * len(fr) / max(1, len(base)), 1),
                         "mean": round(float(np.mean(fr)), 2) if fr else None,
                         "median": round(float(np.median(fr)), 2) if fr else None})
    return rows


def stress_cords(cells, base, tokenizer):
    """Per-instance CORD with p*_B,stress (Appendix C.2): 100 if p_lit <= p*_B and b(p_lit) = 1."""
    out = {}
    for q, c in cells.items():
        cot, ans = base[q].get("baseline_cot") or "", str(base[q].get("baseline_answer") or "")
        hit = _build_direct_pattern(normalize(ans)).search(cot)
        p_star = c["p_star"]
        if hit is not None:
            offs = tokenizer(cot, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
            n = len(offs)
            # Index of the token that completes the answer literal; it is visible once the prefix covers it.
            need = next((i for i, (_, e) in enumerate(offs) if e >= hit.end()), n - 1) + 1
            p_lit = next((p for p in P if max(0, min(n, int(n * p / 100))) >= need), max(P))
            if p_lit <= p_star and c["b"][P.index(p_lit)] >= 0.95:
                p_star = max(P)
        out[q] = cord(c["a_mean"], p_star)
    return out


def stress_test(C, BASE):
    """Table 20 - aligned R1-Distill contrasts with and without the stress-test cutoff."""
    from transformers import AutoTokenizer

    hub = {common.short_name(m): m for m in common.MODELS}
    stressed = {}
    rows = []
    for t in common.TASKS:
        for m1, m2 in PAIRS:
            for m in (m1, m2):
                if (m, t) not in stressed:
                    tok = AutoTokenizer.from_pretrained(hub[INV[m]])
                    stressed[m, t] = stress_cords(C[INV[m], t], BASE[INV[m], t], tok)
            d1, d2, qs = paired_cells(C, m1, m2, t)
            orig = float(np.mean([d1[q]["cord"] - d2[q]["cord"] for q in qs]))
            x = np.array([stressed[m1, t][q] for q in qs])
            y = np.array([stressed[m2, t][q] for q in qs])
            delta = float(np.mean(x - y))
            rows.append({"task": t, "contrast": f"{m1} vs {m2}", "n": len(qs),
                         "original_delta": round(orig, 3), "stress_delta": round(delta, 3),
                         "p": float(wilcoxon(x, y, zero_method="wilcox")[1]),
                         # as reported: ratio of the 3-decimal differences
                         "gap_retained_pct": round(100 * round(delta, 3) / round(orig, 3), 1)})
    return rows


def wilcoxon_all(C):
    """Table 21 - paired Wilcoxon on instance-level CORD for all 15 model pairs per task."""
    names = [SHORT[common.short_name(m)] for m in common.MODELS]
    rows = []
    for t in common.TASKS:
        for m1, m2 in combinations(names, 2):
            d1, d2, qs = paired_cells(C, m1, m2, t)
            rows.append({"task": t, "m1": m1, "m2": m2,
                         **wilcoxon_full([d1[q]["cord"] for q in qs], [d2[q]["cord"] for q in qs])})
    return rows


# ---------------------------------------------------------------- output

def to_markdown(T):
    md = ["# Appendix tables\n"]
    t9 = T["table9"]
    m1, m2, _ = RANK_PAIR
    md += ["## Table 9 - rank-wise CORD (GSM8K)\n", f"| subset | rank | n | {m1} | {m2} | delta | p |",
           "|---|---|---|---|---|---|---|"]
    md += [f"| {r['subset']} | {r['rank']} | {r['n']} | {r[m1]:.3f} | {r[m2]:.3f} | {r['delta']:+.3f} | {r['p']:.1e} |"
           for r in t9["rows"]]
    md.append(f"\ncross-rank Spearman ({t9['cross_rank_model']}): "
              + ", ".join(f"{k} {v:.3f}" for k, v in t9["cross_rank_spearman"].items()))
    md += ["\n## Table 11 - full vs adaptive A-component\n", "| A-component | EA rho | refine AUROC |", "|---|---|---|"]
    md += [f"| {r['a_component']} | {r['ea_rho']:+.3f} | {r.get('refine_auroc', '-')} |" for r in T["table11"]]
    md += ["\n## Table 12 - a_bar vs q*_B per cell\n", "| model | task | n | Pearson r | Spearman rho | NMI |",
           "|---|---|---|---|---|---|"]
    md += [f"| {c['model']} | {c['task']} | {c['n']} | {c['pearson']:.3f} | {c['spearman']:.3f} | {c['nmi']:.3f} |"
           for c in T["table12"]["per_cell"]]
    m12 = T["table12"]["mean_12"]
    md.append(f"| mean (12 cells) | | | {m12['pearson']:.3f} | {m12['spearman']:.3f} | {m12['nmi']:.3f} |")
    md += ["\n## Table 13 - family means\n", "| family | mean r | mean rho | mean NMI |", "|---|---|---|---|"]
    md += [f"| {r['family']} | {r['pearson']:.3f} | {r['spearman']:.3f} | {r['nmi']:.3f} |" for r in T["table13"]]
    md += ["\n## Table 15 - cross-task Kendall tau\n", "| metric | tau |", "|---|---|"]
    md += [f"| {r['metric']} | {r['tau']:+.3f} |" for r in T["table15"]]
    md += ["\n## Table 16 - CORD vs Early Answering Spearman rho\n", "| model | GSM8K | MATH-500 |", "|---|---|---|"]
    md += [f"| {r['model']} | {r['gsm8k']:.3f} | {r['math500']:.3f} |" for r in T["table16"]["rows"]]
    md.append(f"| mean | {T['table16']['mean']:.3f} | |")
    md.append("\n## Table 17 - Early Answering recovery rates\n")
    for t, ps in EA_POSITIONS.items():
        md += [f"**{t}**\n", "| model | " + " | ".join(f"p={p}" for p in ps) + " |", "|---|" + "---|" * len(ps)]
        md += ["| " + r["model"] + " | " + " | ".join(f"{r[f'p{p}']:.3f}" for p in ps) + " |" for r in T["table17"][t]]
        md.append("")
    md += ["## Table 18 - CoT-length quartile upper bounds\n", "| task | contrast | Q1 | Q2 | Q3 | Q4 | Q4/Q1 |",
           "|---|---|---|---|---|---|---|"]
    md += [f"| {r['task']} | {r['contrast']} | " + " | ".join(str(c) for c in r["cuts"]) + f" | {r['q4_over_q1']}x |"
           for r in T["table18"]]
    md += ["\n## Table 19 - answer string in the CoT\n", "| model | task | in CoT % | mean | median |",
           "|---|---|---|---|---|"]
    md += [f"| {r['model']} | {r['task']} | {r['in_cot_pct']} | {r['mean']} | {r['median']} |" for r in T["table19"]]
    if T.get("table20"):
        md += ["\n## Table 20 - early-answer stress test\n", "| task | contrast | original delta | stress delta | p | gap retained |",
               "|---|---|---|---|---|---|"]
        md += [f"| {r['task']} | {r['contrast']} | {r['original_delta']:+.3f} | {r['stress_delta']:+.3f} | "
               f"{r['p']:.0e} | {r['gap_retained_pct']}% |" for r in T["table20"]]
    md += ["\n## Table 21 - all paired Wilcoxon comparisons\n", "| task | contrast | n | delta mean | delta median | W | p |",
           "|---|---|---|---|---|---|---|"]
    md += [f"| {r['task']} | {r['m1']} vs {r['m2']} | {r['n']} | {r['delta_mean']:+.3f} | {r['delta_median']:+.3f} | "
           f"{r['W']:.0f} | {r['p']:.1e} |" for r in T["table21"]]
    return "\n".join(md) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--skip-stress-test", action="store_true", help="skip Table 20 (needs the tokenizers)")
    args = ap.parse_args()
    C, BASE, A_ONLY, EA = {}, {}, {}, {}
    for m in common.MODELS:
        s = common.short_name(m)
        for t in common.TASKS:
            C[s, t], BASE[s, t], A_ONLY[s, t] = load_cell(args.results_dir, m, t)
            EA[s, t] = load_ea(args.results_dir, s, t)
    refine_path = os.path.join(args.results_dir, "refine", common.short_name(mt.REFINE_MODEL),
                               "math500", "self_refine.jsonl")
    refine = {r["query_id"]: r["change_flag"] for r in common.read_jsonl(refine_path)
              if not r.get("skipped_too_long") and r.get("change_flag") is not None} \
        if os.path.exists(refine_path) else {}

    PI = {k: {q: {kk: c[kk] for kk in ("a_mean", "p_star", "cord")} for q, c in v.items()} for k, v in C.items()}
    t6 = mt.ea_spearman(PI, args.results_dir)
    t12, t13 = component_association(C)
    T = {"table9": rank_robustness(C, BASE),
         "table11": adaptive_a(C, EA, refine),
         "table12": t12, "table13": t13,
         "table15": cross_task_tau(C, A_ONLY),
         "table16": {"rows": t6, "mean": round(float(np.mean([r[t] for r in t6 for t in common.TASKS])), 3)},
         "table17": ea_recovery(EA),
         "table18": length_cuts(mt.length_quartiles(PI, BASE)),
         "table19": answer_first_position(BASE),
         "table20": None if args.skip_stress_test else stress_test(C, BASE),
         "table21": wilcoxon_all(C)}
    out = os.path.join(args.results_dir, "tables")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "appendix_tables.json"), "w") as f:
        json.dump(T, f, indent=1)
    with open(os.path.join(out, "appendix_tables.md"), "w") as f:
        f.write(to_markdown(T))
    print(to_markdown(T))


if __name__ == "__main__":
    main()
