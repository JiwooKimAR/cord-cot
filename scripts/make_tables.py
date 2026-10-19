"""Paper Tables 1-6 from the measurement outputs (CPU only).

    python scripts/make_tables.py --results-dir results

Writes ``<results>/tables/tables.json`` and ``tables.md``. Per-instance CORD
uses the A1 cutoff and the token-unit B-axis (paper Sec. 4.1).

Cells (model x task) that have not been measured yet are skipped: each table is
computed from the cells it needs that are present, and the missing cells are
listed in the output.
"""

import argparse
import json
import os
from collections import defaultdict
from decimal import ROUND_HALF_UP, Decimal

import numpy as np
from scipy.stats import spearmanr, wilcoxon

import common

import crr.config as cfg
from crr import cord
from crr.metric.early_answering import ea_shortcut_score_integral
from crr.metric.grid import build_a_matrix, build_b_curve, compute_p_star

P, R = cfg.POSITION_GRID, cfg.RANK_GRID
SHORT = {"DeepSeek-R1-Distill-Qwen-7B": "R1D-Qwen", "DeepSeek-R1-Distill-Llama-8B": "R1D-Llama",
         "Qwen2.5-Math-7B-Instruct": "Qwen-Math", "Qwen2.5-7B-Instruct": "Qwen-7B",
         "deepseek-math-7b-instruct": "DeepSeek-Math", "Llama-3.1-8B-Instruct": "Llama-3.1"}
FAMILY = {"R1D-Qwen": "R1-Distill", "R1D-Llama": "R1-Distill",
          "Qwen-Math": "Math-tuned", "DeepSeek-Math": "Math-tuned",
          "Qwen-7B": "Plain", "Llama-3.1": "Plain"}
# Base-family-aligned R1-Distill vs non-distilled pairs (Tables 2 and 3).
PAIRS = [("R1D-Llama", "Llama-3.1"), ("R1D-Qwen", "Qwen-Math")]
REFINE_MODEL = "Qwen2.5-Math-7B-Instruct"  # Table 4 cell


def by_query(rows, key="query_id"):
    d = defaultdict(list)
    for r in rows:
        d[r[key]].append(r)
    return d


def per_instance(results_dir, model, task):
    """{query_id: {cord, a_mean, p_star}} plus baseline rows for one cell."""
    d = os.path.join(results_dir, common.short_name(model), task)
    base = {r["query_id"]: r for r in common.read_jsonl(f"{d}/baseline.jsonl")}
    a_rows = by_query(common.read_jsonl(f"{d}/experiment_a.jsonl"))
    b_rows = by_query(common.read_jsonl(f"{d}/experiment_b.jsonl"))
    out = {}
    for q in sorted(set(a_rows) & set(b_rows) & set(base)):
        a = build_a_matrix(a_rows[q], q, P, R)
        b = build_b_curve(b_rows[q], q, P, "token")
        if a is None or b is None:
            continue
        p_star = int(compute_p_star(base[q].get("baseline_cot") or "",
                                    str(base[q].get("baseline_answer") or ""), b, P, mode="A1"))
        a_mean = float(a.mean())
        out[q] = {"a_mean": a_mean, "p_star": p_star, "cord": cord(a_mean, p_star)}
    return out, base


def cord_accuracy(PI, BASE):
    """Table 1 - mean CORD and baseline accuracy per cell."""
    rows = []
    for t in common.TASKS:
        for m in common.MODELS:
            s = common.short_name(m)
            if not PI[s, t]:
                continue
            rows.append({"task": t, "model": SHORT[s], "n": len(PI[s, t]),
                         "base_acc": round(float(np.mean([r["baseline_correct"] for r in BASE[s, t].values()])), 3),
                         "cord": round(float(np.mean([r["cord"] for r in PI[s, t].values()])), 3)})
    return rows


def _paired(PI, t, m1, m2, qs=None):
    inv = {v: k for k, v in SHORT.items()}
    d1, d2 = PI[inv[m1], t], PI[inv[m2], t]
    qs = sorted(set(d1) & set(d2)) if qs is None else qs
    x = np.array([d1[q]["cord"] for q in qs])
    y = np.array([d2[q]["cord"] for q in qs])
    p = 1.0 if np.all(x == y) else float(wilcoxon(x, y, zero_method="wilcox")[1])
    return {"n": len(qs), "delta_mean": round(float(np.mean(x - y)), 3), "p": p}


def _pair_present(PI, t, m1, m2):
    inv = {v: k for k, v in SHORT.items()}
    return bool(PI[inv[m1], t]) and bool(PI[inv[m2], t])


def wilcoxon_headline(PI):
    """Table 3 - paired Wilcoxon on instance-level CORD for the aligned pairs."""
    return [{"task": t, "m1": m1, "m2": m2, **_paired(PI, t, m1, m2)}
            for t in common.TASKS for m1, m2 in PAIRS if _pair_present(PI, t, m1, m2)]


def length_quartiles(PI, BASE):
    """Table 2 - CORD difference per quartile of the pair-mean baseline CoT length."""
    inv = {v: k for k, v in SHORT.items()}
    rows = []
    for t in common.TASKS:
        for m1, m2 in PAIRS:
            if not _pair_present(PI, t, m1, m2):
                continue
            s1, s2 = inv[m1], inv[m2]
            qs = sorted(set(PI[s1, t]) & set(PI[s2, t]))
            pl = np.array([(BASE[s1, t][q]["baseline_n_tokens"] + BASE[s2, t][q]["baseline_n_tokens"]) / 2
                           for q in qs])
            cuts = np.quantile(pl, [0.25, 0.5, 0.75])
            qidx = np.digitize(pl, cuts)
            quart = [{"Q": k + 1, **_paired(PI, t, m1, m2, [q for q, g in zip(qs, qidx) if g == k])}
                     for k in range(4)]
            rows.append({"task": t, "m1": m1, "m2": m2,
                         "overall_delta": _paired(PI, t, m1, m2)["delta_mean"],
                         "quartiles": quart, "cuts": [int(c) for c in cuts] + [int(pl.max())]})
    return rows


def nmi(x, y, q=10):
    """Normalized mutual information after rank-based discretization into q bins."""
    def disc(v):
        ranks = np.argsort(np.argsort(v))
        return (ranks * q // len(v)).astype(int)

    def entropy(*cols):
        _, c = np.unique(np.stack(cols), axis=1, return_counts=True)
        pr = c / c.sum()
        return -np.sum(pr * np.log(pr))

    dx, dy = disc(np.asarray(x)), disc(np.asarray(y))
    hx, hy = entropy(dx), entropy(dy)
    return float((hx + hy - entropy(dx, dy)) / max(hx, hy))


def _mean3(values):
    """Mean of 3-decimal values, rounded half-up to 3 decimals (as reported)."""
    mean = sum(Decimal(str(v)) for v in values) / len(values)  # exact decimal mean
    return float(mean.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))


def component_relation(PI):
    """Table 5 - Spearman rho and NMI between a_bar and q*_B per cell, averaged within families."""
    cells, pool = [], defaultdict(lambda: {"rho": [], "nmi": []})
    for m in common.MODELS:
        s = common.short_name(m)
        for t in common.TASKS:
            pi = list(PI[s, t].values())
            if len(pi) < 3:
                continue
            a = np.array([r["a_mean"] for r in pi])
            b = np.array([max(0.0, 1 - r["p_star"] / 100) for r in pi])
            rho, v = float(spearmanr(a, b)[0]), nmi(a, b)
            cells.append({"cell": f"{SHORT[s]} x {t}", "n": len(pi), "spearman": round(rho, 3), "nmi": round(v, 3)})
            f = pool[FAMILY[SHORT[s]]]
            f["rho"].append(round(rho, 3)); f["nmi"].append(round(v, 3))
    fam = [{"family": f, "mean_spearman": _mean3(d["rho"]), "mean_nmi": _mean3(d["nmi"])}
           for f, d in pool.items()]
    if not cells:
        return {"family": [], "per_cell": [], "mean_spearman": None, "mean_nmi": None}
    return {"family": fam, "per_cell": cells,
            "mean_spearman": _mean3([c["spearman"] for c in cells]),
            "mean_nmi": _mean3([c["nmi"] for c in cells])}


def ea_spearman(PI, results_dir):
    """Table 6 - Spearman rho between CORD and the Early Answering score."""
    rows = []
    for m in common.MODELS:
        s = common.short_name(m)
        row = {"model": SHORT[s]}
        for t in common.TASKS:
            if not PI[s, t]:
                continue
            ea = defaultdict(dict)
            for r in common.read_jsonl(os.path.join(results_dir, "ea", f"{s}__{t}.jsonl")):
                if r.get("m_match_baseline") is not None:
                    ea[r["query_id"]][r["p"]] = float(r["m_match_baseline"])
            qs = [q for q in sorted(set(PI[s, t]) & set(ea)) if len(ea[q]) == len(P)]
            if len(qs) < 3:
                continue
            rho, p = spearmanr([PI[s, t][q]["cord"] for q in qs],
                               [ea_shortcut_score_integral(ea[q], P) for q in qs])
            row.update({t: round(float(rho), 3), f"{t}_n": len(qs), f"{t}_p": float(p)})
        if len(row) > 1:
            rows.append(row)
    return rows


def refine_cord(results_dir, task):
    """Table 4 - base vs refined CoT on each axis (Qwen-Math), full set and answer-unchanged subset."""
    s = common.short_name(REFINE_MODEL)
    rd = os.path.join(results_dir, "refine", s, task)
    ref_ax = {}
    for ax, key in (("inverse", "a_match_baseline"), ("mmask", "m_match_baseline"),
                    ("xmmask", "b_match_baseline")):
        d = defaultdict(dict)
        for r in common.read_jsonl(os.path.join(rd, f"{ax}.jsonl")):
            if r.get(key) is not None:
                d[r["qid"]][r["p"]] = float(r[key])
        ref_ax[ax] = d
    cd = os.path.join(results_dir, s, task)
    base_a, base_b = defaultdict(dict), defaultdict(dict)
    for r in common.read_jsonl(f"{cd}/experiment_a.jsonl"):
        if (r.get("rank_position") == "bottom-1" and r.get("condition_type") == "main"
                and not r.get("skipped") and r.get("is_shortcut") is not None):
            base_a[r["query_id"]][r["intervention_point"]] = float(bool(r["is_shortcut"]))
    for r in common.read_jsonl(f"{cd}/experiment_b.jsonl"):
        if (r.get("mask_unit") == "token" and r.get("condition_type") == "main"
                and not r.get("skipped") and r.get("is_shortcut") is not None):
            base_b[r["query_id"]][r["p_visible"]] = float(bool(r["is_shortcut"]))
    ref = {r["query_id"]: r for r in common.read_jsonl(os.path.join(rd, "self_refine.jsonl"))
           if not r.get("skipped_too_long") and r.get("change_flag") is not None}
    if not (ref_ax["inverse"] and ref_ax["xmmask"] and base_a and base_b):
        return None

    def amean(d):
        return float(np.mean(list(d.values()))) if d else None

    def pstar(d):
        hit = [p for p in sorted(d) if d[p] >= 0.95]
        return min(hit) if hit else 100

    qs = sorted(set(ref_ax["inverse"]) & set(ref_ax["xmmask"]) & set(ref_ax["mmask"])
                & set(base_a) & set(base_b) & set(ref))
    unch = {q for q in qs if ref[q]["change_flag"] == 0}
    axes = [("a (A-axis, bottom-1)", lambda q: amean(base_a[q]), lambda q: amean(ref_ax["inverse"][q])),
            ("M-mask (X visible)", lambda q: None, lambda q: amean(ref_ax["mmask"][q])),
            ("X/M-mask (B-axis)", lambda q: amean(base_b[q]), lambda q: amean(ref_ax["xmmask"][q])),
            ("CORD", lambda q: cord(amean(base_a[q]), pstar(base_b[q])),
             lambda q: cord(amean(ref_ax["inverse"][q]), pstar(ref_ax["xmmask"][q])))]
    out = {"task": task, "n_full": len(qs), "n_unchanged": len(unch), "axes": []}
    for name, fb, fr in axes:
        b_full, r_full = [fb(q) for q in qs], [fr(q) for q in qs]
        rec = {"axis": name, "refine_mean": round(float(np.mean(r_full)), 3)}
        ok = [i for i, v in enumerate(b_full) if v is not None]
        if ok:
            bv, rv = [b_full[i] for i in ok], [r_full[i] for i in ok]
            iu = [i for i in ok if qs[i] in unch]
            rec.update(base_mean=round(float(np.mean(bv)), 3), refine_mean=round(float(np.mean(rv)), 3),
                       delta=round(float(np.mean(rv) - np.mean(bv)), 4),
                       spearman=round(float(spearmanr(bv, rv)[0]), 3),
                       base_mean_unchanged=round(float(np.mean([b_full[i] for i in iu])), 3),
                       refine_mean_unchanged=round(float(np.mean([r_full[i] for i in iu])), 3),
                       delta_unchanged=round(float(np.mean([r_full[i] - b_full[i] for i in iu])), 4))
        out["axes"].append(rec)
    return out


def to_markdown(T):
    md = ["# Paper Tables 1-6\n"]
    if T.get("missing_cells"):
        md.append("Not measured (skipped): " + ", ".join(T["missing_cells"])
                  + ". Tables below use only the cells present.\n")
    md += ["## Table 1 - mean CORD and baseline accuracy\n",
          "| task | model | n | base acc | CORD |", "|---|---|---|---|---|"]
    md += [f"| {r['task']} | {r['model']} | {r['n']} | {r['base_acc']:.3f} | {r['cord']:.3f} |" for r in T["table1"]]
    md += ["\n## Table 2 - CORD difference by pair-mean baseline CoT length quartile\n",
           "| task | contrast | overall delta | Q1 | Q2 | Q3 | Q4 |", "|---|---|---|---|---|---|---|"]
    md += [f"| {r['task']} | {r['m1']} vs {r['m2']} | {r['overall_delta']:+.3f} | "
           + " | ".join(f"{q['delta_mean']:+.3f} (p={q['p']:.1e})" for q in r["quartiles"]) + " |"
           for r in T["table2"]]
    if not T["table2"]:
        md.append("(needs both cells of an aligned pair: " + ", ".join(f"{a} / {b}" for a, b in PAIRS) + ")")
    md += ["\n## Table 3 - headline paired Wilcoxon\n", "| task | contrast | n | delta | p |", "|---|---|---|---|---|"]
    md += [f"| {r['task']} | {r['m1']} vs {r['m2']} | {r['n']} | {r['delta_mean']:+.3f} | {r['p']:.1e} |"
           for r in T["table3"]]
    if not T["table3"]:
        md.append("(needs both cells of an aligned pair: " + ", ".join(f"{a} / {b}" for a, b in PAIRS) + ")")
    md.append("\n## Table 4 - CORD on baseline vs refined CoT (Qwen-Math)\n")
    for t4 in T["table4"]:
        if t4 is None:
            continue
        md += [f"**{t4['task']}** - full set n={t4['n_full']}, answer-unchanged n={t4['n_unchanged']}\n",
               "| axis | base | refine | delta | rho | base (unch.) | refine (unch.) | delta (unch.) |",
               "|---|---|---|---|---|---|---|---|"]
        for r in t4["axes"]:
            g = lambda k, f="{:.3f}": f.format(r[k]) if k in r else "-"
            md.append(f"| {r['axis']} | {g('base_mean')} | {r['refine_mean']:.3f} | {g('delta', '{:+.4f}')} | "
                      f"{g('spearman', '{:+.3f}')} | {g('base_mean_unchanged')} | {g('refine_mean_unchanged')} | "
                      f"{g('delta_unchanged', '{:+.4f}')} |")
        md.append("")
    md += ["## Table 5 - relationship between the two CORD components (family means)\n",
           "| family | mean Spearman rho | mean NMI |", "|---|---|---|"]
    md += [f"| {r['family']} | {r['mean_spearman']:.3f} | {r['mean_nmi']:.3f} |" for r in T["table5"]["family"]]
    if T["table5"]["per_cell"]:
        md.append(f"\nall {len(T['table5']['per_cell'])} cells: mean rho = {T['table5']['mean_spearman']:.3f}, "
                  f"mean NMI = {T['table5']['mean_nmi']:.3f}")
    md += ["\n## Table 6 - Spearman rho between CORD and the Early Answering score\n",
           "| model | GSM8K | MATH-500 |", "|---|---|---|"]
    fmt = lambda v: "-" if v is None else f"{v:+.3f}"  # noqa: E731
    md += [f"| {r['model']} | {fmt(r.get('gsm8k'))} | {fmt(r.get('math500'))} |" for r in T["table6"]]
    return "\n".join(md) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results-dir", default="results")
    args = ap.parse_args()
    PI, BASE = {}, {}
    for m in common.MODELS:
        for t in common.TASKS:
            s = common.short_name(m)
            PI[s, t], BASE[s, t] = per_instance(args.results_dir, m, t)
    missing = [f"{SHORT[s]} x {t}" for t in common.TASKS for s in map(common.short_name, common.MODELS)
               if not PI[s, t]]
    if len(missing) == len(PI):
        raise SystemExit(f"no measured cells under {args.results_dir} (need baseline + experiment_a/b)")
    T = {"table1": cord_accuracy(PI, BASE), "table2": length_quartiles(PI, BASE),
         "table3": wilcoxon_headline(PI),
         "table4": [refine_cord(args.results_dir, t) for t in ("math500", "gsm8k")],
         "table5": component_relation(PI), "table6": ea_spearman(PI, args.results_dir)}
    if missing:
        T["missing_cells"] = missing
    out = os.path.join(args.results_dir, "tables")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "tables.json"), "w") as f:
        json.dump(T, f, indent=1)
    with open(os.path.join(out, "tables.md"), "w") as f:
        f.write(to_markdown(T))
    print(to_markdown(T))


if __name__ == "__main__":
    main()
