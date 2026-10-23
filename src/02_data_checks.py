"""
Checks behind the data decisions in Chapter 3.

Section 3.2.1  how many elements were assessed each year (the checklist changed)
Section 3.2.3  why ANC is used as a rate per 1,000 fertile women and not a count
Section 3.4.1  why zero-ANC facility-years are dropped rather than given log(x + 1)
Section 3.6.2  shape of the ICIS score in the finance sample
Section 3.6.4  Gini impurity at the root of the tree

Usage:
    python src/02_data_checks.py
Writes:
    outputs/data_checks.json
"""

import numpy as np
from scipy import stats

from common import (coef, design, fit_lmm, load_eda_panels, load_finance_panel,
                    load_trees_panel, rho, save_json)


def main():
    a, _ = load_eda_panels()
    out = {}

    # The checklist grew from 206 elements in 2017 to 208 in 2018 and 238 in
    # 2020. A score of 80% in 2017 and 80% in 2022 are not out of the same
    # list, which is one reason year is in every model.
    usual = {2018: 208, 2019: 208, 2020: 238, 2021: 238, 2022: 238}
    out["elements"] = {y: dict(share_on_usual_count=(a[a.year == y].total == n).mean(),
                               not_on_usual_count=int((a[a.year == y].total != n).sum()))
                       for y, n in usual.items()}
    out["elements"][2017] = dict(values=sorted(a[a.year == 2017].total.unique().tolist()))

    # The raw count mostly measures clinic size. Dividing by the fertile
    # female population takes out most of that, and with it most of the
    # count's link with score, because bigger clinics also score higher.
    out["size"] = dict(count_vs_headcount=rho(a.anc_first_visits, a.total_headcount),
                       rate_vs_headcount=rho(a.r, a.total_headcount))
    d = a[a.anc_first_visits > 0]
    out["count_vs_rate"] = dict(n=len(d), count_vs_score=rho(d.anc_first_visits, d.score),
                                rate_vs_score=rho(d.r, d.score))

    # Zero ANC rates. Adding 1 before logging is not neutral: these clinics
    # score lower than the rest, so pushing them to the bottom of the log scale
    # builds in a positive link with score. The model with log(ANC + 1) is
    # fitted here only to show that (Table 8).
    t = load_trees_panel()
    zero = t[t.anc_rate <= 0]
    out["zero_anc"] = dict(n=len(zero), facilities=zero.facility_id.nunique(),
                           mean_score=zero.score.mean(),
                           mean_score_rest=t[t.anc_rate > 0].score.mean())
    t = t.copy()
    t["y"] = np.log1p(t.anc_rate)
    t["z"] = (t.score - t.score.mean()) / t.score.std()
    X = design(t, ["z", "log_headcount", "urban", "year"])
    m = fit_lmm(t.y, X, t.facility_id)
    out["log1p_model"] = dict(N=m["N"], icc=m["icc"], **coef(m, X, "z"))
    out["p99_anc_rate"] = t[t.anc_rate > 0].anc_rate.quantile(.99)

    # Objective 2 outcome: left-skewed but no long tail, few at the ceiling
    f = load_finance_panel()
    out["finance_score"] = dict(skew=stats.skew(f.score),
                                share_95_or_more=(f.score >= 95).mean())

    # Gini impurity at the root, G = 2p(1 - p)
    p = t.high.mean()
    out["root_gini"] = dict(p_high=p, gini=2 * p * (1 - p))

    path = save_json(out, "data_checks")
    print("elements: " + ", ".join(f"{y} {v['share_on_usual_count']:.1%}"
                                    for y, v in out["elements"].items() if y != 2017))
    s, c = out["size"], out["count_vs_rate"]
    print(f"headcount vs ANC count rho {s['count_vs_headcount']:.3f}, vs rate {s['rate_vs_headcount']:.3f}")
    print(f"score vs ANC count rho {c['count_vs_score']:.3f}, vs rate {c['rate_vs_score']:.3f} (n = {c['n']:,})")
    z = out["zero_anc"]
    print(f"zero ANC: {z['n']} facility-years, mean score {z['mean_score']:.1f} against {z['mean_score_rest']:.1f}")
    lp = out["log1p_model"]
    print(f"log(ANC + 1) model: {lp['b']:.4f} (CI {lp['lo']:.4f} to {lp['hi']:.4f}), ICC {lp['icc']:.3f}, n = {lp['N']:,}")
    print(f"99th percentile of the ANC rate: {out['p99_anc_rate']:.2f}")
    fs = out["finance_score"]
    print(f"finance-sample score: skew {fs['skew']:.2f}, {fs['share_95_or_more']:.1%} at 95 or above")
    print(f"root of the tree: p = {p:.3f}, Gini = {out['root_gini']['gini']:.3f}")
    print(f"\nWritten to {path}")


if __name__ == "__main__":
    main()
