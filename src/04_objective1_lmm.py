"""
Objective 1: does a higher ICIS score go with a higher ANC first-visit rate?
(Sections 3.6.1 and 5.2)

    log(ANC_it) = b0 + b1 * standardised score_it + b2 * log headcount_it
                  + urbanicity_i + year_t + u_i + e_it

There are no hyperparameters to tune in this model, so it is not
cross-validated. It is checked in these ways instead:

    residuals            shape, Shapiro-Wilk, spread across fitted values
    random intercepts    their spread against the estimated sigma_u
    AIC                  each term dropped in turn, and province added
    other fits           REML, fixed effects with cluster-robust SEs, and the
                         top 1% of ANC rates removed
    detectable effect    the smallest effect the design could pick up,
                         2.80 x SE (5% level, 80% power)

The ownership gap in Section 5.2.5 is estimated at the end.

Usage:
    python src/04_objective1_lmm.py
Writes:
    outputs/objective1.json
    outputs/table3_objective1.csv
    outputs/objective1_residuals.csv, outputs/objective1_blups.csv
"""

import numpy as np
import pandas as pd

from common import (OUT, coef, describe, design, fit_lmm, load_trees_panel,
                    mde, save_json, within_estimator)

TERMS = ["z", "log_headcount", "urban", "year"]


def ownership_gap(d):
    """Municipal against provincial clinics (Section 5.2.5). If the gap were
    only about where municipal clinics are and how big they are, it should
    shrink as urbanicity, headcount and province are added. It does not."""
    d = d[d.ownership.isin(["Gov Province", "Gov Municipality"])].copy()
    d["muni"] = (d.ownership == "Gov Municipality").astype(float)
    specs = {"year only": ["muni", "year"],
             "+ urbanicity": ["muni", "urban", "year"],
             "+ log headcount": ["muni", "log_headcount", "urban", "year"],
             "+ province": ["muni", "log_headcount", "urban", "year", "province"],
             "+ ICIS score": ["muni", "z", "log_headcount", "urban", "year", "province"]}
    res = {}
    for k, terms in specs.items():
        X = design(d, terms)
        c = coef(fit_lmm(d.y, X, d.facility_id), X, "muni")
        c["pct"] = 100 * (np.exp(c["b"]) - 1)
        res[k] = c

    # Almost every municipal clinic is urban, so compare urban clinics only
    urban = d[d.urban_rural == "Urban"]
    X = design(urban, ["muni", "log_headcount", "year", "province"])
    c = coef(fit_lmm(urban.y, X, urban.facility_id), X, "muni")
    c.update(pct=100 * (np.exp(c["b"]) - 1), n=len(urban), n_municipal=int(urban.muni.sum()))
    res["urban only"] = c
    names = {0.0: "provincial", 1.0: "municipal"}
    med = urban.groupby(["province", "muni"]).anc_rate.median().unstack().rename(columns=names)
    res["urban medians"] = dict(all=urban.groupby("muni").anc_rate.median().rename(names).to_dict(),
                                by_province=med.dropna().to_dict("index"))
    return res


def main():
    a = load_trees_panel()
    d = a[a.anc_rate > 0].copy()
    d["y"] = np.log(d.anc_rate)
    # standardised so the coefficient is per one SD of score
    d["z"] = (d.score - d.score.mean()) / d.score.std()

    X = design(d, TERMS)
    m = fit_lmm(d.y, X, d.facility_id)
    score = coef(m, X, "z")
    out = dict(N=m["N"], facilities=m["G"], score=score, icc=m["icc"],
               sigma_u=np.sqrt(m["su2"]), sigma=np.sqrt(m["s2"]), aic=m["aic"])
    table = pd.DataFrame({"coefficient": m["b"], "se": m["se"],
                          "lo": m["b"] - 1.96 * m["se"], "hi": m["b"] + 1.96 * m["se"]},
                         index=X.columns)
    table.to_csv(OUT / "table3_objective1.csv")
    out["coefficients"] = table.to_dict("index")

    # Residuals and random intercepts. The residuals turn out heavy-tailed,
    # which is why the fixed-effects fit with robust SEs below matters.
    out["residuals"] = describe(m["resid"])
    q = pd.qcut(m["fitted"], 5, labels=False)
    out["residual_sd_by_fitted_quintile"] = pd.Series(m["resid"]).groupby(q).std().tolist()
    out["blups"] = describe(m["blup"])

    # The same question asked in other ways
    out["reml"] = coef(fit_lmm(d.y, X, d.facility_id, reml=True), X, "z")
    # urbanicity does not change within a clinic, so it drops out here
    Xfe = X.drop(columns=["const"] + [c for c in X.columns if c.startswith("ur_")])
    out["fixed_effects"] = within_estimator(d.y.values, Xfe.values, d.facility_id.values)
    Xp = design(d, TERMS + ["province"])
    out["with_province"] = coef(fit_lmm(d.y, Xp, d.facility_id), Xp, "z")
    cut = d.anc_rate.quantile(.99)
    t = d[d.anc_rate <= cut]
    Xt = design(t, TERMS)
    out["top1pct_removed"] = dict(N=len(t), cut=cut, **coef(fit_lmm(t.y, Xt, t.facility_id), Xt, "z"))

    # AIC change when a term is dropped (positive means the model gets worse)
    aic = {f"drop {k}": fit_lmm(d.y, design(d, [x for x in TERMS if x != k]),
                                d.facility_id)["aic"] - m["aic"] for k in TERMS}
    aic["add province"] = fit_lmm(d.y, Xp, d.facility_id)["aic"] - m["aic"]
    out["aic_change"] = aic

    out["mde"] = dict(model_se=mde(score["se"]), robust_se=mde(out["fixed_effects"]["se_robust"]))
    out["ownership_gap"] = ownership_gap(d)

    path = save_json(out, "objective1")
    pd.DataFrame({"fitted": m["fitted"], "residual": m["resid"]}) \
        .to_csv(OUT / "objective1_residuals.csv", index=False)
    pd.DataFrame({"blup": m["blup"]}).to_csv(OUT / "objective1_blups.csv", index=False)

    fe, r, b = out["fixed_effects"], out["residuals"], out["blups"]
    print(f"N = {m['N']:,} facility-years, {m['G']:,} facilities\n")
    print(table.round(4).to_string())
    print(f"\nscore: {score['b']:.4f} (95% CI {score['lo']:.4f} to {score['hi']:.4f}), p = {score['p']:.3f}")
    print(f"ICC {m['icc']:.3f}, sigma_u {out['sigma_u']:.3f}, sigma {out['sigma']:.3f}")
    print(f"residuals: sd {r['sd']:.3f}, skew {r['skew']:.2f}, excess kurtosis {r['excess_kurtosis']:.1f}, "
          f"W {r['shapiro_W']:.3f}, IQR {r['q25']:.3f} to {r['q75']:.3f}; sd by fitted quintile "
          + ", ".join(f"{v:.3f}" for v in out["residual_sd_by_fitted_quintile"]))
    print(f"random intercepts: sd {b['sd']:.3f}, skew {b['skew']:.2f}, excess kurtosis "
          f"{b['excess_kurtosis']:.1f}, W {b['shapiro_W']:.3f}")
    print(f"REML {out['reml']['b']:.5f}; fixed effects {fe['b']:.5f} (SE {fe['se_conv']:.5f}, robust "
          f"{fe['se_robust']:.5f}, CI {fe['lo']:.4f} to {fe['hi']:.4f}, p = {fe['p']:.3f}); with province "
          f"{out['with_province']['b']:.4f}; top 1% removed (cut {cut:.2f}) {out['top1pct_removed']['b']:.4f}, "
          f"p = {out['top1pct_removed']['p']:.3f}")
    print(f"MDE {out['mde']['model_se']:.4f} (model SE), {out['mde']['robust_se']:.4f} (robust SE)")
    print("AIC change: " + ", ".join(f"{k} {v:+.1f}" for k, v in aic.items()))
    og = out["ownership_gap"]
    print("ownership gap, municipal against provincial: " + ", ".join(
        f"{k} {v['pct']:.0f}%" for k, v in og.items() if k != "urban medians"))
    print(f"  urban only: n {og['urban only']['n']:,}, municipal {og['urban only']['n_municipal']:,}")
    print("  urban medians (municipal / provincial): " + ", ".join(
        f"{k} {v['municipal']:.2f}/{v['provincial']:.2f}"
        for k, v in [("all", og["urban medians"]["all"])] + list(og["urban medians"]["by_province"].items())))
    print(f"\nWritten to {path}")


if __name__ == "__main__":
    main()
