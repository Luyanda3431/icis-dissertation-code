"""
Objective 2: does spending per visit go with a higher ICIS score?
(Sections 3.6.2 and 5.3)

    score_it = a0 + a1 * log expenditure per visit_it + a2 * log headcount_it
               + urbanicity_i + year_t + v_i + n_it

Fitted on the finance sub-panel: six BAS provinces, no 2021 records, and
KwaZulu-Natal for 2017 only. It is checked the same way as Objective 1.

One extra question matters here. The random-intercept model uses both the
differences between clinics and the changes within them. The fixed-effects
fit uses only the changes within, and adding province takes out the
differences between provinces. Comparing the three shows where the
association comes from (Section 5.3.3).

Usage:
    python src/05_objective2_lmm.py
Writes:
    outputs/objective2.json
    outputs/table7_objective2.csv
    outputs/objective2_residuals.csv, outputs/objective2_blups.csv
"""

import numpy as np
import pandas as pd

from common import (OUT, coef, describe, design, fit_lmm, load_finance_panel,
                    save_json, within_estimator)

TERMS = ["log_exp", "log_headcount", "urban", "year"]


def main():
    f = load_finance_panel()
    f["y"] = f.score

    X = design(f, TERMS)
    m = fit_lmm(f.y, X, f.facility_id)
    c = coef(m, X, "log_exp")
    out = dict(N=m["N"], facilities=m["G"], expenditure=c, icc=m["icc"],
               sigma_v=np.sqrt(m["su2"]), sigma=np.sqrt(m["s2"]), aic=m["aic"],
               obs_per_facility=m["N"] / m["G"])
    table = pd.DataFrame({"coefficient": m["b"], "se": m["se"],
                          "lo": m["b"] - 1.96 * m["se"], "hi": m["b"] + 1.96 * m["se"]},
                         index=X.columns)
    table.to_csv(OUT / "table7_objective2.csv")
    out["coefficients"] = table.to_dict("index")

    out["residuals"] = describe(m["resid"])
    q = pd.qcut(m["fitted"], 5, labels=False)
    out["residual_sd_by_fitted_quintile"] = pd.Series(m["resid"]).groupby(q).std().tolist()
    out["blups"] = describe(m["blup"])

    Xfe = X.drop(columns=["const"] + [k for k in X.columns if k.startswith("ur_")])
    out["fixed_effects"] = within_estimator(f.y.values, Xfe.values, f.facility_id.values)
    Xp = design(f, TERMS + ["province"])
    out["with_province"] = coef(fit_lmm(f.y, Xp, f.facility_id), Xp, "log_exp")
    # spending per visit hardly moves within a clinic, which is part of why the
    # within-facility estimate is so imprecise
    out["median_within_facility_sd_log_exp"] = f.groupby("facility_id").log_exp.std().median()

    aic = {f"drop {k}": fit_lmm(f.y, design(f, [x for x in TERMS if x != k]),
                                f.facility_id)["aic"] - m["aic"] for k in TERMS}
    aic["add province"] = fit_lmm(f.y, Xp, f.facility_id)["aic"] - m["aic"]
    out["aic_change"] = aic

    path = save_json(out, "objective2")
    pd.DataFrame({"fitted": m["fitted"], "residual": m["resid"]}) \
        .to_csv(OUT / "objective2_residuals.csv", index=False)
    pd.DataFrame({"blup": m["blup"]}).to_csv(OUT / "objective2_blups.csv", index=False)

    fe, wp, r, b = out["fixed_effects"], out["with_province"], out["residuals"], out["blups"]
    print(f"N = {m['N']:,} facility-years, {m['G']:,} facilities "
          f"({out['obs_per_facility']:.1f} per facility)\n")
    print(table.round(3).to_string())
    print(f"\nlog expenditure per visit: {c['b']:.3f} (95% CI {c['lo']:.3f} to {c['hi']:.3f}), "
          f"p = {c['p']:.3f}; ICC {m['icc']:.3f}, sigma_v {out['sigma_v']:.2f}, sigma {out['sigma']:.2f}")
    print(f"residuals: skew {r['skew']:.2f}, excess kurtosis {r['excess_kurtosis']:.1f}, W {r['shapiro_W']:.3f}; "
          f"random intercepts: sd {b['sd']:.2f}, skew {b['skew']:.2f}, excess kurtosis {b['excess_kurtosis']:.1f}")
    print(f"fixed effects: {fe['b']:.2f} (robust CI {fe['lo']:.2f} to {fe['hi']:.2f}), p = {fe['p']:.2f}")
    print(f"with province: {wp['b']:.2f} (CI {wp['lo']:.2f} to {wp['hi']:.2f})")
    print(f"median within-facility SD of log expenditure: {out['median_within_facility_sd_log_exp']:.2f}")
    print("AIC change: " + ", ".join(f"{k} {v:+.1f}" for k, v in aic.items()))
    print(f"\nWritten to {path}")


if __name__ == "__main__":
    main()
