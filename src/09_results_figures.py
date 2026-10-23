"""
Figures for Chapter 5 (Figures 15 to 25), drawn from the outputs of scripts
04 to 08.

Usage:
    python src/09_results_figures.py
Writes:
    outputs/figures/fig15_*.png to fig25_*.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FixedFormatter, FixedLocator, NullLocator
from scipy import stats
from sklearn.metrics import roc_curve
from sklearn.tree import plot_tree

from common import (FIG, MODELS, OUT, load_trees_panel, read_json, splits,
                    tree_X)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 12, "axes.titlesize": 13,
    "axes.titleweight": "bold", "axes.labelsize": 12, "xtick.labelsize": 11,
    "ytick.labelsize": 11, "legend.fontsize": 10.5, "axes.spines.top": False,
    "axes.spines.right": False, "savefig.dpi": 300})
COLOUR = {"CART": "#e08a4f", "RF": "#2c6fa8", "GBM": "#5aa66a"}
LABEL = {"CART": "CART", "RF": "Random Forest", "GBM": "Gradient boosting"}


def log_ticks(ax, values, labels):
    ax.xaxis.set_major_locator(FixedLocator(values))
    ax.xaxis.set_major_formatter(FixedFormatter(labels))
    ax.xaxis.set_minor_locator(NullLocator())


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, bbox_inches="tight")
    plt.close(fig)


def lmm_diagnostics(tag, sigma, title, name):
    r = pd.read_csv(OUT / f"{tag}_residuals.csv")
    blup = pd.read_csv(OUT / f"{tag}_blups.csv").blup.values
    fig, ax = plt.subplots(2, 2, figsize=(13, 10))
    ax[0, 0].hexbin(r.fitted, r.residual, gridsize=60, cmap="Blues", mincnt=1, bins="log")
    ax[0, 0].axhline(0, color="#c0392b", lw=1.2)
    ax[0, 0].set(xlabel="Fitted value", ylabel="Within-facility residual",
                 title="Residuals against fitted values")
    for a, x, lab in [(ax[0, 1], r.residual.values, "Residual"),
                      (ax[1, 1], blup, "Intercept")]:
        (osm, osr), (sl, ic, _) = stats.probplot(x, dist="norm")
        a.scatter(osm, osr, s=4, alpha=.4, color="#2c6fa8")
        a.plot(osm, sl * osm + ic, color="#c0392b", lw=1.2)
        a.set(xlabel="Theoretical normal quantile", ylabel=f"{lab} quantile",
              title=f"Normal Q-Q plot of {lab.lower()}s" if lab == "Residual"
              else "Normal Q-Q plot of random intercepts")
    ax[1, 0].hist(blup, bins=50, density=True, color="#7fb3d5", edgecolor="white")
    xs = np.linspace(blup.min(), blup.max(), 200)
    ax[1, 0].plot(xs, stats.norm.pdf(xs, 0, sigma), color="#c0392b", lw=1.5,
                  label=f"Normal, sd = estimated σ ({sigma:.2f})")
    ax[1, 0].legend(frameon=False, loc="upper left")
    ax[1, 0].set(xlabel="Predicted facility intercept (BLUP)", ylabel="Density",
                 title="Distribution of random intercepts")
    fig.suptitle(title, fontweight="bold", fontsize=14)
    save(fig, name)


def coefficient_plot(rows, xlabel, title, name, band=None):
    fig, ax = plt.subplots(figsize=(12, 0.8 * len(rows) + 2.2))
    lo = min(r[2] for r in rows)
    hi = max(r[3] for r in rows)
    if band:
        lo, hi = min(lo, -band), max(hi, band)
    span = hi - lo
    for i, (lab, b, l, h) in enumerate(rows[::-1]):
        ax.plot([l, h], [i, i], color="#2c6fa8", lw=2.4)
        ax.scatter([b], [i], color="#1b4f72", s=60, zorder=3)
        fmt = "{:.4f}" if abs(h) < 0.1 else "{:.2f}"
        ax.text(hi + span * .08, i, (fmt + "  [" + fmt + ", " + fmt + "]").format(b, l, h),
                va="center", fontsize=11)
    ax.axvline(0, color="grey", ls="--", lw=1)
    if band:
        ax.axvspan(-band, band, color="#f5cba7", alpha=.35, lw=0,
                   label=f"Smaller than the minimum detectable effect (|β| < {band:.4f})")
        ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(.4, -.14))
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows[::-1]])
    ax.set(xlabel=xlabel, xlim=(lo - span * .05, hi + span * .55), ylim=(-.6, len(rows) - .4))
    ax.set_title(title, pad=14)
    save(fig, name)


def ci(c):
    return c["b"], c["lo"], c["hi"]


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    o1, o2 = read_json("objective1"), read_json("objective2")
    val, tun, imp = read_json("trees_validation"), read_json("trees_tuning"), read_json("interpretation")

    # ---- mixed models ----------------------------------------------------
    lmm_diagnostics("objective1", o1["sigma_u"], "Objective 1 Mixed-Effects Model: Diagnostics",
                    "fig15_objective1_diagnostics.png")
    lmm_diagnostics("objective2", o2["sigma_v"], "Objective 2 Mixed-Effects Model: Diagnostics",
                    "fig23_objective2_diagnostics.png")
    coefficient_plot(
        [("Random intercept, ML (primary)", *ci(o1["score"])),
         ("Random intercept, REML", *ci(o1["reml"])),
         ("Fixed effects, cluster-robust SE", *ci(o1["fixed_effects"])),
         ("Random intercept + province", *ci(o1["with_province"])),
         ("Random intercept, top 1% of ANC removed", *ci(o1["top1pct_removed"]))],
        "Coefficient on standardised ICIS score (log ANC per 1,000)",
        "Objective 1: ICIS Score Coefficient Across Specifications",
        "fig16_objective1_sensitivity.png", band=o1["mde"]["model_se"])
    coefficient_plot(
        [("Random intercept, ML (primary)", *ci(o2["expenditure"])),
         ("Fixed effects, cluster-robust SE", *ci(o2["fixed_effects"])),
         ("Random intercept + province", *ci(o2["with_province"]))],
        "Coefficient on log expenditure per visit (ICIS percentage points)",
        "Objective 2: Expenditure Coefficient Across Specifications",
        "fig24_objective2_sensitivity.png")

    # ---- validation ------------------------------------------------------
    d = load_trees_panel()
    X, y, g = tree_X(d), d.high.values, d.facility_id.values
    tr, te = splits(d, X, y, g)["group"]
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.8))
    for k, make in MODELS.items():
        m = make().fit(X.iloc[tr], y[tr])
        fpr, tpr, _ = roc_curve(y[te], m.predict_proba(X.iloc[te])[:, 1])
        ax[0].plot(fpr, tpr, color=COLOUR[k], lw=2,
                   label=f"{LABEL[k]} (AUC {val['test'][f'group|{k}']['auc']:.3f})")
        if k == "CART":
            cart = m
    ax[0].plot([0, 1], [0, 1], ls="--", color="grey", lw=1, label="No discrimination")
    ax[0].set(xlabel="False positive rate (1 − specificity)",
              ylabel="True positive rate (sensitivity)", title="ROC curves, facilities held out")
    ax[0].legend(frameon=False, loc="lower right")
    schemes = ["random", "group", "temporal"]
    names = {"random": "Random 70/30\n(leaky)", "group": "Grouped by\nfacility",
             "temporal": "Train 2017–21,\ntest 2022"}
    w = .25
    for i, k in enumerate(MODELS):
        v = [val["test"][f"{s}|{k}"]["auc"] for s in schemes]
        bars = ax[1].bar(np.arange(3) + (i - 1) * w, v, w, color=COLOUR[k], label=LABEL[k])
        for bar, value in zip(bars, v):
            ax[1].text(bar.get_x() + bar.get_width() / 2, value + .003, f"{value:.3f}",
                       ha="center", fontsize=9.5, rotation=90)
    lo = min(val["test"][f"{s}|{k}"]["auc"] for s in schemes for k in MODELS)
    ax[1].set(xticks=range(3), xticklabels=[names[s] for s in schemes],
              ylim=(max(0, lo - .05), 1), ylabel="Test-set AUC",
              title="AUC under three validation designs")
    ax[1].legend(frameon=False, loc="upper left", ncol=3, fontsize=9.5)
    fig.suptitle("Classification Tree Performance", fontweight="bold", fontsize=14)
    save(fig, "fig17_tree_validation.png")

    fig, ax = plt.subplots(figsize=(16, 7.5))
    cols = [c.replace("p_", "").replace("ur_", "").replace("log_headcount", "log headcount")
             .replace("anc_rate", "ANC rate") for c in X.columns]
    plot_tree(cart, max_depth=2, feature_names=cols, class_names=["Low", "High"], filled=True,
              impurity=False, proportion=True, rounded=True, fontsize=11, ax=ax, precision=2)
    ax.set_title("Top Three Levels of the Tuned CART Tree (grouped training set)",
                 fontsize=14, fontweight="bold")
    fig.savefig(FIG / "fig19_cart_top.png", bbox_inches="tight")
    plt.close(fig)

    # ---- tuning ----------------------------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(17, 5.4))
    gap = tun["cart_depth_gap"]
    ax[0].plot(range(len(gap)), [r["train"] for r in gap], "o-", color="#7f8c8d", label="Training folds")
    ax[0].plot(range(len(gap)), [r["test"] for r in gap], "o-", color="#e08a4f", label="Held-out folds")
    ax[0].set(xticks=range(len(gap)), xticklabels=[r["depth"].replace("None", "none") for r in gap],
              xlabel="Maximum depth", ylabel="AUC (five-fold GroupKFold)",
              title="CART: depth and the train–test gap")
    ax[0].legend(frameon=False)
    oob = tun["rf_oob"]
    ax[1].plot([r["n_trees"] for r in oob], [r["oob_auc"] for r in oob], "o-", color="#2c6fa8")
    ax[1].set(xlabel="Number of trees", ylabel="Out-of-bag AUC",
              title="Random Forest: out-of-bag convergence")
    steps = list(range(1, 600, 5)) + [600]
    for lr, c in [("0.01", "#c6dbef"), ("0.05", "#4292c6"), ("0.1", "#08519c"), ("0.3", "#c0392b")]:
        s = tun["gbm_learning_rate"][lr]["test_curve"]
        ax[2].plot(steps[:len(s)], s, color=c, lw=2, label=f"Learning rate {lr}")
    s = tun["gbm_learning_rate"]["0.05"]["train_curve"]
    ax[2].plot(steps[:len(s)], s, color="#4292c6", ls=":", lw=1.6, label="Training, rate 0.05")
    ax[2].set(xlabel="Number of trees", ylabel="AUC", title="Gradient boosting: staged AUC")
    ax[2].legend(frameon=False, fontsize=9.5, loc="lower right")
    fig.suptitle("Tuning Diagnostics for the Three Tree Methods", fontweight="bold", fontsize=14)
    save(fig, "fig18_tree_tuning.png")

    # ---- importance, Shapley, partial dependence -------------------------
    m = imp["main"]
    order = ["Province", "Year", "Log headcount", "ANC rate", "Urbanicity"]
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    measures = [(lambda k: m["impurity_importance"][k], "Impurity (Gini)",
                 "Share of total impurity reduction"),
                (lambda k: m["permutation_importance"][k]["mean"], "Permutation",
                 "Drop in held-out AUC"),
                (lambda k: m["shapley_mean_abs"][k], "Shapley (mean |SHAP|)",
                 "Mean absolute contribution to P(High)")]
    for a, (get, title, xl) in zip(ax, measures):
        v = [get(k) for k in order]
        a.barh(order[::-1], v[::-1],
               color=["#c0392b" if k == "ANC rate" else "#2c6fa8" for k in order[::-1]])
        for i, value in enumerate(v[::-1]):
            a.text(value, i, f" {value:.3f}", va="center", fontsize=10.5)
        a.set(title=title, xlabel=xl, xlim=(0, max(max(v) * 1.3, 1e-3)))
    fig.suptitle("Random Forest Variable Importance: Three Measures", fontweight="bold", fontsize=14)
    save(fig, "fig20_importance.png")

    sh = pd.read_csv(OUT / "shapley_main.csv")
    fig, ax = plt.subplots(2, 2, figsize=(14, 10))
    pv = pd.Series(m["shapley_province_by_province"]).sort_values()
    ax[0, 0].barh(pv.index, pv.values, color=["#c0392b" if v < 0 else "#2c6fa8" for v in pv.values])
    ax[0, 0].axvline(0, color="grey", lw=1)
    ax[0, 0].set(title="Province: mean Shapley contribution",
                 xlabel="Change in P(High) attributed to province")
    yv = pd.Series(m["shapley_year_by_year"]).sort_index()
    ax[0, 1].bar(yv.index.astype(str), yv.values,
                 color=["#c0392b" if v < 0 else "#2c6fa8" for v in yv.values])
    ax[0, 1].axhline(0, color="grey", lw=1)
    ax[0, 1].set(title="Assessment year: mean Shapley contribution", ylabel="Change in P(High)")
    ax[1, 0].scatter(np.expm1(sh.log_headcount), sh["Log headcount"], s=10, alpha=.5, color="#2c6fa8")
    ax[1, 0].set_xscale("log")
    log_ticks(ax[1, 0], [1000, 10000, 100000], ["1,000", "10,000", "100,000"])
    ax[1, 0].axhline(0, color="grey", lw=1)
    ax[1, 0].set(title="Headcount: Shapley value by facility",
                 xlabel="Annual headcount (log scale)", ylabel="Shapley value")
    pos = sh.anc_rate > 0
    ax[1, 1].scatter(sh.anc_rate[pos], sh["ANC rate"][pos], s=10, alpha=.5, color="#c0392b")
    ax[1, 1].set_xscale("log")
    log_ticks(ax[1, 1], [0.01, 0.1, 1], ["0.01", "0.1", "1"])
    ax[1, 1].axhline(0, color="grey", lw=1)
    ax[1, 1].set(title="ANC rate: Shapley value by facility",
                 xlabel="ANC first visits per 1,000 fertile women (log scale)",
                 ylabel="Shapley value", ylim=ax[1, 0].get_ylim())
    fig.suptitle("Shapley Value Attributions, Random Forest (500 held-out facility-years)",
                 fontweight="bold", fontsize=14)
    save(fig, "fig21_shapley.png")

    p = m["partial_dependence"]
    fig, ax = plt.subplots(1, 4, figsize=(19, 5), gridspec_kw={"width_ratios": [1, 1, 1, 1.25]})
    ax[0].plot(p["anc_rate"]["grid"], p["anc_rate"]["pd"], color="#c0392b", lw=2.2)
    ax[0].set(xscale="log", xlabel="ANC per 1,000 (log scale)",
              ylabel="Average predicted P(High)", title="ANC first-visit rate")
    log_ticks(ax[0], [0.05, 0.1, 0.3, 1, 3], ["0.05", "0.1", "0.3", "1", "3"])
    ax[1].plot(np.expm1(p["log_headcount"]["grid"]), p["log_headcount"]["pd"], color="#2c6fa8", lw=2.2)
    ax[1].set(xscale="log", xlabel="Annual headcount (log scale)", title="Headcount")
    log_ticks(ax[1], [5000, 10000, 20000, 50000], ["5,000", "10,000", "20,000", "50,000"])
    ax[2].plot(p["year"]["grid"], p["year"]["pd"], "o-", color="#5aa66a", lw=2.2)
    ax[2].set(xlabel="Assessment year", title="Assessment year")
    pp = pd.Series(m["partial_dependence_province"]).sort_values()
    ax[3].barh(pp.index, pp.values, color="#7fb3d5")
    for i, value in enumerate(pp.values):
        ax[3].text(value, i, f" {value:.2f}", va="center", fontsize=10)
    ax[3].set(xlabel="Average predicted P(High)", title="Province", xlim=(0, min(1, pp.max() * 1.3)))
    fig.suptitle("Partial Dependence, Random Forest", fontweight="bold", fontsize=14)
    save(fig, "fig22_partial_dependence.png")

    fm = imp["finance"]
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.2))
    pe = fm["partial_dependence"]["log_exp"]
    ax[0].plot(np.exp(pe["grid"]), pe["pd"], color="#8e44ad", lw=2.2)
    ax[0].set(xscale="log", xlabel="Expenditure per visit, Rand (log scale)",
              ylabel="Average predicted P(High)", title="Partial dependence: expenditure per visit")
    log_ticks(ax[0], [20, 50, 100, 200, 500], ["R20", "R50", "R100", "R200", "R500"])
    o6 = ["Province", "Urbanicity", "Year", "Log headcount", "Log expenditure", "ANC rate"]
    v = [fm["shapley_mean_abs"][k] for k in o6]
    ax[1].barh(o6[::-1], v[::-1],
               color=["#8e44ad" if k == "Log expenditure" else "#2c6fa8" for k in o6[::-1]])
    for i, value in enumerate(v[::-1]):
        ax[1].text(value, i, f" {value:.3f}", va="center", fontsize=10.5)
    ax[1].set(title="Mean |Shapley value|, finance-sample forest",
              xlabel="Mean absolute contribution to P(High)", xlim=(0, max(v) * 1.3))
    fig.suptitle(f"Finance-Sample Random Forest ({fm['n']:,.0f} facility-years, six provinces)",
                 fontweight="bold", fontsize=14)
    save(fig, "fig25_finance_forest.png")

    print(f"Figures 15 to 25 written to {FIG}")


if __name__ == "__main__":
    main()
