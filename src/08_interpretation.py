"""
How the predictors shape the classification (Sections 3.6.8, 5.2.9, 5.2.10
and 5.3.4).

Importance only says which predictors matter, not how they push the
prediction up or down. For the Random Forest fitted on the grouped training
set this script works out both:

    impurity importance      from the fitted forest, one-hot columns summed
    permutation importance   fall in held-out AUC when a predictor is
                             shuffled (10 repeats)
    Shapley values           exact Shapley values for 500 held-out
                             facility-years, against 100 training rows
    partial dependence       for ANC rate, headcount, year and province

It then does the same for a forest fitted to the finance sub-panel with log
expenditure per visit added, since that is the only place expenditure enters
a tree.

The Shapley values are worked out exactly rather than with the shap package.
With five or six predictors there are only 32 or 64 coalitions, so every one
can be computed. The eight province columns are switched on and off together,
so province is treated as one predictor, and the values add up exactly to
each prediction minus the average (checked below).

Usage:
    python src/08_interpretation.py
Writes:
    outputs/interpretation.json
    outputs/shapley_main.csv, outputs/shapley_finance.csv
"""

from itertools import combinations
from math import factorial

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from common import (OUT, RF, SEED, load_finance_panel, load_trees_panel,
                    predictor_groups, save_json, tree_X)

rng = np.random.default_rng(SEED)


def shapley(model, X, X_explain, X_background, groups):
    """Exact interventional Shapley values over predictor groups.

    v(S) = mean over background rows of f(x_S from the explained row,
                                           x_notS from the background row)
    phi_j = sum over S not containing j of
            |S|! (M - |S| - 1)! / M! * (v(S + j) - v(S))
    """
    names = list(groups)
    M = len(names)
    cols = list(X.columns)
    gidx = {k: [cols.index(c) for c in v] for k, v in groups.items()}
    subsets = [frozenset(s) for r in range(M + 1) for s in combinations(range(M), r)]
    Xe, Xb = X_explain.values, X_background.values
    ne, nb = len(Xe), len(Xb)
    v = {}
    for S in subsets:
        idx = [i for k in S for i in gidx[names[k]]]
        big = np.repeat(Xb[None, :, :], ne, axis=0).copy()
        if idx:
            big[:, :, idx] = Xe[:, None, idx]
        flat = pd.DataFrame(big.reshape(-1, Xb.shape[1]), columns=cols)
        v[S] = model.predict_proba(flat)[:, 1].reshape(ne, nb).mean(1)
    phi = np.zeros((ne, M))
    for j in range(M):
        for S in subsets:
            if j in S:
                continue
            w = factorial(len(S)) * factorial(M - len(S) - 1) / factorial(M)
            phi[:, j] += w * (v[S | {j}] - v[S])
    return pd.DataFrame(phi, columns=names), v[frozenset()], v[frozenset(range(M))]


def permutation_importance(model, X, y, groups, reps=10):
    base = roc_auc_score(y, model.predict_proba(X)[:, 1])
    out = {}
    for k, cs in groups.items():
        drops = []
        for _ in range(reps):
            Xp = X.copy()
            Xp[cs] = X[cs].values[rng.permutation(len(X))]
            drops.append(base - roc_auc_score(y, model.predict_proba(Xp)[:, 1]))
        out[k] = dict(mean=np.mean(drops), sd=np.std(drops))
    return out


def partial_dependence(model, Xs, col, grid):
    res = []
    for value in grid:
        Xc = Xs.copy()
        Xc[col] = value
        res.append(model.predict_proba(Xc)[:, 1].mean())
    return res


def partial_dependence_province(model, Xs):
    pc = [c for c in Xs.columns if c.startswith("p_")]
    res = {}
    for prov in ["Gauteng"] + [c[2:] for c in pc]:
        Xc = Xs.copy()
        Xc[pc] = 0.0
        if prov != "Gauteng":
            Xc["p_" + prov] = 1.0
        res[prov] = model.predict_proba(Xc)[:, 1].mean()
    return res


def analyse(d, X, y, g, tag):
    # the same grouped split as in 06_trees_validation.py
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=.3,
                                    random_state=SEED).split(X, y, g))
    m = RF().fit(X.iloc[tr], y[tr])
    G = predictor_groups(X.columns)
    out = dict(test_auc=roc_auc_score(y[te], m.predict_proba(X.iloc[te])[:, 1]))

    fi = pd.Series(m.feature_importances_, index=X.columns)
    out["impurity_importance"] = {k: fi[v].sum() for k, v in G.items()}
    out["permutation_importance"] = permutation_importance(m, X.iloc[te], y[te], G)

    # 500 held-out rows to explain, against 100 training rows as the
    # "average clinic" background
    ie = rng.choice(te, 500, replace=False)
    ib = rng.choice(tr, 100, replace=False)
    phi, v_empty, v_full = shapley(m, X, X.iloc[ie], X.iloc[ib], G)
    out["shapley_mean_abs"] = phi.abs().mean().to_dict()
    out["shapley_base_value"] = v_empty.mean()
    out["shapley_additivity_max_error"] = np.abs(phi.sum(axis=1).values - (v_full - v_empty)).max()
    dd = d.iloc[ie].reset_index(drop=True)
    out["shapley_province_by_province"] = phi["Province"].groupby(dd.province).mean().to_dict()
    out["shapley_year_by_year"] = phi["Year"].groupby(dd.year).mean().to_dict()
    out["corr_headcount_with_its_shapley"] = np.corrcoef(dd.log_headcount, phi["Log headcount"])[0, 1]
    keep = dd.anc_rate.notna()
    out["corr_anc_with_its_shapley"] = np.corrcoef(dd.anc_rate[keep], phi["ANC rate"][keep])[0, 1]
    extra = {"log_exp": dd.log_exp.values} if "log_exp" in dd else {}
    phi.assign(province=dd.province.values, year=dd.year.values,
               anc_rate=dd.anc_rate.values, log_headcount=dd.log_headcount.values,
               **extra).to_csv(OUT / f"shapley_{tag}.csv", index=False)

    # partial dependence: set one predictor to a value for 1,500 rows and
    # average the predicted chance of High
    Xs = X.iloc[rng.choice(len(X), 1500, replace=False)]
    out["partial_dependence"] = {}
    for col in ["anc_rate", "log_headcount"] + (["log_exp"] if "log_exp" in X else []):
        grid = list(np.nanquantile(X[col], np.linspace(.02, .98, 25)))
        out["partial_dependence"][col] = dict(grid=grid, pd=partial_dependence(m, Xs, col, grid))
    years = list(range(2017, 2023))
    out["partial_dependence"]["year"] = dict(grid=years, pd=partial_dependence(m, Xs, "year", years))
    out["partial_dependence_province"] = partial_dependence_province(m, Xs)
    return out


def main():
    d = load_trees_panel()
    X, y, g = tree_X(d), d.high.values, d.facility_id.values
    res = {"main": analyse(d, X, y, g, "main")}
    print("main forest done", flush=True)

    f = load_finance_panel()
    Xf, yf, gf = tree_X(f, extra=("log_exp",)), f.high.values, f.facility_id.values
    cv = []
    for tr, te in GroupKFold(5).split(Xf, yf, gf):
        m = RF().fit(Xf.iloc[tr], yf[tr])
        cv.append(roc_auc_score(yf[te], m.predict_proba(Xf.iloc[te])[:, 1]))
    res["finance"] = analyse(f, Xf, yf, gf, "finance")
    res["finance"].update(n=len(f), facilities=f.facility_id.nunique(), high_share=yf.mean(),
                          group_kfold_auc=np.mean(cv), group_kfold_sd=np.std(cv),
                          missing_anc_rate=int(f.anc_rate.isna().sum()))

    path = save_json(res, "interpretation")
    for k, r in res.items():
        print(f"\n[{k}] held-out AUC {r['test_auc']:.3f}; Shapley additivity error "
              f"{r['shapley_additivity_max_error']:.1e}")
        print(f"{'':16s}{'impurity':>10s}{'permutation':>13s}{'|Shapley|':>11s}")
        for p in r["impurity_importance"]:
            print(f"{p:16s}{r['impurity_importance'][p]:10.3f}"
                  f"{r['permutation_importance'][p]['mean']:13.3f}{r['shapley_mean_abs'][p]:11.3f}")
    print(f"\nWritten to {path}")


if __name__ == "__main__":
    main()
