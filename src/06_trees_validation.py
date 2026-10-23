"""
Classification trees: how the models are validated (Sections 3.6.3 and 5.2.6,
Tables 5 and 8).

Each clinic appears up to six times, once per year. A random train-test split
therefore puts different years of the same clinic on both sides, and the test
set stops being new data. This script counts how often that happens, then
fits CART, Random Forest and gradient boosting under three designs:

    random    70/30 split ignoring clinics (leaky, kept only for comparison)
    group     70/30 split by clinic, so a clinic is wholly in one side (main)
    temporal  train on 2017-2021, test on 2022

and under five-fold GroupKFold, which is also what the tuning in
07_trees_tuning.py uses.

Usage:
    python src/06_trees_validation.py
Writes:
    outputs/trees_validation.json
"""

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from common import MODELS, load_trees_panel, metrics, save_json, splits, tree_X


def main():
    d = load_trees_panel()
    X, y, g = tree_X(d), d.high.values, d.facility_id.values
    sp = splits(d, X, y, g)
    out = dict(n=len(d), facilities=len(set(g)), high_share=y.mean(),
               # the root split is always a province: the ones with the lowest
               # and highest High shares (Section 5.2.8)
               high_share_by_province=d.groupby("province").high.mean().to_dict())

    # how many test clinics also have rows in training under a random split
    tr, te = sp["random"]
    out["leakage"] = dict(test_facilities=len(set(g[te])),
                          also_in_training=len(set(g[tr]) & set(g[te])))
    tr, te = sp["group"]
    out["group_split"] = dict(train=len(tr), test=len(te),
                              train_facilities=len(set(g[tr])),
                              test_facilities=len(set(g[te])),
                              shared=len(set(g[tr]) & set(g[te])))
    tr, te = sp["temporal"]
    out["temporal_split"] = dict(train=len(tr), test=len(te),
                                 train_high=y[tr].mean(), test_high=y[te].mean())

    res = {}
    for scheme, (tr, te) in sp.items():
        for name, make in MODELS.items():
            m = make().fit(X.iloc[tr], y[tr])
            r = metrics(y[te], m.predict_proba(X.iloc[te])[:, 1])
            r["train_auc"] = roc_auc_score(y[tr], m.predict_proba(X.iloc[tr])[:, 1])
            res[f"{scheme}|{name}"] = r
        # Predicting Low for everyone gets 61% right without finding a single
        # Gold or Platinum clinic, which is why accuracy alone is not enough
        res[f"{scheme}|Majority"] = dict(accuracy=1 - y[te].mean(), auc=.5,
                                         sensitivity=0, specificity=1)
    out["test"] = res

    cv = {}
    for name, make in MODELS.items():
        aucs = []
        for tr, te in GroupKFold(5).split(X, y, g):
            m = make().fit(X.iloc[tr], y[tr])
            aucs.append(roc_auc_score(y[te], m.predict_proba(X.iloc[te])[:, 1]))
        cv[name] = dict(mean=np.mean(aucs), sd=np.std(aucs), folds=aucs)
    out["group_kfold"] = cv

    path = save_json(out, "trees_validation")
    lk = out["leakage"]
    print(f"{out['n']:,} facility-years, {out['facilities']:,} facilities, "
          f"{out['high_share']:.1%} High")
    print("High share by province: " + ", ".join(
        f"{p} {v:.1%}" for p, v in sorted(out["high_share_by_province"].items(), key=lambda t: t[1])))
    print(f"random split: {lk['also_in_training']:,} of {lk['test_facilities']:,} "
          f"test facilities also in training\n")
    print(f"{'':10s}" + "".join(f"{s:>10s}" for s in sp) + f"{'GroupKFold':>12s}")
    for name in MODELS:
        print(f"{name:10s}" + "".join(f"{res[f'{s}|{name}']['auc']:10.3f}" for s in sp)
              + f"{cv[name]['mean']:12.3f}")
    print(f"\nWritten to {path}")


if __name__ == "__main__":
    main()
