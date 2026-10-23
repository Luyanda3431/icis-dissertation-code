"""
Classification trees: choosing the settings (Sections 3.6.4 to 3.6.6, 5.2.7
and Table 6).

Every setting is scored by five-fold GroupKFold AUC, so a clinic is never in
the fold being fitted and the fold being scored at the same time.

CART       grid over depth, minimum split and minimum leaf size; the
           cost-complexity pruning path; training against held-out AUC as
           the tree gets deeper; which predictor the root split uses in each
           fold
Forest     grid over predictors per split and leaf size; out-of-bag AUC as
           trees are added
Boosting   held-out AUC after each tree at four learning rates, on the
           grouped split, and the effect of tree depth
Encoding   one-hot against integer-coded province

Usage:
    python src/07_trees_tuning.py            (takes several minutes)
Writes:
    outputs/trees_tuning.json
"""

from itertools import product

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.tree import DecisionTreeClassifier

from common import (CART, GBM, RF, SEED, load_trees_panel, save_json, splits,
                    tree_X)


def main():
    d = load_trees_panel()
    X, y, g = tree_X(d), d.high.values, d.facility_id.values
    # the same five folds for every setting, so the comparisons are fair
    folds = list(GroupKFold(5).split(X, y, g))

    def cv_auc(make, XX=X):
        a = []
        for tr, te in folds:
            m = make().fit(XX.iloc[tr], y[tr])
            a.append(roc_auc_score(y[te], m.predict_proba(XX.iloc[te])[:, 1]))
        return np.mean(a), np.std(a)

    out = {}

    # ---- CART ------------------------------------------------------------
    # depth, minimum split size and minimum leaf size all stop the tree
    # growing too far (pre-pruning)
    grid = []
    for md, mss, msl in product([3, 5, 7, 10, None], [20, 50, 100], [10, 25, 50]):
        mu, sd = cv_auc(lambda: DecisionTreeClassifier(
            max_depth=md, min_samples_split=mss, min_samples_leaf=msl,
            random_state=SEED))
        grid.append(dict(max_depth=str(md), min_split=mss, min_leaf=msl, auc=mu, sd=sd))
    out["cart_grid"] = grid
    best = max(grid, key=lambda r: r["auc"])
    out["cart_grid_best"] = best
    out["cart_best_by_depth"] = {
        dep: max((r for r in grid if r["max_depth"] == dep), key=lambda r: r["auc"])["auc"]
        for dep in ["3", "5", "7", "10", "None"]}
    out["cart_selected_leaves_full_data"] = CART().fit(X, y).get_n_leaves()
    tr, te = splits(d, X, y, g)["group"]
    sel = CART().fit(X.iloc[tr], y[tr])
    out["cart_selected_grouped_train"] = dict(leaves=sel.get_n_leaves(),
                                              depth=sel.get_depth())

    # Cost-complexity pruning grows a large tree and cuts it back. It is shown
    # to check that pre-pruning lands in the same place. Minimum leaf 25, as
    # in the selected tree.
    path = DecisionTreeClassifier(min_samples_leaf=25, random_state=SEED) \
        .cost_complexity_pruning_path(X, y)
    out["pruning_path"] = dict(n_alphas=len(path.ccp_alphas),
                               smallest=path.ccp_alphas[1],
                               largest=path.ccp_alphas[-2])
    prune = []
    for alpha in [0, 2e-4, 5e-4, 1e-3, 2e-3]:
        a, leaves = [], []
        for f_tr, f_te in folds:
            m = DecisionTreeClassifier(min_samples_leaf=25, ccp_alpha=alpha,
                                       random_state=SEED).fit(X.iloc[f_tr], y[f_tr])
            a.append(roc_auc_score(y[f_te], m.predict_proba(X.iloc[f_te])[:, 1]))
            leaves.append(m.get_n_leaves())
        prune.append(dict(alpha=alpha, auc=np.mean(a), leaves=np.mean(leaves)))
    out["pruning"] = prune

    # training AUC keeps rising with depth; the held-out AUC stops, and the
    # gap between them shows where the tree starts memorising
    gap = []
    for depth in [1, 2, 3, 5, 7, 10, 15, None]:
        a_tr, a_te = [], []
        for f_tr, f_te in folds:
            m = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=25,
                                       random_state=SEED).fit(X.iloc[f_tr], y[f_tr])
            a_tr.append(roc_auc_score(y[f_tr], m.predict_proba(X.iloc[f_tr])[:, 1]))
            a_te.append(roc_auc_score(y[f_te], m.predict_proba(X.iloc[f_te])[:, 1]))
        gap.append(dict(depth=str(depth), train=np.mean(a_tr), test=np.mean(a_te)))
    out["cart_depth_gap"] = gap

    # does the first split stay the same when the data change?
    cols = list(X.columns)
    stab = []
    for f_tr, _ in folds:
        t = CART().fit(X.iloc[f_tr], y[f_tr]).tree_
        stab.append(dict(root=cols[t.feature[0]],
                         second_level=[cols[t.feature[t.children_left[0]]],
                                       cols[t.feature[t.children_right[0]]]]))
    out["cart_root_by_fold"] = stab
    print("CART done", flush=True)

    # ---- Random Forest ---------------------------------------------------
    # max_features 1.0 is bagging (every predictor at every split); 1 is a
    # single random predictor per split
    rf_grid = []
    for mf in ["sqrt", 0.5, 1.0, 1]:
        for msl in [5, 20, 50]:
            mu, sd = cv_auc(lambda: RF(300, mf, msl))
            rf_grid.append(dict(max_features=str(mf), min_leaf=msl, auc=mu, sd=sd))
    out["rf_grid"] = rf_grid
    # out-of-bag AUC is used only to decide how many trees are enough
    oob = []
    for n in [25, 50, 100, 200, 300, 500]:
        m = RandomForestClassifier(n_estimators=n, max_features="sqrt",
                                   min_samples_leaf=20, oob_score=True,
                                   random_state=SEED, n_jobs=-1).fit(X, y)
        oob.append(dict(n_trees=n, oob_auc=roc_auc_score(y, m.oob_decision_function_[:, 1])))
    out["rf_oob"] = oob
    print("Forest done", flush=True)

    # ---- Gradient boosting -----------------------------------------------
    # learning rate and number of trees work against each other, so the
    # held-out AUC is tracked after every added tree
    checkpoints = [25, 50, 100, 200, 300, 400, 500, 600]
    boost = {}
    for lr in [0.01, 0.05, 0.1, 0.3]:
        m = GBM(600, 3, lr).fit(X.iloc[tr], y[tr])
        s_te = [roc_auc_score(y[te], p) for p in m.staged_decision_function(X.iloc[te])]
        r = dict(test_curve=s_te[::5] + [s_te[-1]],
                 test_at={k: s_te[k - 1] for k in checkpoints},
                 best_n_trees=int(np.argmax(s_te) + 1), best_auc=max(s_te))
        if lr == 0.05:
            s_tr = [roc_auc_score(y[tr], p) for p in m.staged_decision_function(X.iloc[tr])]
            r["train_curve"] = s_tr[::5] + [s_tr[-1]]
            r["train_at"] = {k: s_tr[k - 1] for k in checkpoints}
        boost[str(lr)] = r
    out["gbm_learning_rate"] = boost
    out["gbm_depth"] = {dp: roc_auc_score(y[te], GBM(300, dp, 0.05).fit(X.iloc[tr], y[tr])
                                          .predict_proba(X.iloc[te])[:, 1])
                        for dp in [1, 2, 3, 5]}
    print("Boosting done", flush=True)

    # ---- province encoding -----------------------------------------------
    # integer codes put the provinces in an order that means nothing; if the
    # trees leaned on that order the two codings would differ
    Xi = tree_X(d, "integer")
    out["encoding"] = {"CART": dict(onehot=cv_auc(CART)[0], integer=cv_auc(CART, Xi)[0]),
                       "RF": dict(onehot=cv_auc(RF)[0], integer=cv_auc(RF, Xi)[0])}

    path = save_json(out, "trees_tuning")
    print(f"\nCART grid optimum: depth {best['max_depth']}, min split "
          f"{best['min_split']}, min leaf {best['min_leaf']}, AUC {best['auc']:.3f}")
    print("pruning: " + ", ".join(f"alpha {r['alpha']:g} -> {r['leaves']:.0f} leaves, "
                                   f"AUC {r['auc']:.3f}" for r in prune))
    rb = max(rf_grid, key=lambda r: r["auc"])
    print(f"forest grid optimum: max_features {rb['max_features']}, min leaf "
          f"{rb['min_leaf']}, AUC {rb['auc']:.3f}")
    print("boosting best: " + ", ".join(f"lr {k}: {v['best_auc']:.3f} at {v['best_n_trees']} trees"
                                        for k, v in boost.items()))
    print(f"\nWritten to {path}")


if __name__ == "__main__":
    main()
