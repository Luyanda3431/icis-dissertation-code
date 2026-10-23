"""
Pieces shared by the numbered scripts.

The file paths, the design matrices, the mixed-effects model, the
fixed-effects check, the three tree models and the train-test splits are all
kept here, so each numbered script only has to say what it is estimating.

Outputs go to outputs/. Set ICIS_OUTPUT_DIR to write them somewhere else, for
example when running on the real data, so nothing from the real data ends up
inside the repository.
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize_scalar
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get("ICIS_OUTPUT_DIR", ROOT / "outputs"))
FIG = OUT / "figures"
SEED = 123
HIGH = ["Gold", "Platinum"]


CATEGORIES = ["Not achieved", "Silver", "Gold", "Platinum"]
CATEGORY_COLOUR = {"Not achieved": "#d9534f", "Silver": "#6fa8dc",
                   "Gold": "#f0ad4e", "Platinum": "#5cb85c"}
PROVINCE_COLOUR = dict(zip(
    ["Eastern Cape", "Free State", "Gauteng", "KwaZulu-Natal", "Limpopo",
     "Mpumalanga", "North West", "Northern Cape", "Western Cape"],
    ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
     "#e377c2", "#7f7f7f", "#bcbd22"]))
BAS_EXCLUDED = ["Free State", "North West", "Western Cape"]


def plot_style():
    """Same look for every figure in the dissertation."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 12, "axes.titlesize": 13,
        "axes.titleweight": "bold", "axes.labelsize": 12, "xtick.labelsize": 11,
        "ytick.labelsize": 11, "legend.fontsize": 10.5, "axes.spines.top": False,
        "axes.spines.right": False, "savefig.dpi": 300})
    return plt


def rho(x, y):
    """Spearman correlation, ignoring missing values."""
    return float(stats.spearmanr(x, y, nan_policy="omit")[0])


# --------------------------------------------------------------------------
# reading and writing
# --------------------------------------------------------------------------

def load_trees_panel():
    """ICIS-ANC panel after the exclusions (Section 3.3.4). Used by the tree
    models and, with zero-ANC rows dropped, by Objective 1."""
    d = pd.read_csv(OUT / "panel_anc.csv")
    return d.reset_index(drop=True)


def load_finance_panel():
    """Finance sub-panel used for Objective 2 (Section 3.3.4)."""
    return pd.read_csv(OUT / "panel_finance.csv").reset_index(drop=True)


def load_eda_panels():
    """Both panels as loaded, before the model exclusions. Chapter 4 and the
    appendices describe these."""
    a = pd.read_csv(OUT / "eda_anc.csv")
    a["r"] = a["anc_first_visit_per_1000_"]
    f = pd.read_csv(OUT / "eda_finance.csv")
    f["province"] = f.province.replace({"Kwazulu-Natal": "KwaZulu-Natal"})
    return a, f


def save_json(obj, name):
    def conv(o):
        if isinstance(o, dict):
            return {str(k): conv(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [conv(v) for v in o]
        if isinstance(o, np.ndarray):
            return conv(o.tolist())
        if isinstance(o, (np.floating, np.integer)):
            return float(o)
        return o
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.json"
    path.write_text(json.dumps(conv(obj), indent=1))
    return path


def read_json(name):
    return json.loads((OUT / f"{name}.json").read_text())


# --------------------------------------------------------------------------
# mixed-effects model
# --------------------------------------------------------------------------

def design(d, terms):
    """Design matrix with an intercept. 'urban', 'year' and 'province' become
    indicator columns, with Urban, 2017 and Gauteng as the reference levels.
    Anything else is used as a numeric column."""
    parts = [pd.Series(1.0, index=d.index, name="const")]
    for t in terms:
        if t == "urban":
            parts.append(pd.get_dummies(d.urban_rural, prefix="ur")
                         .drop(columns="ur_Urban").astype(float))
        elif t == "year":
            parts.append(pd.get_dummies(d.year, prefix="y")
                         .drop(columns="y_2017").astype(float))
        elif t == "province":
            parts.append(pd.get_dummies(d.province, prefix="p")
                         .drop(columns="p_Gauteng").astype(float))
        else:
            parts.append(d[t].astype(float))
    return pd.concat(parts, axis=1)


def fit_lmm(y, X, groups, reml=False):
    """Random-intercept linear mixed model, y_it = x_it b + u_i + e_it.

    Fitted by maximum likelihood, or REML if asked. The only thing that has
    to be searched for is the ratio lam = sigma_u^2 / sigma^2. Once lam is
    fixed, the model is ordinary least squares on partly demeaned data,
    y_it - theta_i * ybar_i with theta_i = 1 - sqrt(1 / (1 + n_i * lam)), so
    b and sigma^2 follow directly. This is the same model lme4::lmer and
    statsmodels MixedLM fit. Writing it out keeps the estimation visible and
    needs nothing beyond numpy and scipy.

    Returns the coefficients and standard errors, the two variance
    components, the ICC, the AIC, the predicted facility intercepts (BLUPs)
    and the within-facility residuals y - xb - u_i.
    """
    y = np.asarray(y, float)
    X = np.asarray(X, float)
    _, inv = np.unique(np.asarray(groups), return_inverse=True)
    ni = np.bincount(inv)
    N, p = X.shape
    ybar = np.bincount(inv, y) / ni
    Xbar = np.vstack([np.bincount(inv, X[:, j]) for j in range(p)]).T / ni[:, None]

    def gls(lam):
        th = 1 - np.sqrt(1 / (1 + ni * lam))
        ys = y - th[inv] * ybar[inv]
        Xs = X - th[inv][:, None] * Xbar[inv]
        XtX = Xs.T @ Xs
        b = np.linalg.solve(XtX, Xs.T @ ys)
        r = ys - Xs @ b
        return b, r @ r, XtX

    def negloglik(loglam):
        lam = np.exp(loglam)
        _, rss, XtX = gls(lam)
        logdet_v = np.sum(np.log1p(ni * lam))
        if reml:
            s2 = rss / (N - p)
            ll = -0.5 * ((N - p) * np.log(2 * np.pi * s2) + logdet_v + (N - p)
                         + np.linalg.slogdet(XtX / s2)[1])
        else:
            s2 = rss / N
            ll = -0.5 * (N * np.log(2 * np.pi * s2) + logdet_v + N)
        return -ll

    opt = minimize_scalar(negloglik, bounds=(-10, 10), method="bounded",
                          options={"xatol": 1e-9})
    lam = np.exp(opt.x)
    b, rss, XtX = gls(lam)
    s2 = rss / ((N - p) if reml else N)
    su2 = lam * s2
    se = np.sqrt(np.diag(s2 * np.linalg.inv(XtX)))
    blup = (ni * lam / (1 + ni * lam)) * (ybar - Xbar @ b)
    fitted = X @ b + blup[inv]
    return dict(b=b, se=se, s2=s2, su2=su2, icc=su2 / (su2 + s2),
                loglik=-opt.fun, aic=2 * opt.fun + 2 * (p + 2),
                blup=blup, resid=y - fitted, fitted=fitted,
                N=N, G=len(ni), ni=ni)


def coef(m, X, name):
    """Estimate, SE, 95% CI and two-sided p for one coefficient."""
    j = list(X.columns).index(name)
    b, se = m["b"][j], m["se"][j]
    return dict(b=b, se=se, lo=b - 1.96 * se, hi=b + 1.96 * se,
                p=2 * stats.norm.sf(abs(b / se)))


def within_estimator(y, X, groups):
    """Fixed-effects (within-facility) estimate for the first column of X.

    Each variable has its facility mean taken off, so only changes within a
    clinic are used. Anything that never changes for a clinic, such as
    urbanicity, drops out and should be left out of X. The cluster-robust
    standard error (clustered on facility) does not assume normal or
    constant-variance residuals, and uses the usual small-sample factor
    G/(G-1) * (N-1)/(N-K).
    """
    g = np.asarray(groups)
    d = pd.DataFrame(np.asarray(X, float))
    d["_y"] = np.asarray(y, float)
    W = d - d.groupby(g).transform("mean")
    Xm = W.drop(columns="_y").values
    ym = W["_y"].values
    Xm = Xm[:, Xm.std(0) > 1e-12]
    b = np.linalg.lstsq(Xm, ym, rcond=None)[0]
    e = ym - Xm @ b
    A = np.linalg.inv(Xm.T @ Xm)
    meat = np.zeros((Xm.shape[1],) * 2)
    for ix in pd.Series(np.arange(len(g))).groupby(g).indices.values():
        s = Xm[ix].T @ e[ix]
        meat += np.outer(s, s)
    G, N, K = len(np.unique(g)), len(ym), Xm.shape[1]
    V_rob = G / (G - 1) * (N - 1) / (N - K) * A @ meat @ A
    V_conv = e @ e / (N - G - K) * A
    se_r = np.sqrt(V_rob[0, 0])
    return dict(b=b[0], se_robust=se_r, se_conv=np.sqrt(V_conv[0, 0]),
                lo=b[0] - 1.96 * se_r, hi=b[0] + 1.96 * se_r,
                p=2 * stats.norm.sf(abs(b[0] / se_r)))


def describe(x, seed=1):
    """Mean, SD, skew, excess kurtosis, quartiles and a Shapiro-Wilk W. The
    Shapiro-Wilk test is run on a random 5,000 rows because it is not
    reliable on samples larger than that."""
    x = np.asarray(x)
    sub = np.random.default_rng(seed).choice(x, min(5000, len(x)), replace=False)
    return dict(mean=x.mean(), sd=x.std(ddof=1), skew=stats.skew(x),
                excess_kurtosis=stats.kurtosis(x), shapiro_W=stats.shapiro(sub)[0],
                q25=np.quantile(x, .25), q75=np.quantile(x, .75))


def mde(se, alpha=0.05, power=0.8):
    """Smallest effect the design could detect: (z_{1-a/2} + z_power) * SE,
    which is about 2.80 SE at the 5% level with 80% power."""
    return (stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)) * se


# --------------------------------------------------------------------------
# tree models
# --------------------------------------------------------------------------

def tree_X(d, encoding="onehot", extra=()):
    """Predictors for the tree models (Section 3.6.3): ANC rate, log
    headcount, year, urbanicity and province, plus any extra columns.
    Province is one-hot coded by default. The integer coding is only used to
    show it makes almost no difference."""
    base = d[["anc_rate", "log_headcount", "year"] + list(extra)].astype(float).copy()
    if encoding == "onehot":
        ur = pd.get_dummies(d.urban_rural, prefix="ur").drop(columns="ur_Urban")
        pr = pd.get_dummies(d.province, prefix="p").drop(columns="p_Gauteng")
        return pd.concat([base, ur, pr], axis=1).astype(float)
    base["urban_enc"] = pd.Categorical(d.urban_rural).codes
    base["province_enc"] = pd.Categorical(d.province).codes
    return base


# the one-hot columns for a variable are treated as one predictor, so that
# province counts once rather than eight times
_GROUP_RULES = {
    "Province": lambda c: c.startswith("p_") or c == "province_enc",
    "Urbanicity": lambda c: c.startswith("ur_") or c == "urban_enc",
    "ANC rate": lambda c: c == "anc_rate",
    "Log headcount": lambda c: c == "log_headcount",
    "Year": lambda c: c == "year",
    "Log expenditure": lambda c: c == "log_exp",
}


def predictor_groups(columns):
    return {k: [c for c in columns if f(c)]
            for k, f in _GROUP_RULES.items() if any(f(c) for c in columns)}


# The settings below are the ones chosen in 07_trees_tuning.py (Table 6).

def CART(**kw):
    p = dict(max_depth=10, min_samples_split=100, min_samples_leaf=25,
             random_state=SEED)
    p.update(kw)
    return DecisionTreeClassifier(**p)


def RF(n=300, max_features="sqrt", min_samples_leaf=20, **kw):
    return RandomForestClassifier(n_estimators=n, max_features=max_features,
                                  min_samples_leaf=min_samples_leaf,
                                  random_state=SEED, n_jobs=-1, **kw)


def GBM(n=300, depth=3, lr=0.05):
    return GradientBoostingClassifier(n_estimators=n, max_depth=depth,
                                      learning_rate=lr, random_state=SEED)


MODELS = {"CART": CART, "RF": RF, "GBM": GBM}


def metrics(y, prob, threshold=0.5):
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(accuracy=accuracy_score(y, pred), auc=roc_auc_score(y, prob),
                sensitivity=tp / (tp + fn), specificity=tn / (tn + fp))


def splits(d, X, y, g):
    """The three train-test designs in Section 3.6.3.

    random   : 70/30, ignores which clinic a row belongs to (only used to
               show the leakage)
    group    : 70/30 by clinic, each clinic wholly in training or testing
    temporal : train on 2017-2021, test on 2022
    """
    idx = np.arange(len(y))
    r_tr, r_te = train_test_split(idx, test_size=.3, stratify=y, random_state=SEED)
    g_tr, g_te = next(GroupShuffleSplit(n_splits=1, test_size=.3,
                                        random_state=SEED).split(X, y, g))
    yr = d.year.values
    return {"random": (r_tr, r_te), "group": (g_tr, g_te),
            "temporal": (idx[yr < 2022], idx[yr == 2022])}
