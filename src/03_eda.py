"""
Exploratory data analysis (Chapter 4).

I did the exploratory work in R first. This is the same analysis in Python so
that the whole dissertation runs from one set of scripts. It uses the panels
as loaded, before the model exclusions, which is what Chapter 4 describes.

Every number quoted in Chapter 4 is printed here under its section heading,
and Tables 1 and 2 and Figures 1 to 14 are written to outputs/.

Usage:
    python src/03_eda.py
Writes:
    outputs/eda.json
    outputs/table1_descriptives.csv, outputs/table2_facilities_by_year.csv
    outputs/figures/fig01_*.png to fig14_*.png
"""

import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, MaxNLocator
from scipy import stats

from common import (CATEGORIES, CATEGORY_COLOUR, FIG, OUT, PROVINCE_COLOUR,
                    load_eda_panels, plot_style, rho, save_json)

plt = plot_style()
YEARS = list(range(2017, 2023))
PROVS = list(PROVINCE_COLOUR)
SHORT = {"Eastern Cape": "EC", "Free State": "FS", "Gauteng": "GP",
         "KwaZulu-Natal": "KZN", "Limpopo": "LP", "Mpumalanga": "MP",
         "North West": "NW", "Northern Cape": "NC", "Western Cape": "WC"}
thousands = FuncFormatter(lambda v, p: f"{v:,.0f}")


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, bbox_inches="tight")
    plt.close(fig)


def describe(x):
    x = x.dropna()
    return dict(n=len(x), mean=x.mean(), sd=x.std(), median=x.median(),
                q1=x.quantile(.25), q3=x.quantile(.75), min=x.min(), max=x.max())


def tables(a, f):
    """Table 1 (descriptive statistics) and Table 2 (facilities by year)."""
    rows = {"ICIS score (%)": a.score,
            "ANC per 1,000 fertile women": a.r,
            "ANC first visits (count)": a.anc_first_visits,
            "Total headcount (contacts)": a.total_headcount,
            # NNV was only assessed from 2020
            "NNV score (2020-2022 only)": a[a.year >= 2020].non_negotiable_vital,
            "Expenditure per visit (R)": f.Exp_per_visit,
            "Expenditure per 1,000 population (R)": f.exp_per_1000_pop,
            "Total expenditure (R thousands)": f.expenditure / 1000}
    t1 = pd.DataFrame({k: describe(v) for k, v in rows.items()}).T
    t1.to_csv(OUT / "table1_descriptives.csv")
    t2 = pd.crosstab(a.province, a.year, margins=True, margins_name="Total")
    t2.to_csv(OUT / "table2_facilities_by_year.csv")
    missing = a[["score", "r", "anc_first_visits", "total_headcount", "province",
                 "urban_rural", "ownership"]].isna().sum().sum()
    return t1, t2, int(missing)


def score_distributions(a, out):
    """Section 4.3: national and provincial score distributions (Figures 1, 2)."""
    med = a.groupby("year").score.median()
    out["score_median_by_year"] = med.to_dict()
    out["share_below_50_by_year"] = (a.score < 50).groupby(a.year).mean().to_dict()

    fig, ax = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    for axi, y in zip(ax.flat, YEARS):
        s = a[a.year == y].score
        axi.hist(s, bins=np.arange(10, 102, 2), color="#4a90b8", alpha=.85, edgecolor="white")
        axi.axvline(s.median(), color="black", ls="--", lw=1.5, label=f"Median {s.median():.1f}%")
        axi.axvline(70, color="grey", ls=":", lw=1.5, label="70% threshold")
        axi.set(title=f"{y} (n = {len(s):,})", xlabel="ICIS score (%)" if y > 2019 else "")
        axi.legend(frameon=False, fontsize=9, loc="upper left")
    fig.suptitle("ICIS Score Distribution by Year (2017–2022)", fontweight="bold", fontsize=14)
    save(fig, "fig01_score_by_year.png")

    g = a.groupby("province").score
    out["province_score"] = pd.DataFrame({"median": g.median(), "sd": g.std(),
                                          "mean": g.mean(), "n": g.size()}).to_dict("index")
    out["share_below_50"] = dict(national=(a.score < 50).mean(),
                                 by_province=(a.score < 50).groupby(a.province).mean().to_dict())
    fig, ax = plt.subplots(3, 3, figsize=(15, 11), sharex=True)
    for axi, p in zip(ax.flat, PROVS):
        s = a[a.province == p].score
        axi.hist(s, bins=np.arange(10, 102, 3), density=True, color=PROVINCE_COLOUR[p],
                 alpha=.8, edgecolor="white")
        axi.axvline(s.median(), color="black", ls="--", lw=1.5, label=f"Median {s.median():.0f}%")
        axi.axvline(70, color="grey", ls=":", lw=1.5, label="70% threshold")
        axi.set(title=f"{p}\nn = {len(s):,}, SD = {s.std():.1f}pp", ylabel="Density")
        axi.legend(frameon=False, fontsize=9, loc="upper left")
    for axi in ax[2]:
        axi.set_xlabel("ICIS score (%)")
    fig.suptitle("ICIS Score Distribution by Province (2017–2022 pooled)", fontweight="bold", fontsize=14)
    save(fig, "fig02_score_by_province.png")


def categories_and_nnv(a, out):
    """Sections 4.3.3 and 4.3.4: category shares and the NNV rule (Figures 3, 4)."""
    ct = pd.crosstab(a.year, a.category).reindex(columns=CATEGORIES, fill_value=0)
    out["category_counts"] = ct.to_dict("index")
    share = ct.div(ct.sum(axis=1), axis=0) * 100
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.6))
    ax[0].stackplot(YEARS, [share[c] for c in CATEGORIES], labels=CATEGORIES,
                    colors=[CATEGORY_COLOUR[c] for c in CATEGORIES], alpha=.9)
    ax[0].set(xlim=(2017, 2022), ylim=(0, 100), ylabel="Share of assessed clinics (%)",
              xlabel="Assessment year", title="Category share")
    w = .2
    for i, c in enumerate(CATEGORIES):
        ax[1].bar(np.array(YEARS) + (i - 1.5) * w, ct[c], w, color=CATEGORY_COLOUR[c], label=c)
    ax[1].axvspan(2019.5, 2020.5, color="#f4cccc", alpha=.5, zorder=0, label="COVID year (2020)")
    h, l = ax[1].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(.5, -.06))
    ax[1].set(ylabel="Number of clinics", xlabel="Assessment year", xticks=YEARS,
              title="Number of clinics by category")
    ax[1].yaxis.set_major_formatter(thousands)
    fig.suptitle("National ICIS Category Distribution Over Time (2017–2022)", fontweight="bold", fontsize=14)
    fig.tight_layout(rect=(0, .04, 1, 1))
    fig.savefig(FIG / "fig03_categories.png", bbox_inches="tight")
    plt.close(fig)

    n = a[a.year >= 2020]
    full = (n.non_negotiable_vital == 100).groupby(n.year).mean()
    n22 = n[n.year == 2022]
    out["nnv"] = dict(full_compliance_by_year=full.to_dict(),
                      full_compliance_by_category_2022=(n22.non_negotiable_vital == 100)
                      .groupby(n22.category).mean().to_dict())
    # Before 2020 the category follows the composite score. From 2020 it
    # follows the component thresholds, so the lowest Platinum score drops.
    out["lowest_platinum_score"] = a[a.category == "Platinum"].groupby("year").score.min().to_dict()
    # Where did the 2019 Gold and Silver clinics end up in 2022?
    c19 = a[(a.year == 2019) & a.category.isin(["Gold", "Silver"])].facility_id
    m22 = n22[n22.facility_id.isin(c19)]
    na22 = m22[m22.category == "Not achieved"]
    out["gold_silver_2019_in_2022"] = dict(
        n=len(m22), platinum=int((m22.category == "Platinum").sum()),
        not_achieved=len(na22), not_achieved_failing_nnv=int((na22.non_negotiable_vital < 100).sum()))
    na_all = n22[n22.category == "Not achieved"].score
    out["not_achieved_2022_score"] = dict(median=na_all.median(), max=na_all.max())

    cols = ["#e07b54", "#6aaa6a", "#4a90b8"]
    fig, ax = plt.subplots(1, 3, figsize=(15, 5.6), gridspec_kw={"width_ratios": [1, 1.1, 1.2]})
    bars = ax[0].bar([str(y) for y in full.index], full.values * 100, color=cols, width=.65)
    for b, v in zip(bars, full.values * 100):
        ax[0].text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.1f}%", ha="center", fontweight="bold")
    ax[0].set(ylim=(0, 100), ylabel="% of facilities", xlabel="Assessment year",
              title="Facilities with full NNV\ncompliance (100%)")
    data = [n[n.year == y].non_negotiable_vital.values for y in (2020, 2021, 2022)]
    vp = ax[1].violinplot(data, showmedians=True, widths=.8)
    for body, c in zip(vp["bodies"], cols):
        body.set_facecolor(c)
        body.set_alpha(.6)
    ax[1].set(xticks=[1, 2, 3], xticklabels=["2020", "2021", "2022"], ylabel="NNV score (%)",
              xlabel="Assessment year", ylim=(-3, 108), title="NNV score distribution")
    grp = pd.cut(n22.non_negotiable_vital, [-1, 0, 99.99, 100], labels=["NNV = 0", "NNV 1–99%", "NNV = 100%"])
    tb = pd.crosstab(n22.category, grp, normalize="index").reindex(CATEGORIES).fillna(0) * 100
    bottom = np.zeros(4)
    for c, colour in zip(tb.columns, ["#d9534f", "#f5b460", "#5cb85c"]):
        ax[2].bar(range(4), tb[c], bottom=bottom, color=colour, label=c, width=.65)
        bottom += tb[c].values
    ax[2].set(xticks=range(4), xticklabels=["Not\nachieved", "Silver", "Gold", "Platinum"],
              ylim=(0, 110), ylabel="% of facilities", title="NNV compliance by category, 2022")
    ax[2].legend(loc="lower center", bbox_to_anchor=(.5, -.42), ncol=3, frameon=False)
    fig.suptitle("Non-Negotiable Vital (NNV) Compliance, 2020–2022", fontweight="bold", fontsize=14)
    save(fig, "fig04_nnv.png")


def stability_and_covid(a, out):
    """Section 4.4: score volatility and the 2020 dip (Figures 5, 6)."""
    # Volatility is the SD of each clinic's own scores, for clinics seen in at
    # least four years so the SD means something
    g = a.groupby("facility_id").agg(sd=("score", "std"), k=("score", "size"),
                                     p=("province", "first"), u=("urban_rural", "first"))
    g = g[g.k >= 4]
    out["volatility"] = dict(n=len(g), median=g.sd.median(),
                             by_province=g.groupby("p").sd.median().to_dict(),
                             by_urbanicity=g.groupby("u").sd.median().to_dict())
    order = g.groupby("p").sd.median().sort_values(ascending=False).index
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.8), gridspec_kw={"width_ratios": [1.6, 1]})
    bp = ax[0].boxplot([g[g.p == p].sd for p in order], patch_artist=True, widths=.6,
                       medianprops=dict(color="black", lw=2), flierprops=dict(ms=2, alpha=.4))
    for box, p in zip(bp["boxes"], order):
        box.set_facecolor(PROVINCE_COLOUR[p])
        box.set_alpha(.8)
    ax[0].axhline(g.sd.median(), ls="--", color="grey", label=f"National median SD = {g.sd.median():.1f}pp")
    ax[0].legend(frameon=False, loc="upper right")
    ax[0].set(xticklabels=[SHORT[p] for p in order], ylabel="Within-facility score SD (pp)",
              title="By province (sorted by median)")
    urb = ["Rural", "Peri-Urban", "Urban"]
    bp = ax[1].boxplot([g[g.u == u].sd for u in urb], patch_artist=True, widths=.55,
                       medianprops=dict(color="black", lw=2), flierprops=dict(ms=2, alpha=.4))
    for box, c in zip(bp["boxes"], ["#a9744f", "#e3bb4f", "#4f7fa8"]):
        box.set_facecolor(c)
        box.set_alpha(.8)
    ax[1].set(xticklabels=[f"{u}\n(n = {(g.u == u).sum():,})" for u in urb],
              ylabel="Within-facility score SD (pp)", title="By urbanicity", ylim=ax[0].get_ylim())
    fig.suptitle("Within-Facility ICIS Score Volatility (2017–2022)", fontweight="bold", fontsize=14)
    save(fig, "fig05_volatility.png")

    # Year-on-year changes for clinics seen in consecutive years
    s = a.sort_values(["facility_id", "year"]).copy()
    by = s.groupby("facility_id")
    s["prev_score"], s["prev_year"], s["prev_cat"] = by.score.shift(), by.year.shift(), by.category.shift()
    c = s[s.year - s.prev_year == 1].copy()
    c["d"] = c.score - c.prev_score
    small = c[c.d.abs() < 5]
    out["year_on_year"] = dict(
        share_declined=(c.d < 0).groupby(c.year).mean().to_dict(),
        share_improved=(c.d > 0).groupby(c.year).mean().to_dict(),
        small_change_crosses_category=(small.category != small.prev_cat).mean(),
        small_change_crosses_category_before_2020=(
            (small.category != small.prev_cat)[small.year <= 2019]).mean())

    mean_score = a.groupby("year").score.mean()
    anc_q = a.groupby("year").r.quantile([.25, .5, .75]).unstack()
    out["national_mean_score_by_year"] = mean_score.to_dict()
    out["national_median_anc_by_year"] = anc_q[.5].to_dict()
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.4))
    ax[0].plot(YEARS, mean_score, "o-", color="#2c6fa8", lw=2.4)
    for y, v in mean_score.items():
        ax[0].text(y, v + .4, f"{v:.1f}", ha="center", fontsize=10)
    ax[0].set(xlabel="Assessment year", ylabel="Mean ICIS score (%)", title="National mean ICIS score")
    ax[1].fill_between(YEARS, anc_q[.25], anc_q[.75], color="#c0392b", alpha=.15, label="Interquartile range")
    ax[1].plot(YEARS, anc_q[.5], "o-", color="#c0392b", lw=2.4, label="Median")
    for y, v in anc_q[.5].items():
        ax[1].text(y, v + .03, f"{v:.2f}", ha="center", fontsize=10)
    ax[1].set(xlabel="Assessment year", ylabel="ANC per 1,000 fertile women",
              title="National ANC first-visit rate")
    ax[1].legend(frameon=False)
    for axi in ax:
        axi.axvspan(2019.5, 2020.5, color="#f4cccc", alpha=.5, zorder=0)
    fig.suptitle("ICIS Scores and ANC First-Visit Rates Around COVID-19 (2017–2022)",
                 fontweight="bold", fontsize=14)
    save(fig, "fig06_covid.png")


def anc_patterns(a, out):
    """Section 4.5: ANC rate by province and the very high rates (Figure 7)."""
    out["province_anc_median"] = a.groupby("province").r.median().to_dict()
    hi = a[a.r > 3]
    out["anc_above_3"] = dict(n=len(hi), share=len(hi) / len(a), above_10=int((a.r > 10).sum()),
                              by_province=hi.province.value_counts().to_dict(),
                              urban_share=(hi.urban_rural == "Urban").mean())
    fig, ax = plt.subplots(3, 3, figsize=(15, 11), sharex=True)
    for axi, p in zip(ax.flat, PROVS):
        s = a[a.province == p].r
        axi.hist(s.clip(upper=6), bins=np.arange(0, 6.1, .15), density=True,
                 color=PROVINCE_COLOUR[p], alpha=.8, edgecolor="white")
        axi.axvline(s.median(), color="black", ls="--", lw=1.5, label=f"Median {s.median():.2f}")
        axi.set(title=f"{p} (n = {len(s):,})", ylabel="Density")
        axi.legend(frameon=False, fontsize=9)
    for axi in ax[2]:
        axi.set_xlabel("ANC per 1,000 fertile women (values above 6 shown at 6)")
    fig.suptitle("ANC First-Visit Rate Distribution by Province (2017–2022 pooled)",
                 fontweight="bold", fontsize=14)
    save(fig, "fig07_anc_by_province.png")


def icis_anc(a, out):
    """Section 4.6: how score and ANC rate relate (Figures 8 to 11)."""
    res = {}
    # national score deciles, so "lowest decile" means the same score range
    # in every province
    decile = pd.qcut(a.score, 10, labels=False)
    fig, ax = plt.subplots(3, 3, figsize=(15, 12))
    for axi, p in zip(ax.flat, PROVS):
        s = a[a.province == p]
        r_, pv = stats.spearmanr(s.score, s.r)
        dec_med = s.r.groupby(decile[s.index]).median()
        res[p] = dict(rho=r_, p=pv, n=len(s), lowest_decile_median=dec_med.iloc[0],
                      highest_decile_median=dec_med.iloc[-1])
        axi.scatter(s.score, s.r, s=4, alpha=.35, color=PROVINCE_COLOUR[p], rasterized=True)
        b = np.polyfit(s.score, s.r, 1)
        xs = np.linspace(s.score.min(), s.score.max(), 50)
        axi.plot(xs, np.polyval(b, xs), color="black", lw=1.5)
        bins = pd.cut(s.score, 15)
        bm = s.groupby(bins, observed=True).agg(x=("score", "median"), y=("r", "median"), n=("r", "size"))
        bm = bm[bm.n >= 20]
        axi.plot(bm.x, bm.y, ls="--", color="grey", lw=1.5)
        axi.set(title=p, ylim=(0, 8), xlabel="ICIS score (%)", ylabel="ANC per 1,000")
        axi.text(.03, .92, f"ρ = {r_:.2f}, n = {len(s):,}", transform=axi.transAxes, fontsize=10)
    fig.suptitle("ICIS Score versus ANC First-Visit Rate by Province (2017–2022)\n"
                 "black = linear fit, grey dashed = binned median", fontweight="bold", fontsize=14)
    save(fig, "fig08_score_vs_anc_by_province.png")
    out["province_score_anc"] = res

    # North West: is the positive link within years, and within facilities?
    nw = a[a.province == "North West"]
    out["north_west"] = dict(
        rho_by_year={y: rho(g.score, g.r) for y, g in nw.groupby("year")},
        median_score_2017=nw[nw.year == 2017].score.median(),
        median_anc_2017=nw[nw.year == 2017].r.median(),
        median_anc_2022=nw[nw.year == 2022].r.median())

    # Ownership
    own = {}
    for o, g in a[a.ownership.isin(["Gov Province", "Gov Municipality"])].groupby("ownership"):
        own[o] = dict(n=len(g), median_score=g.score.median(),
                      platinum_share=(g.category == "Platinum").mean(),
                      median_anc=g.r.median(), urban_share=(g.urban_rural == "Urban").mean())
    out["ownership"] = own
    po = a[a.ownership.isin(["Gov Province", "Gov Municipality"])]
    lab = {"Gov Province": "Provincial", "Gov Municipality": "Municipal"}
    fig, ax = plt.subplots(1, 3, figsize=(16, 5.4))
    for i, (col, ttl, ylab) in enumerate([("score", "ICIS score", "ICIS score (%)"),
                                          ("r", "ANC first-visit rate", "ANC per 1,000 fertile women")]):
        data = [po[po.ownership == o][col] for o in lab]
        bp = ax[i].boxplot(data, patch_artist=True, widths=.55, showfliers=False,
                           medianprops=dict(color="black", lw=2))
        for box, c in zip(bp["boxes"], ["#7fb3d5", "#f0ad4e"]):
            box.set_facecolor(c)
        ax[i].set(xticklabels=[f"{lab[o]}\n(median {po[po.ownership == o][col].median():.2f})"
                               if col == "r" else f"{lab[o]}\n(median {po[po.ownership == o][col].median():.0f}%)"
                               for o in lab], ylabel=ylab, title=ttl)
    tb = pd.crosstab(po.ownership, po.category, normalize="index").reindex(
        index=list(lab), columns=CATEGORIES).fillna(0) * 100
    bottom = np.zeros(2)
    for c in CATEGORIES:
        ax[2].bar([lab[o] for o in tb.index], tb[c], bottom=bottom, color=CATEGORY_COLOUR[c], label=c)
        bottom += tb[c].values
    ax[2].set(ylabel="% of facility-years", title="Category composition")
    ax[2].legend(frameon=False, fontsize=9.5, loc="upper center", bbox_to_anchor=(.5, -.1), ncol=4)
    fig.suptitle("Municipal Facilities Score Higher but Serve Fewer ANC Patients",
                 fontweight="bold", fontsize=14)
    save(fig, "fig09_ownership.png")

    # Score quintiles
    q = pd.qcut(a.score, 5, labels=False)
    qs = a.groupby(q).agg(low=("score", "min"), high=("score", "max"), median_anc=("r", "median"),
                          q1=("r", lambda x: x.quantile(.25)), q3=("r", lambda x: x.quantile(.75)),
                          n=("r", "size"))
    out["quintiles"] = qs.to_dict("index")
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.4))
    labels = [f"Q{i + 1}\n{r.low:.0f}–{r.high:.0f}%" for i, r in qs.iterrows()]
    ax[0].errorbar(range(5), qs.median_anc, yerr=[qs.median_anc - qs.q1, qs.q3 - qs.median_anc],
                   fmt="o", color="#2c6fa8", capsize=6, lw=2, ms=8)
    for i, v in enumerate(qs.median_anc):
        ax[0].text(i + .1, v, f"{v:.3f}", va="center", fontsize=10)
    ax[0].set(xticks=range(5), xticklabels=labels, ylabel="ANC per 1,000 fertile women",
              title="Median and interquartile range")
    ax[1].boxplot([a.r[q == i].clip(upper=5) for i in range(5)], showfliers=False, widths=.55,
                  medianprops=dict(color="black", lw=2))
    ax[1].set(xticklabels=labels, ylabel="ANC per 1,000 fertile women", title="Distribution by quintile")
    fig.suptitle("ANC First-Visit Rates Across ICIS Score Quintiles", fontweight="bold", fontsize=14)
    save(fig, "fig10_quintiles.png")

    # Change in each facility from 2017 to 2022
    pv = a.pivot_table(index="facility_id", columns="year", values=["score", "r", "total_headcount"])
    prov = a.groupby("facility_id").province.first()
    ch = pd.DataFrame({"ds": pv["score"][2022] - pv["score"][2017],
                       "dr": pv["r"][2022] - pv["r"][2017],
                       "dh": pv["total_headcount"][2022] - pv["total_headcount"][2017],
                       "province": prov}).dropna(subset=["ds"])
    both = ch.dropna(subset=["dr"])
    rose, fell = both[both.ds > 0], both[both.ds < 0]
    out["within_facility_change"] = dict(
        rho_anc=rho(both.ds, both.dr), n_anc=len(both),
        rho_headcount=rho(ch.ds, ch.dh), n_headcount=int(ch.dh.notna().sum()),
        score_rose_anc_fell=(rose.dr < 0).mean(), score_rose_anc_rose=(rose.dr > 0).mean(),
        score_fell_anc_fell=(fell.dr < 0).mean(), score_fell_anc_rose=(fell.dr > 0).mean(),
        median_anc_change_gain_20_or_more=both[both.ds >= 20].dr.median(),
        median_anc_change_within_5=both[both.ds.abs() <= 5].dr.median(),
        median_score_gain_by_province=ch.groupby("province").ds.median().to_dict(),
        median_anc_change_by_province=both.groupby("province").dr.median().to_dict())
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.8))
    for p in PROVS:
        s = ch[ch.province == p]
        ax[0].scatter(s.ds, s.dr, s=8, alpha=.5, color=PROVINCE_COLOUR[p], label=SHORT[p])
        ax[1].scatter(s.ds, s.dh, s=8, alpha=.5, color=PROVINCE_COLOUR[p])
    for axi in ax:
        axi.axhline(0, color="grey", lw=1)
        axi.axvline(0, color="grey", lw=1)
    w = out["within_facility_change"]
    ax[0].set(xlabel="Change in ICIS score, 2017 to 2022 (pp)", ylabel="Change in ANC per 1,000",
              title=f"ANC rate (ρ = {w['rho_anc']:.3f}, n = {w['n_anc']:,})",
              ylim=both.dr.quantile([.005, .995]).values)
    ax[1].set(xlabel="Change in ICIS score, 2017 to 2022 (pp)", ylabel="Change in total headcount",
              title=f"Headcount (ρ = {w['rho_headcount']:.3f}, n = {w['n_headcount']:,})")
    ax[1].yaxis.set_major_formatter(thousands)
    ax[0].legend(frameon=False, ncol=3, fontsize=9, markerscale=2)
    fig.suptitle("Within-Facility Changes, 2017 to 2022", fontweight="bold", fontsize=14)
    save(fig, "fig11_within_facility_change.png")


def correlations(a, f, out):
    """Section 4.6.5: Spearman matrix on the finance sub-dataset (Figure 12)."""
    m = f.merge(a[["facility_id", "year", "r"]], on=["facility_id", "year"])
    m["cat"] = m.category.map({c: i for i, c in enumerate(CATEGORIES)})
    cols = ["score", "r", "total_headcount", "cat", "Exp_per_visit"]
    names = ["ICIS score", "ANC per 1,000", "Headcount", "ICIS category\n(numeric)", "Expenditure\nper visit"]
    R = m[cols].corr(method="spearman")
    out["correlation_matrix"] = dict(n=len(m), rho=R.to_dict())
    fig, ax = plt.subplots(figsize=(9, 7.6))
    im = ax.imshow(R.values, cmap="RdBu_r", vmin=-1, vmax=1)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{R.values[i, j]:.3f}", ha="center", va="center", fontsize=11,
                    color="white" if abs(R.values[i, j]) > .55 else "black")
    ax.set(xticks=range(5), yticks=range(5), xticklabels=names, yticklabels=names)
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig.colorbar(im, ax=ax, fraction=.046, pad=.04).set_label("Spearman ρ")
    ax.set_title(f"Spearman Correlation Matrix: Key Study Variables\nFinance sub-dataset (n = {len(m):,})")
    save(fig, "fig12_correlations.png")


def expenditure(f, out):
    """Section 4.7: spending per visit and per 1,000 people (Figures 13, 14)."""
    pm = f.groupby("province").agg(per_visit=("Exp_per_visit", "median"),
                                   per_1000=("exp_per_1000_pop", "median"),
                                   score=("score", "median"), headcount=("total_headcount", "median"))
    py = f.groupby(["province", "year"]).Exp_per_visit.median().unstack()
    out["expenditure"] = dict(
        national_median_per_visit=f.Exp_per_visit.median(),
        by_province=pm.to_dict("index"),
        rho_per_visit_vs_score=rho(pm.per_visit, pm.score),
        rho_per_1000_vs_score=rho(pm.per_1000, pm.score),
        median_per_visit_by_province_year=py.to_dict("index"))

    fig, ax = plt.subplots(figsize=(11, 5.6))
    order = pm.per_visit.sort_values(ascending=False).index
    year_colour = dict(zip([2017, 2018, 2019, 2020, 2022],
                           ["#c6dbef", "#9ecae1", "#6baed6", "#c0392b", "#08519c"]))
    for i, p in enumerate(order):
        for y, v in py.loc[p].dropna().items():
            ax.scatter(i + (y - 2019.5) * .06, v, color=year_colour[y], s=60,
                       label=str(y) if i == 0 or (p == order[1] and y != 2017) else None)
        ax.scatter([i], [pm.per_visit[p]], marker="D", color="black", s=70, zorder=3,
                   label="All years" if i == 0 else None)
    ax.axhline(f.Exp_per_visit.median(), ls="--", color="grey",
               label=f"National median R{f.Exp_per_visit.median():.0f}")
    ax.set(xticks=range(len(order)), xticklabels=order, ylabel="Median expenditure per visit (R)",
           title="Expenditure per Outpatient Visit by Province, 2017–2022\n"
                 "points = yearly medians, diamond = all-year median (no BAS extract for 2021)")
    ax.legend(frameon=False)
    save(fig, "fig13_expenditure_by_province.png")

    fig, ax = plt.subplots(1, 3, figsize=(16, 5.8))
    for k, (col, ttl, raw) in enumerate([("per_visit", "Expenditure per outpatient visit", "Exp_per_visit"),
                                         ("per_1000", "Expenditure per 1,000 population", "exp_per_1000_pop")]):
        q = pm.sort_values(col, ascending=False)
        ax[k].barh(q.index, q[col], color=[PROVINCE_COLOUR[p] for p in q.index])
        ax[k].axvline(f[raw].median(), ls="--", color="grey", label=f"National median R{f[raw].median():,.0f}")
        for i, v in enumerate(q[col]):
            ax[k].text(v + q[col].max() * .01, i, f"R{v:,.0f}", va="center", fontsize=10.5)
        ax[k].set(xlabel="Rand", title=ttl + "\n(median, all years)", xlim=(0, q[col].max() * 1.22))
        ax[k].legend(frameon=False, loc="lower right")
        ax[k].xaxis.set_major_locator(MaxNLocator(4))
        ax[k].xaxis.set_major_formatter(thousands)
    for p, r in pm.iterrows():
        ax[2].scatter(r.per_1000, r.score, s=140, color=PROVINCE_COLOUR[p], zorder=3)
        ax[2].annotate(p, (r.per_1000, r.score), xytext=(6, 6), textcoords="offset points", fontsize=10.5)
    e = out["expenditure"]
    ax[2].set(xlabel="Expenditure per 1,000 population (R)", ylabel="Median ICIS score (%)",
              title=f"Per-capita spending vs ICIS score\n(Spearman ρ = {e['rho_per_1000_vs_score']:.2f}, six provinces)",
              xlim=(0, pm.per_1000.max() * 1.3))
    ax[2].xaxis.set_major_formatter(thousands)
    fig.suptitle("Two Measures of Expenditure Tell Opposite Stories", fontweight="bold", fontsize=14)
    save(fig, "fig14_per_visit_vs_per_capita.png")


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    a, f = load_eda_panels()
    out = dict(n=len(a), facilities=a.facility_id.nunique(),
               finance_n=len(f), finance_provinces=f.province.nunique())
    t1, t2, missing = tables(a, f)
    out["missing_primary"] = missing
    score_distributions(a, out)
    categories_and_nnv(a, out)
    stability_and_covid(a, out)
    anc_patterns(a, out)
    icis_anc(a, out)
    correlations(a, f, out)
    expenditure(f, out)
    path = save_json(out, "eda")

    # print what Chapter 4 quotes, in the order it quotes it
    pd.set_option("display.width", 140)
    print(f"Panel: {out['n']:,} facility-years, {out['facilities']:,} facilities; "
          f"finance {out['finance_n']:,} facility-years, {out['finance_provinces']} provinces; "
          f"missing values on primary variables: {missing}\n")
    print("Table 1\n" + t1.round(2).to_string() + "\n")
    print("Table 2\n" + t2.to_string() + "\n")
    print("4.3.1 median score by year: " + ", ".join(f"{y} {v:.1f}" for y, v in out["score_median_by_year"].items()))
    ps = pd.DataFrame(out["province_score"]).T
    print("4.3.2 province median / SD:\n" + ps[["median", "sd"]].round(1).to_string())
    print(f"      share below 50%: national {out['share_below_50']['national']:.1%}, "
          f"Limpopo {out['share_below_50']['by_province'].get('Limpopo', np.nan):.1%}")
    cc = out["category_counts"]
    print(f"4.3.3 Not achieved 2017 {cc[2017]['Not achieved']:,}, Platinum 2017 {cc[2017]['Platinum']:,}, "
          f"Platinum 2022 {cc[2022]['Platinum']:,}, Not achieved 2022 {cc[2022]['Not achieved']:,}")
    nv = out["nnv"]
    print("4.3.4 full NNV: " + ", ".join(f"{y} {v:.1%}" for y, v in nv["full_compliance_by_year"].items())
          + "; 2022 by category: " + ", ".join(f"{k} {v:.1%}" for k, v in nv["full_compliance_by_category_2022"].items()))
    print("      lowest Platinum score: " + ", ".join(f"{y} {v:.0f}" for y, v in out["lowest_platinum_score"].items()))
    gs = out["gold_silver_2019_in_2022"]
    print(f"      2019 Gold/Silver seen in 2022: {gs['n']:,}, Platinum {gs['platinum']:,} "
          f"({gs['platinum'] / gs['n']:.0%}), Not achieved {gs['not_achieved']} "
          f"({gs['not_achieved'] / gs['n']:.0%}, {gs['not_achieved_failing_nnv']} failed NNV)")
    print(f"      2022 Not achieved score: median {out['not_achieved_2022_score']['median']:.0f}, "
          f"max {out['not_achieved_2022_score']['max']:.0f}")
    v = out["volatility"]
    print(f"4.4.1 volatility: n = {v['n']:,}, median {v['median']:.1f}pp; "
          + ", ".join(f"{SHORT[k]} {x:.1f}" for k, x in sorted(v["by_province"].items(), key=lambda t: -t[1]))
          + "; " + ", ".join(f"{k} {x:.1f}" for k, x in v["by_urbanicity"].items()))
    yy = out["year_on_year"]
    print(f"      changes under 5pp that cross a category: {yy['small_change_crosses_category']:.1%} "
          f"(before 2020 {yy['small_change_crosses_category_before_2020']:.1%})")
    print("4.4.2 mean score: " + ", ".join(f"{y} {s:.1f}" for y, s in out["national_mean_score_by_year"].items()))
    print("      share declined: " + ", ".join(f"{y} {s:.1%}" for y, s in yy["share_declined"].items())
          + f"; improved 2020-21 {yy['share_improved'].get(2021, np.nan):.1%}")
    print("      median ANC: " + ", ".join(f"{y} {s:.2f}" for y, s in out["national_median_anc_by_year"].items()))
    print("4.5   median ANC by province: " + ", ".join(f"{SHORT[k]} {x:.2f}" for k, x in out["province_anc_median"].items()))
    h = out["anc_above_3"]
    print(f"      above 3.0: {h['n']} ({h['share']:.1%}), above 10: {h['above_10']}; "
          + ", ".join(f"{k} {x}" for k, x in list(h["by_province"].items())[:3]))
    print("4.6.1 Spearman by province:")
    for p, r in out["province_score_anc"].items():
        print(f"      {p:14s} rho {r['rho']:+.2f} (p = {r['p']:.2g}), lowest/highest decile median "
              f"{r['lowest_decile_median']:.2f}/{r['highest_decile_median']:.2f}")
    nw = out["north_west"]
    wf = out["within_facility_change"]
    print(f"      North West: 2017 median score {nw['median_score_2017']:.0f}, median gain "
          f"{wf['median_score_gain_by_province']['North West']:+.1f}pp, within-year rho "
          f"{min(nw['rho_by_year'].values()):.3f} to {max(nw['rho_by_year'].values()):.3f}, median ANC change "
          f"{wf['median_anc_change_by_province']['North West']:+.3f}, median ANC {nw['median_anc_2017']:.2f} "
          f"(2017) and {nw['median_anc_2022']:.2f} (2022)")
    print("      median score gain by province: " + ", ".join(
        f"{SHORT[k]} {x:+.1f}" for k, x in sorted(wf["median_score_gain_by_province"].items(), key=lambda t: -t[1])))
    for o, r in out["ownership"].items():
        print(f"4.6.2 {o}: n {r['n']:,}, median score {r['median_score']:.0f}, Platinum {r['platinum_share']:.0%}, "
              f"median ANC {r['median_anc']:.3f}, urban {r['urban_share']:.0%}")
    print("4.6.3 quintiles: " + "; ".join(f"Q{i + 1} {r['low']:.0f}-{r['high']:.0f}% median {r['median_anc']:.3f}"
                                         for i, r in out["quintiles"].items()))
    print(f"4.6.4 within-facility rho {wf['rho_anc']:.3f} (n = {wf['n_anc']:,}); headcount rho "
          f"{wf['rho_headcount']:.3f} (n = {wf['n_headcount']:,}); score rose: ANC fell "
          f"{wf['score_rose_anc_fell']:.0%}, rose {wf['score_rose_anc_rose']:.0%}; score fell: ANC fell "
          f"{wf['score_fell_anc_fell']:.0%}, rose {wf['score_fell_anc_rose']:.0%}")
    R = pd.DataFrame(out["correlation_matrix"]["rho"])
    print("4.6.5 correlations (n = {:,}):\n{}".format(out["correlation_matrix"]["n"], R.round(3).to_string()))
    e = out["expenditure"]
    print(f"4.7   national median per visit R{e['national_median_per_visit']:.0f}; across provinces rho "
          f"per visit vs score {e['rho_per_visit_vs_score']:.2f}, per 1,000 vs score {e['rho_per_1000_vs_score']:.2f}")
    print(pd.DataFrame(e["by_province"]).T.round(0).to_string())
    print(pd.DataFrame(e["median_per_visit_by_province_year"]).T.round(0).to_string())
    print(f"\nWritten to {path}")


if __name__ == "__main__":
    main()
