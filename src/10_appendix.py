"""
Appendix A (diagnostic checks) and Appendix B (extra exploratory analysis).

Like Chapter 4, these use the panels as loaded. Each block below is one
appendix section, and the numbers in the figure captions are printed and
saved to outputs/appendix.json.

Usage:
    python src/10_appendix.py
Writes:
    outputs/appendix.json
    outputs/figures/fig26_*.png to fig47_*.png
"""

import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

from common import (BAS_EXCLUDED, CATEGORIES, CATEGORY_COLOUR, FIG,
                    PROVINCE_COLOUR, load_eda_panels, plot_style, rho, save_json)

plt = plot_style()
BLUE, RED, GREEN = "#2c6fa8", "#c0392b", "#5aa66a"
RANK = {c: i for i, c in enumerate(CATEGORIES)}
thousands = FuncFormatter(lambda v, p: f"{v:,.0f}")
COMPONENTS = {
    "administration": "Administration",
    "integrated_clinical_services_management_icsm": "ICSM",
    "medicines_supplies_and_laboratory_services": "Medicines & laboratory",
    "human_resources_for_health": "Human resources for health",
    "support_services": "Support services",
    "infrastructure": "Infrastructure",
    "health_information_management": "Health information",
    "communication": "Communication",
    "district_health_system_support": "District health support",
    "implementing_partners_and_stakeholders": "Implementing partners"}


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, bbox_inches="tight")
    plt.close(fig)


def binned_median(ax, x, y, bins=15):
    q = pd.cut(x, bins)
    m = pd.DataFrame({"x": x, "y": y}).groupby(q, observed=True) \
          .agg(x=("x", "median"), y=("y", "median"), n=("y", "size"))
    m = m[m.n >= 20]
    ax.plot(m.x, m.y, "o-", color="black", lw=2, ms=5, label="Binned median")


def kaplan_meier(a, fac, ids, target):
    """Share of clinics that have reached `target` (or higher) by each
    assessment number. A clinic that never gets there is censored at its last
    assessment."""
    s = a[a.facility_id.isin(ids)]
    hit = s[s.category.map(RANK) >= RANK[target]].groupby("facility_id").t.min()
    T = pd.Series(fac.loc[ids, "nobs"].values, index=ids)
    E = pd.Series(0, index=ids)
    T.loc[hit.index] = hit.values
    E.loc[hit.index] = 1
    surv, curve = 1.0, [(0, 0.0)]
    for t in range(1, 7):
        at_risk = (T >= t).sum()
        events = ((T == t) & (E == 1)).sum()
        if at_risk:
            surv *= 1 - events / at_risk
        curve.append((t, 1 - surv))
    return curve


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    a, f_raw = load_eda_panels()
    a = a.sort_values(["facility_id", "year"]).reset_index(drop=True)
    num = {}

    # A.1 count against rate, for facility-years with at least one visit
    d = a[a.anc_first_visits > 0]
    num["A1"] = dict(n=len(d), rho_count=rho(d.anc_first_visits, d.score), rho_rate=rho(d.r, d.score))
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.4))
    for axi, col, lab, ttl in [(ax[0], "anc_first_visits", "ANC first visits (count)", "Raw count"),
                               (ax[1], "r", "ANC first visits per 1,000 fertile women",
                                "Rate per 1,000 fertile women")]:
        axi.scatter(d.score, d[col], s=3, alpha=.15, color=BLUE, rasterized=True)
        binned_median(axi, d.score, d[col])
        axi.set(ylim=(0, d[col].quantile(.99)), xlabel="ICIS score (%)", ylabel=lab,
                title=f"{ttl}: ρ = {rho(d.score, d[col]):.3f}")
        axi.legend(frameon=False, loc="upper left")
    ax[0].yaxis.set_major_formatter(thousands)
    fig.suptitle(f"ANC First Visits against ICIS Score (n = {len(d):,} facility-years)",
                 fontweight="bold", fontsize=14)
    save(fig, "fig26_A1_count_vs_rate.png")

    # year-on-year pairs, used by A.2, A.3, B.1 and B.2
    g = a.groupby("facility_id")
    a["ps"], a["py"], a["pc"], a["pr"] = g.score.shift(), g.year.shift(), g.category.shift(), g.r.shift()
    c = a[a.year - a.py == 1].copy()
    c["d"] = c.score - c.ps
    pairs = sorted(c.year.unique())
    pair_label = {y: f"{y - 1}→{y}" for y in pairs}

    # A.2 was 2020 a real dip? If it were an artefact the changes would look
    # like the other years. They do not: far more clinics fell, and fell further.
    share = pd.DataFrame({"Fell >5pp": (c.d < -5).groupby(c.year).mean(),
                          "Within ±5pp": c.d.abs().le(5).groupby(c.year).mean(),
                          "Rose >5pp": (c.d > 5).groupby(c.year).mean()}) * 100
    num["A2"] = dict(any_decline=((c.d < 0).groupby(c.year).mean() * 100).to_dict(),
                     fell_more_than_5=share["Fell >5pp"].to_dict())
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.4))
    colours = dict(zip(pairs, ["#bdbdbd", "#9ecae1", RED, "#4292c6", "#08519c"]))
    for y in pairs:
        ax[0].hist(c[c.year == y].d.clip(-50, 50), bins=np.arange(-50, 52, 2), density=True,
                   histtype="step", lw=2.2 if y == 2020 else 1.5, color=colours[y], label=pair_label[y])
    ax[0].axvline(0, color="grey", ls="--", lw=1)
    ax[0].set(xlabel="Change in ICIS score from previous year (pp)", ylabel="Density",
              title="Distribution of score changes")
    ax[0].legend(frameon=False)
    w = .27
    for i, (col, cl) in enumerate(zip(share.columns, [RED, "#bdbdbd", GREEN])):
        bars = ax[1].bar(np.arange(len(pairs)) + (i - 1) * w, share[col], w, color=cl, label=col)
        for b, v in zip(bars, share[col]):
            ax[1].text(b.get_x() + b.get_width() / 2, v + .8, f"{v:.0f}%", ha="center", fontsize=9.5)
    ax[1].set(xticks=range(len(pairs)), xticklabels=[pair_label[y] for y in pairs],
              ylabel="% of facilities", title="Direction of change by year pair", ylim=(0, 62))
    ax[1].legend(frameon=False, ncol=3, loc="upper left")
    fig.suptitle("Year-on-Year ICIS Score Changes, 2017–2022", fontweight="bold", fontsize=14)
    save(fig, "fig27_A2_covid_check.png")

    # A.3 how often a clinic changes category from one assessment to the
    # next, and how often a small score change is enough to do it. The
    # transition matrix uses every pair of consecutive assessments, which
    # includes the 2018 to 2020 step for the Limpopo clinics with no 2019 record.
    cons = a.dropna(subset=["pc"])
    tm = pd.crosstab(cons.pc, cons.category, normalize="index").reindex(index=CATEGORIES, columns=CATEGORIES)
    num["A3_transition_matrix"] = tm.to_dict("index")
    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(tm.values * 100, cmap="Blues", vmin=0, vmax=100)
    for i in range(4):
        for j in range(4):
            v = tm.values[i, j] * 100
            ax.text(j, i, f"{v:.1f}%", ha="center", va="center", color="white" if v > 55 else "black")
    ax.set(xticks=range(4), yticks=range(4), xticklabels=CATEGORIES, yticklabels=CATEGORIES,
           xlabel="Category the following year", ylabel="Category this year",
           title="Year-on-Year Category Transitions (row %)")
    fig.colorbar(im, ax=ax, fraction=.046, pad=.04)
    save(fig, "fig28_A3a_transitions.png")

    c["cross"] = c.category != c.pc
    bands = pd.cut(c.d.abs(), [-.01, 4.999, 15, 200], labels=["Under 5pp", "5 to 15pp", "Over 15pp"])
    cr = c.groupby(bands, observed=True).cross.agg(["mean", "size"])
    pre = c[c.year <= 2019]
    num["A3"] = dict(small_change_crosses=c[c.d.abs() < 5].cross.mean() * 100,
                     small_change_crosses_before_2020=pre[pre.d.abs() < 5].cross.mean() * 100,
                     by_band=(cr["mean"] * 100).to_dict(), n_by_band=cr["size"].to_dict())
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.4))
    bars = ax[0].bar(cr.index.astype(str), cr["mean"] * 100, color=["#f0ad4e", BLUE, GREEN], width=.6)
    for b, v, n in zip(bars, cr["mean"] * 100, cr["size"]):
        ax[0].text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.1f}%\n(n = {n:,})", ha="center", fontsize=10.5)
    ax[0].set(ylim=(0, 75), xlabel="Size of year-on-year score change", ylabel="% that change category",
              title="Category changes by size of score change")
    for mask, colour, lab in [(c.cross, RED, "Changed category"), (~c.cross, "#7fb3d5", "Same category")]:
        ax[1].hist(c[mask].d.clip(-50, 50), bins=np.arange(-50, 52, 2), density=True, alpha=.6,
                   color=colour, label=f"{lab} (n = {mask.sum():,})")
    ax[1].axvline(0, color="grey", ls="--", lw=1)
    ax[1].set(xlabel="Change in ICIS score (pp)", ylabel="Density",
              title="Score changes: category changers and stayers")
    ax[1].legend(frameon=False)
    fig.suptitle("Category Changes and the Size of Score Changes, 2017–2022", fontweight="bold", fontsize=14)
    save(fig, "fig29_A3b_small_changes.png")

    # A.4 are the six BAS provinces like the three without finance data?
    exc = a.province.isin(BAS_EXCLUDED)
    num["A4"] = {col: dict(bas=a[~exc][col].median(), excluded=a[exc][col].median())
                 for col in ["score", "r", "total_headcount"]}
    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    for axi, col, lab, bins, scale in [
            (ax[0], "score", "ICIS score (%)", np.arange(10, 102, 3), 1),
            (ax[1], "r", "ANC first visits per 1,000 fertile women", np.arange(0, 5.1, .15), 1),
            (ax[2], "total_headcount", "Annual headcount (thousands)", np.arange(0, 101, 3), 1000)]:
        for mask, colour, name in [(~exc, GREEN, "BAS provinces"), (exc, RED, "Excluded provinces")]:
            x = a[mask][col] / scale
            axi.hist(x.clip(upper=bins[-1]), bins=bins, density=True, alpha=.55, color=colour,
                     label=f"{name} (median {x.median():.2f})" if col == "r"
                     else f"{name} (median {x.median():.1f})")
        axi.set(xlabel=lab, ylabel="Density")
        axi.legend(frameon=False, fontsize=9.5)
    ax[0].set_title("ICIS score")
    ax[1].set_title("ANC first-visit rate")
    ax[2].set_title("Headcount")
    fig.suptitle("Six BAS-Covered Provinces Compared with Free State, North West and Western Cape",
                 fontweight="bold", fontsize=14)
    save(fig, "fig30_A4_finance_representativeness.png")

    # A.5 time to reach each category, by urbanicity (Kaplan-Meier)
    a["t"] = a.groupby("facility_id").cumcount() + 1
    fac = a.groupby("facility_id").agg(u=("urban_rural", "first"), nobs=("t", "max"))
    num["A5"] = {}
    fig, ax = plt.subplots(1, 3, figsize=(17, 5.2))
    urb_colour = {"Urban": BLUE, "Peri-Urban": "#e3bb4f", "Rural": "#a9744f"}
    for axi, target in zip(ax, ["Silver", "Gold", "Platinum"]):
        for u, colour in urb_colour.items():
            ids = fac[fac.u == u].index
            curve = kaplan_meier(a, fac, ids, target)
            axi.step([p[0] for p in curve], [p[1] * 100 for p in curve], where="post", color=colour,
                     lw=2.2, label=f"{u} (n = {len(ids):,})")
            num["A5"][f"{target}|{u}"] = curve[-1][1]
        curve = kaplan_meier(a, fac, fac.index, target)
        axi.step([p[0] for p in curve], [p[1] * 100 for p in curve], where="post", color="black",
                 lw=1.4, ls="--", label="All clinics")
        num["A5"][f"{target}|All"] = curve[-1][1]
        axi.set(title=f"Reached {target} or higher" if target != "Platinum" else "Reached Platinum",
                xlabel="Assessment number (1 = first assessment)", ylabel="% of clinics",
                ylim=(0, 100), xticks=range(0, 7))
    ax[0].legend(frameon=False, loc="lower right", fontsize=9.5)
    fig.suptitle("Time to Reach Each ICIS Category, by Urbanicity (Kaplan–Meier)", fontweight="bold", fontsize=14)
    save(fig, "fig31_A5_kaplan_meier.png")

    # B.1 does last year's score predict this year's ANC rate?
    num["B1"] = dict(n=len(c), rho_current=rho(c.score, c.r), rho_previous=rho(c.ps, c.r))
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.4), sharey=True)
    for axi, col, ttl in [(ax[0], "score", "Current-year score"), (ax[1], "ps", "Previous-year score")]:
        axi.scatter(c[col], c.r, s=3, alpha=.12, color=BLUE, rasterized=True)
        binned_median(axi, c[col], c.r)
        axi.set(xlabel=f"{ttl} (%)", title=f"{ttl}: ρ = {rho(c[col], c.r):.3f}", ylim=(0, c.r.quantile(.99)))
        axi.legend(frameon=False, loc="upper left")
    ax[0].set_ylabel("ANC first visits per 1,000 fertile women")
    fig.suptitle(f"ANC First-Visit Rate against Current and Previous-Year ICIS Score (n = {len(c):,})",
                 fontweight="bold", fontsize=14)
    save(fig, "fig32_B1_lagged_score.png")

    # B.2 does moving up a category change the ANC rate?
    num["B2"] = {}
    fig, ax = plt.subplots(1, 3, figsize=(17, 5.4))
    for axi, (c0, c1), colour in zip(ax, [("Not achieved", "Silver"), ("Silver", "Gold"),
                                          ("Gold", "Platinum")], [BLUE, "#f0ad4e", GREEN]):
        s = c[(c.pc == c0) & (c.category == c1)]
        dd = s.r - s.pr
        top = np.nanquantile(np.r_[s.r, s.pr], .99)
        axi.scatter(s.pr, s.r, s=8, alpha=.4, color=colour)
        axi.plot([0, top], [0, top], ls="--", color="grey", lw=1)
        axi.set(xlim=(0, top), ylim=(0, top), xlabel="ANC per 1,000, year before upgrade",
                ylabel="ANC per 1,000, year of upgrade",
                title=f"{c0} → {c1}\n(n = {len(s):,}; mean change {dd.mean():+.3f})")
        num["B2"][f"{c0} -> {c1}"] = dict(n=len(s), mean_change=dd.mean(), share_up=(dd > 0).mean())
    fig.suptitle("ANC First-Visit Rate Before and After a Category Upgrade", fontweight="bold", fontsize=14)
    save(fig, "fig33_B2_upgrades.png")

    # B.3 provincial medians over time
    pv = a.pivot_table(index="year", columns="province", values=["score", "r"], aggfunc="median")
    num["B3"] = dict(rho_change=rho(pv["score"].loc[2022] - pv["score"].loc[2017],
                                    pv["r"].loc[2022] - pv["r"].loc[2017]))
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.8))
    for p, colour in PROVINCE_COLOUR.items():
        ax[0].plot(pv.index, pv["score"][p], "o-", color=colour, lw=2, ms=4, label=p)
        ax[1].plot(pv.index, pv["r"][p], "o-", color=colour, lw=2, ms=4)
    for axi in ax:
        axi.axvspan(2019.5, 2020.5, color="#f4cccc", alpha=.5, zorder=0)
    ax[0].set(xlabel="Assessment year", ylabel="Median ICIS score (%)", title="Median ICIS score by province")
    ax[1].set(xlabel="Assessment year", ylabel="Median ANC per 1,000 fertile women",
              title="Median ANC first-visit rate by province")
    fig.legend(*ax[0].get_legend_handles_labels(), loc="lower center", ncol=5, frameon=False,
               bbox_to_anchor=(.5, -.08))
    fig.suptitle("Provincial ICIS Score and ANC First-Visit Rate, 2017–2022", fontweight="bold", fontsize=14)
    save(fig, "fig34_B3_provincial_trajectories.png")

    # finance panel with duplicates removed, as in the models
    f = f_raw.sort_values("expenditure", ascending=False).drop_duplicates(["facility_id", "year"])
    f = f[f.Exp_per_visit > 0].copy()
    ff = f.merge(a[["facility_id", "year", "r"]], on=["facility_id", "year"])

    # B.4 what goes with a high ANC rate? For province, a correlation ratio on
    # the ranks stands in for a correlation.
    rk = a.r.rank()
    eta = np.sqrt(((rk.groupby(a.province).transform("mean") - rk.mean()) ** 2).sum()
                  / ((rk - rk.mean()) ** 2).sum())
    vals = {"Headcount": rho(a.total_headcount, a.r), "Province (correlation ratio)": float(eta),
            "Previous-year ICIS score": num["B1"]["rho_previous"], "ICIS score": rho(a.score, a.r),
            "Expenditure per visit": rho(ff.Exp_per_visit, ff.r)}
    num["B4"] = vals
    fig, ax = plt.subplots(figsize=(11, 4.8))
    keys = list(vals)[::-1]
    ax.barh(keys, [vals[k] for k in keys],
            color=[RED if vals[k] < 0 else (BLUE if abs(vals[k]) > .1 else "#bdbdbd") for k in keys])
    for i, k in enumerate(keys):
        ax.text(vals[k] + (.01 if vals[k] >= 0 else -.01), i, f"{vals[k]:.3f}", va="center",
                ha="left" if vals[k] >= 0 else "right")
    ax.axvline(0, color="grey", lw=1)
    ax.set(xlim=(-.4, .72), xlabel="Spearman ρ with ANC per 1,000 (province: correlation ratio on ranks)",
           title="Correlates of the ANC First-Visit Rate")
    save(fig, "fig35_B4_anc_correlates.png")

    # B.5 score change 2017 to 2022 by urbanicity and province
    fp = a.pivot_table(index="facility_id", columns="year", values="score")
    change = (fp[2022] - fp[2017]).dropna()
    info = a.groupby("facility_id").agg(u=("urban_rural", "first"), p=("province", "first"))
    ch = info.join(change.rename("d"), how="inner")
    num["B5"] = dict(by_urbanicity=ch.groupby("u").d.median().to_dict(),
                     by_province_urbanicity=ch.groupby(["p", "u"]).d.median().unstack().to_dict("index"))
    fig, ax = plt.subplots(1, 2, figsize=(16, 5.6), gridspec_kw={"width_ratios": [1, 2.2]})
    urb = ["Rural", "Peri-Urban", "Urban"]
    bp = ax[0].boxplot([ch[ch.u == u].d for u in urb], patch_artist=True, widths=.55, showfliers=False,
                       medianprops=dict(color="black", lw=2))
    for box, colour in zip(bp["boxes"], ["#a9744f", "#e3bb4f", BLUE]):
        box.set_facecolor(colour)
        box.set_alpha(.8)
    ax[0].axhline(0, color="grey", lw=1)
    ax[0].set(xticklabels=[f"{u}\n(median {ch[ch.u == u].d.median():+.1f}pp)" for u in urb],
              ylabel="Change in ICIS score, 2017 to 2022 (pp)", title="By urbanicity")
    tab = ch.groupby(["p", "u"]).d.median().unstack().reindex(columns=urb)
    x = np.arange(len(tab))
    for i, (u, colour) in enumerate(zip(urb, ["#a9744f", "#e3bb4f", BLUE])):
        ax[1].bar(x + (i - 1) * .27, tab[u], .27, color=colour, label=u)
    ax[1].axhline(0, color="grey", lw=1)
    ax[1].set(xticks=x, xticklabels=[p.replace(" ", "\n") for p in tab.index],
              ylabel="Median change (pp)", title="By province and urbanicity")
    ax[1].legend(frameon=False)
    fig.suptitle("ICIS Score Change 2017 to 2022 by Urbanicity and Province", fontweight="bold", fontsize=14)
    save(fig, "fig36_B5_change_by_urbanicity.png")

    # B.6 trajectory groups by score change
    groups = pd.cut(change, [-200, -5, 5, 15, 200], right=False,
                    labels=["Decliners", "Stagnant", "Moderate improvers", "Strong improvers"])
    fm = a.groupby("facility_id").agg(anc=("r", "mean"), hc=("total_headcount", "mean"))
    G = pd.DataFrame({"d": change, "g": groups}).join(fm)
    order = ["Strong improvers", "Moderate improvers", "Stagnant", "Decliners"]
    num["B6"] = dict(n=G.g.value_counts().to_dict(),
                     anc_median=G.groupby("g", observed=True).anc.median().to_dict(),
                     headcount_median=G.groupby("g", observed=True).hc.median().to_dict())
    fig, ax = plt.subplots(1, 3, figsize=(17, 5.4))
    for axi, col, lab, cap in [(ax[0], "d", "Change in ICIS score, 2017 to 2022 (pp)", None),
                               (ax[1], "anc", "Mean ANC per 1,000 fertile women", G.anc.quantile(.98)),
                               (ax[2], "hc", "Mean annual headcount", G.hc.quantile(.98))]:
        bp = axi.boxplot([G[G.g == o][col].clip(upper=cap) if cap else G[G.g == o][col] for o in order],
                         patch_artist=True, widths=.6, medianprops=dict(color="black", lw=2),
                         flierprops=dict(ms=2, alpha=.4))
        for box, colour in zip(bp["boxes"], [BLUE, "#7fb3d5", "#f0ad4e", RED]):
            box.set_facecolor(colour)
            box.set_alpha(.8)
        axi.set_xticklabels([f"{o.replace(' ', chr(10))}\n(n = {(G.g == o).sum():,})" for o in order], fontsize=10)
        axi.set_ylabel(lab)
    ax[2].yaxis.set_major_formatter(thousands)
    ax[0].set_title("Score change")
    ax[1].set_title("ANC first-visit rate")
    ax[2].set_title("Headcount")
    fig.suptitle("Facility Score Trajectory Groups, 2017–2022", fontweight="bold", fontsize=14)
    save(fig, "fig37_B6_trajectory_groups.png")

    # B.7 clinics that never leave Not achieved against those that stay high
    cats = a.groupby("facility_id").category.agg(list)
    low = cats[cats.apply(lambda l: len(l) >= 4 and all(x == "Not achieved" for x in l))].index
    high = cats[cats.apply(lambda l: sum(x in ("Gold", "Platinum") for x in l) >= 3)].index
    A, B = a[a.facility_id.isin(low)], a[a.facility_id.isin(high)]
    fA, fB = A.groupby("facility_id").first(), B.groupby("facility_id").first()
    num["B7"] = dict(n=[len(low), len(high)], anc_median=[A.r.median(), B.r.median()],
                     rural=[(fA.urban_rural == "Rural").mean(), (fB.urban_rural == "Rural").mean()],
                     eastern_cape=[(fA.province == "Eastern Cape").mean(), (fB.province == "Eastern Cape").mean()],
                     headcount_median=[A.total_headcount.median(), B.total_headcount.median()])
    la, lb = f"Persistently Not Achieved (n = {len(low):,})", f"Sustained high performers (n = {len(high):,})"
    fig, ax = plt.subplots(2, 2, figsize=(14, 10))
    for axi, col, lab, bins, scale in [
            (ax[0, 0], "r", "ANC first visits per 1,000 fertile women", np.arange(0, 5.1, .15), 1),
            (ax[0, 1], "total_headcount", "Annual headcount (thousands)", np.arange(0, 101, 3), 1000)]:
        for s, colour, name in [(A, RED, la), (B, GREEN, lb)]:
            x = s[col] / scale
            axi.hist(x.clip(upper=bins[-1]), bins=bins, density=True, alpha=.55, color=colour,
                     label=f"{name.split(' (')[0]} (median {x.median():.2f})")
        axi.set(xlabel=lab, ylabel="Density")
        axi.legend(frameon=False, fontsize=9.5)
    ax[0, 0].set_title("ANC first-visit rate")
    ax[0, 1].set_title("Headcount")
    urb = ["Rural", "Peri-Urban", "Urban"]
    for frame, off, colour, name in [(fA, -.19, RED, la), (fB, .19, GREEN, lb)]:
        v = frame.urban_rural.value_counts(normalize=True).reindex(urb).fillna(0) * 100
        bars = ax[1, 0].bar(np.arange(3) + off, v, .38, color=colour, label=name)
        for b, vv in zip(bars, v):
            ax[1, 0].text(b.get_x() + b.get_width() / 2, vv + 1, f"{vv:.0f}%", ha="center", fontsize=10)
    ax[1, 0].set(xticks=range(3), xticklabels=urb, ylabel="% of clinics in group", title="Urbanicity", ylim=(0, 85))
    ax[1, 0].legend(frameon=False, fontsize=9.5)
    provs = sorted(PROVINCE_COLOUR)
    for frame, off, colour in [(fA, -.19, RED), (fB, .19, GREEN)]:
        v = frame.province.value_counts(normalize=True).reindex(provs).fillna(0) * 100
        ax[1, 1].bar(np.arange(len(provs)) + off, v, .38, color=colour)
    ax[1, 1].set(xticks=range(len(provs)), xticklabels=["EC", "FS", "GP", "KZN", "LP", "MP", "NW", "NC", "WC"],
                 ylabel="% of clinics in group", title="Province")
    fig.suptitle("Persistently Not Achieved Clinics and Sustained High Performers", fontweight="bold", fontsize=14)
    save(fig, "fig38_B7_low_vs_high.png")

    # B.8 expenditure against score, province by province
    f["log_exp"] = np.log(f.Exp_per_visit)
    num["B8"] = {p: rho(s.log_exp, s.score) for p, s in f.groupby("province")}
    fig, ax = plt.subplots(2, 3, figsize=(16, 9.5), sharex=True, sharey=True)
    for axi, (p, s) in zip(ax.flat, f.groupby("province")):
        axi.scatter(s.log_exp, s.score, s=4, alpha=.25, color=PROVINCE_COLOUR[p], rasterized=True)
        binned_median(axi, s.log_exp, s.score)
        axi.set(title=f"{p}{' (2017 only)' if p == 'KwaZulu-Natal' else ''}\nρ = {num['B8'][p]:+.2f}, n = {len(s):,}",
                xlabel="Log expenditure per visit", ylabel="ICIS score (%)")
    fig.suptitle("Log Expenditure per Visit against ICIS Score by Province", fontweight="bold", fontsize=14)
    save(fig, "fig39_B8_expenditure_by_province.png")

    # B.9 change in spending against change in score
    f = f.sort_values(["facility_id", "year"])
    fe = f.pivot_table(index="facility_id", columns="year", values=["Exp_per_visit", "score"])
    m = pd.DataFrame({"de": fe["Exp_per_visit"][2022] - fe["Exp_per_visit"][2017],
                      "ds": fe["score"][2022] - fe["score"][2017]}).dropna()
    cuts = f.Exp_per_visit.quantile([1 / 3, 2 / 3]).values
    f["band"] = pd.cut(f.Exp_per_visit, [0, cuts[0], cuts[1], 1e9],
                       labels=["Lowest third", "Middle third", "Highest third"])
    bm = f.groupby(["year", "band"], observed=True).score.median().unstack()
    num["B9"] = dict(rho=rho(m.de, m.ds), n=len(m), band_cuts=cuts.tolist(),
                     median_2017=bm.loc[2017].to_dict(), median_2022=bm.loc[2022].to_dict())
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.4))
    lim = m.de.quantile([.01, .99]).values
    mm = m[m.de.between(*lim)]
    ax[0].scatter(mm.de, mm.ds, s=8, alpha=.35, color=BLUE)
    ax[0].axhline(0, color="grey", lw=1)
    ax[0].axvline(0, color="grey", lw=1)
    ax[0].set(xlabel="Change in expenditure per visit, 2017 to 2022 (R)", ylabel="Change in ICIS score (pp)",
              title=f"ρ = {num['B9']['rho']:.3f}, n = {len(m):,}")
    for band, colour in zip(bm.columns, [GREEN, BLUE, RED]):
        ax[1].plot(bm.index, bm[band], "o-", color=colour, lw=2.2, label=band)
    ax[1].axvspan(2019.5, 2020.5, color="#f4cccc", alpha=.5, zorder=0)
    ax[1].set(xlabel="Assessment year (no BAS records for 2021)", ylabel="Median ICIS score (%)",
              title="Median score by spending band", xticks=sorted(f.year.unique()))
    ax[1].legend(frameon=False, title=f"below R{cuts[0]:.0f} / R{cuts[0]:.0f}–R{cuts[1]:.0f} / above R{cuts[1]:.0f}",
                 title_fontsize=9.5, fontsize=10)
    fig.suptitle("Change in Expenditure and Change in ICIS Score", fontweight="bold", fontsize=14)
    save(fig, "fig40_B9_expenditure_change.png")

    # B.10 does the previous record's spending predict this year's score?
    f["pe"] = f.groupby("facility_id").Exp_per_visit.shift()
    lag = f.dropna(subset=["pe"])
    num["B10"] = dict(n=len(lag), rho_current=rho(lag.Exp_per_visit, lag.score),
                      rho_previous=rho(lag.pe, lag.score))
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.4), sharey=True)
    top = lag.Exp_per_visit.quantile(.99)
    for axi, col, ttl in [(ax[0], "Exp_per_visit", "Current expenditure per visit"),
                          (ax[1], "pe", "Expenditure per visit, previous record")]:
        s = lag[lag[col] <= top]
        axi.scatter(s[col], s.score, s=4, alpha=.15, color=BLUE, rasterized=True)
        binned_median(axi, s[col], s.score)
        axi.set(xlabel=f"{ttl} (R)", title=f"{ttl}: ρ = {rho(lag[col], lag.score):.3f}")
        axi.legend(frameon=False, loc="lower right")
    ax[0].set_ylabel("ICIS score (%)")
    fig.suptitle(f"ICIS Score against Current and Previous Expenditure per Visit (n = {len(lag):,})",
                 fontweight="bold", fontsize=14)
    save(fig, "fig41_B10_lagged_expenditure.png")

    # B.11 six views of expenditure
    ff["decile"] = pd.qcut(ff.Exp_per_visit, 10, labels=False) + 1
    num["B11"] = dict(rho_exp_anc=rho(ff.Exp_per_visit, ff.r),
                      rho_exp_headcount=rho(ff.Exp_per_visit, ff.total_headcount),
                      median_by_category=ff.groupby("category").Exp_per_visit.median().to_dict(),
                      median_by_urbanicity=ff.groupby("urban_rural").Exp_per_visit.median().to_dict(),
                      score_by_decile=ff.groupby("decile").score.median().to_dict())
    fig, ax = plt.subplots(2, 3, figsize=(17, 10))
    top = ff.Exp_per_visit.quantile(.99)
    ax[0, 0].boxplot([ff[ff.category == k].Exp_per_visit.clip(upper=top) for k in CATEGORIES],
                     showfliers=False, widths=.55, medianprops=dict(color="black", lw=2))
    ax[0, 0].set(xticklabels=CATEGORIES, ylabel="Expenditure per visit (R)", title="By ICIS category")
    for p, s in ff.groupby("province"):
        med = s.groupby("year").Exp_per_visit.median()
        ax[0, 1].plot(med.index, med.values, "o-", color=PROVINCE_COLOUR[p], label=p)
    ax[0, 1].set(xlabel="Year", ylabel="Median expenditure per visit (R)", title="Provincial trends")
    ax[0, 1].legend(frameon=False, fontsize=9)
    dm = ff.groupby("decile").score.median()
    ax[0, 2].bar(dm.index, dm.values, color="#7fb3d5")
    ax[0, 2].set(xlabel="Expenditure-per-visit decile", ylabel="Median ICIS score (%)",
                 title="ICIS score by expenditure decile", ylim=(dm.min() - 10, 100))
    urb = ["Rural", "Peri-Urban", "Urban"]
    ax[1, 0].boxplot([ff[ff.urban_rural == u].Exp_per_visit.clip(upper=top) for u in urb],
                     showfliers=False, widths=.55, medianprops=dict(color="black", lw=2))
    ax[1, 0].set(xticklabels=urb, ylabel="Expenditure per visit (R)", title="By urbanicity")
    for axi, col, lab in [(ax[1, 1], "r", "ANC per 1,000 fertile women"),
                          (ax[1, 2], "total_headcount", "Annual headcount")]:
        s = ff[ff.Exp_per_visit <= top]
        axi.scatter(s.Exp_per_visit, s[col], s=3, alpha=.15, color=BLUE, rasterized=True)
        axi.set(xlabel="Expenditure per visit (R)", ylabel=lab,
                title=f"Against {'the ANC rate' if col == 'r' else 'headcount'} (ρ = {rho(ff.Exp_per_visit, ff[col]):.3f})",
                ylim=(0, s[col].quantile(.99)))
    ax[1, 2].yaxis.set_major_formatter(thousands)
    fig.suptitle("Expenditure Patterns: Six-Panel Summary", fontweight="bold", fontsize=14)
    save(fig, "fig42_B11_expenditure_summary.png")

    # B.12 which components go with the ANC rate among Not achieved clinics?
    na = a[a.category == "Not achieved"]
    sv = pd.Series({lab: rho(na[col], na.r) for col, lab in COMPONENTS.items()}).sort_values()
    num["B12"] = dict(n=len(na), rho=sv.to_dict())
    fig, ax = plt.subplots(figsize=(11, 5.6))
    ax.barh(sv.index, sv.values, color=[RED if v < 0 else BLUE for v in sv.values])
    for i, v in enumerate(sv.values):
        ax.text(v + (.003 if v >= 0 else -.003), i, f"{v:.3f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=10.5)
    ax.axvline(0, color="grey", lw=1)
    ax.set(xlim=(min(-.16, sv.min() - .03), max(.11, sv.max() + .03)),
           xlabel="Spearman ρ with ANC per 1,000 fertile women",
           title=f"ICIS Components and the ANC Rate, Not Achieved Facilities (n = {len(na):,})")
    save(fig, "fig43_B12_component_correlations.png")

    # B.13 component profiles by category (radar charts)
    prof = a.groupby("category")[list(COMPONENTS)].mean().reindex(CATEGORIES)
    num["B13"] = dict(means=prof.to_dict("index"),
                      platinum_minus_gold=(prof.loc["Platinum"] - prof.loc["Gold"]).to_dict(),
                      silver_minus_not_achieved=(prof.loc["Silver"] - prof.loc["Not achieved"]).to_dict())
    angles = np.linspace(0, 2 * np.pi, len(COMPONENTS), endpoint=False).tolist()
    fig, ax = plt.subplots(1, 2, figsize=(15, 7), subplot_kw=dict(polar=True))
    for axi, pair in zip(ax, [("Not achieved", "Silver"), ("Gold", "Platinum")]):
        for k in pair:
            v = prof.loc[k].tolist()
            axi.plot(angles + angles[:1], v + v[:1], color=CATEGORY_COLOUR[k], lw=2, label=k)
            axi.fill(angles + angles[:1], v + v[:1], color=CATEGORY_COLOUR[k], alpha=.15)
        axi.set_xticks(angles)
        axi.set_xticklabels(list(COMPONENTS.values()), fontsize=9)
        axi.set_ylim(0, 100)
        axi.set_title(f"{pair[0]} against {pair[1]}", pad=20)
        axi.legend(frameon=False, loc="lower right", bbox_to_anchor=(1.15, -.1))
    fig.suptitle("ICIS Component Profiles by Category (mean score, 2017–2022)", fontweight="bold", fontsize=14)
    save(fig, "fig44_B13_component_profiles.png")

    # B.14 vital items score by category
    vt = a.groupby(["year", "category"]).vital.mean().unstack().reindex(columns=CATEGORIES)
    num["B14"] = dict(mean_by_year=vt.to_dict(),
                      median=a.groupby("category").vital.median().to_dict())
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.4))
    for k in CATEGORIES:
        ax[0].plot(vt.index, vt[k], "o-", color=CATEGORY_COLOUR[k], lw=2.2, label=k)
    ax[0].axvspan(2019.5, 2020.5, color="#f4cccc", alpha=.5, zorder=0)
    ax[0].set(xlabel="Assessment year", ylabel="Mean vital items score (%)",
              title="Mean vital items score by category", ylim=(50, 105))
    ax[0].legend(frameon=False, loc="lower left")
    bp = ax[1].boxplot([a[a.category == k].vital for k in CATEGORIES], patch_artist=True, widths=.55,
                       medianprops=dict(color="black", lw=2), flierprops=dict(ms=2, alpha=.3))
    for box, k in zip(bp["boxes"], CATEGORIES):
        box.set_facecolor(CATEGORY_COLOUR[k])
        box.set_alpha(.8)
    ax[1].set_xticklabels([f"{k}\n(median {a[a.category == k].vital.median():.0f}%)" for k in CATEGORIES])
    ax[1].set(ylabel="Vital items score (%)", title="Distribution by category, 2017–2022 pooled")
    fig.suptitle("Vital Items Score by ICIS Category", fontweight="bold", fontsize=14)
    save(fig, "fig45_B14_vital_items.png")

    # B.15 every district side by side. Each measure is scaled 0 to 1 across
    # districts so the four can share an axis.
    dist = a.groupby(["province", "district"]).agg(score=("score", "mean"), anc=("r", "mean"),
                                                   headcount=("total_headcount", "mean"))
    # BAS spells district names differently, so expenditure is matched to a
    # district through the facility instead of by name
    home = a.drop_duplicates("facility_id").set_index("facility_id").district
    exp = f.assign(district=f.facility_id.map(home)).groupby("district").Exp_per_visit.median()
    dist = dist.join(exp.rename("exp"), on="district")
    scaled = (dist - dist.min()) / (dist.max() - dist.min())
    scaled = scaled.sort_values(["province", "score"])
    num["B15"] = dict(n_districts=len(dist), rho_score_anc=rho(dist.score, dist.anc),
                      rho_score_anc_within_province={p: rho(s.score, s.anc) for p, s in dist.groupby(level=0)})
    fig, ax = plt.subplots(figsize=(18, 7))
    x = np.arange(len(scaled))
    for i, (col, colour, lab) in enumerate([("score", BLUE, "ICIS score"), ("anc", RED, "ANC per 1,000"),
                                            ("headcount", GREEN, "Headcount"), ("exp", "#8e44ad", "Expenditure per visit")]):
        ax.bar(x + (i - 1.5) * .2, scaled[col], .2, color=colour, label=lab)
    edges = np.cumsum(scaled.groupby(level=0, sort=False).size().values)
    for e in edges[:-1]:
        ax.axvline(e - .5, color="grey", lw=1)
    ax.set(xticks=x, xlim=(-.6, len(x) - .4), ylabel="Scaled value (0 = lowest district, 1 = highest)",
           title="District-Level Profile by Province, All Districts\n")
    ax.set_xticklabels([d.replace(" District Municipality", "").replace(" Metropolitan Municipality", "")
                         .replace(" Municipality", "") for d in scaled.index.get_level_values(1)],
                       rotation=90, fontsize=8)
    ax.legend(frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(.5, 1.0))
    save(fig, "fig46_B15_districts.png")

    # B.16 headcount over time
    hc = a.groupby("year").total_headcount.median()
    idx = a.groupby(["year", "province"]).total_headcount.median().unstack()
    idx = idx / idx.loc[2017] * 100
    num["B16"] = dict(national_median=hc.to_dict(),
                      change_2020=hc.loc[2020] / hc.loc[2019] - 1,
                      change_2022_vs_2019=hc.loc[2022] / hc.loc[2019] - 1)
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.4))
    ax[0].plot(hc.index, hc.values, "o-", color=BLUE, lw=2.4)
    for y, v in hc.items():
        ax[0].text(y, v * 1.01, f"{v:,.0f}", ha="center", fontsize=10)
    ax[0].set(xlabel="Assessment year", ylabel="Median annual headcount per facility",
              title="National median headcount")
    ax[0].yaxis.set_major_formatter(thousands)
    for p, colour in PROVINCE_COLOUR.items():
        ax[1].plot(idx.index, idx[p], "o-", color=colour, lw=2, ms=4, label=p)
    ax[1].axhline(100, color="grey", ls="--", lw=1)
    ax[1].set(xlabel="Assessment year", ylabel="Median headcount (2017 = 100)", title="By province")
    ax[1].legend(frameon=False, fontsize=9, ncol=2)
    for axi in ax:
        axi.axvspan(2019.5, 2020.5, color="#f4cccc", alpha=.5, zorder=0)
    fig.suptitle("Headcount Trends, 2017–2022", fontweight="bold", fontsize=14)
    save(fig, "fig47_B16_headcount.png")

    path = save_json(num, "appendix")
    p = lambda v: f"{v:.3f}"
    print(f"A.1 n = {num['A1']['n']:,}; rho count {p(num['A1']['rho_count'])}, rate {p(num['A1']['rho_rate'])}")
    print("A.2 any decline: " + ", ".join(f"{y} {v:.1f}%" for y, v in num["A2"]["any_decline"].items())
          + "; fell >5pp: " + ", ".join(f"{y} {v:.1f}%" for y, v in num["A2"]["fell_more_than_5"].items()))
    print("A.3 stay in category: " + ", ".join(f"{k} {num['A3_transition_matrix'][k][k]:.1%}" for k in CATEGORIES)
          + f"; under 5pp cross {num['A3']['small_change_crosses']:.1f}% "
            f"(before 2020 {num['A3']['small_change_crosses_before_2020']:.1f}%)")
    print("A.4 medians BAS / excluded: " + ", ".join(f"{k} {v['bas']:.2f}/{v['excluded']:.2f}" for k, v in num["A4"].items()))
    print("A.5 reached by sixth assessment: " + ", ".join(f"{k} {v:.0%}" for k, v in num["A5"].items()))
    print(f"B.1 n = {num['B1']['n']:,}; rho current {p(num['B1']['rho_current'])}, previous {p(num['B1']['rho_previous'])}")
    print("B.2 " + "; ".join(f"{k}: n {v['n']}, mean change {v['mean_change']:+.3f}" for k, v in num["B2"].items()))
    print(f"B.3 rho of provincial changes {num['B3']['rho_change']:.2f}")
    print("B.4 " + ", ".join(f"{k} {v:.3f}" for k, v in num["B4"].items()))
    print("B.5 median change: " + ", ".join(f"{k} {v:+.1f}" for k, v in num["B5"]["by_urbanicity"].items()))
    print("    rural by province: " + ", ".join(f"{k} {v['Rural']:+.1f}" for k, v in num["B5"]["by_province_urbanicity"].items()))
    print("B.6 " + ", ".join(f"{k} {v:,}" for k, v in num["B6"]["n"].items()))
    b7 = num["B7"]
    print(f"B.7 n {b7['n']}, median ANC {b7['anc_median'][0]:.2f}/{b7['anc_median'][1]:.2f}, rural "
          f"{b7['rural'][0]:.0%}/{b7['rural'][1]:.0%}, Eastern Cape {b7['eastern_cape'][0]:.0%}/{b7['eastern_cape'][1]:.0%}")
    print("B.8 " + ", ".join(f"{k} {v:+.2f}" for k, v in num["B8"].items()))
    print(f"B.9 rho {num['B9']['rho']:.3f}, n {num['B9']['n']:,}; 2022 band medians "
          + ", ".join(f"{k} {v:.1f}" for k, v in num["B9"]["median_2022"].items()))
    print(f"B.10 n {num['B10']['n']:,}; rho current {p(num['B10']['rho_current'])}, previous {p(num['B10']['rho_previous'])}")
    print(f"B.11 rho with ANC {num['B11']['rho_exp_anc']:.3f}, with headcount {num['B11']['rho_exp_headcount']:.3f}")
    print("B.12 " + ", ".join(f"{k} {v:+.3f}" for k, v in num["B12"]["rho"].items()))
    print("B.13 Platinum minus Gold: " + ", ".join(f"{v:+.1f}" for v in num["B13"]["platinum_minus_gold"].values())
          + "; Silver minus Not achieved: " + ", ".join(f"{v:+.1f}" for v in num["B13"]["silver_minus_not_achieved"].values()))
    print("B.14 Platinum mean by year: " + ", ".join(f"{y} {v:.1f}" for y, v in num["B14"]["mean_by_year"]["Platinum"].items())
          + "; medians " + ", ".join(f"{k} {v:.0f}" for k, v in num["B14"]["median"].items()))
    print(f"B.15 {num['B15']['n_districts']} districts, rho score vs ANC across districts {num['B15']['rho_score_anc']:.2f}")
    print(f"B.16 national median headcount change in 2020 {num['B16']['change_2020']:+.1%}, "
          f"2022 against 2019 {num['B16']['change_2022_vs_2019']:+.1%}")
    print(f"\nWritten to {path}")


if __name__ == "__main__":
    main()
