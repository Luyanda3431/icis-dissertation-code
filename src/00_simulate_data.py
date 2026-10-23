"""
Simulated ICIS-ANC and finance panels.

The real assessment, DHIS and BAS data cannot be shared, so this script makes
a fake panel with the same columns and roughly the same shape. Every other
script runs on it, so the whole analysis can be followed without any real
clinic appearing anywhere.

The simulated data will not give the numbers in the dissertation, and it is
not meant to. Every number in the dissertation comes from the real data.

The province settings below are rounded provincial summaries (number of
clinics, typical score, typical ANC rate, typical headcount). They say nothing
about any single facility.

Usage:
    python src/00_simulate_data.py
Writes:
    data/sim_anc_panel.csv
    data/sim_finance_panel.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 2026
YEARS = range(2017, 2023)
OUT = Path(__file__).resolve().parents[1] / "data"

# Set this above zero to build a real link between score and ANC into the
# fake data. The model in 04_objective1_lmm.py should then pick it up, which
# is a simple check that the code can find an effect when there is one.
SCORE_ON_ANC = 0.0

# province: (clinics, score mean, score sd, median ANC per 1,000,
#            median headcount, share rural, share municipal, districts)
PROVINCES = {
    "Eastern Cape":  (727, 70.1, 14.3, 0.387, 13379, 0.640, 0.000, 8),
    "Free State":    (210, 75.1, 11.2, 1.013, 19850, 0.149, 0.000, 5),
    "Gauteng":       (327, 83.6,  8.0, 0.550, 39361, 0.025, 0.608, 5),
    "KwaZulu-Natal": (586, 81.8, 10.8, 0.860, 27014, 0.621, 0.100, 11),
    "Limpopo":       (454, 73.2, 14.0, 0.619, 23178, 0.852, 0.000, 5),
    "Mpumalanga":    (231, 74.1, 14.0, 0.552, 22049, 0.598, 0.000, 3),
    "North West":    (263, 72.4, 16.2, 0.487, 14863, 0.698, 0.000, 4),
    "Northern Cape": (128, 72.9, 12.7, 1.915, 12897, 0.277, 0.000, 5),
    "Western Cape":  (189, 80.6, 12.1, 0.677, 21516, 0.073, 0.360, 6),
}

# national mean score by year
YEAR_EFFECT = {2017: 71.0, 2018: 75.1, 2019: 75.5,
               2020: 73.6, 2021: 76.8, 2022: 82.5}

# number of elements assessed each year; the checklist changed in 2018 and
# again in 2020 (Section 3.2.1)
N_ELEMENTS = {2017: 206, 2018: 208, 2019: 208, 2020: 238, 2021: 238, 2022: 238}

# the ten ICIS components, as named in the assessment extract
COMPONENTS = ["administration", "integrated_clinical_services_management_icsm",
              "medicines_supplies_and_laboratory_services",
              "human_resources_for_health", "support_services", "infrastructure",
              "health_information_management", "communication",
              "district_health_system_support",
              "implementing_partners_and_stakeholders"]

# BAS records exist for six provinces, there is no 2021 extract, and
# KwaZulu-Natal only has 2017 (Section 3.2.4)
BAS_PROVINCES = ["Eastern Cape", "Gauteng", "KwaZulu-Natal",
                 "Limpopo", "Mpumalanga", "Northern Cape"]


def categorise(score, nnv):
    """ICIS category. From 2020 a facility that does not score 100% on the
    non-negotiable vital (NNV) elements is Not achieved, whatever else it
    scores."""
    if nnv is not None and nnv < 100:
        return "Not achieved"
    if score >= 90:
        return "Platinum"
    if score >= 80:
        return "Gold"
    if score >= 70:
        return "Silver"
    return "Not achieved"


def simulate_anc(rng):
    rows = []
    fid = 100000
    for prov, (n_fac, s_mu, s_sd, anc_med, hc_med,
               p_rural, p_muni, n_dist) in PROVINCES.items():

        districts = [f"{prov} District {i + 1}" for i in range(n_dist)]
        # fertile female population of each district, the same in every year
        dist_pop = {d: rng.lognormal(np.log(250_000), 0.45) for d in districts}

        for _ in range(n_fac):
            fid += 1
            district = rng.choice(districts)
            u = rng.random()
            urban = ("Rural" if u < p_rural
                     else "Peri-Urban" if u < p_rural + 0.08 else "Urban")
            ownership = ("Gov Municipality" if rng.random() < p_muni
                         else "Gov Province")

            # Facility effects. In the real panel almost all the variation in
            # the ANC rate is between facilities (ICC 0.944) and much less of
            # the score variation is (ICC 0.411), so the ANC effect is larger.
            u_score = rng.normal(0, s_sd * 0.64)
            u_anc = rng.normal(0, 0.86)
            u_comp = rng.normal(0, 6, len(COMPONENTS))

            hc_base = hc_med * np.exp(rng.normal(0, 0.55))
            if urban == "Rural":
                hc_base *= 0.75

            for year in YEARS:
                score = (s_mu + (YEAR_EFFECT[year] - 75.7) + u_score
                         + rng.normal(0, 6.5))
                score = float(np.clip(round(score), 14, 100))

                # NNV is only assessed from 2020 (the extract holds 0 before
                # that), and higher-scoring clinics are more likely to pass it
                if year < 2020:
                    nnv = 0.0
                else:
                    p_pass = 1 / (1 + np.exp(-(score - 72) / 8))
                    nnv = float(rng.choice([0, 33, 67, 100],
                                           p=[(1 - p_pass) * 0.45,
                                              (1 - p_pass) * 0.30,
                                              (1 - p_pass) * 0.25, p_pass]))

                vital = float(np.clip(score + rng.normal(5, 8), 0, 100))
                comps = np.clip(score + u_comp + rng.normal(0, 7, len(COMPONENTS)), 0, 100)

                # headcount drops in 2020, as it did nationally
                headcount = float(max(1, round(hc_base * np.exp(rng.normal(0, 0.12))
                                               * (0.89 if year >= 2020 else 1.0))))

                log_anc = (np.log(anc_med) + u_anc
                           + SCORE_ON_ANC * (score - s_mu) / s_sd
                           + rng.normal(0, 0.19))
                anc_rate = float(np.exp(log_anc))
                # a few facility-years have no ANC visits at all, and in the
                # real panel these clinics score lower than the rest
                if rng.random() < 0.0055 * (1.8 if score < 70 else 1.0):
                    anc_rate = 0.0

                fem_pop = dist_pop[district]
                row = {
                    "facility_id": fid,
                    "facility": f"Clinic {fid}",
                    "province": prov,
                    "district": district,
                    "urban_rural": urban,
                    "ownership": ownership,
                    "year": year,
                    "score": score,
                    "non_negotiable_vital": nnv,
                    "vital": round(vital),
                    "total": N_ELEMENTS[year],
                    "category": categorise(score, None if year < 2020 else nnv),
                    "total_headcount": headcount,
                    "anc_first_visits": round(anc_rate * fem_pop / 1000),
                    "anc_first_visit_per_1000_": anc_rate,
                    "female_15_49_pop": fem_pop,
                    "total_pop": fem_pop * 3.9,
                }
                row.update({c: round(float(v)) for c, v in zip(COMPONENTS, comps)})
                rows.append(row)
    return pd.DataFrame(rows)


def simulate_finance(anc, rng):
    """Expenditure per visit for the BAS provinces. Spending per visit falls
    with headcount, as it does in the real data (Figure 12), and has a weak
    positive link with score."""
    f = anc[anc.province.isin(BAS_PROVINCES) & (anc.year != 2021)]
    f = f[(f.province != "KwaZulu-Natal") | (f.year == 2017)]
    # about half of the eligible facility-years have a usable BAS record
    f = f.sample(frac=0.55, random_state=SEED).sort_values(
        ["facility_id", "year"]).copy()

    log_exp = (np.log(219)
               + 0.010 * (f.score - 75.7)
               - 0.35 * (np.log(f.total_headcount) - np.log(20385))
               + rng.normal(0, 0.45, len(f)))
    f["Exp_per_visit"] = np.exp(log_exp)
    f["headcount"] = f.total_headcount
    f["expenditure"] = f.Exp_per_visit * f.headcount
    f["exp_per_1000_pop"] = f.expenditure / f.total_pop * 1000

    # The BAS extract spells the province "Kwazulu-Natal", and a few
    # facility-years come through twice because two BAS cost centres matched
    # one facility. Both are copied here so the cleaning step has something
    # to fix.
    f["province"] = f.province.replace({"KwaZulu-Natal": "Kwazulu-Natal"})
    dup = f[f.province == "Gauteng"].sample(10, random_state=SEED).copy()
    for c in ["expenditure", "Exp_per_visit", "exp_per_1000_pop"]:
        dup[c] *= 0.05
    f = pd.concat([f, dup]).sort_values(["facility_id", "year"])
    return f.reset_index(drop=True)


def add_rows_to_exclude(anc):
    """A few rows that 01_build_panel.py should drop: province-aided clinics,
    which are outside the ICRM programme, and one facility-year with a score
    of zero, which means it was not assessed."""
    extra = anc.sample(6, random_state=SEED).copy()
    extra.loc[extra.index[:5], "ownership"] = "Province Aided"
    extra.loc[extra.index[5], "score"] = 0.0
    extra["facility_id"] = np.arange(900001, 900007)
    extra["facility"] = [f"Clinic {i}" for i in extra.facility_id]
    return pd.concat([anc, extra], ignore_index=True)


def main():
    rng = np.random.default_rng(SEED)
    OUT.mkdir(parents=True, exist_ok=True)

    anc = simulate_anc(rng)
    fin = simulate_finance(anc, rng)
    anc = add_rows_to_exclude(anc)

    anc.to_csv(OUT / "sim_anc_panel.csv", index=False)
    fin.to_csv(OUT / "sim_finance_panel.csv", index=False)

    print(f"ANC panel     {len(anc):,} facility-years, "
          f"{anc.facility_id.nunique():,} facilities, {anc.province.nunique()} provinces")
    print(f"              score mean {anc.score.mean():.1f}, sd {anc.score.std():.1f}; "
          f"median ANC rate {anc['anc_first_visit_per_1000_'].median():.2f}; "
          f"{(anc['anc_first_visit_per_1000_'] <= 0).sum()} zero-ANC rows")
    print(f"              Gold or Platinum {anc.category.isin(['Gold', 'Platinum']).mean():.1%}")
    print(f"Finance panel {len(fin):,} facility-years, {fin.facility_id.nunique():,} "
          f"facilities; median R{fin.Exp_per_visit.median():.0f} per visit")
    print(f"\nWritten to {OUT}")


if __name__ == "__main__":
    main()
