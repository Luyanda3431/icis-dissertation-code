"""
Build the analysis panels (Section 3.3).

Reads either the simulated panel or the real extracts, applies the exclusions
in Section 3.3.3, and writes the samples the other scripts use. It also writes
a reconciliation table so the counts in Section 3.3.4 can be checked against
what the code actually does.

Chapter 4 and the appendices describe the panel as it was loaded, before the
model exclusions, so a copy of the loaded data is written as well.

Usage:
    python src/01_build_panel.py --source sim
    python src/01_build_panel.py --source real --anc ANC.xlsx --finance FINANCE.xlsx

The real extracts can be .xlsx (sheets anc_with_pop_clean and
finance_with_pop_clean), .csv or .pkl.

Writes:
    outputs/eda_anc.csv, outputs/eda_finance.csv   panels as loaded (Chapter 4)
    outputs/panel_anc.csv                          tree models and Objective 1
    outputs/panel_finance.csv                      Objective 2
    outputs/sample_reconciliation.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common import HIGH, OUT, ROOT

KEEP_ANC = ["facility_id", "facility", "province", "district", "urban_rural",
            "ownership", "year", "score", "category", "total_headcount",
            "anc_first_visit_per_1000_"]
KEEP_FIN = ["facility_id", "province", "district", "urban_rural", "ownership",
            "year", "score", "category", "total_headcount", "headcount",
            "expenditure", "Exp_per_visit"]


def read_any(path, sheet):
    path = Path(path)
    if path.suffix == ".xlsx":
        return pd.read_excel(path, sheet_name=sheet)
    if path.suffix == ".pkl":
        return pd.read_pickle(path)
    return pd.read_csv(path)


def load(source, anc_path, fin_path):
    if source == "sim":
        return (pd.read_csv(ROOT / "data" / "sim_anc_panel.csv"),
                pd.read_csv(ROOT / "data" / "sim_finance_panel.csv"))
    return (read_any(anc_path, "anc_with_pop_clean"),
            read_any(fin_path, "finance_with_pop_clean"))


def build(anc, fin):
    steps = []

    def note(frame, stage, df):
        steps.append(dict(frame=frame, stage=stage, observations=len(df),
                          facilities=df.facility_id.nunique()))

    # ICIS-ANC panel
    anc = anc[KEEP_ANC].copy()
    note("ANC", "as loaded", anc)
    # Province-aided clinics are not part of the ICRM programme, and a score
    # of zero means the clinic was not assessed (Section 3.3.3)
    anc = anc[anc.ownership != "Province Aided"]
    note("ANC", "province-aided excluded", anc)
    anc = anc[anc.score > 0]
    note("ANC", "zero scores excluded", anc)

    anc["anc_rate"] = anc["anc_first_visit_per_1000_"]
    anc["log_headcount"] = np.log1p(anc.total_headcount)
    anc["high"] = anc.category.isin(HIGH).astype(int)
    anc = anc.dropna(subset=["anc_rate", "log_headcount", "urban_rural",
                             "province"]).reset_index(drop=True)
    note("ANC", "tree estimation sample", anc)
    # log(0) is undefined, so Objective 1 uses only rows with a positive ANC
    # rate (Section 3.4.1). The drop happens in 04_objective1_lmm.py; the
    # count is recorded here.
    note("ANC", "Objective 1 sample (ANC rate > 0)", anc[anc.anc_rate > 0])

    # Finance sub-panel
    fin = fin[KEEP_FIN].copy()
    fin["province"] = fin.province.replace({"Kwazulu-Natal": "KwaZulu-Natal"})
    note("Finance", "as loaded", fin)
    # Ten facility-years match two BAS records. I keep the one with the larger
    # expenditure (Section 3.3.4).
    fin = (fin.sort_values("expenditure", ascending=False)
              .drop_duplicates(["facility_id", "year"]))
    note("Finance", "duplicate facility-years removed", fin)
    fin = fin[(fin.Exp_per_visit > 0) & (fin.headcount > 0)]
    fin = fin.dropna(subset=["score", "urban_rural"]).copy()
    fin["log_exp"] = np.log(fin.Exp_per_visit)
    fin["log_headcount"] = np.log1p(fin.total_headcount)
    fin["high"] = fin.category.isin(HIGH).astype(int)
    # the ANC rate is added for the finance-sample forest (Section 5.3.4)
    fin = fin.merge(anc[["facility_id", "year", "anc_rate"]],
                    on=["facility_id", "year"], how="left").reset_index(drop=True)
    note("Finance", "Objective 2 estimation sample", fin)

    return anc, fin, pd.DataFrame(steps)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", choices=["sim", "real"], default="sim")
    p.add_argument("--anc", default="ANC.xlsx")
    p.add_argument("--finance", default="FINANCE.xlsx")
    a = p.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    anc_raw, fin_raw = load(a.source, a.anc, a.finance)
    anc_raw.to_csv(OUT / "eda_anc.csv", index=False)
    fin_raw.to_csv(OUT / "eda_finance.csv", index=False)

    anc, fin, recon = build(anc_raw, fin_raw)
    anc.to_csv(OUT / "panel_anc.csv", index=False)
    fin.to_csv(OUT / "panel_finance.csv", index=False)
    recon.to_csv(OUT / "sample_reconciliation.csv", index=False)

    print(f"source: {a.source}\n")
    print(recon.to_string(index=False))
    zero = anc[anc.anc_rate <= 0]
    print(f"\nfacility-years with no ANC visits: {len(zero)}")
    if len(zero):
        print(f"  their mean ICIS score  {zero.score.mean():.1f}")
        print(f"  mean score of the rest {anc[anc.anc_rate > 0].score.mean():.1f}")
    print(f"share Gold or Platinum: {anc.high.mean():.1%}")
    print(f"\nWritten to {OUT}")


if __name__ == "__main__":
    main()
