# Predicting the Responsiveness of Health Services with Ideal Clinic Indicator Scores

Code for my MSc dissertation in the Department of Statistical Sciences,
University of Cape Town.

**Author:** Luyanda Cebekhulu
**Supervisors:** Prof. Sheetal Silal, Dr Sebnem Er

---

## What this is

The dissertation asks two questions. Does a clinic that improves its Ideal
Clinic Indicator Score (ICIS) also see more women for a first antenatal care
(ANC) visit? And is spending per visit related to ICIS scores? It answers them
with mixed-effects models and classification trees, after an exploratory
chapter.

This repository has the code for all of it: the data checks in Chapter 3, the
exploratory analysis in Chapter 4, the models in Chapter 5 and the extra
analyses in the appendices. Every table and figure in those chapters is made
by one of the scripts below, so the steps can be followed and checked.

## What is not here

**The data.** The ICIS assessments, DHIS antenatal counts and BAS expenditure
records are not public and cannot be shared. So every script runs on a
**simulated panel** made by `src/00_simulate_data.py`. It has:

- made-up clinics ("Clinic 100001", ...) in made-up districts;
- the same columns, nine provinces and six years (2017 to 2022);
- province-level score, ANC and headcount levels set from rounded provincial
  summaries, so the scale is realistic;
- the same gaps as the real finance data (six provinces, no 2021,
  KwaZulu-Natal for 2017 only) and the same cleaning problems (province-aided
  rows, a zero score, duplicated finance records, a misspelt province name).

The simulated data will not give the numbers in the dissertation, and it is
not meant to. Every number in the dissertation comes from the real data.

In the simulation the ANC rate is made independently of the ICIS score,
because the dissertation finds no link, so the Objective 1 estimate on the
simulated data should be close to zero. Setting `SCORE_ON_ANC` above zero in
`00_simulate_data.py` builds in a real effect, and the model should then find
it.

## Running it

```bash
pip install -r requirements.txt

python src/00_simulate_data.py               # makes data/sim_*.csv
python src/01_build_panel.py --source sim    # cleaning and the samples
python src/02_data_checks.py                 # Chapter 3 checks
python src/03_eda.py                         # Chapter 4
python src/04_objective1_lmm.py              # Section 5.2, mixed model
python src/05_objective2_lmm.py              # Section 5.3, mixed model
python src/06_trees_validation.py            # under a minute
python src/07_trees_tuning.py                # several minutes
python src/08_interpretation.py              # about a minute
python src/09_results_figures.py             # Chapter 5 figures
python src/10_appendix.py                    # Appendices A and B
```

Each script prints the numbers the dissertation quotes, under the section they
belong to, and writes them to `outputs/` as JSON and CSV. Figures go to
`outputs/figures/`, named by their figure number in the dissertation
(`fig01_...` to `fig47_...`).

With access to the real extracts, replace the first two steps with

```bash
export ICIS_OUTPUT_DIR=/somewhere/outside/the/repo
python src/01_build_panel.py --source real --anc ANC.xlsx --finance FINANCE.xlsx
```

and run the rest as they are. `ICIS_OUTPUT_DIR` keeps everything made from
the real data outside the repository.

## Scripts

| Script | Dissertation | What it does |
|---|---|---|
| `common.py` | | Shared code: the mixed-effects model, the fixed-effects check, the tree models, the train-test splits |
| `00_simulate_data.py` | | Makes the simulated panel |
| `01_build_panel.py` | 3.3 | Exclusions, finance de-duplication, the estimation samples and a table of counts at each step |
| `02_data_checks.py` | 3.2, 3.4 | Checklist size by year, ANC count against rate, why zero-ANC rows are dropped and not given log(x + 1) |
| `03_eda.py` | Chapter 4 | Tables 1 and 2, Figures 1 to 14 and every number in the chapter |
| `04_objective1_lmm.py` | 3.6.1, 5.2.1 to 5.2.5 | Objective 1 model (Table 3) with its checks, the smallest detectable effect and the ownership gap |
| `05_objective2_lmm.py` | 3.6.2, 5.3.1 to 5.3.3 | Objective 2 model (Table 7) with the same checks, and the within against between comparison |
| `06_trees_validation.py` | 3.6.3, 5.2.6 | The leakage in a random split, and the three tree methods under grouped, temporal and GroupKFold validation (Table 5) |
| `07_trees_tuning.py` | 3.6.4 to 3.6.6, 5.2.7 | Grid searches, the pruning path, out-of-bag and boosting curves, province coding (Table 6) |
| `08_interpretation.py` | 3.6.8, 5.2.9, 5.2.10, 5.3.4 | Impurity and permutation importance, Shapley values, partial dependence, and the finance-sample forest |
| `09_results_figures.py` | Chapter 5 | Figures 15 to 25 |
| `10_appendix.py` | Appendices A, B | Figures 26 to 47 and the numbers in their captions |

## Notes on method

**Fitting the mixed models.** `common.fit_lmm` fits the random-intercept model
by maximum likelihood (or REML). Only the ratio of the two variances has to
be searched for; given that ratio, the model is least squares on partly
demeaned data. It is the same model `lme4::lmer` and `statsmodels` `MixedLM`
fit, written out so the estimation is visible.

**No cross-validation for the mixed models.** They have no hyperparameters,
so cross-validation would not inform any choice. They are checked with
residual and random-intercept diagnostics and AIC instead.

**Validation of the trees.** Each clinic appears up to six times, so a random
train-test split puts the same clinic on both sides. In the real panel 2,713
of the 2,720 test clinics were also in training. `06_trees_validation.py`
measures this and uses splits by clinic and by year instead. All tuning uses
five-fold GroupKFold.

**Exact Shapley values.** With five or six predictors there are only 32 or 64
coalitions, so `08_interpretation.py` computes every one. The eight province
columns move together, so province is treated as one predictor.

**Zero ANC rates.** A few facility-years record no first visits, and log(0)
is undefined. Adding 1 first is not neutral, because those clinics score
lower than the rest, so it creates a positive link with score. They are left
out of the Objective 1 model; `02_data_checks.py` shows the log(ANC + 1)
result for comparison.

**Duplicated finance records.** Ten facility-years match two BAS records; the
one with the larger expenditure is kept.

**One retained row.** As Section 3.3.3 notes, one province-aided
facility-year in 2022 stayed in the panel the dissertation's results were
produced from. By default `01_build_panel.py` applies the exclusion and drops
it. Adding `--as-reported` keeps it, so the published numbers can be
reproduced exactly.
