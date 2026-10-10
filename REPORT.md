# Assignment 1 Report

*Delete this italic guidance as you fill in each section. You'll be asked to
defend any of this without your code in front of you — write only what you
can actually explain.*

- **Name**: Anna Serra
- **Student ID**: 18815
- **Email**: [aserrag.ieu2022@student.ie.edu](mailto:aserrag.ieu2022@student.ie.edu)
- **Group**: BBADBA 5B

## Dataset

My dataset is [StartUp Investments (Crunchbase)](https://www.kaggle.com/datasets/arindam235/startup-investments-crunchbase/data) from Kaggle. It is a 2014 snapshot and an export of Crunchbase data on venture-backed companies. The file has 54,294 lines and 39 columns, of which 4,856 are completely empty, so there are **49,438 startups**; each non-empty row is one company. Restricting to founding years 2000–2008 (the modeling cohort below) leaves **12,214** companies.

Its columns cover identity and location (`name`, `market`, `country_code`, …), founding and funding dates (`founded_year`, `first_funding_at`, `last_funding_at`), amounts by round type (`seed`, `venture`, `round_A`–`round_H`, `funding_total_usd`, …), and a categorical `status` (`acquired` / `operating` / `closed`).

I picked it because the business question — who is likely to be acquired — maps cleanly onto `status`, the funding columns leave room for non-trivial feature engineering, and founding/funding dates make a time-aware split the natural choice. The raw file is large for the assignment’s “commit and retrain” guide, so the modeling cohort is restricted rather than shipping every row.

## Business / real-life framing

This is an M&A advisory boutique with limited banker time that has to decide which venture-backed startups to approach for a sell-side mandate. A model will score how likely a startup is to be acquired, relative to peers of similar age, from its funding profile. Bankers use that score to prioritize outreach; they review the list often (weekly or when capacity frees up), not once a year.

My target is the status variable, where `acquired = 1`, and `operating` or `closed = 0`. There is no acquisition date, so a “acquired within X years” label cannot be defined — only the 2014 status snapshot. Raw acquisition rates also collapse with youth (about 22% for 1999 founders vs 0.6% for 2013) because younger companies simply have not had time to exit. To make the label usable I kept companies founded from 2000 to 2008 (at least six years of exposure by the snapshot) and I added `company_age` as an exposure control so the model does not treat “young” as “never acquired.”

For the split, I will train on the founding cohorts of 2000–2005, validate on 2006, and test on 2007–2008 (~4,598 startups). Scoring newer startups from what happened to older ones matches how the boutique would actually use the model. A random split would mix cohorts, let the model see companies founded later than those it is meant to rank, and hide the real drift in acquisition rates across the split (~16% → ~13.6% → ~9.7%). The time split surfaces that honestly. Trade-offs I accept: test ages (6–7) sit outside the training age range (9–14), so any age coefficient is an extrapolation; funding totals may include rounds raised after an acquisition; and mean predicted probability on test ends up slightly **low** (about 0.086 vs actual 0.097), mostly because a positive `company_age` coefficient pulls the younger test cohorts down — the opposite of the “probabilities will run high” intuition from base-rate drift alone.

I estimated the costs of errors as follows:


| Error          | What happens                                                | Illustrative cost |
| -------------- | ----------------------------------------------------------- | ----------------- |
| False positive | Pitch a company that is never acquired                      | **€3,000** — roughly a day of associate research + deck prep |
| False negative | Miss a company that does get acquired (without our mandate) | **€37,500** expected — a €150,000 sell-side success fee (about 1–2% of a ~€10M deal) × a **25% win-rate** assumption. Missing an acquirer only costs the fee if we would have pitched *and* won; not every seller hires an adviser, and we would not win every pitch. |


FN cost ≫ FP cost (~12.5×), so on pure expected-cost grounds we would want high recall. The unconstrained cost-minimising probability threshold is about `FP / (FP + FN) ≈ 3,000 / 40,500 ≈ 0.07`, which would mean pitching almost every startup — far beyond what a small boutique can do. **Capacity is therefore the binding constraint; the euro costs score each choice of *k*, they do not set *k*.**

In production the firm can pitch on the order of **~200 companies a year**. The test set is two founding cohorts (~4,598 startups), not one year of deal flow, so I translate that capacity into a **fixed share: the top 5%** of scores on whatever set is being ranked (~230 names on the test set; about 200 on a 4,000-company annual list). The operating rule is **pitch the top-*k***, not an absolute probability cutoff. That choice also sidesteps the calibration issue above: whether probabilities run a bit high or a bit low, a fixed probability threshold would be misleading, whereas top-*k* depends only on ranking and is unaffected by that level shift.

The dashboard metric is expected business cost over the ranked list at a chosen *k* (default *k* = top 5%):

`cost = (# FP among pitched) × €3,000 + (# FN among not pitched) × €37,500`

Precision@*k* is the companion view: of the names we can afford to pitch, how many are real acquirers?

## Data preparation & feature engineering

Dataset-level findings (raw structure, acquisition rate by founding year that justifies the 2000–2008 window) use all rows — they describe how the export is built, not what predicts acquisition. Variable-level choices (funding, rounds, leakage, pace, market, geography: exploration 1.7–1.12) use the **training cohorts only** (founded 2000–2005), so validation and test never drove those design decisions.

**Cleaning.** The file is latin-1 with a few stray spaces in headers. I dropped the 4,856 completely empty rows, parsed `funding_total_usd` from text (Indian digit grouping and `-` for missing → NaN), parsed founding/funding dates with impossible values coerced to missing, stripped `market`, dropped rows without `status` or `founded_year`, and deduplicated on `permalink` (names can collide; the permalink is the id). That leaves **37,563** labelled companies; restricting to founding years 2000–2008 then leaves the **12,214** in the modeling window.

**Target, exposure, split.** As in the framing: `acquired = 1` vs operating/closed = 0; `company_age = 2014 − founded_year`; train 2000–2005 / val 2006 / test 2007–2008. Age controls for exposure **within training** so the model does not treat “young” as “never acquired” among the cohorts it learns from. On test it only shifts 2007 vs 2008 relative to each other, using a coefficient learned on older companies (ages 9–14) — an extrapolation I accept and name; it does **not** make absolute scores “comparable” across the train/test age gap.

**Funding features.** A missing `funding_total_usd` always came with zeros in every round column and a lower acquisition rate, so missing means “nothing disclosed,” not a random hole: I kept those rows, added a `funding_disclosed` flag, and filled the total with 0. Amounts span orders of magnitude with heavy right skew, so size enters as `log1p` of total, seed, angel, venture, and debt. Mix is `share_venture`, `share_debt_financing`, and `share_seed` of typed funding. Lettered rounds are only partly nested (about **61%** of companies with a C also had a B), so I do not treat them as eight independent dummies: `max_round` (A=1 … H=8, else 0) summarises progress in one ordered number. I also keep flags for seed/angel/venture/debt/grant, a count of distinct round types, and `funding_rounds`. Early pace is years from founding to first funding (negatives clipped to 0 — data-entry or pre-incorporation money — rather than dropping the company) and `log_funding_per_round`. **Post-IPO equity/debt are excluded from every funding feature:** that money only exists after an IPO, which is itself an exit.

**Sector and geography.** Hundreds of sparse `market` values cannot be one-hot encoded, so I hand-mapped the common markets into nine fixed sectors (`software_it`, `health_life_sciences`, …); everything else, including missing, is `other` (reference level, no column). For geography: `is_usa`, one column each for the five largest non-US countries **chosen on the training rows only** (GBR, CHN, CAN, FRA, ISR), `is_hub` for SF Bay Area / Boston / New York City / London, and `n_categories` from `category_list`. I dropped `state_code` (~⅓ missing). Exploration already shows `is_hub` as a strong univariate separator; how funding vs sector/geo share the fitted score is discussed with the coefficients below.

**Date features and the final set.** I also built `funding_span_years` and `rounds_per_year`, but they can leak if acquired companies stop raising after the deal (plausible from exploration, not proven by a shorter span alone). On validation, adding them lifted ROC-AUC by only **+0.006**. A paired bootstrap (1,000 redraws of the validation set) put the 95% interval of that gain at **[−0.005, +0.017]**, which includes zero — indistinguishable from noise. Rule in code: keep the date features only if the whole interval sits above zero; otherwise prefer the safer set. **The three final models therefore use the 37 features without the date features.**

## Modeling: three implementations, one model

The target is binary, so the model is **logistic regression**. Features are standardised with a `StandardScaler` **fitted on the training rows only**, then applied to validation and test — needed so gradient descent sees log-funding (~14) and 0/1 flags on a common scale.

I train the **same** model three ways on that matrix: scikit-learn, a from-scratch PyTorch loop (raw tensors, autograd, manual update), and `torch.nn.Linear` + `BCEWithLogitsLoss` + `torch.optim.SGD`. Comparability is enforced, not assumed: the same 37 features and cohort split; the same L2 penalty mapped from sklearn’s `C` as `λ = 1/(C·n_train)` with the bias unpenalised (PyTorch does **not** get its own `C`); `float64` throughout; both PyTorch runs start at zero weights.

**Tune sparingly.** Effort went into features (13 exploration sections, 37 engineered columns), not a grid search over the model. Only the two hyperparameters the brief allows get small grids:

| Hyperparameter | Where | Criterion | Grid |
| --- | --- | --- | --- |
| Regularisation `C` | 5.2 | Best validation ROC-AUC (ranking metric for the boutique) | 0.001, 0.01, 0.1, 1, 10 |
| Learning rate (both PyTorch versions) | 5.4 | Lowest training loss after a 500-step trial | 0.01, 0.1, 0.5, 1.0 |

Chosen values: **C = 10**, **lr = 1.0**. `C` barely matters — validation ROC-AUC is **0.7610** at `C = 1` vs **0.7618** at `C = 10`. The learning rate is chosen on **training** loss on purpose: it only changes how fast gradient descent reaches the optimum, not which optimum; with 100,000 steps, every non-diverging rate would land in the same place. The step count itself was **not** tuned for score — section 5.7 raised it until PyTorch matched sklearn. One small inconsistency: the date-feature bootstrap (4.3) still uses sklearn’s default `C = 1`, while the final models use `C = 10`; given how flat validation AUC is in `C`, that does not change the drop decision.

**Test results** (4,598 startups founded 2007–2008; top 5% = 230 pitches; cost = FP × €3,000 + FN × €37,500):

| Model | ROC-AUC | Precision@5% | Acquirers pitched | Cost |
| --- | ---: | ---: | ---: | ---: |
| Baseline: base rate (random ranking) | 0.500 | 9.7% | 22 | €16.55M |
| Baseline: rank by total funding | 0.579 | 9.1% | 21 | €16.60M |
| scikit-learn | 0.718 | 27.8% | 64 | €14.86M |
| PyTorch (manual loop) | 0.718 | 27.8% | 64 | €14.86M |
| PyTorch (nn + optim) | 0.718 | 27.8% | 64 | €14.86M |

The model’s pitch list hits acquirers about **2.9×** as often as random. Ranking by funding alone is no better than random on precision@5%. Most of the euro cost is the **383 acquirers left outside the top 230** — capacity binds, as the framing says, so the models mainly rearrange who fills the fixed pitch slots.

**Do the three agree?** After a first run they did not: the two PyTorch versions matched each other, but some weights were up to ~0.48 from scikit-learn and training loss was slightly higher. That was a hard optimisation problem, not a code bug. The design matrix is badly conditioned (condition number of `X′X/n` ≈ **7,300**; e.g. `has_angel` and `log_angel` have **r ≈ 0.996**), so plain full-batch gradient descent creeps along a narrow valley while sklearn’s solver reaches the optimum directly. At 20,000 steps PyTorch had not arrived; at **100,000** steps all three share the same objective, weights within ~**3e-4**, test probabilities within ~**8e-6**, and identical metrics. Section 5.7 of `train.py` prints that diagnosis.

**Reading the coefficients.** I do not interpret signs one feature at a time. Correlated groups split their effect unpredictably — e.g. `log_funding_total` ≈ +3.54 while `log_funding_per_round` ≈ −2.25 partly cancel because they overlap. Safer to speak in families: funding size/mix/stage, pace to first funding, sector dummies, and geography (including `is_hub`). Individually large coefficients inside a collinear pair are not stable stories. Artifacts (models, scaler, test predictions, feature table, results, coefficients, loss curves) are regenerated by every `main.py train` run and are git-ignored; the dashboard loads them rather than retraining.

## Limitations & next steps

1. **No acquisition date (1.2).** The label is “acquired by the 2014 snapshot,” not “within X years of founding,” which is why rates collapse from ~23% (1999 founders) to 0.6% (2013). **Fix:** join an acquisitions table with deal dates and redefine the target as acquired within a fixed window (e.g. five years), censoring companies that have not yet reached that age.

2. **Exposure drift and age extrapolation (2.7, 5.8).** Acquisition rates fall across the split (16.0% → 13.6% → 9.7%), and `company_age` is learned on ages 9–14 then applied to 6–7 — mean predicted probability on test runs **low** (0.086 vs 0.097). **Fix:** score within age strata (or use time-varying / landmark models) so the age effect is never extrapolated outside the training support; with deal dates, one could also match exposure windows across cohorts.

3. **Funding observed after the outcome (1.9).** Totals and round amounts may include money raised after an acquisition; the date-feature test found no clear leak from span alone, but the amounts themselves cannot be cut off at a decision date. **Fix:** rebuild features from round-level data truncated at a hypothetical pitch date (or at acquisition date for positives), so every input would have been known when the boutique decides whom to approach.

4. **Correlated features and a linear model (5.7).** Condition number of `X′X/n` ≈ 7,300 (`has_angel` / `log_angel` r ≈ 0.996) makes individual coefficients unreadable and forced 100,000 GD steps; logistic regression also cannot learn interactions (e.g. hub × sector) without hand-built terms. **Fix:** drop or combine near-duplicate funding encodings, then try a small tree / gradient-boosted model (or explicitly add a few interaction features) and re-check calibration and precision@5%.

5. **Thin validation and assumed business costs.** One extra hit moves validation precision@5% by ~0.011, and the date-feature gain sat inside noise (−0.005 to +0.017); the €150k fee and 25% win rate shape the euro column but not the ranking. **Fix:** bootstrap or nested validation for ranking metrics, and replace assumed costs with the boutique’s actual pitch cost and historical win rate (or treat cost as a dashboard sensitivity slider only).

6. **Stale, messy export.** A 2014 Crunchbase snapshot misses today’s funding landscape; train also had ~100 “funded before founded” gaps (clipped to 0), 22% of raw rows lacked `founded_year` (dropped), and markets were hand-mapped into sectors. **Fix:** refresh from a current company/funding dump with cleaner dates, and replace the hand sector map with a supervised or embedding-based market encoding refit on training only.

## Generative AI use disclosure

