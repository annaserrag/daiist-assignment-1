# Assignment 1 Report

- **Name**: Anna Serra
- **Student ID**: 18815
- **Email**: [aserrag.ieu2022@student.ie.edu](mailto:aserrag.ieu2022@student.ie.edu)
- **Group**: BBADBA 5B

## Dataset

My dataset is [StartUp Investments (Crunchbase)](https://www.kaggle.com/datasets/arindam235/startup-investments-crunchbase/data) from Kaggle. It is a 2014 snapshot and an export of Crunchbase data on venture-backed companies. The file has 54,294 lines and 39 columns, of which 4,856 are completely empty, so there are **49,438 startups**; each non-empty row is one company. Restricting to founding years 2000–2008 (the modeling cohort below) leaves **12,214** companies.

Its columns cover identity and location (`name`, `market`, `country_code`, …), founding and funding dates (`founded_year`, `first_funding_at`, `last_funding_at`), amounts by round type (`seed`, `venture`, `round_A`–`round_H`, `funding_total_usd`, …), and a categorical `status` (`acquired` / `operating` / `closed`).

I picked it because the business question — who is likely to be acquired — maps cleanly onto `status`, the funding columns leave room for non-trivial feature engineering, and founding/funding dates make a time-aware split the natural choice. The full **~12.5 MB** export is committed in the repo (under the ~20 MB guide); the row count is above the rough ~20k guide, so the model uses the **12,214** companies in the 2000–2008 window rather than every startup in the file.

## Business / real-life framing

This is an M&A advisory boutique with limited banker time that has to decide which venture-backed startups to approach for a sell-side mandate. A model will score how likely a startup is to be acquired, relative to peers of similar age, from its funding profile. Bankers use that score to prioritize outreach; they review the list often (weekly or when capacity frees up), not once a year.

My target is the status variable, where `acquired = 1`, and `operating` or `closed = 0`. There is no acquisition date, so a “acquired within X years” label cannot be defined — only the 2014 status snapshot. Raw acquisition rates also collapse with youth (**22.4%** for 1999 founders vs 0.6% for 2013, on all non-empty rows before cleaning) because younger companies simply have not had time to exit. To make the label usable I kept companies founded from 2000 to 2008 (at least six years of exposure by the snapshot) and I added `company_age` as an exposure control so the model does not treat “young” as “never acquired.”

For the split, I will train on the founding cohorts of 2000–2005, validate on 2006, and test on 2007–2008 (~4,598 startups). Scoring newer startups from what happened to older ones matches how the boutique would actually use the model. A random split would mix cohorts, let the model see companies founded later than those it is meant to rank, and hide the real drift in acquisition rates across the split (~16% → ~13.6% → ~9.7%). The time split surfaces that honestly. Trade-offs I accept: test ages (6–7) sit outside the training age range (9–14), so any age coefficient is an extrapolation; funding totals may include rounds raised after an acquisition; and mean predicted probability on test ends up slightly **low** (about 0.086 vs actual 0.097). Base-rate drift alone would push predictions *high*; a check that sets every test company’s `company_age` to 9 (the youngest training age) raises the mean prediction from 0.086 to **0.135**, above the actual 0.097. So the positive age coefficient **over-corrects** when extrapolated to the younger test cohorts — that is why the probabilities run low.

I estimated the costs of errors as follows:


| Error          | What happens                                                | Illustrative cost |
| -------------- | ----------------------------------------------------------- | ----------------- |
| False positive | Pitch a company that is never acquired                      | **€3,000** — roughly a day of associate research + deck prep |
| False negative | Miss a company that does get acquired (without our mandate) | **€37,500** expected — a €150,000 sell-side success fee (about 1–2% of a ~€10M deal) × a **25% win-rate** assumption. Missing an acquirer only costs the fee if we would have pitched *and* won; not every seller hires an adviser, and we would not win every pitch. |


FN cost ≫ FP cost (~12.5×), so on pure expected-cost grounds we would want high recall. The unconstrained cost-minimising probability threshold is about `FP / (FP + FN) ≈ 3,000 / 40,500 ≈ 0.074`, which on the test set would mean pitching **43.9%** of startups (2,018 of 4,598) — about **9×** the boutique’s top-5% capacity of 230. **Capacity is therefore the binding constraint; the euro costs score each choice of *k*, they do not set *k*.**

In production the firm can pitch on the order of **~200 companies a year**. The test set is two founding cohorts (~4,598 startups), not one year of deal flow, so I translate that capacity into a **fixed share: the top 5%** of scores on whatever set is being ranked (~230 names on the test set; about 200 on a 4,000-company annual list). The operating rule is **pitch the top-*k***, not an absolute probability cutoff. That choice also sidesteps the calibration issue above: whether probabilities run a bit high or a bit low, a fixed probability threshold would be misleading, whereas top-*k* depends only on ranking and is unaffected by that level shift.

The dashboard metric is expected business cost over the ranked list at a chosen *k* (default *k* = top 5%):

`cost = (# FP among pitched) × €3,000 + (# FN among not pitched) × €37,500`

Precision@*k* is the companion view: of the names we can afford to pitch, how many are real acquirers?

## Data preparation & feature engineering

Dataset-level findings (raw structure, acquisition rate by founding year that justifies the 2000–2008 window) use all rows — they describe how the export is built, not what predicts acquisition. Variable-level choices (funding, rounds, leakage, pace, market, geography: exploration 1.7–1.12) use the **training cohorts only** (founded 2000–2005), so validation and test never drove those design decisions.

**Cleaning.** The file is latin-1 with a few stray spaces in headers. I dropped the 4,856 completely empty rows, parsed `funding_total_usd` from text (Indian digit grouping and `-` for missing → NaN), parsed founding/funding dates with impossible values coerced to missing, stripped `market`, dropped rows without `status` or `founded_year`, and deduplicated on `permalink` (names can collide; the permalink is the id). That leaves **37,563** labelled companies; restricting to founding years 2000–2008 then leaves the **12,214** in the modeling window.

**Target, exposure, split.** As in the framing: `acquired = 1` vs operating/closed = 0; `company_age = 2014 − founded_year`; train 2000–2005 / val 2006 / test 2007–2008. Age controls for exposure **within training** so the model does not treat “young” as “never acquired” among the cohorts it learns from. On test it only shifts 2007 vs 2008 relative to each other, using a coefficient learned on older companies (ages 9–14) — an extrapolation I accept and name; it does **not** make absolute scores “comparable” across the train/test age gap.

**Funding features.** A missing `funding_total_usd` always came with zeros in every round column and a lower acquisition rate, so missing means “nothing disclosed,” not a random hole: I kept those rows, added a `funding_disclosed` flag, and filled the total with 0. Amounts span orders of magnitude with heavy right skew, so size enters as `log1p` of total, seed, angel, venture, and debt. Mix is `share_venture`, `share_debt_financing`, and `share_seed` of typed funding. Lettered rounds are only partly nested (about **61%** of companies with a C also had a B); rather than eight sparse, correlated round dummies I keep `max_round` (A=1 … H=8, else 0) so progress is one ordered number. I also keep flags for seed/angel/venture/debt/grant, a count of distinct round types, and `funding_rounds`. Early pace is years from founding to first funding (negatives clipped to 0 — data-entry or pre-incorporation money — rather than dropping the company) and `log_funding_per_round`. **Post-IPO equity/debt are excluded from every funding feature:** that money only exists after an IPO, which is itself an exit.

**Sector and geography.** Hundreds of sparse `market` values cannot be one-hot encoded, so I hand-mapped the common markets into nine fixed sectors (`software_it`, `health_life_sciences`, …); everything else, including missing, is `other` (reference level, no column). For geography: `is_usa`, one column each for the five largest non-US countries **chosen on the training rows only** (GBR, CHN, CAN, FRA, ISR), `is_hub` for SF Bay Area / Boston / New York City / London, and `n_categories` from `category_list`. I dropped `state_code` (~⅓ missing). Exploration already shows `is_hub` as a strong univariate separator; I do not claim from the fitted coefficients alone how much funding vs sector/geo “drives” the score, because those features are collinear (see modeling).

**Date features and the final set.** I also built `funding_span_years` and `rounds_per_year`, but they can leak if acquired companies stop raising after the deal (plausible from exploration, not proven by a shorter span alone). On validation, adding them lifted ROC-AUC by only **+0.006**. A paired bootstrap (1,000 redraws of the validation set) put the 95% interval of that gain at **[−0.005, +0.017]**, which includes zero — indistinguishable from noise. Rule in code: keep the date features only if the whole interval sits above zero; otherwise prefer the safer set. **The three final models therefore use the 37 features without the date features.**

## Modeling: three implementations, one model

The target is binary, so the model is **logistic regression**. Features are standardised with a `StandardScaler` **fitted on the training rows only**, then applied to validation and test — needed so gradient descent sees log-funding (~14) and 0/1 flags on a common scale.

I train the **same** model three ways on that matrix: scikit-learn, a from-scratch PyTorch loop (raw tensors, autograd, manual update), and `torch.nn.Linear` + `BCEWithLogitsLoss` + `torch.optim.SGD`. Comparability is enforced, not assumed: the same 37 features and cohort split; the same L2 penalty mapped from sklearn’s `C` as `λ = 1/(C·n_train)` with the bias unpenalised (PyTorch does **not** get its own `C`); `float64` throughout; both PyTorch runs start at zero weights.

**Tune sparingly.** Effort went into features (13 exploration sections, 37 engineered columns), not a grid search over the model. Only the two hyperparameters the brief allows get small grids:

| Hyperparameter | Where | Criterion | Grid |
| --- | --- | --- | --- |
| Regularisation `C` | 5.2 | Best validation ROC-AUC (ranking metric for the boutique) | 0.001, 0.01, 0.1, 1, 10 |
| Learning rate (both PyTorch versions) | 5.4 | Lowest training loss after a 500-step trial | 0.01, 0.1, 0.5, 1.0 |

Chosen values: **C = 10**, **lr = 1.0**. `C` barely matters — validation ROC-AUC is **0.7610** at `C = 1` vs **0.7618** at `C = 10`. The learning rate is chosen on **training** loss on purpose: it only changes how fast gradient descent reaches the optimum, not which optimum; with 100,000 steps, every non-diverging rate would land in the same place. The step count itself was **not** tuned for score — section 5.7 raised it until PyTorch matched sklearn. The date-feature probes in 4.2–4.3 use the same `C = 10` as the final models.

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

The biggest hole in the label is still that we never see *when* a company was acquired — only that it had been by the 2014 snapshot (1.2). That is why acquisition rates look healthy for 1999 founders (22.4% on all non-empty rows) and almost vanish for 2013 (0.6%): younger companies simply have not had time. With more data I would join an acquisitions table that has deal dates, redefine the target as “acquired within five years,” and censor anyone who has not reached that age yet.

That same clock problem shows up in the split. Rates drift down from 16.0% in train to 13.6% in validation to 9.7% in test (2.7), and `company_age` was learned on ages 9–14 then applied to 6–7. On test the mean predicted probability runs a bit **low** (0.086 vs 0.097) because the age coefficient over-corrects when extrapolated — setting every test age to 9 lifts the mean prediction to 0.135, above the actual rate (5.8). Next time I would score within age strata, or use a landmark / time-varying setup so the model never has to invent an age effect outside the range it saw.

Even for companies we do label, the funding features are not cleanly “known at pitch time.” Totals may include rounds raised after the acquisition (1.9). The date-feature check did not find a clear leak from funding span, but that does not fix the amounts themselves. The honest upgrade is round-level data truncated at a hypothetical decision date (or at the acquisition date for positives), so every input would have been available when the boutique chooses whom to approach.

On the modeling side, the design matrix is a mess of near-duplicates — condition number of `X′X/n` around 7,300, with `has_angel` and `log_angel` at r ≈ 0.996 — which is why individual coefficients are not worth reading and why plain gradient descent needed 100,000 steps to match sklearn (5.7). Logistic regression also cannot pick up interactions like hub × sector unless I build them by hand. I would drop or merge the redundant funding encodings, then try a small tree or gradient-boosted model (or a few explicit interactions) and re-check calibration and precision@5%.

A couple of evaluation caveats sit on top of that. Validation is thin: one extra hit moves precision@5% by about 0.011, and the date-feature gain was inside noise (−0.005 to +0.017). The euro figures also lean on assumed pitch cost, fee, and a 25% win rate — they score choices of *k*, they do not invent the ranking. I would bootstrap the ranking metrics and, for a real boutique, plug in their actual costs (or leave cost as a sensitivity slider on the dashboard).

Finally, the export itself is old and imperfect. It is a 2014 Crunchbase snapshot, so the funding world has moved on; in train, about a hundred companies were funded “before founding” (clipped to 0), 22% of raw rows had no founding year (dropped), and markets were hand-mapped into sectors. Refreshing from a current dump with cleaner dates, and replacing the hand sector map with something fit on training only (supervised or embeddings), would be the obvious next data pass.

## Generative AI use disclosure

I used **Claude** (via Cursor) heavily on the code side: it wrote most of `train.py` — exploration, preprocessing, feature engineering, the three logistic-regression implementations, evaluation, and saving artifacts — and it proposed design options and reviewed drafts of this report against the pipeline. **Cursor** is listed as a co-author on the REPORT.md commits in git.

What I did myself was the substance of the assignment decisions: which dataset and business framing to use, how to define the target and founding cohorts, the cost assumptions and top-5% rule, the hand sector mapping, how to treat missing funding, and the date-feature keep/drop rule (including asking for the validation bootstrap). I also wrote and edited this REPORT.md so that every claim is something I can explain without the code in front of me. Understating that split would be worse than disclosing the AI help.
