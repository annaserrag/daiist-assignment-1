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

For the split, I will train on the founding cohorts of 2000–2005, validate on 2006, and test on 2007–2008 (~4,598 startups). Scoring newer startups from what happened to older ones matches how the boutique would actually use the model. A random split would mix cohorts, let the model see companies founded later than those it is meant to rank, and hide the real drift in acquisition rates across the split (~16% → ~13.6% → ~9.7%). The time split surfaces that honestly. Trade-offs I accept: test cohorts have had less calendar time (predicted probabilities will run high), test ages (6–7) sit outside the training age range (9–14), and funding totals may include rounds raised after an acquisition.

I estimated the costs of errors as follows:


| Error          | What happens                                                | Illustrative cost |
| -------------- | ----------------------------------------------------------- | ----------------- |
| False positive | Pitch a company that is never acquired                      | **€3,000** — roughly a day of associate research + deck prep |
| False negative | Miss a company that does get acquired (without our mandate) | **€37,500** expected — a €150,000 sell-side success fee (about 1–2% of a ~€10M deal) × a **25% win-rate** assumption. Missing an acquirer only costs the fee if we would have pitched *and* won; not every seller hires an adviser, and we would not win every pitch. |


FN cost ≫ FP cost (~12.5×), so on pure expected-cost grounds we would want high recall. The unconstrained cost-minimising probability threshold is about `FP / (FP + FN) ≈ 3,000 / 40,500 ≈ 0.07`, which would mean pitching almost every startup — far beyond what a small boutique can do. **Capacity is therefore the binding constraint; the euro costs score each choice of *k*, they do not set *k*.**

In production the firm can pitch on the order of **~200 companies a year**. The test set is two founding cohorts (~4,598 startups), not one year of deal flow, so I translate that capacity into a **fixed share: the top 5%** of scores on whatever set is being ranked (~230 names on the test set; about 200 on a 4,000-company annual list). The operating rule is **pitch the top-*k***, not an absolute probability cutoff. That choice also addresses the calibration problem named above: because test probabilities run high from shorter exposure, a fixed probability threshold would be misleading, whereas top-*k* depends only on ranking and is unaffected by that level shift.

The dashboard metric is expected business cost over the ranked list at a chosen *k* (default *k* = top 5%):

`cost = (# FP among pitched) × €3,000 + (# FN among not pitched) × €37,500`

Precision@*k* is the companion view: of the names we can afford to pitch, how many are real acquirers?

## Data preparation & feature engineering

Every cleaning and feature choice below rests on a finding from exploration on the **training cohorts only** (founded 2000–2005), so validation and test never drove the design.

**Cleaning.** The file is latin-1 with a few stray spaces in headers. I dropped the 4,856 completely empty rows, parsed `funding_total_usd` from text (Indian digit grouping and `-` for missing → NaN), parsed founding/funding dates with impossible values coerced to missing, stripped `market`, dropped rows without `status` or `founded_year`, and deduplicated on `permalink` (names can collide; the permalink is the id). That leaves the 12,214 companies in the 2000–2008 modeling window.

**Target, exposure, split.** As in the framing: `acquired = 1` vs operating/closed = 0; `company_age = 2014 − founded_year` as an exposure control so “young” is not read as “never acquired”; train 2000–2005 / val 2006 / test 2007–2008. Age goes into the model as a control rather than ranking within age strata — the point is to keep scores comparable when test ages (6–7) sit outside the training range (9–14).

**Funding profile (the core signal).** A missing `funding_total_usd` always came with zeros in every round column and a lower acquisition rate, so missing means “nothing disclosed,” not a random hole: I kept those rows, added a `funding_disclosed` flag, and filled the total with 0. Amounts span orders of magnitude with heavy right skew, so size enters as `log1p` of total, seed, angel, venture, and debt. Mix is `share_venture`, `share_debt_financing`, and `share_seed` of typed funding. Lettered rounds are nested (almost every C had a B), so I keep `max_round` (A=1 … H=8, else 0) instead of eight separate dummies, plus flags for seed/angel/venture/debt/grant, a count of distinct round types, and `funding_rounds`. Early pace is years from founding to first funding (negatives clipped to 0 — data-entry or pre-incorporation money — rather than dropping the company) and `log_funding_per_round`. **Post-IPO equity/debt are excluded from every funding feature:** that money only exists after an IPO, which is itself an exit.

**Sector and geography (context bankers already use).** The score is driven by the funding profile; sector and location are controls, not the product. Hundreds of sparse `market` values cannot be one-hot encoded, so I hand-mapped the common markets into nine fixed sectors (`software_it`, `health_life_sciences`, …); everything else, including missing, is `other` (reference level, no column). For geography: `is_usa`, one column each for the five largest non-US countries **chosen on the training rows only** (GBR, CHN, CAN, FRA, ISR), `is_hub` for SF Bay Area / Boston / New York City / London, and `n_categories` from `category_list`. I dropped `state_code` (~⅓ missing).

**Date features and the final set.** I also built `funding_span_years` and `rounds_per_year`, but they can leak if acquired companies stop raising after the deal (plausible from exploration, not proven by a shorter span alone). On validation, adding them lifted ROC-AUC by only **+0.006**. A paired bootstrap (1,000 redraws of the validation set) put the 95% interval of that gain at **[−0.005, +0.017]**, which includes zero — indistinguishable from noise. Rule in code: keep the date features only if the whole interval sits above zero; otherwise prefer the safer set. **The three final models therefore use the 37 features without the date features.**

## Modeling: three implementations, one model

*Which model (linear or logistic regression) and why. A results table
comparing scikit-learn, the manual PyTorch loop, and the standard
torch.nn.Module/torch.optim workflow, on the same test set, against the
naive baseline. Do the three agree? If not, why not?*

## Limitations & next steps

*Real limitations you found, and concretely how you'd address each one with
more time or data — not generic hedging.*

## Generative AI use disclosure

