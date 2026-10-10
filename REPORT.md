# Assignment 1 Report

*Delete this italic guidance as you fill in each section. You'll be asked to
defend any of this without your code in front of you — write only what you
can actually explain.*

- **Name**: Anna Serra
- **Student ID**: 18815
- **Email**: [aserrag.ieu2022@student.ie.edu](mailto:aserrag.ieu2022@student.ie.edu)
- **Group**: BBADBA 5B

## Dataset

My dataset is [StartUp Investments (Crunchbase)](https://www.kaggle.com/datasets/arindam235/startup-investments-crunchbase/data) from Kaggle. It is a 2014 snapshot of venture-backed companies scraped from Crunchbase and it has 54k rows and 39 columns, where each row represents one startup.

It's columns cover identity and location (`name`, `market`, `country_code`, …), founding and funding dates (`founded_year`, `first_funding_at`, `last_funding_at`), amounts by round type (`seed`, `venture`, `round_A`–`round_H`, `funding_total_usd`, …), and a categorical `status` (`acquired` / `operating` / `closed`).

I picked it because the business question, which is who is likely to be acquired, maps cleanly onto `status`, and because the columns leave room for non-trivial feature engineering, and founding/funding dates force a time-aware split instead of a lazy random one. The raw file is large for the assignment’s “commit and retrain” guide, so the modeling cohort is restricted, rather than shipping every row.

## Business / real-life framing

This is an M&A advisory boutique with limited banker time that has to decide which venture-backed startups to approach for a sell-side mandate. A model will score how likely a startup is to be acquired, relative to peers of similar age, from its funding profile. Bankers use that score to prioritize outreach; they review the list often (weekly or when capacity frees up), not once a year.

My target is the status variable, where `acquired = 1`, and `operating` or `closed = 0`. There is no acquisition date, so a “acquired within X years” label cannot be defined — only the 2014 status snapshot. Raw acquisition rates also collapse with youth (about 22% for 1999 founders vs 0.6% for 2013) because younger companies simply have not had time to exit. To make the label usable I kept companies founded from 2000 to 2008 (at least six years of exposure by the snapshot) and I added `company_age` as an exposure control so the model does not treat “young” as “never acquired.”

For the split, I will train on the founding cohorts of 2000–2005, I will validate with those of 2006, and I will test with 2007–2008. Scoring newer startups from what happened to older ones matches how the boutique would actually use the model. A random split would mix cohorts, let the model see companies founded later than those it is meant to rank, and hide the real drift in acquisition rates across the split (~16% → ~13.6% → ~9.7%). The time split surfaces that honestly. And with that I am accepting some trade-offs, like the test cohorts have had less calendar time (probabilities will run high), test ages (6–7) sit outside the training age range (9–14), and funding totals may include rounds raised after an acquisition.

I estimated the costs of the thresholds to be the following:


| Error          | What happens                                                | Illustrative cost                                                                                                                                  |
| -------------- | ----------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| False positive | Pitch a company that is never acquired                      | €3,000 roughly a day of associate research + deck prep                                                                                             |
| False negative | Miss a company that does get acquired (without our mandate) | €150,000 a modest sell-side success fee (on the order of 1–2% of a ~€10M deal; real boutique fees often sit in this ballpark for smaller VC exits) |


Y ≫ X (~50×), so missing a true acquirer hurts far more than wasting a pitch. That pushes the operating point toward higher recall (accept more FPs to catch more acquirers). The firm also has a hard capacity constraint, say ~200 pitches per year, so the practical rule is not an arbitrary probability cutoff but pitch the top-k scores (k ≈ annual capacity). The dashboard metric I optimize is therefore expected business cost at a chosen threshold / top-k:

`cost = (# FP) × €3,000 + (# FN) × €150,000`

with the default threshold set so that the selected set size matches capacity while that cost (and recall among true acquirers) stays acceptable. Precision in the top-k is the companion view: of the 200 we can afford to pitch, how many are real acquirers?

## Data preparation & feature engineering

*What you engineered and why, and any data-quality decisions you made along
the way — e.g. "segment X had defective data, so I excluded it and used a
population-average default for scope Y at inference time; the impact of
that choice is Z."*

## Modeling: three implementations, one model

*Which model (linear or logistic regression) and why. A results table
comparing scikit-learn, the manual PyTorch loop, and the standard
torch.nn.Module/torch.optim workflow, on the same test set, against the
naive baseline. Do the three agree? If not, why not?*

## Limitations & next steps

*Real limitations you found, and concretely how you'd address each one with
more time or data — not generic hedging.*

## Generative AI use disclosure

