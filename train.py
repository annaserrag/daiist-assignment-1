"""
Assignment 1 — training pipeline.

Business question (see REPORT.md): an M&A advisory boutique ranks venture-backed
startups by how likely they are to be acquired, relative to peers of the same age,
from their funding profile, and pitches the top 5% of the list.

Run with `uv run python main.py train`. The script is meant to be read top to bottom:

    Part 1  Data exploration      what the raw data looks like and how variables behave
    Part 2  Preprocessing         cleaning, target, cohorts and split, each step
                                  justified by a finding in Part 1
    Part 3  Feature engineering   features derived from the Part 1 findings
    Part 4  Feature checks        sanity checks and the date-feature comparison

Section numbers in comments (e.g. "see 1.6") point back to the finding a step rests on.
"""

# %% Setup
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

pd.set_option("display.width", 120)

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "investments_VC.csv"

SNAPSHOT_YEAR = 2014           # last funding dates end on 2015-01-01 (see 1.5)
COHORTS = range(2000, 2009)    # founded 2000-2008 (see 1.6)
TRAIN_YEARS = range(2000, 2006)
VAL_YEARS = [2006]
TEST_YEARS = [2007, 2008]
TOP_SHARE = 0.05               # the boutique pitches the top 5% of the ranked list

FUNDING_TYPES = [
    "seed", "angel", "venture", "debt_financing", "convertible_note", "grant",
    "equity_crowdfunding", "product_crowdfunding", "private_equity", "undisclosed",
    "secondary_market",
]
LETTERED_ROUNDS = ["round_A", "round_B", "round_C", "round_D",
                   "round_E", "round_F", "round_G", "round_H"]
POST_IPO = ["post_ipo_equity", "post_ipo_debt"]
LINE = "=" * 78


# =================================================================================
# PART 1 — DATA EXPLORATION
# =================================================================================

# %% 1.1 Raw structure
# The export is latin-1, not UTF-8 (reading it as UTF-8 fails), and a few column
# headers carry stray spaces.
raw = pd.read_csv(DATA_PATH, encoding="latin-1")
raw.columns = raw.columns.str.strip()

print(f"\n{LINE}\n1.1 Raw structure\n{LINE}")
print(f"rows x columns: {raw.shape[0]:,} x {raw.shape[1]}")
print(f"completely empty rows: {raw.isna().all(axis=1).sum():,}")
print("\ncolumn types:")
print(raw.dtypes.value_counts().to_string())
print("\ncolumns:", ", ".join(raw.columns))

# Everything below looks at the non-empty rows only.
explore = raw.dropna(how="all")

# %% 1.2 The target: status
print(f"\n{LINE}\n1.2 The target variable: status\n{LINE}")
print(explore["status"].value_counts(dropna=False).to_string())
print("\ncolumns that could date an acquisition:",
      [c for c in explore.columns if "acqui" in c.lower()] or "none")
print("-> status is a single snapshot: we know THAT a company was acquired, not WHEN.")

# %% 1.3 Missing values
print(f"\n{LINE}\n1.3 Missing values (% of non-empty rows)\n{LINE}")
missing = (explore.isna().mean() * 100).round(1)
print(missing[missing > 0].sort_values(ascending=False).to_string())

# %% 1.4 How funding is stored
print(f"\n{LINE}\n1.4 How funding_total_usd is stored\n{LINE}")
print("sample values:", explore["funding_total_usd"].head(5).tolist())
print(f"values that are '-': {(explore['funding_total_usd'].str.strip() == '-').sum():,}")
print("-> text with Indian digit grouping (' 17,50,000 ' = 1,750,000) and '-' for missing.")
print("   The round columns (seed, venture, round_A, ...) are already numeric:")
print(explore[["seed", "venture", "round_A"]].describe().loc[["min", "50%", "max"]].to_string())

# A parsed copy, used only to look at the data in this part (Part 2 does it for real).
funding_peek = pd.to_numeric(
    explore["funding_total_usd"].str.strip().str.replace(",", "").replace("-", np.nan),
    errors="coerce",
)

# %% 1.5 Dates
print(f"\n{LINE}\n1.5 Dates\n{LINE}")
dates_peek = {c: pd.to_datetime(explore[c], errors="coerce")
              for c in ("founded_at", "first_funding_at", "last_funding_at")}
print("earliest first_funding_at (raw text):", explore["first_funding_at"].min())
print("latest last_funding_at:", explore["last_funding_at"].max(),
      f"-> the snapshot was taken around {SNAPSHOT_YEAR}")
print("founded_year range:", int(explore["founded_year"].min()), "-",
      int(explore["founded_year"].max()))

# %% 1.6 Acquisition rate by founding year
acquired_peek = (explore["status"] == "acquired").astype(int)

print(f"\n{LINE}\n1.6 Acquisition rate by founding year\n{LINE}")
by_year = (
    pd.DataFrame({"founded_year": explore["founded_year"], "acquired": acquired_peek,
                  "closed": (explore["status"] == "closed").astype(int)})
    .query("1995 <= founded_year <= @SNAPSHOT_YEAR")
    .groupby("founded_year")
    .agg(startups=("acquired", "size"), acquired_rate=("acquired", "mean"),
         closed_rate=("closed", "mean"))
)
by_year.index = by_year.index.astype(int)
print(by_year.round(3).to_string())
print("-> the rate collapses for young cohorts: they have simply had less time to be")
print("   acquired (right-censoring). A label that mixes all cohorts would mostly")
print("   teach the model to recognise young companies.")

# %% 1.7 From here on, look only at the labelled training cohorts (founded 2000-2005,
# status known), so no decision below is informed by the validation or test companies.
in_train = explore["founded_year"].isin(TRAIN_YEARS) & explore["status"].notna()
tr = explore[in_train]
tr_acquired = acquired_peek[in_train]
tr_funding = funding_peek[in_train]
base_rate = tr_acquired.mean()

print(f"\n{LINE}\n1.7 Funding amounts (training cohorts 2000-2005)\n{LINE}")
tr_missing = tr_funding.isna()
print(f"companies: {len(tr):,} | acquired rate: {base_rate:.1%}")
print(f"funding_total_usd missing: {tr_missing.sum():,} ({tr_missing.mean():.1%})")
print(f"acquired rate | missing: {tr_acquired[tr_missing].mean():.1%}"
      f" | present: {tr_acquired[~tr_missing].mean():.1%}")
print("missing totals with an amount in any round column:",
      int((tr.loc[tr_missing, FUNDING_TYPES + LETTERED_ROUNDS].sum(axis=1) > 0).sum()))
print("-> a missing total means 'nothing disclosed' and is informative, not random.\n")

amounts = pd.concat([tr_funding.rename("funding_total_usd"),
                     tr[["seed", "angel", "venture", "debt_financing"]]], axis=1)
print(pd.DataFrame({
    "share > 0": (amounts > 0).mean(),
    "median": amounts.median(),
    "mean": amounts.mean(),
    "max": amounts.max(),
    "skew": amounts.skew(),
}).to_string(float_format="{:,.2f}".format))
print("-> amounts span several orders of magnitude with heavy right skew and many zeros.")

# %% 1.8 Funding rounds
print(f"\n{LINE}\n1.8 Funding round types and stages (training cohorts)\n{LINE}")
present = tr[FUNDING_TYPES + LETTERED_ROUNDS].gt(0)
print(pd.DataFrame({
    "share of companies": present.mean(),
    "acquired rate if present": [tr_acquired[present[c]].mean() if present[c].any()
                                 else np.nan for c in present.columns],
}).round(3).to_string())
print(f"\noverall acquired rate: {base_rate:.1%}")
has_c = present["round_C"]
print(f"companies with a round C that also had a round B: "
      f"{present.loc[has_c, 'round_B'].mean():.0%}")
print("funding_rounds:", tr["funding_rounds"].describe()[["min", "50%", "max"]].to_dict())
print("-> lettered rounds are largely nested (C usually follows B): the furthest stage")
print("   reached says more than each round separately.")

# %% 1.9 Leakage checks
print(f"\n{LINE}\n1.9 Leakage checks (training cohorts)\n{LINE}")
post_ipo = tr[POST_IPO].gt(0).any(axis=1)
print(f"companies with post-IPO funding: {post_ipo.sum()}"
      f" (acquired rate {tr_acquired[post_ipo].mean():.1%})")
print("-> post-IPO money only exists after an IPO, which is itself an exit outcome.")
span_peek = (dates_peek["last_funding_at"] - dates_peek["first_funding_at"]).dt.days[in_train] / 365.25
print("\nfunding span (last - first funding, years) by acquired:")
print(span_peek.groupby(tr_acquired).describe()[["mean", "50%", "max"]].round(2).to_string())
print("-> if acquired companies stopped raising after the deal, their span would be")
print("   shorter. It is not, so the leak is plausible but unproven: Part 4 tests it.")

# %% 1.10 Founding-to-first-funding gap
print(f"\n{LINE}\n1.10 Years from founding to first funding (training cohorts)\n{LINE}")
gap_peek = (dates_peek["first_funding_at"] - dates_peek["founded_at"]).dt.days[in_train] / 365.25
print(gap_peek.describe()[["min", "25%", "50%", "75%", "max"]].round(2).to_string())
print(f"negative gaps (funded before founded): {(gap_peek < 0).sum()},"
      f" down to {gap_peek.min():.1f} years")
print("median gap by acquired:", gap_peek.clip(lower=0).groupby(tr_acquired).median().round(2).to_dict())
print("-> negatives are data-entry errors or pre-incorporation money; acquired companies")
print("   tend to raise sooner after founding.")

# %% 1.11 Market
print(f"\n{LINE}\n1.11 Market (training cohorts)\n{LINE}")
market_counts = tr["market"].str.strip().value_counts()
print(f"distinct markets: {len(market_counts)} | missing: {tr['market'].isna().sum()}")
print(f"top 10 cover {market_counts.head(10).sum() / len(tr):.0%},"
      f" top 60 cover {market_counts.head(60).sum() / len(tr):.0%},"
      f" markets with < 10 companies: {(market_counts < 10).sum()}")
print("\nthe 30 largest markets:")
print(market_counts.head(30).to_string())
print("-> far too many sparse categories to one-hot encode directly.")

# %% 1.12 Geography and breadth
print(f"\n{LINE}\n1.12 Geography and breadth (training cohorts)\n{LINE}")
by_country = (pd.DataFrame({"country": tr["country_code"], "acquired": tr_acquired})
              .groupby("country")["acquired"].agg(companies="size", acquired_rate="mean")
              .sort_values("companies", ascending=False))
print(f"countries: {len(by_country)}")
print(by_country.head(8).round(3).to_string())
print("\nlargest regions:")
by_region = (pd.DataFrame({"region": tr["region"], "acquired": tr_acquired})
             .groupby("region")["acquired"].agg(companies="size", acquired_rate="mean")
             .sort_values("companies", ascending=False))
print(by_region.head(8).round(3).to_string())
print(f"\nstate_code missing for {tr['state_code'].isna().mean():.0%} of rows")
tags_peek = tr["category_list"].fillna("").str.count(r"[^|]+")  # "|A|B|" -> 2 tags
print("category tags per company:", tags_peek.describe()[["mean", "50%", "max"]].round(2).to_dict())
print("-> the US dominates and a handful of startup hubs stand out; the long tail of")
print("   countries is too thin to model one by one.")

# %% 1.13 Duplicates
print(f"\n{LINE}\n1.13 Duplicates\n{LINE}")
print("duplicated permalinks:", explore["permalink"].duplicated().sum(),
      "| duplicated names:", explore["name"].duplicated().sum(),
      "(different companies can share a name; the permalink is the identifier)")


# =================================================================================
# PART 2 — PREPROCESSING
# =================================================================================
print(f"\n{LINE}\nPART 2 — Preprocessing\n{LINE}")

# %% 2.1 Drop the completely empty rows (1.1).
df = raw.dropna(how="all").copy()
print(f"non-empty rows: {len(df):,}")

# %% 2.2 Parse funding_total_usd from text (1.4). A '-' becomes NaN, kept for now.
df["funding_total_usd"] = pd.to_numeric(
    df["funding_total_usd"].str.strip().str.replace(",", "").replace("-", np.nan),
    errors="coerce",
)

# %% 2.3 Parse dates (1.5). errors="coerce" turns impossible dates (year 0001) into NaT.
for col in ("founded_at", "first_funding_at", "last_funding_at"):
    df[col] = pd.to_datetime(df[col], errors="coerce")
df["market"] = df["market"].str.strip()

# %% 2.4 Drop rows that cannot be labelled (no status, 1.2) or placed in a founding
# cohort (no founded_year, 1.3), and duplicate permalinks (1.13).
df = df.dropna(subset=["status", "founded_year"])
df["founded_year"] = df["founded_year"].astype(int)
df = df.drop_duplicates(subset="permalink")
print(f"after dropping missing status / founded_year and duplicates: {len(df):,}")

# %% 2.5 Target: acquired = 1, operating or closed = 0 (1.2).
df["acquired"] = (df["status"] == "acquired").astype(int)

# %% 2.6 Keep founding cohorts 2000-2008 only (1.6): every company has had at least
# six years to be acquired by the snapshot, so the label means roughly the same thing
# for all of them. company_age records how long each one has had (exposure control).
df = df[df["founded_year"].isin(COHORTS)].reset_index(drop=True)
df["company_age"] = SNAPSHOT_YEAR - df["founded_year"]
print(f"founding cohorts {COHORTS.start}-{COHORTS.stop - 1}: {len(df):,}")

# %% 2.7 Time-based split by founding cohort (1.6): score newer startups using what
# happened to older ones. A random split would mix cohorts and hide the drift.
is_train = df["founded_year"].isin(TRAIN_YEARS)
is_val = df["founded_year"].isin(VAL_YEARS)
is_test = df["founded_year"].isin(TEST_YEARS)

print(f"\n{'split':<7}{'founded':<11}{'rows':>7}{'acquired':>10}{'rate':>8}{'age':>9}")
for name, mask in (("train", is_train), ("val", is_val), ("test", is_test)):
    part = df[mask]
    print(f"{name:<7}{f'{part.founded_year.min()}-{part.founded_year.max()}':<11}"
          f"{len(part):>7,}{part.acquired.sum():>10,}{part.acquired.mean():>8.1%}"
          f"{part.company_age.min():>6}-{part.company_age.max()}")
print("the rate drifts down with less exposure: probabilities will run high on test,")
print("while the top-k ranking rule is unaffected.")


# =================================================================================
# PART 3 — FEATURE ENGINEERING
# =================================================================================
print(f"\n{LINE}\nPART 3 — Feature engineering\n{LINE}")

# Features are computed row by row, so building them on all cohorts at once is safe.
# The only thing learned from data (which countries get their own column, 3.6) is
# taken from the training rows.
features = pd.DataFrame(index=df.index)

# %% 3.1 Exposure control (1.6, 2.6).
features["company_age"] = df["company_age"]

# %% 3.2 Funding size and mix (1.7). A missing total always comes with no amount in
# any round column and lower acquisition rates: keep the rows, flag them, fill with 0.
# Amounts are heavily skewed with many zeros, so they enter as log1p(amount).
features["funding_disclosed"] = df["funding_total_usd"].notna().astype(int)
funding_total = df["funding_total_usd"].fillna(0.0)
features["log_funding_total"] = np.log1p(funding_total)
for col in ("seed", "angel", "venture", "debt_financing"):
    features[f"log_{col}"] = np.log1p(df[col])

# The mix: what share of the typed funding came from each source. post_ipo_* are left
# out of every funding feature (1.9).
typed_total = df[FUNDING_TYPES].sum(axis=1)
for col in ("venture", "debt_financing", "seed"):
    features[f"share_{col}"] = np.where(typed_total > 0, df[col] / typed_total.where(typed_total > 0, 1), 0.0)

# %% 3.3 Stage reached (1.8). Lettered rounds are nested, so keep the furthest one:
# A = 1 ... H = 8, none = 0. Round types become flags and a count of distinct types.
lettered = df[LETTERED_ROUNDS].gt(0).to_numpy()
features["max_round"] = np.where(
    lettered.any(axis=1), len(LETTERED_ROUNDS) - np.argmax(lettered[:, ::-1], axis=1), 0)
for col, name in (("seed", "has_seed"), ("angel", "has_angel"), ("venture", "has_venture"),
                  ("debt_financing", "has_debt"), ("grant", "has_grant")):
    features[name] = (df[col] > 0).astype(int)
features["n_round_types"] = df[FUNDING_TYPES].gt(0).sum(axis=1)
features["funding_rounds"] = df["funding_rounds"]

# %% 3.4 Early pace (1.10). Negative gaps are clipped to 0 ("funded at founding")
# rather than dropping the companies.
years_to_first = (df["first_funding_at"] - df["founded_at"]).dt.days / 365.25
features["years_to_first_funding"] = years_to_first.clip(lower=0)
features["log_funding_per_round"] = np.log1p(funding_total / df["funding_rounds"].clip(lower=1))

# %% 3.5 Sector (1.11). The most frequent markets are mapped by hand into nine broad
# sectors; anything else, and missing markets, is "other" (the reference level, so
# it gets no column of its own). The mapping is fixed in advance, not learned.
SECTORS = {
    "software_it": ["Software", "Enterprise Software", "SaaS", "Cloud Computing",
                    "Analytics", "Security", "Web Hosting", "Web Development",
                    "Information Technology", "Technology", "Internet", "Email"],
    "hardware_semis": ["Hardware + Software", "Hardware", "Semiconductors",
                       "Nanotechnology", "Networking", "Manufacturing"],
    "health_life_sciences": ["Biotechnology", "Health Care", "Health and Wellness",
                             "Medical", "Medical Devices", "Pharmaceuticals"],
    "mobile_telecom": ["Mobile", "Wireless", "Telecommunications", "Messaging"],
    "consumer_web_media": ["Curated Web", "Social Media", "News", "Search", "Games",
                           "Music", "Video", "Photography", "Media", "Digital Media",
                           "Entertainment"],
    "commerce_marketing": ["E-Commerce", "Advertising", "Fashion", "Public Relations",
                           "Sales and Marketing", "Customer Service"],
    "finance": ["Finance", "Financial Services", "Payments"],
    "cleantech_mobility": ["Clean Technology", "Automotive", "Transportation",
                           "Public Transportation"],
    "services_verticals": ["Consulting", "Education", "Hospitality", "Travel",
                           "Real Estate", "Sports", "Design", "Business Services"],
}
market_to_sector = {market: sector for sector, markets in SECTORS.items() for market in markets}
sector = df["market"].map(market_to_sector).fillna("other")
for name in SECTORS:
    features[f"sector_{name}"] = (sector == name).astype(int)

print("acquired rate by sector (train):")
print(df[is_train].groupby(sector[is_train])["acquired"]
      .agg(companies="size", acquired_rate="mean")
      .sort_values("companies", ascending=False).round(3).to_string())

# %% 3.6 Geography and breadth (1.12). One flag for the US, one per large non-US
# country (taken from the training rows), and one for the four largest startup hubs.
# state_code is not used: it is missing for about a third of companies.
top_countries = (df.loc[is_train & (df["country_code"] != "USA"), "country_code"]
                 .value_counts().head(5).index.tolist())
features["is_usa"] = (df["country_code"] == "USA").astype(int)
for country in top_countries:
    features[f"country_{country}"] = (df["country_code"] == country).astype(int)
hub_regions = ["SF Bay Area", "Boston", "New York City", "London"]
features["is_hub"] = df["region"].isin(hub_regions).astype(int)
features["n_categories"] = df["category_list"].fillna("").str.count(r"[^|]+")
print("\ncountries with their own column (from train):", top_countries)
print("hub regions:", hub_regions)

# %% 3.7 Date features (1.9). Built here, but only used in the variant that tests
# whether they leak the outcome (Part 4).
span = (df["last_funding_at"] - df["first_funding_at"]).dt.days / 365.25
features["funding_span_years"] = span
features["rounds_per_year"] = df["funding_rounds"] / span.clip(lower=1)
DATE_FEATURES = ["funding_span_years", "rounds_per_year"]

features = features.astype(float)
FEATURES_ALL = list(features.columns)
FEATURES_NO_DATES = [c for c in FEATURES_ALL if c not in DATE_FEATURES]
y = df["acquired"].to_numpy()


# =================================================================================
# PART 4 — FEATURE CHECKS
# =================================================================================

# %% 4.1 No missing values, sensible ranges, and how each feature differs by target
print(f"\n{LINE}\n4.1 Engineered features on train: mean by target\n{LINE}")
print(f"{len(FEATURES_ALL)} features | missing values train/val/test:",
      [int(features[m].isna().sum().sum()) for m in (is_train, is_val, is_test)])
train_features = features[is_train]
print(pd.DataFrame({
    "mean": train_features.mean(),
    "min": train_features.min(),
    "max": train_features.max(),
    "mean | acquired": train_features[df.loc[is_train, "acquired"] == 1].mean(),
    "mean | not acquired": train_features[df.loc[is_train, "acquired"] == 0].mean(),
}).round(2).to_string())

# %% 4.2 Do the date features help? Same model with and without them, scored on
# validation; the test set stays untouched until the final comparison.
print(f"\n{LINE}\n4.2 Date features: with vs without (validation)\n{LINE}")
k_val = int(round(TOP_SHARE * is_val.sum()))
variant_scores = {}
print(f"{'variant':<22}{'features':>9}{'val ROC-AUC':>13}{'val P@5%':>10}")
for label, columns in (("without date features", FEATURES_NO_DATES),
                       ("with date features", FEATURES_ALL)):
    scaler = StandardScaler().fit(features.loc[is_train, columns])
    probe = LogisticRegression(max_iter=5000).fit(
        scaler.transform(features.loc[is_train, columns]), y[is_train])
    val_scores = probe.predict_proba(scaler.transform(features.loc[is_val, columns]))[:, 1]
    variant_scores[label] = val_scores
    top = np.argsort(-val_scores)[:k_val]
    print(f"{label:<22}{len(columns):>9}{roc_auc_score(y[is_val], val_scores):>13.3f}"
          f"{y[is_val][top].mean():>10.3f}")
print(f"\nvalidation base rate {y[is_val].mean():.1%}; the top 5% is {k_val} companies,"
      f" so one extra hit moves P@5% by {1 / k_val:.3f}")

# %% 4.3 Pick the feature set for the three final models. A small gain on 1,770
# validation companies can be noise, so measure it: resample the validation set with
# replacement 1,000 times and recompute the ROC-AUC gain each time (a paired
# bootstrap: both variants are scored on the same resampled companies). The date
# features are kept only if the whole 95% interval of the gain is above zero;
# otherwise the safer set wins, because they also carry the leakage risk from 1.9.
print(f"\n{LINE}\n4.3 Choosing the feature set for the final models\n{LINE}")
rng = np.random.default_rng(0)
y_val = y[is_val]
gains = []
for _ in range(1000):
    sample = rng.integers(0, len(y_val), len(y_val))
    if y_val[sample].min() == y_val[sample].max():
        continue  # a resample with a single class has no ROC-AUC
    gains.append(roc_auc_score(y_val[sample], variant_scores["with date features"][sample])
                 - roc_auc_score(y_val[sample], variant_scores["without date features"][sample]))
gain_low, gain_high = np.percentile(gains, [2.5, 97.5])
observed_gain = (roc_auc_score(y_val, variant_scores["with date features"])
                 - roc_auc_score(y_val, variant_scores["without date features"]))
print(f"ROC-AUC gain from the date features: {observed_gain:+.3f}"
      f" (95% bootstrap interval {gain_low:+.3f} to {gain_high:+.3f})")

if gain_low > 0:
    FINAL_FEATURES = FEATURES_ALL
    print("-> the gain is clearly above noise: the final models use the date features.")
else:
    FINAL_FEATURES = FEATURES_NO_DATES
    print("-> the interval includes zero: the gain is indistinguishable from noise, and")
    print("   the date features carry a leakage risk (1.9). The final models use the")
    print("   set without them.")
print(f"final feature set: {len(FINAL_FEATURES)} features")

# Next: train the same logistic regression three ways (scikit-learn, manual PyTorch
# loop, nn.Module + torch.optim), compare against a naive baseline, and save
# everything app.py needs.
