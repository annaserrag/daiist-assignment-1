"""
Assignment 1 — training pipeline.

Business question (see REPORT.md): an M&A advisory boutique ranks venture-backed
startups by how likely they are to be acquired, relative to peers of the same age,
from their funding profile, and pitches the top 5% of the list.

Run with `uv run python main.py train`. The file reads top to bottom: every
cleaning or feature-engineering step is preceded by the exploration output that
motivates it, so the printed log doubles as the evidence for each decision.

    1. Load the raw export and look at its structure
    2. Clean it
    3. Define the target, choose the cohorts, split by time
    4. Explore the variables on the training split
    5. Build the features
    6. Check the features and compare the date-feature variants
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "investments_VC.csv"

SNAPSHOT_YEAR = 2014  # the export's last funding dates end on 2015-01-01
COHORTS = range(2000, 2009)  # founded 2000-2008: every company has had >= 6 years
TRAIN_YEARS = range(2000, 2006)
VAL_YEARS = [2006]
TEST_YEARS = [2007, 2008]
TOP_SHARE = 0.05  # the boutique pitches the top 5% of the ranked list

# Round types the funding mix and stage features are built from. post_ipo_* are left
# out on purpose: they only exist after an IPO, which is itself an outcome (section 4.3).
FUNDING_TYPES = [
    "seed", "angel", "venture", "debt_financing", "convertible_note", "grant",
    "equity_crowdfunding", "product_crowdfunding", "private_equity", "undisclosed",
    "secondary_market",
]
LETTERED_ROUNDS = ["round_A", "round_B", "round_C", "round_D",
                   "round_E", "round_F", "round_G", "round_H"]

# Hand-made mapping of the most frequent markets into broad sectors (section 4.5).
# Anything not listed, and missing markets, falls into "other".
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
MARKET_TO_SECTOR = {m: s for s, markets in SECTORS.items() for m in markets}

HUB_REGIONS = {"SF Bay Area", "Boston", "New York City", "London"}

# Built from the last funding date. Acquired companies stop raising, so these may
# partly encode the outcome; section 6 fits the model with and without them.
DATE_FEATURES = ["funding_span_years", "rounds_per_year"]


def section(title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def acquired_rate(part):
    return f"{part['acquired'].mean():.1%}"


# ---------------------------------------------------------------------------------
# 1. Load the raw export
# ---------------------------------------------------------------------------------

def load_raw(path=DATA_PATH):
    """Read the export as-is. It is latin-1, not UTF-8 (UTF-8 decoding fails)."""
    df = pd.read_csv(path, encoding="latin-1")
    df.columns = df.columns.str.strip()  # some headers carry stray spaces
    return df


def explore_raw(raw):
    section("1. Raw export: structure")
    print(f"rows x columns: {raw.shape[0]:,} x {raw.shape[1]}")
    empty = raw.isna().all(axis=1).sum()
    print(f"completely empty rows: {empty:,}  -> dropped, they carry no information")
    df = raw.dropna(how="all")

    print("\ncolumn types:")
    print(df.dtypes.value_counts().to_string())

    print("\nstatus (the future target):")
    print(df["status"].value_counts(dropna=False).to_string())

    print("\nmissing values, % of non-empty rows:")
    missing = (df.isna().mean() * 100).round(1)
    print(missing[missing > 0].sort_values(ascending=False).to_string())

    print("\nfunding_total_usd is stored as text, with Indian digit grouping and '-':")
    print(df["funding_total_usd"].head(5).tolist(), "...",
          (df["funding_total_usd"].str.strip() == "-").sum(), "values are '-'")

    print("\ndate sanity: earliest first_funding_at =", df["first_funding_at"].min(),
          "| latest last_funding_at =", df["last_funding_at"].max(),
          f"(-> snapshot taken around {SNAPSHOT_YEAR})")
    print("duplicated permalinks:", df["permalink"].duplicated().sum(),
          "| duplicated names:", df["name"].duplicated().sum())
    print("columns that would date an acquisition:",
          [c for c in df.columns if "acqui" in c.lower()] or "none")


# ---------------------------------------------------------------------------------
# 2. Clean
# ---------------------------------------------------------------------------------

def parse_usd(value):
    """'  17,50,000 ' -> 1750000.0; '-' or blank -> NaN."""
    if pd.isna(value):
        return np.nan
    text = str(value).strip().replace(",", "")
    return float(text) if text not in ("", "-") else np.nan


def clean(raw):
    """Parse types; drop rows that cannot be labelled or placed in a cohort; dedupe."""
    df = raw.dropna(how="all").copy()
    df["funding_total_usd"] = df["funding_total_usd"].map(parse_usd)
    for col in ("founded_at", "first_funding_at", "last_funding_at"):
        # errors="coerce" turns impossible dates (e.g. year 0001) into NaT
        df[col] = pd.to_datetime(df[col], errors="coerce")
    df["market"] = df["market"].str.strip()

    df = df.dropna(subset=["status", "founded_year"])  # no label / no cohort
    df["founded_year"] = df["founded_year"].astype(int)
    df = df.drop_duplicates(subset="permalink")
    return df.reset_index(drop=True)


def explore_cleaning(raw, cleaned):
    section("2. After cleaning")
    print(f"non-empty rows: {len(raw.dropna(how='all')):,} -> after cleaning: {len(cleaned):,}")
    print("(dropped: missing status, missing founded_year, duplicate permalinks)")
    print("funding_total_usd parsed: min",
          f"{cleaned['funding_total_usd'].min():,.0f}", "| median",
          f"{cleaned['funding_total_usd'].median():,.0f}", "| max",
          f"{cleaned['funding_total_usd'].max():,.0f}")


# ---------------------------------------------------------------------------------
# 3. Target, cohorts and the time-based split
# ---------------------------------------------------------------------------------

def add_target(df):
    """acquired = 1, operating or closed = 0."""
    df = df.copy()
    df["acquired"] = (df["status"] == "acquired").astype(int)
    return df


def explore_target_by_cohort(df):
    section("3.1 Acquisition rate by founding year")
    print("status is a single snapshot with no acquisition date, so younger cohorts have")
    print("simply had less time to be acquired (right-censoring):\n")
    by_year = (
        df[df["founded_year"].between(1995, SNAPSHOT_YEAR)]
        .groupby("founded_year")
        .agg(startups=("acquired", "size"), acquired_rate=("acquired", "mean"),
             closed_rate=("status", lambda s: (s == "closed").mean()))
    )
    print(by_year.round(3).to_string())
    print("\n-> keep cohorts founded 2000-2008 (>= 6 years of exposure by the snapshot)")
    print("   and add company_age as an exposure control.")


def select_cohorts(df):
    df = df[df["founded_year"].isin(COHORTS)].copy()
    df["company_age"] = SNAPSHOT_YEAR - df["founded_year"]
    return df.reset_index(drop=True)


def time_split(df):
    """Train on older cohorts, validate and test on progressively younger ones."""
    train = df[df["founded_year"].isin(TRAIN_YEARS)].reset_index(drop=True)
    val = df[df["founded_year"].isin(VAL_YEARS)].reset_index(drop=True)
    test = df[df["founded_year"].isin(TEST_YEARS)].reset_index(drop=True)
    return train, val, test


def explore_split(train, val, test):
    section("3.2 Time-based split")
    print(f"{'split':<7}{'founded':<11}{'rows':>7}{'acquired':>10}{'rate':>8}{'age':>9}")
    for name, part in (("train", train), ("val", val), ("test", test)):
        years = part["founded_year"]
        print(f"{name:<7}{f'{years.min()}-{years.max()}':<11}{len(part):>7,}"
              f"{part['acquired'].sum():>10,}{acquired_rate(part):>8}"
              f"{part['company_age'].min():>6}-{part['company_age'].max()}")
    print("\nthe rate drifts down across the split (less exposure for younger cohorts):")
    print("probabilities will run high on test; the top-k ranking rule is unaffected.")


def load_splits(path=DATA_PATH):
    """The whole of sections 1-3 without the exploration output (used by app.py)."""
    return time_split(select_cohorts(add_target(clean(load_raw(path)))))


# ---------------------------------------------------------------------------------
# 4. Explore the variables (training split only, so nothing is learned from val/test)
# ---------------------------------------------------------------------------------

def explore_funding(train):
    section("4.1 Funding amounts: missing totals and skew")
    missing = train["funding_total_usd"].isna()
    print(f"funding_total_usd missing: {missing.sum():,} ({missing.mean():.1%} of train)")
    print(f"acquired rate | missing: {acquired_rate(train[missing])}"
          f" | present: {acquired_rate(train[~missing])}")
    print("rows with a missing total but some amount in a round column:",
          int((train.loc[missing, FUNDING_TYPES + LETTERED_ROUNDS].sum(axis=1) > 0).sum()))
    print("-> a missing total means 'nothing disclosed', and it is informative:")
    print("   keep the rows, fill the total with 0 and add a funding_disclosed flag.\n")

    amounts = train[["funding_total_usd", "seed", "angel", "venture", "debt_financing"]]
    stats = pd.DataFrame({
        "share > 0": (amounts > 0).mean(),
        "median": amounts.median(),
        "mean": amounts.mean(),
        "max": amounts.max(),
        "skew": amounts.skew(),
    })
    print(stats.to_string(float_format=lambda v: f"{v:,.2f}"))
    print("\n-> amounts span several orders of magnitude with heavy right skew:")
    print("   use log1p(amount), which also maps the many zeros to 0.")


def explore_rounds(train):
    section("4.2 Funding rounds: types, stage reached and number of rounds")
    present = train[FUNDING_TYPES + LETTERED_ROUNDS].gt(0)
    rates = pd.DataFrame({
        "share of companies": present.mean(),
        "acquired rate if present": [
            train.loc[present[c], "acquired"].mean() if present[c].any() else np.nan
            for c in present.columns
        ],
    })
    print(rates.round(3).to_string())
    print(f"\noverall acquired rate on train: {acquired_rate(train)}")
    print("funding_rounds:", train["funding_rounds"].describe()[["min", "50%", "max"]].to_dict())
    print("-> lettered rounds are nested (a round_C company usually had A and B), so the")
    print("   stage reached is summarised as max_round (A=1 ... H=8); round types become")
    print("   has_* flags, shares of total, and a count of distinct types.")


def explore_leakage(train):
    section("4.3 Leakage checks")
    post_ipo = (train["post_ipo_equity"] > 0) | (train["post_ipo_debt"] > 0)
    print(f"companies with post-IPO funding: {post_ipo.sum()}"
          f" (acquired rate {acquired_rate(train[post_ipo])})")
    print("-> post_ipo_* only exist after an IPO, an exit outcome: excluded from features.")
    span = (train["last_funding_at"] - train["first_funding_at"]).dt.days / 365.25
    print("\nfunding span (last - first funding), years, by target:")
    print(span.groupby(train["acquired"]).describe()[["mean", "50%", "max"]].round(2).to_string())
    print("-> if acquired companies stopped raising early, their span would be shorter.")
    print("   The leak is plausible but not obvious here: section 6 tests it directly.")


def explore_dates(train):
    section("4.4 Founding-to-first-funding gap")
    gap = (train["first_funding_at"] - train["founded_at"]).dt.days / 365.25
    print("years from founding to first funding:")
    print(gap.describe()[["min", "25%", "50%", "75%", "max"]].round(2).to_string())
    print(f"negative gaps (funded before founded): {(gap < 0).sum()} rows, "
          f"down to {gap.min():.1f} years")
    print("-> negatives are data-entry errors or pre-incorporation money: clip at 0")
    print("   ('funded at founding') rather than drop the companies.")
    print("acquired vs not, median gap (years):",
          gap.clip(lower=0).groupby(train["acquired"]).median().round(2).to_dict())


def explore_market(train):
    section("4.5 Market (sector)")
    counts = train["market"].value_counts()
    print(f"distinct markets: {len(counts)} | missing: {train['market'].isna().sum()}")
    print(f"top 10 cover {counts.head(10).sum() / len(train):.0%}, "
          f"top 60 cover {counts.head(60).sum() / len(train):.0%}")
    print(f"markets with fewer than 10 companies: {(counts < 10).sum()}")
    print("-> far too many sparse categories for one-hot encoding: map them into")
    print("   9 broad sectors plus 'other'. Rates per sector:\n")
    sector = train["market"].map(MARKET_TO_SECTOR).fillna("other")
    print(train.groupby(sector)["acquired"].agg(companies="size", acquired_rate="mean")
          .sort_values("companies", ascending=False).round(3).to_string())


def explore_geography(train):
    section("4.6 Geography and breadth")
    print(f"countries: {train['country_code'].nunique()} | top 7:")
    by_country = train.groupby("country_code")["acquired"].agg(companies="size", acquired_rate="mean")
    print(by_country.sort_values("companies", ascending=False).head(7).round(3).to_string())
    print("-> one is_usa flag plus the 5 biggest non-US countries; the long tail is too thin.")
    hub = train["region"].isin(HUB_REGIONS)
    print(f"\nstartup hubs {sorted(HUB_REGIONS)}: {hub.mean():.1%} of train,"
          f" acquired rate {acquired_rate(train[hub])} vs {acquired_rate(train[~hub])} elsewhere")
    print(f"state_code missing for {train['state_code'].isna().mean():.0%} of rows:"
          " not used; region and country are more complete.")
    n_tags = train["category_list"].fillna("").str.strip("|").str.split("|").map(
        lambda tags: len([t for t in tags if t]))
    print("\nnumber of category tags per company:",
          n_tags.describe()[["mean", "50%", "max"]].round(2).to_dict())


# ---------------------------------------------------------------------------------
# 5. Feature engineering
# ---------------------------------------------------------------------------------

class FeatureBuilder:
    """fit() on the training split, then transform() any split the same way.

    The only thing learned from data is which non-US countries get their own column.
    """

    def __init__(self, n_countries=5, include_date_features=False):
        self.n_countries = n_countries
        self.include_date_features = include_date_features

    def fit(self, train):
        non_us = train.loc[train["country_code"] != "USA", "country_code"]
        self.countries_ = non_us.value_counts().head(self.n_countries).index.tolist()
        self.feature_names_ = list(self.transform(train).columns)
        return self

    def transform(self, df):
        f = pd.DataFrame(index=df.index)

        # Exposure control (3.1): years the company has had to be acquired.
        f["company_age"] = df["company_age"]

        # Funding size & mix (4.1): missing total = nothing disclosed -> flag + 0; log1p.
        f["funding_disclosed"] = df["funding_total_usd"].notna().astype(int)
        total = df["funding_total_usd"].fillna(0.0)
        f["log_funding_total"] = np.log1p(total)
        for col in ("seed", "angel", "venture", "debt_financing"):
            f[f"log_{col}"] = np.log1p(df[col])
        typed = df[FUNDING_TYPES].sum(axis=1)
        for col in ("venture", "debt_financing", "seed"):
            f[f"share_{col}"] = np.where(typed > 0, df[col] / typed.where(typed > 0, 1), 0.0)

        # Stage reached (4.2): highest lettered round with money in it (A=1 ... H=8).
        lettered = df[LETTERED_ROUNDS].gt(0).to_numpy()
        f["max_round"] = np.where(lettered.any(axis=1),
                                  len(LETTERED_ROUNDS) - np.argmax(lettered[:, ::-1], axis=1),
                                  0)
        for col, name in (("seed", "has_seed"), ("angel", "has_angel"),
                          ("venture", "has_venture"), ("debt_financing", "has_debt"),
                          ("grant", "has_grant")):
            f[name] = (df[col] > 0).astype(int)
        f["n_round_types"] = df[FUNDING_TYPES].gt(0).sum(axis=1)
        f["funding_rounds"] = df["funding_rounds"]

        # Early pace (4.4): negative gaps clipped to 0.
        years_to_first = (df["first_funding_at"] - df["founded_at"]).dt.days / 365.25
        f["years_to_first_funding"] = years_to_first.clip(lower=0)
        f["log_funding_per_round"] = np.log1p(total / df["funding_rounds"].clip(lower=1))

        # Geography & breadth (4.6).
        f["is_usa"] = (df["country_code"] == "USA").astype(int)
        for country in self.countries_:
            f[f"country_{country}"] = (df["country_code"] == country).astype(int)
        f["is_hub"] = df["region"].isin(HUB_REGIONS).astype(int)
        f["n_categories"] = (
            df["category_list"].fillna("").str.strip("|").str.split("|")
            .map(lambda tags: len([t for t in tags if t]))
        )

        # Sector one-hot (4.5). "other" is the reference level, so it gets no column.
        sector = df["market"].map(MARKET_TO_SECTOR).fillna("other")
        for name in SECTORS:
            f[f"sector_{name}"] = (sector == name).astype(int)

        # Date features (4.3): only in the variant that tests the leakage question.
        if self.include_date_features:
            span = (df["last_funding_at"] - df["first_funding_at"]).dt.days / 365.25
            f["funding_span_years"] = span
            f["rounds_per_year"] = df["funding_rounds"] / span.clip(lower=1)

        return f.astype(float)


# ---------------------------------------------------------------------------------
# 6. Feature checks and the date-feature comparison
# ---------------------------------------------------------------------------------

def precision_at_top(y, scores, share=TOP_SHARE):
    """Of the top `share` of the ranked list, the fraction that were really acquired."""
    k = max(1, int(round(share * len(y))))
    top = np.argsort(-scores)[:k]
    return y[top].mean()


def explore_features(train, val, test):
    section("6.1 Engineered features on train: missing values and mean by target")
    builder = FeatureBuilder(include_date_features=True).fit(train)
    X_train = builder.transform(train)
    nans = [int(builder.transform(p).isna().sum().sum()) for p in (train, val, test)]
    print(f"{len(builder.feature_names_)} features | missing values train/val/test: {nans}")
    summary = pd.DataFrame({
        "mean": X_train.mean(),
        "min": X_train.min(),
        "max": X_train.max(),
        "mean | acquired": X_train[train["acquired"] == 1].mean(),
        "mean | not acquired": X_train[train["acquired"] == 0].mean(),
    })
    print(summary.round(2).to_string())


def compare_date_feature_variants(train, val):
    section("6.2 Do the date features help? (scored on validation; test stays untouched)")
    print(f"{'variant':<22}{'features':>9}{'val ROC-AUC':>13}{'val P@5%':>10}")
    for include in (False, True):
        builder = FeatureBuilder(include_date_features=include).fit(train)
        scaler = StandardScaler().fit(builder.transform(train))
        model = LogisticRegression(max_iter=5000).fit(
            scaler.transform(builder.transform(train)), train["acquired"])
        scores = model.predict_proba(scaler.transform(builder.transform(val)))[:, 1]
        y = val["acquired"].to_numpy()
        label = "with date features" if include else "without date features"
        print(f"{label:<22}{len(builder.feature_names_):>9}"
              f"{roc_auc_score(y, scores):>13.3f}{precision_at_top(y, scores):>10.3f}")
    k = int(round(TOP_SHARE * len(val)))
    print(f"\nvalidation base rate {acquired_rate(val)}; the top 5% is {k} companies,"
          " so one extra hit moves P@5% by"
          f" {1 / k:.3f}")


if __name__ == "__main__":
    raw = load_raw()
    explore_raw(raw)

    cleaned = clean(raw)
    explore_cleaning(raw, cleaned)

    labelled = add_target(cleaned)
    explore_target_by_cohort(labelled)
    train, val, test = time_split(select_cohorts(labelled))
    explore_split(train, val, test)

    explore_funding(train)
    explore_rounds(train)
    explore_leakage(train)
    explore_dates(train)
    explore_market(train)
    explore_geography(train)

    explore_features(train, val, test)
    compare_date_feature_variants(train, val)

    # Next: train the same logistic regression three ways (scikit-learn, manual
    # PyTorch loop, nn.Module + torch.optim), compare against a naive baseline,
    # and save everything app.py needs.
