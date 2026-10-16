
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "modeling_sample.csv"
OUTPUT_DIR = BASE_DIR / "eda_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid")

print("Loading prepared loan dataset...")
df = pd.read_csv(DATA_PATH, low_memory=False)

print(f"\nDataset shape: {df.shape}")
print(f"Duplicate rows: {df.duplicated().sum()}")

# 1. Target distribution
print("\n===== LOAN OUTCOME DISTRIBUTION =====")
print(df["default"].value_counts())
print((df["default"].value_counts(normalize=True) * 100).round(2))

plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="default")
plt.xticks([0, 1], ["Fully Paid", "Charged Off"])
plt.title("Loan Outcome Distribution")
plt.xlabel("Loan Outcome")
plt.ylabel("Number of Loans")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "01_outcome_distribution.png", dpi=150)
plt.close()

# 2. Missing values
print("\n===== MISSING VALUES =====")
missing = (
    df.isna().mean().mul(100)
    .sort_values(ascending=False)
    .rename("missing_percent")
)
print(missing[missing > 0].round(2).head(20))

# 3. Numerical summary
numeric_cols = [
    "loan_amnt", "annual_inc", "dti",
    "fico_range_low", "fico_range_high",
    "delinq_2yrs", "revol_util", "revol_bal"
]

print("\n===== NUMERICAL SUMMARY =====")
print(df[numeric_cols].describe().round(2).to_string())

# 4. Default rate by DTI band
dti = df["dti"].where(df["dti"].between(0, 100))
dti_band = pd.cut(
    dti,
    bins=[0, 10, 20, 30, 40, 50, 100],
    include_lowest=True
)

dti_rates = df.groupby(dti_band, observed=True)["default"].agg(
    default_rate="mean",
    loan_count="size"
)
dti_rates["default_rate"] *= 100

print("\n===== DEFAULT RATE BY DTI BAND =====")
print(dti_rates.round(2))

plt.figure(figsize=(9, 5))
sns.barplot(
    x=dti_rates.index.astype(str),
    y=dti_rates["default_rate"]
)
plt.title("Charged-Off Rate by Debt-to-Income Ratio")
plt.xlabel("DTI Band")
plt.ylabel("Charged-Off Rate (%)")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "02_default_by_dti.png", dpi=150)
plt.close()

# 5. Default rate by FICO range
fico = df.groupby("fico_range_low", dropna=False)["default"].agg(
    default_rate="mean",
    loan_count="size"
)
fico["default_rate"] *= 100

print("\n===== DEFAULT RATE BY FICO RANGE =====")
print(fico.round(2).to_string())

plt.figure(figsize=(10, 5))
sns.lineplot(
    x=fico.index,
    y=fico["default_rate"],
    marker="o"
)
plt.title("Charged-Off Rate by Lower FICO Score")
plt.xlabel("Lower FICO Score")
plt.ylabel("Charged-Off Rate (%)")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "03_default_by_fico.png", dpi=150)
plt.close()

# 6. Default rate by loan term
term_rates = df.groupby("term", dropna=False)["default"].agg(
    default_rate="mean",
    loan_count="size"
)
term_rates["default_rate"] *= 100

print("\n===== DEFAULT RATE BY LOAN TERM =====")
print(term_rates.round(2))

plt.figure(figsize=(8, 5))
sns.barplot(
    data=df,
    x="term",
    y="default",
    errorbar=None
)
plt.title("Charged-Off Rate by Loan Term")
plt.xlabel("Loan Term")
plt.ylabel("Charged-Off Rate")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "04_default_by_term.png", dpi=150)
plt.close()

print(f"\nEDA complete. Charts saved to: {OUTPUT_DIR}")
