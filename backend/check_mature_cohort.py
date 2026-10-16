
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
FILE = BASE_DIR / "data" / "cleaned_sample.csv"

df = pd.read_csv(FILE, low_memory=False)

df["issue_d"] = pd.to_datetime(
    df["issue_d"], errors="coerce"
)

# Convert terms such as "36 months" to integer month counts.
df["term_months"] = pd.to_numeric(
    df["term"].astype("string").str.extract(r"(\d+)")[0],
    errors="coerce",
)

# The original dataset runs through December 2018.
snapshot_date = pd.Timestamp("2018-12-31")

# Conservative scheduled-maturity rule:
# loan term + 6-month outcome buffer.
df["expected_maturity_date"] = (
    df["issue_d"]
    + pd.to_datetime(df["issue_d"].dt.strftime("%Y-%m-01"))
      .map(lambda d: pd.DateOffset(months=0))
      if False else df["issue_d"]
)

# Calculate maturity dates row by row.
df["expected_maturity_date"] = [
    date + pd.DateOffset(months=int(months) + 6)
    if pd.notna(date) and pd.notna(months)
    else pd.NaT
    for date, months in zip(df["issue_d"], df["term_months"])
]

mature = df[
    df["expected_maturity_date"] <= snapshot_date
].copy()

print("\n--- MATURITY CHECK ---")
print("All cleaned loans:", len(df))
print("Loans meeting maturity rule:", len(mature))
print("Loans excluded as not yet mature:", len(df) - len(mature))

print("\nMature loans by issue year:")
print(
    mature.groupby(mature["issue_d"].dt.year)["default"]
    .agg(["count", "mean"])
    .rename(columns={"count": "loan_count", "mean": "default_rate"})
    .assign(default_rate_percent=lambda x: x["default_rate"] * 100)
    .drop(columns="default_rate")
    .round(2)
    .to_string()
)

output = BASE_DIR / "data" / "mature_loan_sample.csv"
mature.drop(
    columns=["term_months", "expected_maturity_date"]
).to_csv(output, index=False)

print(f"\nSaved mature cohort to: {output}")
print("\nMaturity check completed.")
