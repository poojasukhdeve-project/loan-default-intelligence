
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
FILE = BASE_DIR / "data" / "cleaned_sample.csv"

df = pd.read_csv(FILE, low_memory=False)

df["issue_d"] = pd.to_datetime(
    df["issue_d"], errors="coerce"
)

# Inspect available loan outcomes in the original dataset.
original_file = BASE_DIR / "data" / "accepted_2007_to_2018Q4.csv.gz"

outcomes = pd.read_csv(
    original_file,
    usecols=["issue_d", "loan_status"],
    low_memory=False,
)

outcomes["issue_d"] = pd.to_datetime(
    outcomes["issue_d"], format="%b-%Y", errors="coerce"
)

print("\n--- ORIGINAL OUTCOMES BY ISSUE YEAR ---")
print(
    pd.crosstab(
        outcomes["issue_d"].dt.year,
        outcomes["loan_status"],
    ).to_string()
)

print("\n--- FINAL DATASET COUNTS BY ISSUE YEAR ---")
print(
    df.groupby(df["issue_d"].dt.year)["default"]
    .agg(["count", "mean"])
    .rename(columns={
        "count": "loan_count",
        "mean": "default_rate",
    })
    .assign(
        default_rate_percent=lambda x: x["default_rate"] * 100
    )
    .drop(columns="default_rate")
    .round(2)
    .to_string()
)

print("\nOutcome maturity inspection completed.")
