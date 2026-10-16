
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
FILE = BASE_DIR / "data" / "cleaned_sample.csv"

df = pd.read_csv(FILE, low_memory=False)

# Annual income groups
df["income_band"] = pd.cut(
    df["annual_inc"],
    bins=[0, 25000, 50000, 100000, 250000, 1000000, float("inf")],
    labels=[
        "0-25k",
        "25k-50k",
        "50k-100k",
        "100k-250k",
        "250k-1m",
        "Above 1m",
    ],
    include_lowest=True,
)

# Revolving balance groups
df["revol_bal_band"] = pd.cut(
    df["revol_bal"],
    bins=[-1, 0, 10000, 50000, 100000, 500000, float("inf")],
    labels=[
        "Zero",
        "1-10k",
        "10k-50k",
        "50k-100k",
        "100k-500k",
        "Above 500k",
    ],
)

print("\n--- DEFAULT RATE BY ANNUAL INCOME ---")
print(
    df.groupby("income_band", observed=False)["default"]
    .agg(["count", "mean"])
    .assign(default_rate_percent=lambda x: x["mean"] * 100)
    .drop(columns="mean")
    .round(2)
)

print("\n--- DEFAULT RATE BY REVOLVING BALANCE ---")
print(
    df.groupby("revol_bal_band", observed=False)["default"]
    .agg(["count", "mean"])
    .assign(default_rate_percent=lambda x: x["mean"] * 100)
    .drop(columns="mean")
    .round(2)
)

print("\nAnalysis completed.")
