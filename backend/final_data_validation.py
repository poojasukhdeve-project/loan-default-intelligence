
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
FILE = BASE_DIR / "data" / "cleaned_sample.csv"

df = pd.read_csv(FILE, low_memory=False)

print("\n========== FINAL DATA VALIDATION ==========")

# 1. Dataset dimensions
print("\n1. Dataset shape:", df.shape)

# 2. Target validation
print("\n2. Target distribution:")
print(df["default"].value_counts().sort_index())

assert df["default"].notna().all(), "Target contains missing values."
assert set(df["default"].unique()).issubset({0, 1}), (
    "Target contains values other than 0 and 1."
)

# 3. Invalid values after cleaning
print("\n3. Invalid-value checks:")

if "annual_inc" in df.columns:
    print("Annual income <= 0:", (df["annual_inc"] <= 0).sum())

if "dti" in df.columns:
    print("DTI below 0:", (df["dti"] < 0).sum())
    print("DTI >= 100:", (df["dti"] >= 100).sum())

if "revol_util" in df.columns:
    print("Utilization below 0:", (df["revol_util"] < 0).sum())
    print("Utilization above 100:", (df["revol_util"] > 100).sum())

# 4. Remaining missing values
print("\n4. Missing values:")
print(df.isna().sum()[df.isna().sum() > 0].sort_values(ascending=False))

# 5. Date validation
if "issue_d" in df.columns:
    dates = pd.to_datetime(df["issue_d"], errors="coerce")
    print("\n5. Issue date range:")
    print("Earliest:", dates.min())
    print("Latest:", dates.max())
    print("Missing/unparseable dates:", dates.isna().sum())

# 6. Duplicate check
print("\n6. Exact duplicate rows:", df.duplicated().sum())

print("\n========== VALIDATION COMPLETED ==========")
