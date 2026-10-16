
from pathlib import Path
import pandas as pd
import numpy as np

# --------------------------------------------------
# 1. Set project paths
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "data" / "modeling_sample.csv"
OUTPUT_FILE = BASE_DIR / "data" / "cleaned_sample.csv"
REPORT_DIR = BASE_DIR / "eda_outputs"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading modeling dataset...")
df = pd.read_csv(INPUT_FILE, low_memory=False)

print(f"Original shape: {df.shape}")
print("\nOriginal columns:")
print(df.columns.tolist())

# --------------------------------------------------
# 2. Remove exact duplicate rows
# --------------------------------------------------
duplicate_count = int(df.duplicated().sum())
print(f"\nExact duplicate rows: {duplicate_count}")

if duplicate_count:
    df = df.drop_duplicates().copy()

# --------------------------------------------------
# 3. Validate the target column
# --------------------------------------------------
if "default" not in df.columns:
    raise ValueError("The required 'default' target column is missing.")

print("\nTarget values before cleaning:")
print(df["default"].value_counts(dropna=False))

# Keep only valid binary target values: 0 and 1.
df["default"] = pd.to_numeric(df["default"], errors="coerce")
invalid_target_count = int(
    (~df["default"].isin([0, 1])).sum()
)

print(f"Rows with invalid target values: {invalid_target_count}")

df = df[df["default"].isin([0, 1])].copy()
df["default"] = df["default"].astype("int8")

# --------------------------------------------------
# 4. Clean numeric columns
# --------------------------------------------------
numeric_columns = [
    "loan_amnt",
    "annual_inc",
    "dti",
    "fico_range_low",
    "fico_range_high",
    "delinq_2yrs",
    "inq_last_6mths",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "revol_util",
    "total_acc",
]

for column in numeric_columns:
    if column in df.columns:
        df[column] = pd.to_numeric(
            df[column], errors="coerce"
        )

# Income must be positive.
if "annual_inc" in df.columns:
    invalid_income = df["annual_inc"] <= 0
    print(
        "Zero/negative annual income values:",
        int(invalid_income.sum())
    )
    df.loc[invalid_income, "annual_inc"] = np.nan

# DTI should be nonnegative and below 100 for this project.
if "dti" in df.columns:
    invalid_dti = (df["dti"] < 0) | (df["dti"] >= 100)
    print("Invalid DTI values:", int(invalid_dti.sum()))
    df.loc[invalid_dti, "dti"] = np.nan

# Revolving utilization is represented as a percentage.
if "revol_util" in df.columns:
    invalid_util = (
        (df["revol_util"] < 0)
        | (df["revol_util"] > 100)
    )
    print(
        "Invalid revolving utilization values:",
        int(invalid_util.sum())
    )
    df.loc[invalid_util, "revol_util"] = np.nan

# Check logical consistency of the FICO range.
if {"fico_range_low", "fico_range_high"}.issubset(df.columns):
    invalid_fico = (
        (df["fico_range_low"] > df["fico_range_high"])
        | (df["fico_range_low"] < 300)
        | (df["fico_range_high"] > 850)
    )
    print("Invalid FICO ranges:", int(invalid_fico.sum()))

    df.loc[
        invalid_fico,
        ["fico_range_low", "fico_range_high"]
    ] = np.nan

# --------------------------------------------------
# 5. Standardize text/categorical columns
# --------------------------------------------------
categorical_columns = [
    "term",
    "emp_length",
    "home_ownership",
    "verification_status",
    "purpose",
]

for column in categorical_columns:
    if column in df.columns:
        df[column] = (
            df[column]
            .astype("string")
            .str.strip()
        )

        df[column] = df[column].replace(
            {"": pd.NA, "nan": pd.NA, "None": pd.NA}
        )

# Standardize loan term, e.g. " 36 months" -> "36 months".
if "term" in df.columns:
    df["term"] = df["term"].str.replace(
        r"\s+", " ", regex=True
    )

# --------------------------------------------------
# 6. Parse loan issue date
# --------------------------------------------------
if "issue_d" in df.columns:
    df["issue_d"] = pd.to_datetime(
        df["issue_d"],
        format="%b-%Y",
        errors="coerce"
    )

    print(
        "\nInvalid or missing issue dates:",
        int(df["issue_d"].isna().sum())
    )

# --------------------------------------------------
# 7. Review extreme values without deleting them
# --------------------------------------------------
for column in ["annual_inc", "loan_amnt", "revol_bal"]:
    if column in df.columns:
        print(f"\n{column} percentiles:")
        print(
            df[column].quantile(
                [0.50, 0.90, 0.95, 0.99, 0.995, 1.00]
            )
        )

# Do not automatically remove high incomes or balances.
# Investigate extreme values before deciding how to handle them.

# --------------------------------------------------
# 8. Save the cleaned data
# --------------------------------------------------
df.to_csv(OUTPUT_FILE, index=False)

print("\nCleaning completed.")
print(f"Cleaned shape: {df.shape}")
print(f"Saved to: {OUTPUT_FILE}")

# --------------------------------------------------
# 9. Save a data-quality report
# --------------------------------------------------
quality = pd.DataFrame({
    "column": df.columns,
    "data_type": [
        str(df[column].dtype) for column in df.columns
    ],
    "missing_count": [
        int(df[column].isna().sum()) for column in df.columns
    ],
    "missing_percent": [
        round(df[column].isna().mean() * 100, 3)
        for column in df.columns
    ],
    "unique_values": [
        int(df[column].nunique(dropna=True))
        for column in df.columns
    ],
})

quality_file = REPORT_DIR / "cleaning_quality_report.csv"
quality.to_csv(quality_file, index=False)

print(f"Quality report saved to: {quality_file}")

print("\nFinal target distribution:")
print(df["default"].value_counts())
print("\nFinal target percentages:")
print((df["default"].value_counts(normalize=True) * 100).round(2))
