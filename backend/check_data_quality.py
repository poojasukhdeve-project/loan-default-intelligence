
from pathlib import Path
import pandas as pd

DATA_PATH = (
    Path(__file__).parent
    / "data"
    / "accepted_2007_to_2018Q4.csv.gz"
)

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

print("Loading dataset for data quality checks...")

# Read the complete dataset in chunks to limit memory usage.
total_rows = 0
duplicate_ids = 0
missing_status = 0
status_counts = {}
missing_counts = {}
numeric_stats = {
    "loan_amnt": [],
    "annual_inc": [],
    "int_rate": [],
    "dti": [],
    "revol_util": [],
}

for chunk in pd.read_csv(
    DATA_PATH,
    compression="gzip",
    chunksize=100_000,
    low_memory=False,
):
    total_rows += len(chunk)

    # Check duplicate loan IDs
    if "id" in chunk.columns:
        duplicate_ids += int(chunk["id"].duplicated().sum())

    # Count missing values by column
    missing = chunk.isna().sum()
    for col, count in missing.items():
        missing_counts[col] = missing_counts.get(col, 0) + int(count)

    # Loan outcome distribution
    if "loan_status" in chunk.columns:
        missing_status += int(chunk["loan_status"].isna().sum())
        counts = chunk["loan_status"].value_counts(dropna=False)
        for status, count in counts.items():
            key = str(status)
            status_counts[key] = status_counts.get(key, 0) + int(count)

    # Basic numeric validation
    for col in numeric_stats:
        if col in chunk.columns:
            values = pd.to_numeric(chunk[col], errors="coerce").dropna()
            numeric_stats[col].append(values.describe())

print("\n===== DATASET SIZE =====")
print(f"Rows: {total_rows:,}")
print(f"Columns: {len(missing_counts)}")

print("\n===== DUPLICATE IDs WITHIN CHUNKS =====")
print(duplicate_ids)

print("\n===== MISSING VALUES: TOP 20 COLUMNS =====")
missing_series = pd.Series(missing_counts).sort_values(ascending=False)
missing_pct = (missing_series / total_rows * 100).round(2)
print(pd.DataFrame({
    "missing_count": missing_series,
    "missing_percent": missing_pct,
}).head(20).to_string())

print("\n===== LOAN STATUS COUNTS =====")
for status, count in sorted(
    status_counts.items(), key=lambda item: item[1], reverse=True
):
    print(f"{status}: {count:,} ({count / total_rows:.2%})")

print("\n===== MISSING LOAN STATUS =====")
print(missing_status)

print("\n===== NUMERIC VALUE CHECKS =====")
print("Inspect these ranges for impossible or suspicious values:")
for col, chunks in numeric_stats.items():
    if chunks:
        print(f"\n{col}:")
        for stats in chunks:
            print(stats.to_string())
