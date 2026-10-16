
from pathlib import Path
import pandas as pd

# Locate the compressed LendingClub dataset
data_path = (
    Path(__file__).parent
    / "data"
    / "accepted_2007_to_2018Q4.csv.gz"
)

if not data_path.exists():
    raise FileNotFoundError(f"Dataset not found: {data_path}")

print("Reading loan outcomes...")

# Read only the loan_status column to save memory
df = pd.read_csv(
    data_path,
    usecols=["loan_status"],
    compression="gzip",
    low_memory=False,
)

# Count every recorded loan status
counts = df["loan_status"].value_counts(dropna=False)

print("\n===== LOAN STATUS COUNTS =====")
print(counts.to_string())

print("\n===== LOAN STATUS PERCENTAGES =====")
percentages = (
    df["loan_status"]
    .value_counts(normalize=True, dropna=False)
    .mul(100)
    .round(2)
)
print(percentages.to_string())

print("\n===== TOTAL RECORDS =====")
print(f"{len(df):,} loans")
