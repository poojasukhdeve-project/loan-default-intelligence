
from pathlib import Path
import pandas as pd

BASE = Path(__file__).parent
SOURCE = BASE / "data" / "accepted_2007_to_2018Q4.csv.gz"
OUTPUT = BASE / "data" / "modeling_sample.csv"

FEATURES = [
    "loan_amnt",
    "term",
    "int_rate",
    "installment",
    "emp_length",
    "home_ownership",
    "annual_inc",
    "verification_status",
    "purpose",
    "dti",
    "delinq_2yrs",
    "fico_range_low",
    "fico_range_high",
    "inq_last_6mths",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "revol_util",
    "total_acc",
    "issue_d",
    "loan_status",
]

if not SOURCE.exists():
    raise FileNotFoundError(f"Dataset not found: {SOURCE}")

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
first_write = True
total_saved = 0

for chunk in pd.read_csv(
    SOURCE,
    compression="gzip",
    usecols=FEATURES,
    chunksize=100_000,
    low_memory=False,
):
    # Use only clearly defined main outcomes for this first prototype.
    chunk = chunk[
        chunk["loan_status"].isin(["Fully Paid", "Charged Off"])
    ].copy()

    if chunk.empty:
        continue

    chunk["default"] = (
        chunk["loan_status"] == "Charged Off"
    ).astype("int8")

    chunk = chunk.drop(columns=["loan_status"])

    chunk.to_csv(
        OUTPUT,
        mode="w" if first_write else "a",
        header=first_write,
        index=False,
    )

    first_write = False
    total_saved += len(chunk)

print(f"Saved records: {total_saved:,}")
print(f"Output file: {OUTPUT}")
print("Target counts:")
print(pd.read_csv(OUTPUT, usecols=["default"])["default"].value_counts())
