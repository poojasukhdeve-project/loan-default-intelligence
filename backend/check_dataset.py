
from pathlib import Path
import pandas as pd

DATA_PATH = (
    Path(__file__).parent
    / "data"
    / "accepted_2007_to_2018Q4.csv.gz"
)

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

print("Dataset found:", DATA_PATH)
print("File size (GB):", round(DATA_PATH.stat().st_size / 1e9, 2))

# Read only a small sample to avoid loading the entire dataset
df = pd.read_csv(DATA_PATH, compression="gzip", nrows=5)

print("\nSample rows:")
print(df.head())

print("\nColumns:")
print(df.columns.tolist())
