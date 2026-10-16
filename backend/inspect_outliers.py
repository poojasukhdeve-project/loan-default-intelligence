
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
FILE = BASE_DIR / "data" / "cleaned_sample.csv"

df = pd.read_csv(FILE, low_memory=False)

columns = ["annual_inc", "revol_bal", "dti", "revol_util"]

for column in columns:
    if column not in df.columns:
        continue

    values = pd.to_numeric(df[column], errors="coerce").dropna()

    print(f"\n--- {column} ---")
    print("Minimum:", values.min())
    print("Median:", values.median())
    print("95th percentile:", values.quantile(0.95))
    print("99th percentile:", values.quantile(0.99))
    print("99.5th percentile:", values.quantile(0.995))
    print("Maximum:", values.max())

    print("\nTop 10 largest values:")
    print(values.nlargest(10).to_list())

print("\nOutlier inspection completed.")
