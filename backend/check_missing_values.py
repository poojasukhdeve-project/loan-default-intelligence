
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
file_path = BASE_DIR / "data" / "cleaned_sample.csv"

df = pd.read_csv(file_path, low_memory=False)

missing = pd.DataFrame({
    "missing_count": df.isna().sum(),
    "missing_percentage": (df.isna().mean() * 100).round(3),
})

missing = missing.sort_values(
    "missing_percentage", ascending=False
)

print("\n--- MISSING VALUES BY COLUMN ---")
print(missing.to_string())

print("\n--- COLUMNS WITH MISSING VALUES ---")
print(missing[missing["missing_count"] > 0].to_string())

output_file = BASE_DIR / "eda_outputs" / "missing_values_report.csv"
missing.to_csv(output_file)

print(f"\nReport saved to: {output_file}")
