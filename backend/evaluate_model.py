
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

BASE_DIR = Path(__file__).resolve().parent

df = pd.read_csv(
    BASE_DIR / "data" / "cleaned_sample.csv",
    low_memory=False,
)

df["issue_d"] = pd.to_datetime(
    df["issue_d"], errors="coerce"
)

model = joblib.load(
    BASE_DIR / "models" / "loan_default_model.joblib"
)

numeric_features = [
    "loan_amnt", "annual_inc", "dti",
    "fico_range_low", "fico_range_high",
    "delinq_2yrs", "inq_last_6mths",
    "open_acc", "pub_rec", "revol_bal",
    "revol_util", "total_acc",
]

categorical_features = [
    "term", "emp_length", "home_ownership",
    "verification_status", "purpose",
]

features = numeric_features + categorical_features

# Evaluate the same 2016-2018 period used for testing.
test = df[
    (df["issue_d"] >= "2016-01-01")
    & (df["default"].notna())
].copy()

test["year"] = test["issue_d"].dt.year

print("\n--- OUTCOME DISTRIBUTION BY YEAR ---")

for year, group in test.groupby("year"):
    print(
        f"{year}: loans={len(group):,}, "
        f"charged_off_rate={group['default'].mean():.2%}"
    )

print("\n--- MODEL PERFORMANCE BY YEAR ---")

for year, group in test.groupby("year"):
    X = group[features]
    y = group["default"].astype(int)

    probabilities = model.predict_proba(X)[:, 1]

    if y.nunique() < 2:
        print(f"{year}: not enough outcome classes to calculate metrics")
        continue

    print(
        f"{year}: ROC-AUC={roc_auc_score(y, probabilities):.4f}, "
        f"Average Precision={average_precision_score(y, probabilities):.4f}"
    )

print("\nEvaluation completed.")
