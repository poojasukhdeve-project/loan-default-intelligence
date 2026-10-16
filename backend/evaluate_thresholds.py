
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "mature_loan_sample.csv"
MODEL_FILE = BASE_DIR / "models" / "loan_default_mature_model.joblib"

df = pd.read_csv(DATA_FILE, low_memory=False)
df["issue_d"] = pd.to_datetime(df["issue_d"], errors="coerce")

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

# Use the same held-out period as the mature-cohort model.
test = df[
    (df["issue_d"] >= "2013-01-01")
    & (df["default"].notna())
].copy()

model = joblib.load(MODEL_FILE)

X_test = test[features]
y_test = test["default"].astype(int)
probabilities = model.predict_proba(X_test)[:, 1]

print("ROC-AUC:", round(roc_auc_score(y_test, probabilities), 4))
print("Average precision:",
      round(average_precision_score(y_test, probabilities), 4))

print("\n--- THRESHOLD COMPARISON ---")

for threshold in [0.20, 0.30, 0.40, 0.50, 0.60, 0.70]:
    predictions = (probabilities >= threshold).astype(int)

    print(f"\nThreshold: {threshold:.2f}")
    print("Precision:", round(
        precision_score(y_test, predictions, zero_division=0), 3
    ))
    print("Recall:", round(
        recall_score(y_test, predictions, zero_division=0), 3
    ))
    print("F1-score:", round(
        f1_score(y_test, predictions, zero_division=0), 3
    ))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions))

print("\nThreshold evaluation completed.")
