
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# Project paths
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "mature_loan_sample.csv"
MODEL_PATH = BASE_DIR / "models" / "loan_default_mature_model.joblib"

# Chronological split boundaries
VALIDATION_START = pd.Timestamp("2011-01-01")
TEST_START = pd.Timestamp("2013-01-01")

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    df = pd.read_csv(DATA_PATH, low_memory=False)

    # Parse issue date
    df["issue_d"] = pd.to_datetime(
        df["issue_d"], errors="coerce"
    )

    df = df.dropna(subset=["issue_d", "default"]).copy()
    df["default"] = df["default"].astype(int)

    # Check the required date range
    print("Dataset rows:", len(df))
    print("Date range:", df["issue_d"].min(), "to", df["issue_d"].max())

    # Validation: 2011–2012
    validation = df[
        (df["issue_d"] >= VALIDATION_START)
        & (df["issue_d"] < TEST_START)
    ].copy()

    # Final test: 2013 onward
    test = df[df["issue_d"] >= TEST_START].copy()

    print("\nValidation rows:", len(validation))
    print("Final test rows:", len(test))

    if validation.empty or test.empty:
        raise ValueError(
            "Validation or test set is empty. Check issue_d dates."
        )

    model = joblib.load(MODEL_PATH)

    # Use the features expected by the saved pipeline.
    feature_columns = list(model.feature_names_in_)

    missing = [
        col for col in feature_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(f"Missing model features: {missing}")

    X_val = validation[feature_columns]
    y_val = validation["default"]

    # Threshold selection happens on validation only.
    probabilities = model.predict_proba(X_val)[:, 1]

    print("\nVALIDATION METRICS")
    print("------------------")
    print("ROC-AUC:", round(roc_auc_score(y_val, probabilities), 4))
    print(
        "Average precision:",
        round(average_precision_score(y_val, probabilities), 4),
    )

    results = []

    for threshold in np.arange(0.10, 0.81, 0.05):
        predictions = (probabilities >= threshold).astype(int)

        results.append({
            "threshold": round(float(threshold), 2),
            "precision": precision_score(
                y_val, predictions, zero_division=0
            ),
            "recall": recall_score(
                y_val, predictions, zero_division=0
            ),
            "f1": f1_score(
                y_val, predictions, zero_division=0
            ),
        })

    results_df = pd.DataFrame(results)
    results_df = results_df.round(4)

    output_path = BASE_DIR / "validation_threshold_results.csv"
    results_df.to_csv(output_path, index=False)

    print("\nTHRESHOLD COMPARISON")
    print(results_df.to_string(index=False))
    print("\nResults saved to:", output_path)

    # Report the best validation F1 threshold.
    best_row = results_df.loc[results_df["f1"].idxmax()]
    print("\nBest validation threshold by F1:")
    print(best_row.to_string())

    print(
        "\nThe final test set has NOT been evaluated. "
        "Keep it untouched until model and threshold decisions are final."
    )

if __name__ == "__main__":
    main()
