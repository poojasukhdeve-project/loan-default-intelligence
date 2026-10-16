
from pathlib import Path

import joblib
import pandas as pd

from sklearn.base import clone
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "mature_loan_sample.csv"
EXISTING_MODEL_PATH = BASE_DIR / "models" / "loan_default_mature_model.joblib"
OUTPUT_MODEL_PATH = BASE_DIR / "models" / "loan_default_validation_model.joblib"
OUTPUT_RESULTS_PATH = BASE_DIR / "validation_threshold_results.csv"

TRAIN_END = pd.Timestamp("2011-01-01")
VALIDATION_END = pd.Timestamp("2013-01-01")


def main():
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df["issue_d"] = pd.to_datetime(df["issue_d"], errors="coerce")
    df = df.dropna(subset=["issue_d", "default"]).copy()
    df["default"] = df["default"].astype(int)

    # Strict chronological split
    train = df[df["issue_d"] < TRAIN_END].copy()
    validation = df[
        (df["issue_d"] >= TRAIN_END)
        & (df["issue_d"] < VALIDATION_END)
    ].copy()
    final_test = df[df["issue_d"] >= VALIDATION_END].copy()

    print("Training rows:", len(train))
    print("Training dates:", train["issue_d"].min(), "to", train["issue_d"].max())
    print("Training default rate:", round(train["default"].mean(), 4))

    print("\nValidation rows:", len(validation))
    print("Validation default rate:", round(validation["default"].mean(), 4))

    print("\nFinal test rows:", len(final_test))
    print("Final test dates:", final_test["issue_d"].min(), "to", final_test["issue_d"].max())

    if train.empty or validation.empty or final_test.empty:
        raise ValueError("One of the chronological splits is empty.")

    if train["default"].nunique() < 2 or validation["default"].nunique() < 2:
        raise ValueError("Training and validation must each contain both target classes.")

    # Reuse the existing pipeline configuration, but fit from scratch.
    template = joblib.load(EXISTING_MODEL_PATH)
    feature_columns = list(template.feature_names_in_)
    model = clone(template)

    X_train = train[feature_columns]
    y_train = train["default"]
    X_val = validation[feature_columns]
    y_val = validation["default"]

    print("\nTraining a fresh model...")
    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_val)[:, 1]

    print("\nVALIDATION PERFORMANCE")
    print("ROC-AUC:", round(roc_auc_score(y_val, probabilities), 4))
    print(
        "Average precision:",
        round(average_precision_score(y_val, probabilities), 4),
    )

    rows = []
    for threshold in [round(x / 100, 2) for x in range(10, 81, 5)]:
        predictions = (probabilities >= threshold).astype(int)
        rows.append({
            "threshold": threshold,
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

    results = pd.DataFrame(rows).round(4)
    results.to_csv(OUTPUT_RESULTS_PATH, index=False)

    print("\nTHRESHOLD RESULTS")
    print(results.to_string(index=False))

    best = results.loc[results["f1"].idxmax()]
    print("\nBest validation threshold by F1:")
    print(best.to_string())

    # Save the freshly trained model for subsequent final testing.
    joblib.dump(model, OUTPUT_MODEL_PATH)
    print("\nFresh model saved to:", OUTPUT_MODEL_PATH)
    print("Threshold results saved to:", OUTPUT_RESULTS_PATH)
    print("\nFinal test data was not used for fitting or threshold selection.")


if __name__ == "__main__":
    main()
