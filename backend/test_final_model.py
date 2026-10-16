
from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    brier_score_loss,
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "mature_loan_sample.csv"
MODEL_PATH = BASE_DIR / "models" / "loan_default_validation_model.joblib"
REPORT_PATH = BASE_DIR / "final_test_report.txt"

TEST_START = pd.Timestamp("2013-01-01")
THRESHOLD = 0.55


def main():
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df["issue_d"] = pd.to_datetime(df["issue_d"], errors="coerce")
    df = df.dropna(subset=["issue_d", "default"]).copy()
    df["default"] = df["default"].astype(int)

    test = df[df["issue_d"] >= TEST_START].copy()

    if test.empty:
        raise ValueError("Final test dataset is empty.")

    model = joblib.load(MODEL_PATH)
    features = list(model.feature_names_in_)

    missing = [col for col in features if col not in test.columns]
    if missing:
        raise ValueError(f"Missing model features: {missing}")

    X_test = test[features]
    y_test = test["default"]

    if y_test.nunique() < 2:
        raise ValueError("Final test data must contain both target classes.")

    # Predict probabilities and classify using the validation-selected threshold.
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= THRESHOLD).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test, predictions, labels=[0, 1]
    ).ravel()

    report = f"""
FINAL TEST REPORT
=================

Test period: {test['issue_d'].min().date()} to {test['issue_d'].max().date()}
Test rows: {len(test):,}
Observed default rate: {y_test.mean():.4%}
Threshold selected on validation data: {THRESHOLD:.2f}

Ranking metrics
---------------
ROC-AUC: {roc_auc_score(y_test, probabilities):.4f}
Average precision: {average_precision_score(y_test, probabilities):.4f}

Classification metrics
----------------------
Accuracy:  {accuracy_score(y_test, predictions):.4f}
Precision: {precision_score(y_test, predictions, zero_division=0):.4f}
Recall:    {recall_score(y_test, predictions, zero_division=0):.4f}
F1 score:  {f1_score(y_test, predictions, zero_division=0):.4f}
Brier score: {brier_score_loss(y_test, probabilities):.4f}

Confusion matrix
----------------
True negatives:  {tn:,}
False positives: {fp:,}
False negatives: {fn:,}
True positives:  {tp:,}

Classification report
---------------------
{classification_report(y_test, predictions, target_names=['Fully Paid', 'Charged Off'], zero_division=0)}
"""

    print(report)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print("Report saved to:", REPORT_PATH)


if __name__ == "__main__":
    main()
