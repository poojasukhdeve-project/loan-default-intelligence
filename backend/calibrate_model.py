
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.calibration import calibration_curve

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "mature_loan_sample.csv"
MODEL_PATH = BASE_DIR / "models" / "loan_default_validation_model.joblib"
CALIBRATOR_PATH = BASE_DIR / "models" / "loan_default_calibrator.joblib"

OUTPUT_PATH = BASE_DIR / "calibrated_test_results.csv"

TRAIN_END = pd.Timestamp("2011-01-01")
TEST_START = pd.Timestamp("2013-01-01")


def logit(probabilities):
    probabilities = np.clip(probabilities, 1e-6, 1 - 1e-6)
    return np.log(probabilities / (1 - probabilities)).reshape(-1, 1)


def main():
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df["issue_d"] = pd.to_datetime(df["issue_d"], errors="coerce")
    df = df.dropna(subset=["issue_d", "default"]).copy()
    df["default"] = df["default"].astype(int)

    validation = df[
        (df["issue_d"] >= TRAIN_END)
        & (df["issue_d"] < TEST_START)
    ].copy()

    test = df[df["issue_d"] >= TEST_START].copy()

    model = joblib.load(MODEL_PATH)
    features = list(model.feature_names_in_)

    X_val = validation[features]
    y_val = validation["default"]
    X_test = test[features]
    y_test = test["default"]

    # Get probabilities from the model trained before 2011.
    val_prob = model.predict_proba(X_val)[:, 1]
    test_prob = model.predict_proba(X_test)[:, 1]

    # Fit Platt-style sigmoid calibration on validation predictions only.
    calibrator = LogisticRegression()
    calibrator.fit(logit(val_prob), y_val)

    joblib.dump(calibrator, CALIBRATOR_PATH)
    print("Calibrator saved to:", CALIBRATOR_PATH)

    calibrated_val = calibrator.predict_proba(logit(val_prob))[:, 1]
    calibrated_test = calibrator.predict_proba(logit(test_prob))[:, 1]

    print("Validation rows:", len(validation))
    print("Test rows:", len(test))

    print("\nCALIBRATION COMPARISON")
    print("Test observed default rate:", round(y_test.mean(), 4))
    print("Raw mean predicted probability:", round(test_prob.mean(), 4))
    print("Calibrated mean predicted probability:", round(calibrated_test.mean(), 4))
    print("Raw test Brier score:", round(brier_score_loss(y_test, test_prob), 4))
    print("Calibrated test Brier score:", round(brier_score_loss(y_test, calibrated_test), 4))
    print("Raw test ROC-AUC:", round(roc_auc_score(y_test, test_prob), 4))
    print("Calibrated test ROC-AUC:", round(roc_auc_score(y_test, calibrated_test), 4))

    observed, predicted = calibration_curve(
        y_test, calibrated_test, n_bins=10, strategy="quantile"
    )

    results = pd.DataFrame({
        "mean_calibrated_probability": predicted,
        "observed_default_rate": observed,
    })
    results.to_csv(OUTPUT_PATH, index=False)
    print("\nCalibrated test reliability:")
    print(results.round(4).to_string(index=False))
    print("\nSaved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
