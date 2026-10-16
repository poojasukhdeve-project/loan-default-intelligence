
from pathlib import Path

import joblib
import pandas as pd
import numpy as np

from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "mature_loan_sample.csv"
MODEL_PATH = BASE_DIR / "models" / "loan_default_model.joblib"
OUTPUT_PATH = BASE_DIR / "calibration_results.csv"

TEST_START = pd.Timestamp("2013-01-01")


def main():
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df["issue_d"] = pd.to_datetime(df["issue_d"], errors="coerce")
    df = df.dropna(subset=["issue_d", "default"]).copy()
    df["default"] = df["default"].astype(int)

    test = df[df["issue_d"] >= TEST_START].copy()

    model = joblib.load(MODEL_PATH)
    features = list(model.feature_names_in_)

    X_test = test[features]
    y_test = test["default"]
    probabilities = model.predict_proba(X_test)[:, 1]

    # Divide predictions into ten bins and compare predicted versus
    # observed default rates in each bin.
    observed_rate, mean_predicted = calibration_curve(
        y_test,
        probabilities,
        n_bins=10,
        strategy="quantile",
    )

    result = pd.DataFrame({
        "mean_predicted_probability": mean_predicted,
        "observed_default_rate": observed_rate,
    })

    result["difference"] = (
        result["observed_default_rate"]
        - result["mean_predicted_probability"]
    )

    result.to_csv(OUTPUT_PATH, index=False)

    print("Test rows:", len(test))
    print("Mean predicted probability:", round(probabilities.mean(), 4))
    print("Observed default rate:", round(y_test.mean(), 4))
    print("Brier score:", round(brier_score_loss(y_test, probabilities), 4))
    print("\nCalibration by probability group:")
    print(result.round(4).to_string(index=False))
    print("\nSaved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
