
from pathlib import Path

import joblib
import pandas as pd

from sklearn.base import clone

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "mature_loan_sample.csv"
TEMPLATE_PATH = BASE_DIR / "models" / "loan_default_mature_model.joblib"
OUTPUT_PATH = BASE_DIR / "models" / "loan_default_best_model.joblib"

TRAIN_END = pd.Timestamp("2011-01-01")
C_VALUE = 0.01


def main():
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df["issue_d"] = pd.to_datetime(df["issue_d"], errors="coerce")
    df = df.dropna(subset=["issue_d", "default"]).copy()
    df["default"] = df["default"].astype(int)

    train = df[df["issue_d"] < TRAIN_END].copy()

    if train.empty or train["default"].nunique() < 2:
        raise ValueError("Training data is empty or missing a target class.")

    template = joblib.load(TEMPLATE_PATH)
    features = list(template.feature_names_in_)
    model = clone(template)

    # Use the best regularization value found on validation data.
    model.steps[-1][1].set_params(C=C_VALUE)

    X_train = train[features]
    y_train = train["default"]

    print("Training rows:", len(train))
    print("Training period:", train["issue_d"].min(), "to", train["issue_d"].max())
    print("Logistic Regression C:", C_VALUE)

    model.fit(X_train, y_train)
    joblib.dump(model, OUTPUT_PATH)

    print("\nBest candidate model saved to:", OUTPUT_PATH)
    print("Do not use the final test data to tune this model.")


if __name__ == "__main__":
    main()
