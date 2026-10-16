
from pathlib import Path

import joblib
import pandas as pd

from sklearn.base import clone
from sklearn.metrics import roc_auc_score, average_precision_score

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "mature_loan_sample.csv"
TEMPLATE_PATH = BASE_DIR / "models" / "loan_default_mature_model.joblib"

TRAIN_END = pd.Timestamp("2011-01-01")
VALIDATION_END = pd.Timestamp("2013-01-01")


def main():
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df["issue_d"] = pd.to_datetime(df["issue_d"], errors="coerce")
    df = df.dropna(subset=["issue_d", "default"]).copy()
    df["default"] = df["default"].astype(int)

    train = df[df["issue_d"] < TRAIN_END].copy()
    validation = df[
        (df["issue_d"] >= TRAIN_END)
        & (df["issue_d"] < VALIDATION_END)
    ].copy()

    template = joblib.load(TEMPLATE_PATH)
    features = list(template.feature_names_in_)

    X_train = train[features]
    y_train = train["default"]
    X_val = validation[features]
    y_val = validation["default"]

    results = []

    # Compare a small set of regularization strengths.
    for c_value in [0.01, 0.1, 1.0, 10.0]:
        model = clone(template)

        # Access the Logistic Regression estimator in the pipeline.
        classifier = model.steps[-1][1]
        classifier.set_params(C=c_value)

        print(f"Training Logistic Regression with C={c_value}...")
        model.fit(X_train, y_train)

        probabilities = model.predict_proba(X_val)[:, 1]

        results.append({
            "C": c_value,
            "roc_auc": roc_auc_score(y_val, probabilities),
            "average_precision": average_precision_score(
                y_val, probabilities
            ),
        })

    results_df = pd.DataFrame(results).round(4)
    results_df = results_df.sort_values(
        "average_precision", ascending=False
    )

    print("\nVALIDATION COMPARISON")
    print(results_df.to_string(index=False))

    output_path = BASE_DIR / "logistic_tuning_results.csv"
    results_df.to_csv(output_path, index=False)
    print("\nSaved:", output_path)


if __name__ == "__main__":
    main()
