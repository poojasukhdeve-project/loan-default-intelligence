
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "mature_loan_sample.csv"
MODEL_PATH = BASE_DIR / "models" / "loan_default_validation_model.joblib"

def main():
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df["issue_d"] = pd.to_datetime(df["issue_d"], errors="coerce")
    df = df.dropna(subset=["issue_d", "default"]).copy()
    df["default"] = df["default"].astype(int)

    # Check how many loans are available in each year.
    df["issue_year"] = df["issue_d"].dt.year

    print("\nDATA DISTRIBUTION BY YEAR")
    print("-------------------------")
    yearly = df.groupby("issue_year").agg(
        loan_count=("default", "size"),
        default_rate=("default", "mean"),
        default_count=("default", "sum"),
    )
    print(yearly.round(4).to_string())

    # Evaluate the saved model by year for diagnostic purposes.
    model = joblib.load(MODEL_PATH)
    features = list(model.feature_names_in_)

    print("\nMODEL PERFORMANCE BY YEAR")
    print("-------------------------")

    rows = []
    for year, group in df.groupby("issue_year"):
        if len(group) < 100 or group["default"].nunique() < 2:
            continue

        probabilities = model.predict_proba(group[features])[:, 1]

        rows.append({
            "year": year,
            "loans": len(group),
            "default_rate": group["default"].mean(),
            "roc_auc": roc_auc_score(
                group["default"], probabilities
            ),
            "average_precision": average_precision_score(
                group["default"], probabilities
            ),
        })

    results = pd.DataFrame(rows).round(4)
    print(results.to_string(index=False))

    output_path = BASE_DIR / "yearly_model_diagnostics.csv"
    results.to_csv(output_path, index=False)
    print("\nSaved:", output_path)

    print(
        "\nDiagnostic only: use these results to understand performance, "
        "not to claim independent final-test performance or select a "
        "new threshold."
    )

if __name__ == "__main__":
    main()
