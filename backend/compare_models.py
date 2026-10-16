
from pathlib import Path

import joblib
import pandas as pd

from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
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

    numeric_features = train[features].select_dtypes(
        include=["number"]
    ).columns.tolist()
    categorical_features = [
        col for col in features if col not in numeric_features
    ]

    # HistGradientBoosting requires numeric input, so encode
    # categorical columns with one-hot encoding first.
    preprocessing = ColumnTransformer(
        transformers=[
            (
                "num",
                SimpleImputer(strategy="median"),
                numeric_features,
            ),
            (
                "cat",
                Pipeline([
                    (
                        "imputer",
                        SimpleImputer(strategy="constant", fill_value="Unknown"),
                    ),
                    (
                        "encoder",
                        OneHotEncoder(
                            handle_unknown="ignore",
                            sparse_output=False,
                        ),
                    ),
                ]),
                categorical_features,
            ),
        ],
        remainder="drop",
    )

    model = Pipeline([
        ("preprocessing", preprocessing),
        (
            "classifier",
            HistGradientBoostingClassifier(
                max_iter=150,
                learning_rate=0.08,
                max_leaf_nodes=15,
                l2_regularization=1.0,
                random_state=42,
            ),
        ),
    ])

    X_train = train[features]
    y_train = train["default"]
    X_val = validation[features]
    y_val = validation["default"]

    print("Training rows:", len(train))
    print("Validation rows:", len(validation))
    print("\nTraining HistGradientBoosting...")

    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_val)[:, 1]

    print("\nVALIDATION RESULTS")
    print("ROC-AUC:", round(roc_auc_score(y_val, probabilities), 4))
    print(
        "Average precision:",
        round(average_precision_score(y_val, probabilities), 4),
    )

    output_path = BASE_DIR / "models" / "loan_default_hgb_model.joblib"
    joblib.dump(model, output_path)
    print("\nSaved model to:", output_path)
    print("The final test data was not used.")


if __name__ == "__main__":
    main()
