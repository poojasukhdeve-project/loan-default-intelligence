
from pathlib import Path
import json

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "mature_loan_sample.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

CUTOFF_DATE = pd.Timestamp("2013-01-01")

df = pd.read_csv(DATA_FILE, low_memory=False)
df["issue_d"] = pd.to_datetime(df["issue_d"], errors="coerce")
df = df.dropna(subset=["issue_d", "default"]).copy()
df["default"] = df["default"].astype(int)

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

train_df = df[df["issue_d"] < CUTOFF_DATE].copy()
test_df = df[df["issue_d"] >= CUTOFF_DATE].copy()

if train_df.empty or test_df.empty:
    raise ValueError("Training or testing dataset is empty.")

X_train = train_df[features]
y_train = train_df["default"]
X_test = test_df[features]
y_test = test_df["default"]

print("\n--- MATURE COHORT SPLIT ---")
print("Training rows:", len(train_df))
print("Testing rows:", len(test_df))
print("Training dates:", train_df["issue_d"].min(), "to",
      train_df["issue_d"].max())
print("Testing dates:", test_df["issue_d"].min(), "to",
      test_df["issue_d"].max())
print("Training default rate:", round(y_train.mean(), 4))
print("Testing default rate:", round(y_test.mean(), 4))

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="constant",
                              fill_value="Unknown")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features),
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        max_iter=300,
        solver="lbfgs",
        class_weight="balanced",
        random_state=42,
    )),
])

print("\nTraining mature-cohort model...")
model.fit(X_train, y_train)

probabilities = model.predict_proba(X_test)[:, 1]
predictions = (probabilities >= 0.5).astype(int)

print("\n--- MATURE COHORT RESULTS ---")
print("ROC-AUC:", round(roc_auc_score(y_test, probabilities), 4))
print("Average precision:",
      round(average_precision_score(y_test, probabilities), 4))
print("\nClassification report:")
print(classification_report(
    y_test, predictions,
    target_names=["Fully Paid", "Charged Off"],
    zero_division=0,
))
print("Confusion matrix:")
print(confusion_matrix(y_test, predictions))

model_path = MODEL_DIR / "loan_default_mature_model.joblib"
joblib.dump(model, model_path)

metadata = {
    "model_type": "LogisticRegression",
    "dataset": "mature_loan_sample.csv",
    "cutoff_date": str(CUTOFF_DATE.date()),
    "training_rows": int(len(train_df)),
    "testing_rows": int(len(test_df)),
    "roc_auc": float(roc_auc_score(y_test, probabilities)),
    "average_precision": float(
        average_precision_score(y_test, probabilities)
    ),
    "numeric_features": numeric_features,
    "categorical_features": categorical_features,
}

metadata_path = MODEL_DIR / "mature_model_metadata.json"
with open(metadata_path, "w", encoding="utf-8") as file:
    json.dump(metadata, file, indent=2)

print("\nSaved model:", model_path)
print("Saved metadata:", metadata_path)
print("\nMature-cohort training completed.")
