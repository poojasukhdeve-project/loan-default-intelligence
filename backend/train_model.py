
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


# --------------------------------------------------
# 1. Paths and settings
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "cleaned_sample.csv"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Train on loans issued before 2016.
# Test on loans issued in 2016 or later.
CUTOFF_DATE = pd.Timestamp("2016-01-01")

print("Loading cleaned dataset...")
df = pd.read_csv(DATA_FILE, low_memory=False)

df["issue_d"] = pd.to_datetime(
    df["issue_d"], errors="coerce"
)

df = df.dropna(subset=["issue_d", "default"]).copy()
df["default"] = df["default"].astype(int)

# --------------------------------------------------
# 2. Select features known at loan issuance
# --------------------------------------------------
numeric_features = [
    "loan_amnt",
    "annual_inc",
    "dti",
    "fico_range_low",
    "fico_range_high",
    "delinq_2yrs",
    "inq_last_6mths",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "revol_util",
    "total_acc",
]

categorical_features = [
    "term",
    "emp_length",
    "home_ownership",
    "verification_status",
    "purpose",
]

features = numeric_features + categorical_features

missing_columns = [
    col for col in features if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing expected columns: {missing_columns}"
    )

# Exclude issue_d from model features; use it only for splitting.
# Initially exclude int_rate and installment because they may
# encode prior lender pricing/underwriting decisions.

train_df = df[df["issue_d"] < CUTOFF_DATE].copy()
test_df = df[df["issue_d"] >= CUTOFF_DATE].copy()

if train_df.empty or test_df.empty:
    raise ValueError(
        "Train or test set is empty. Check issue_d and cutoff."
    )

X_train = train_df[features]
y_train = train_df["default"]

X_test = test_df[features]
y_test = test_df["default"]

print("\n--- SPLIT SUMMARY ---")
print("Training rows:", len(train_df))
print("Testing rows:", len(test_df))
print(
    "Training date range:",
    train_df["issue_d"].min(),
    "to",
    train_df["issue_d"].max(),
)
print(
    "Testing date range:",
    test_df["issue_d"].min(),
    "to",
    test_df["issue_d"].max(),
)

print("\nTraining default rate:", round(y_train.mean(), 4))
print("Testing default rate:", round(y_test.mean(), 4))

# --------------------------------------------------
# 3. Preprocess numeric and categorical features
# --------------------------------------------------
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

# --------------------------------------------------
# 4. Train Logistic Regression
# --------------------------------------------------
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
            max_iter=300,
            solver="lbfgs",
            class_weight="balanced",
            random_state=42,
    )),
])

print("\nTraining Logistic Regression...")
model.fit(X_train, y_train)

# --------------------------------------------------
# 5. Evaluate on later loans
# --------------------------------------------------
print("\nGenerating test predictions...")
probabilities = model.predict_proba(X_test)[:, 1]

# 0.5 threshold for the initial baseline.
predictions = (probabilities >= 0.5).astype(int)

print("\n--- MODEL RESULTS ---")
print("ROC-AUC:", round(
    roc_auc_score(y_test, probabilities), 4
))
print("Average precision:", round(
    average_precision_score(y_test, probabilities), 4
))

print("\nClassification report:")
print(classification_report(
    y_test,
    predictions,
    target_names=["Fully Paid", "Charged Off"],
    zero_division=0,
))

print("Confusion matrix:")
print(confusion_matrix(y_test, predictions))

# --------------------------------------------------
# 6. Save model and metadata
# --------------------------------------------------
model_path = MODEL_DIR / "loan_default_model.joblib"
joblib.dump(model, model_path)

metadata = {
    "model_type": "LogisticRegression",
    "target": "default",
    "positive_class": 1,
    "cutoff_date": str(CUTOFF_DATE.date()),
    "numeric_features": numeric_features,
    "categorical_features": categorical_features,
    "training_rows": int(len(train_df)),
    "testing_rows": int(len(test_df)),
    "roc_auc": float(roc_auc_score(y_test, probabilities)),
    "average_precision": float(
        average_precision_score(y_test, probabilities)
    ),
}

metadata_path = MODEL_DIR / "model_metadata.json"
with open(metadata_path, "w", encoding="utf-8") as file:
    json.dump(metadata, file, indent=2)

print("\nSaved model:", model_path)
print("Saved metadata:", metadata_path)
print("\nStep 4 baseline training completed.")
