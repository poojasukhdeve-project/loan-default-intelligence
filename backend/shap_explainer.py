import warnings
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

warnings.filterwarnings("ignore", category=FutureWarning)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "loan_default_validation_model.joblib"
DATA_PATH = BASE_DIR / "data" / "modeling_sample.csv"

BACKGROUND_ROWS = 1000

model = joblib.load(MODEL_PATH)
preprocessor = model.named_steps["preprocessor"]
classifier = model.named_steps["classifier"]

INPUT_COLUMNS = list(model.feature_names_in_)

# Map transformed feature names to their original input fields.
FEATURE_NAMES = preprocessor.get_feature_names_out()

# The two FICO fields carry the same information, so they are reported
# as one factor instead of taking two of the top-5 slots.
FICO_FIELDS = ("fico_range_low", "fico_range_high")
FICO_LABEL = "fico_score"


@lru_cache(maxsize=1)
def _get_explainer() -> shap.LinearExplainer:
    """Build the SHAP explainer once and reuse it for every request."""
    background_df = pd.read_csv(
        DATA_PATH,
        usecols=INPUT_COLUMNS,
        nrows=BACKGROUND_ROWS,
    )[INPUT_COLUMNS]

    background_transformed = preprocessor.transform(background_df)
    return shap.LinearExplainer(classifier, background_transformed)


def explain_prediction(inputs: dict, max_factors: int = 5) -> list[dict]:
    """Return local SHAP contributions for the fitted binary classifier."""

    input_df = pd.DataFrame([inputs])[INPUT_COLUMNS]
    transformed = preprocessor.transform(input_df)

    shap_result = _get_explainer()(transformed)
    values = np.asarray(shap_result.values)

    # Select the positive/default class if SHAP returns multiple outputs.
    if values.ndim == 3:
        values = values[0, :, 1] if values.shape[-1] > 1 else values[0, :, 0]
    elif values.ndim == 2:
        values = values[0]

    grouped = {name: 0.0 for name in INPUT_COLUMNS}

    for transformed_name, impact in zip(FEATURE_NAMES, values):
        cleaned = str(transformed_name).split("__")[-1]

        # Numeric features have matching names; categorical features
        # may have encoded names such as term_ 36 months.
        matches = [
            original for original in INPUT_COLUMNS
            if cleaned == original or cleaned.startswith(original + "_")
        ]

        if matches:
            # Use the longest match to avoid ambiguous prefixes.
            original = max(matches, key=len)
            grouped[original] += float(impact)

    # Merge the FICO low/high pair into a single factor.
    fico_total = sum(grouped.pop(name, 0.0) for name in FICO_FIELDS)
    grouped[FICO_LABEL] = fico_total

    ranked = sorted(
        grouped.items(),
        key=lambda item: abs(item[1]),
        reverse=True,
    )[:max_factors]

    return [
        {"feature": feature, "impact": round(impact, 6)}
        for feature, impact in ranked
    ]