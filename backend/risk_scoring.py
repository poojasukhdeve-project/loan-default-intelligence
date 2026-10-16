
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "loan_default_validation_model.joblib"
CALIBRATOR_PATH = BASE_DIR / "models" / "loan_default_calibrator.joblib"


def calculate_risk_score(default_probability: float) -> int:
    """Custom project score, not an official FICO score."""
    if not 0 <= default_probability <= 1:
        raise ValueError("Probability must be between 0 and 1.")

    score = round(900 - 600 * default_probability)
    return max(300, min(900, score))


def calculate_expected_loss(
    default_probability: float,
    exposure_at_default: float,
    loss_given_default: float = 0.45,
) -> float:
    """Illustrative expected-loss estimate."""
    if not 0 <= default_probability <= 1:
        raise ValueError("Probability must be between 0 and 1.")
    if exposure_at_default < 0:
        raise ValueError("Exposure cannot be negative.")
    if not 0 <= loss_given_default <= 1:
        raise ValueError("LGD must be between 0 and 1.")

    return round(
        default_probability * exposure_at_default * loss_given_default,
        2,
    )


def predict_loan(loan_features: dict) -> dict:
    """
    Predict default risk for one loan.

    loan_features must contain all input features expected by the
    saved model. Missing and extra feature names are checked below.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if not CALIBRATOR_PATH.exists():
        raise FileNotFoundError(
            f"Calibrator not found: {CALIBRATOR_PATH}"
        )

    model = joblib.load(MODEL_PATH)
    calibrator = joblib.load(CALIBRATOR_PATH)

    required_features = list(model.feature_names_in_)

    missing = [
        name for name in required_features
        if name not in loan_features
    ]
    extra = [
        name for name in loan_features
        if name not in required_features
    ]

    if missing:
        raise ValueError(f"Missing required features: {missing}")

    if extra:
        raise ValueError(f"Unexpected features: {extra}")

    loan_df = pd.DataFrame(
        [{name: loan_features[name] for name in required_features}],
        columns=required_features,
    )

    raw_probability = float(model.predict_proba(loan_df)[0, 1])

    # Apply the same logit transformation used during calibration.
    clipped = np.clip(raw_probability, 1e-6, 1 - 1e-6)
    logit_probability = np.log(
        clipped / (1 - clipped)
    )

    calibrated_probability = float(
        calibrator.predict_proba([[logit_probability]])[0, 1]
    )

    loan_amount = float(loan_features["loan_amnt"])

    return {
        "raw_default_probability": round(raw_probability, 4),
        "default_probability": round(calibrated_probability, 4),
        "risk_score": calculate_risk_score(calibrated_probability),
        "loan_amount": loan_amount,
        "expected_loss": calculate_expected_loss(
            calibrated_probability,
            loan_amount,
        ),
        "lgd_assumption": 0.45,
        "score_type": "Custom illustrative score, not FICO",
    }


if __name__ == "__main__":
    print("Risk scoring module loaded.")
    print("Model exists:", MODEL_PATH.exists())
    print("Calibrator exists:", CALIBRATOR_PATH.exists())
    print("Ready to connect loan inputs to predictions.")
