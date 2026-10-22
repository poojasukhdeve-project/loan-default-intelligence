import math
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "loan_default_validation_model.joblib"
CALIBRATOR_PATH = BASE_DIR / "models" / "loan_default_calibrator.joblib"

# ---------------------------------------------------------------------------
# Score scale (log-odds scorecard)
# ---------------------------------------------------------------------------
SCORE_MIN = 300
SCORE_MAX = 850
BASE_SCORE = 600   # score at the anchor default rate
BASE_PD = 0.20     # anchor: roughly the dataset's average default rate
PDO = 80           # points needed to double the good:bad odds

LGD_ASSUMPTION = 0.45


def calculate_risk_score(default_probability: float) -> int:
    """Custom project score (log-odds scorecard), not an official FICO score."""
    if not 0 <= default_probability <= 1:
        raise ValueError("Probability must be between 0 and 1.")

    pd_ = min(max(default_probability, 1e-6), 1 - 1e-6)
    odds_good = (1 - pd_) / pd_
    base_odds = (1 - BASE_PD) / BASE_PD
    factor = PDO / math.log(2)

    score = BASE_SCORE + factor * (math.log(odds_good) - math.log(base_odds))
    return int(round(max(SCORE_MIN, min(SCORE_MAX, score))))


def get_risk_band(score: int) -> str:
    """Map a custom score to a readable risk band."""
    if score >= 720:
        return "Low risk"
    if score >= 620:
        return "Moderate risk"
    if score >= 520:
        return "Elevated risk"
    return "High risk"


def calculate_expected_loss(
    default_probability: float,
    exposure_at_default: float,
    loss_given_default: float = LGD_ASSUMPTION,
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


@lru_cache(maxsize=1)
def _load_artifacts():
    """Load the model and calibrator once and reuse them across requests."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if not CALIBRATOR_PATH.exists():
        raise FileNotFoundError(f"Calibrator not found: {CALIBRATOR_PATH}")

    return joblib.load(MODEL_PATH), joblib.load(CALIBRATOR_PATH)


def predict_loan(loan_features: dict) -> dict:
    """
    Predict default risk for one loan.

    loan_features must contain all input features expected by the
    saved model. Missing and extra feature names are checked below.
    """
    model, calibrator = _load_artifacts()

    required_features = list(model.feature_names_in_)

    missing = [name for name in required_features if name not in loan_features]
    extra = [name for name in loan_features if name not in required_features]

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
    logit_probability = np.log(clipped / (1 - clipped))

    calibrated_probability = float(
        calibrator.predict_proba([[logit_probability]])[0, 1]
    )

    loan_amount = float(loan_features["loan_amnt"])
    risk_score = calculate_risk_score(calibrated_probability)

    return {
        "raw_default_probability": round(raw_probability, 4),
        "default_probability": round(calibrated_probability, 4),
        "risk_score": risk_score,
        "score_min": SCORE_MIN,
        "score_max": SCORE_MAX,
        "risk_band": get_risk_band(risk_score),
        "loan_amount": loan_amount,
        "expected_loss": calculate_expected_loss(
            calibrated_probability,
            loan_amount,
        ),
        "lgd_assumption": LGD_ASSUMPTION,
        "score_type": "Custom illustrative score, not FICO",
    }


if __name__ == "__main__":
    print("Risk scoring module loaded.")
    print("Model exists:", MODEL_PATH.exists())
    print("Calibrator exists:", CALIBRATOR_PATH.exists())

    # Quick sanity check of the score mapping
    for p in (0.02, 0.05, 0.10, 0.172, 0.30, 0.40, 0.60, 0.80):
        s = calculate_risk_score(p)
        print(f"PD {p:>5.1%} -> score {s} ({get_risk_band(s)})")