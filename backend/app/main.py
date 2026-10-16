from typing import Literal
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict

app = FastAPI(title="Loan Default Intelligence API", version="0.1.0", description="Starter API contract; replace the transparent demo scorer with your trained LendingClub model.")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

class ScoreRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    loan_amnt: float = Field(gt=0)
    annual_inc: float = Field(gt=0)
    dti: float = Field(ge=0, le=100)
    delinq_2yrs: int = Field(ge=0)
    open_acc: int = Field(ge=0)
    pub_rec: int = Field(ge=0)
    revol_bal: float = Field(ge=0)
    revol_util: float = Field(ge=0, le=200)
    total_acc: int = Field(ge=0)
    term: Literal["36 months", "60 months"]
    emp_length: str
    home_ownership: str
    verification_status: str
    purpose: str
    exposure: float = Field(gt=0)
    lgd: float = Field(ge=0, le=1)

class Factor(BaseModel):
    feature: str
    impact: float

class ScoreResponse(BaseModel):
    credit_risk_score: int
    score_scale: str = "300-900"
    predicted_default_probability: float
    risk_band: str
    score_type: str
    top_factors: list[Factor]
    explanation: str
    expected_loss: float
    disclaimer: str

@app.get("/health")
def health():
    return {"status": "ok", "mode": "demo"}

@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest):
    """DEMO ONLY: transparent heuristic to validate frontend/API wiring. Not a trained model."""
    # Simple illustrative heuristic. Replace with model.predict_proba and calibrated score mapping.
    logit = -3.0 + 0.045 * req.dti + 0.000012 * req.loan_amnt - 0.000004 * req.annual_inc + 0.32 * req.delinq_2yrs + 0.16 * req.pub_rec + 0.012 * req.revol_util
    p = 1 / (1 + __import__("math").exp(-max(-20, min(20, logit))))
    score_value = round(900 - 600 * p)
    band = "Low risk" if score_value >= 780 else "Medium-low risk" if score_value >= 660 else "Medium risk" if score_value >= 540 else "High risk"
    factors = [Factor(feature="dti", impact=0.045 * req.dti), Factor(feature="loan_amnt", impact=0.000012 * req.loan_amnt), Factor(feature="annual_inc", impact=-0.000004 * req.annual_inc), Factor(feature="delinq_2yrs", impact=0.32 * req.delinq_2yrs), Factor(feature="revol_util", impact=0.012 * req.revol_util)]
    factors.sort(key=lambda item: abs(item.impact), reverse=True)
    return ScoreResponse(credit_risk_score=max(300, min(900, score_value)), predicted_default_probability=round(p, 6), risk_band=band, score_type="demo heuristic — not a trained model", top_factors=factors[:3], explanation=f"Demo heuristic estimates a {p:.1%} probability for the configured outcome. This is not a LendingClub-trained prediction; replace the heuristic with a validated model before drawing conclusions.", expected_loss=round(p * req.exposure * req.lgd, 2), disclaimer="Educational software demo only. Not a real credit score, lending decision, or financial advice.")
