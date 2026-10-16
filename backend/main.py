
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from risk_scoring import predict_loan

app = FastAPI(
    title="Loan Default Intelligence API",
    description="Loan default risk and expected-loss estimation",
    version="1.0.0",
)


class LoanApplication(BaseModel):
    loan_amnt: float = Field(gt=0)
    annual_inc: float = Field(gt=0)
    dti: float = Field(ge=0, lt=100)
    fico_range_low: int = Field(ge=300, le=850)
    fico_range_high: int = Field(ge=300, le=850)
    delinq_2yrs: int = Field(ge=0)
    inq_last_6mths: float = Field(ge=0)
    open_acc: int = Field(ge=0)
    pub_rec: int = Field(ge=0)
    revol_bal: float = Field(ge=0)
    revol_util: float = Field(ge=0, le=100)
    total_acc: int = Field(ge=0)
    term: str
    emp_length: str
    home_ownership: str
    verification_status: str
    purpose: str


@app.get("/")
def root():
    return {"message": "Loan Default Intelligence API is running"}


@app.post("/predict")
def predict(application: LoanApplication):
    try:
        return predict_loan(application.model_dump())
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
