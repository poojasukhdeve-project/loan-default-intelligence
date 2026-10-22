from shap_explainer import explain_prediction
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from risk_scoring import predict_loan

app = FastAPI(
    title="Loan Default Intelligence API",
    description="Loan default risk and expected-loss estimation",
    version="1.0.0",
)

# Allow your React frontend to communicate with FastAPI
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
        inputs = application.model_dump()
        result = predict_loan(inputs)

        factors = explain_prediction(inputs)
        result["top_factors"] = factors

        return result

    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction or SHAP explanation failed: {exc}",
        )
