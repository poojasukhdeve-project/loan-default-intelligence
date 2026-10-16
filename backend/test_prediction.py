
from risk_scoring import predict_loan

# Illustrative test applicant; these are not real borrower details.
sample_loan = {
    "loan_amnt": 10000,
    "annual_inc": 65000,
    "dti": 18.5,
    "fico_range_low": 680,
    "fico_range_high": 684,
    "delinq_2yrs": 0,
    "inq_last_6mths": 1,
    "open_acc": 8,
    "pub_rec": 0,
    "revol_bal": 12000,
    "revol_util": 35.0,
    "total_acc": 20,
    "term": " 36 months",
    "emp_length": "5 years",
    "home_ownership": "RENT",
    "verification_status": "Verified",
    "purpose": "debt_consolidation",
}

if __name__ == "__main__":
    result = predict_loan(sample_loan)

    print("\nLOAN DEFAULT INTELLIGENCE")
    print("=" * 35)

    for key, value in result.items():
        if key in ("raw_default_probability", "default_probability"):
            print(f"{key}: {value:.2%}")
        else:
            print(f"{key}: {value}")
