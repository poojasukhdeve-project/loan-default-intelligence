# Loan Default Intelligence

An AI-powered credit-risk scorecard built with **React, TypeScript and FastAPI**, backed by a model trained on **LendingClub historical loan data (2007 to 2018 Q4)**. A four-step loan application feeds the prediction API and returns a custom credit score, a calibrated default probability, an expected-loss figure, SHAP-based decision factors and a plain-English explanation.

> **Research demonstration only.** The score is a custom illustrative score, **not** an official FICO, CIBIL or Experian score. Do not use this app to approve, deny or price credit.

---

## Features

- **Four-step application form:** Credit Scores, Applicant Profile, Loan Details and Stability Indicators, with a progress tracker and validation.
- **Custom credit score** on a 300–850 scale with a risk band (for example, "Moderate Risk").
- **Calibrated default probability** from a model trained on LendingClub loan outcomes.
- **Annual payment ratio** (annualized monthly payment divided by income).
- **Expected loss** = default probability × loan exposure × an assumed 45% loss given default (LGD).
- **Decision factors (SHAP)** showing which features raise or lower the model output, computed from the fitted model.
- **Plain-English explanation** that can be switched on or off in the form.
- **Clear disclaimers** throughout: research use only and not legally reviewed adverse-action reasons.

## Machine learning pipeline

| Stage | What was done | Scripts |
|---|---|---|
| Data checks | Dataset inspection, missing-value and data-quality reports, outlier analysis | `check_dataset.py`, `check_missing_values.py`, `check_data_quality.py`, `inspect_outliers.py`, `analyze_outliers.py` |
| Cleaning and preparation | Cleaned dataset, modeling sample and mature-loan sample | `clean_dataset.py`, `prepare_dataset.py`, `final_data_validation.py` |
| Outcome maturity | Checked loan status and restricted training to loans with settled outcomes to avoid labeling unfinished loans | `inspect_loan_status.py`, `check_outcome_maturity.py`, `check_mature_cohort.py` |
| EDA | Outcome distribution and default rates by DTI, FICO and loan term (charts in `eda_outputs/`) | `eda.py` |
| Training and tuning | Logistic regression, gradient boosting and mature-cohort models, with logistic tuning | `train_model.py`, `train_best_model.py`, `train_mature_model.py`, `train_validate_model.py`, `tune_logistic.py` |
| Model comparison and diagnostics | Model comparison, yearly diagnostics, threshold evaluation | `compare_models.py`, `diagnose_model.py`, `evaluate_model.py`, `evaluate_thresholds.py` |
| Calibration | Probability calibration and calibration checks | `calibrate_model.py`, `check_calibration.py` |
| Explainability | SHAP explanations from the fitted model | `shap_explainer.py` |
| Scoring and serving | Score mapping and the FastAPI prediction service | `risk_scoring.py`, `app/main.py` |

Saved models and metadata are stored in `backend/models/` (`*.joblib` and `*_metadata.json`).

## Model details

| Item | Value |
|---|---|
| Dataset | LendingClub accepted loans, 2007 to 2018 Q4 (`accepted_2007_to_2018Q4.csv.gz`) |
| Target | Charged Off (default) vs. Fully Paid, using mature loans with settled outcomes |
| Split | Chronological by issue date. Test set: loans issued 2013-01-01 to 2015-06-01 (396,032 loans, 14.17% charged off) |
| Calibration | Probabilities calibrated with `calibrate_model.py` and checked by risk decile in `check_calibration.py` |
| ROC-AUC | 0.615 (test) |
| Average precision (PR-AUC) | 0.196 (test), against a 0.142 base default rate |
| Operating threshold | 0.55, selected on validation data (best validation F1) |
| Precision / Recall / F1 at threshold | 0.198 / 0.424 / 0.270 (test) |
| Confusion matrix (test) | TN 243,219, FP 96,690, FN 32,316, TP 23,807 |
| Fairness checks | Not yet evaluated |

**How to read these results.** The model ranks borrowers better than chance (ROC-AUC 0.615 vs. 0.5), and flagged loans default about 1.4 times as often as the portfolio average (precision 0.198 vs. a 0.142 base rate). That is modest discriminatory power, so the app is a research demonstration and not a decision tool. On the test set the model catches about 42% of defaults at the cost of many false alarms. Recall and precision at the 0.55 threshold were higher on the validation period (0.55 and 0.253) than on the later test period, which points to drift over time.

In the calibration check, predicted probabilities tracked observed default rates closely in most risk deciles (for example, 5.9% predicted vs. 5.5% observed in the lowest decile) but over-predicted the riskiest decile (31.5% vs. 23.1%).

## How the inputs are used

| Step | Fields | How they are used |
|---|---|---|
| 1. Credit Scores | Bureau Score 1, 2, 3 | The average is sent to the model as the FICO range (valid range 300–850). |
| 2. Applicant | Age, occupation, income source, employer type, education, marital status | UI reference only; not sent to the model. |
| 2. Applicant | Years at current job | Mapped to employment length. |
| 3. Loan Details | Loan amount, annual income, debt-to-income ratio, loan term, purpose, home ownership, income verification | Sent to the model. |
| 3. Loan Details | Monthly payment (EMI) | Used only to compute the annual payment ratio. |
| 4. Stability | Delinquencies, open accounts, public records, revolving balance and utilization, total accounts | Credit-history inputs for the model. |
| 4. Stability | Days since phone change / ID issued / registration, region population, region credit rating, display currency | UI reference only unless your API maps them; the currency setting does not convert model inputs. |

Keeping demographic and reference-only fields out of the model is the safer design for lending use cases.

## Tech stack

| Layer | Tools |
|---|---|
| Frontend | React, TypeScript, Vite |
| Backend | Python, FastAPI, Uvicorn |
| ML | scikit-learn, SHAP, joblib, pandas |
| Testing | pytest |
| Deployment | Docker, Docker Compose |

## Project structure

```
loan-default-intelligence/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   └── main.py              # FastAPI service
│   ├── data/                    # LendingClub data and samples (not committed)
│   ├── eda_outputs/             # EDA charts and quality reports
│   ├── models/                  # Saved models, calibrator and metadata
│   ├── tests/test_api.py
│   ├── *.py                     # Data prep, training, calibration, SHAP, evaluation scripts
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.tsx
│   │   ├── api.ts
│   │   ├── main.tsx
│   │   ├── styles.css
│   │   ├── types.ts
│   │   └── vite-env.d.ts
│   ├── .env.example
│   ├── index.html
│   ├── package.json
│   ├── tsconfig.json
│   └── vite.config.ts
├── docker-compose.yml
├── .gitignore
└── README.md
```

## Getting started

### Prerequisites

- Python 3.10+
- Node.js 18+ and npm
- Docker (optional)

### 1. Start the API

```bash
cd backend
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux:        source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

- API docs: http://127.0.0.1:8000/docs
- Health check: http://127.0.0.1:8000/health

### 2. Start the React UI (second terminal)

```bash
cd frontend
npm install
# Optional: copy .env.example to .env.local and set
# VITE_API_BASE_URL=http://127.0.0.1:8000
npm run dev
```

Open the URL printed by Vite, normally http://localhost:5173.

### 3. Run the tests

From `backend/` with the virtual environment active:

```bash
pytest -q
```

### 4. Run the API with Docker

From the repository root:

```bash
docker compose up --build
```

The API runs on port 8000. Run the frontend separately with `npm run dev`.

### Retraining the model (optional)

Download the LendingClub accepted-loans file, place it in `backend/data/`, then run the preparation, training, calibration and SHAP scripts listed in the pipeline table above. Check that the license of your data source allows your use.

## Known limitations

- The credit score is a custom illustrative mapping, not a bureau score.
- Predictive power is modest (test ROC-AUC 0.615), and the highest-risk decile is over-predicted after calibration.
- Expected loss uses a fixed 45% LGD assumption, not a modeled value.
- The model learns from 2007 to 2018 LendingClub loans, so it may not reflect current borrowers or economic conditions.
- LendingClub accepted loans only, so applicants who were rejected are not represented (selection bias).
- Decision factors are research explanations, not legally reviewed adverse-action reasons.
- Reference-only fields (age, region and similar) are not used for prediction.
- Fairness and subgroup performance have not yet been evaluated.

## Roadmap

- Add subgroup and fairness analysis.
- Show calibration plots in the app.
- Replace negative day counts with a "years ago" input.
- Add authentication, rate limits and monitoring before any real users.
- Add a frontend container to Docker Compose.

## Deployment

- **Frontend:** deploy `frontend/` to Vercel or Netlify and set `VITE_API_BASE_URL` to your deployed API URL.
- **API:** deploy `backend/` to Render, Railway, Fly.io or any container host. Restrict CORS to your frontend domain in production.
- **Large files:** keep the raw dataset and large model files out of Git (use `.gitignore`, Git LFS or a release asset).
- **Before real users:** add authentication, rate limits, controlled request logging, monitoring and a data-retention policy. Never commit secrets or applicant personal data.

## Disclaimer

This project is for learning and demonstration only. Scores, probabilities and explanations are illustrative and are not financial, credit or legal advice.
