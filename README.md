# Loan Default Intelligence

A full-stack credit-risk scorecard prototype built with **React, TypeScript and FastAPI**. Enter an applicant's details and the app shows a custom 300–900 display score, an estimated default probability, an expected-loss figure, a factor breakdown and a plain-English explanation.

> **Project status: prototype / demo.** The bundled API uses a **demo heuristic** to validate the frontend-backend wiring. It is **not trained on LendingClub data**, it is **not a bureau score**, and it **must not be used to make lending decisions**. See [Model status](#model-status) for details and [Roadmap](#roadmap-connecting-a-real-model) for how to connect a real model.

---

## Features

- **Applicant form** for loan and borrower inputs
- **Custom 300–900 display score** (a project score, not a FICO, CIBIL or Experian score)
- **Estimated default probability** and **expected-loss tile**
- **SHAP-ready factor list**, shaped to accept real per-feature contributions once a trained model is connected
- **Adverse-action reason placeholder** (needs legal review before any real use)
- **Plain-English explanation** section for non-technical readers
- **FastAPI backend** with auto-generated docs, a health endpoint and pytest tests
- **Docker support** for the API

## Tech stack

| Layer | Tools |
|---|---|
| Frontend | React, TypeScript, Vite |
| Backend | Python, FastAPI, Uvicorn |
| Testing | pytest |
| Deployment | Docker, Docker Compose |

## Model status

| Item | Status |
|---|---|
| Frontend UI and API wiring | Working |
| Scoring logic | Demo heuristic only |
| Trained on LendingClub data | No |
| Real SHAP values | No (the factor list is a placeholder) |
| Probability calibration and validation | Not done |
| Fairness review | Not done |

The initial result shown in the UI is illustrative. The UI-only bureau, age and region fields are intentionally **not** sent to the LendingClub endpoint.

## Project structure

```
loan-default-intelligence/
├── frontend/
│   ├── .env.example
│   ├── index.html
│   ├── package.json
│   ├── tsconfig.json
│   ├── vite.config.ts
│   └── src/
│       ├── App.tsx
│       ├── api.ts
│       ├── main.tsx
│       ├── styles.css
│       └── types.ts
├── backend/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── app/
│   │   ├── __init__.py
│   │   └── main.py
│   └── tests/test_api.py
├── docker-compose.yml
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

The API runs on port 8000. Run the frontend separately with `npm run dev`, or add a frontend container for a fully containerized setup.

## Roadmap: connecting a real model

1. **Choose the data.** Select a public LendingClub loan-performance dataset and verify its license and target definition.
2. **Define the target.** Fix the prediction point and target (for example, charged-off versus fully paid). Exclude records whose outcome would not be known at prediction time.
3. **Split by time.** Split by issue date, not only randomly, to reduce temporal leakage.
4. **Train a pipeline.** Build a preprocessing and classification pipeline. Save the exact ordered feature list and categorical mappings in a versioned artifact.
5. **Calibrate and evaluate.** Calibrate probabilities on a validation set. Report ROC-AUC, PR-AUC, Brier score, calibration, recall at the chosen threshold and subgroup performance.
6. **Replace the heuristic.** Swap the logic in `backend/app/main.py` for the saved pipeline's `predict_proba`, and make sure request fields and preprocessing match training exactly.
7. **Add real SHAP.** Compute SHAP values from the same fitted model and return the actual per-feature contributions. Do not label coefficients or heuristics as SHAP.
8. **Document the score mapping.** The starter mapping is illustrative. A 300–900 score here is a custom project score, not a bureau score.
9. **Review adverse-action logic.** Have reason codes reviewed legally before presenting them as compliant with any lending regulation.

## Deployment

- **Frontend:** deploy `frontend/` to Vercel or Netlify and set `VITE_API_BASE_URL` to your deployed API URL.
- **API:** deploy `backend/` to Render, Railway, Fly.io or any container host. Restrict CORS to your frontend domain in production.
- **Before real users:** add authentication, rate limits, controlled request logging, monitoring and a data-retention policy. Never commit secrets or applicant personal data.

## Disclaimer

This project is for learning and demonstration only. Scores, probabilities and explanations it produces are illustrative and are not financial, credit or legal advice. Do not use it to approve, deny or price credit.

## License

Add a license of your choice (for example, MIT) before sharing publicly.
