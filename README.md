# Loan Default Intelligence — React + TypeScript + FastAPI

A navy-and-white credit-risk scorecard UI inspired by the supplied reference. It includes an applicant form, custom 300–900 display score, estimated default probability, expected-loss tile, SHAP-ready factor list, adverse-action-reason placeholder, and plain-English explanation section.

## Important model caveat

The bundled API is a **demo heuristic** used only to validate frontend/backend wiring. It is not trained on LendingClub data, not a bureau score, and must not be used to make lending decisions. The initial result shown in the UI is also illustrative. The UI-only bureau/age/region fields are intentionally not sent to the LendingClub endpoint. Before using real predictions, define the target and time split, train/calibrate a model, align API features with the fitted pipeline, generate SHAP values from that model, and validate performance and fairness.

## Folder structure

```text
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

## Run locally in VS Code

### 1. Start the API

```bash
cd backend
python -m venv .venv
# Windows PowerShell: .venv\\Scripts\\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Open http://127.0.0.1:8000/docs to inspect the API. Health endpoint: http://127.0.0.1:8000/health.

### 2. Start the React UI in a second terminal

```bash
cd frontend
npm install
# Optional: copy .env.example to .env.local and set VITE_API_BASE_URL=http://127.0.0.1:8000
npm run dev
```

Open the Vite URL printed in the terminal, normally http://localhost:5173.

### 3. Run backend tests

From `backend/` with the virtual environment active:

```bash
pytest -q
```

### 4. Run API with Docker

From the repository root:

```bash
docker compose up --build
```

The API runs on port 8000. Run the frontend separately with `npm run dev`, or add a frontend container for a full containerized deployment.

## Connect a real LendingClub model

1. Select a public LendingClub loan-performance dataset and verify its license and target definition.
2. Define the prediction point and target (for example, default/charged-off versus fully paid); exclude records whose outcome would not be known at prediction time.
3. Split by issue date rather than random-only split to reduce temporal leakage.
4. Train a preprocessing + classification pipeline. Keep the exact ordered feature list and categorical mappings in a versioned artifact.
5. Calibrate probabilities on a validation set; evaluate ROC-AUC, PR-AUC, Brier score, calibration, recall at an operating threshold, and subgroup performance.
6. Replace the heuristic in `backend/app/main.py` with the saved pipeline's `predict_proba`. Ensure request fields and preprocessing match training exactly.
7. Compute SHAP explanations from the same fitted model and return actual per-feature contributions. Do not label arbitrary coefficients/heuristics as SHAP.
8. Document score mapping explicitly. The starter mapping is illustrative; a 300–900 score is a custom project score, not a FICO/CIBIL/Experian score.
9. Use legally reviewed adverse-action reason logic before presenting any reason as compliant with a lending regulation.

## Deployment outline

- Frontend: deploy `frontend/` to Vercel or Netlify; configure `VITE_API_BASE_URL` to your deployed API URL.
- API: deploy `backend/` to Render, Railway, Fly.io, or a container host; configure production CORS for your frontend domain only.
- Never commit secrets or applicant personal data. Add authentication, rate limits, request logging controls, monitoring, and data-retention policy before accepting real users.
