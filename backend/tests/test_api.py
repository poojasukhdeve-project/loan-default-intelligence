from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)
PAYLOAD = {
    "loan_amnt": 12000, "annual_inc": 72000, "dti": 18.5, "delinq_2yrs": 0,
    "open_acc": 8, "pub_rec": 0, "revol_bal": 8500, "revol_util": 32,
    "total_acc": 18, "term": "36 months", "emp_length": "5 years",
    "home_ownership": "MORTGAGE", "verification_status": "Verified",
    "purpose": "debt_consolidation", "exposure": 12000, "lgd": 0.45
}

def test_health():
    response = client.get('/health')
    assert response.status_code == 200
    assert response.json()['status'] == 'ok'

def test_score_contract_and_ranges():
    response = client.post('/score', json=PAYLOAD)
    assert response.status_code == 200
    data = response.json()
    assert 300 <= data['credit_risk_score'] <= 900
    assert 0 <= data['predicted_default_probability'] <= 1
    assert data['expected_loss'] >= 0
    assert isinstance(data['top_factors'], list)

def test_invalid_dti_rejected():
    payload = {**PAYLOAD, 'dti': 140}
    assert client.post('/score', json=payload).status_code == 422
