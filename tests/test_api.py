from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.api.serve_model import app

client = TestClient(app)

def test_home_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {
        "message": "Fraud Detection API is Running."
    }

def test_predict_valid_transaction():
    payload = {
        "features": {
            "purchase_value": 50,
            "age": 30,
            "time_since_signup_seconds": 400000,
            "device_user_count": 1,
            "country_enc": 23
        }
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200

    data = response.json()

    assert "prediction" in data
    assert "fraud_probability" in data
    assert "risk_level" in data
    assert "is_fraud" in data

    assert isinstance(data["prediction"], int)
    assert isinstance(data["fraud_probability"], float)
    assert isinstance(data["is_fraud"], bool)
    assert data["risk_level"] in {"Low", "Medium", "High"}

    assert 0.0 <= data["fraud_probability"] <= 1.0

def test_predict_malformed_input():
    payload = {"features": "This is not a dictionary"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
