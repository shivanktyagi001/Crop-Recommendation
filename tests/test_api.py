
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_home():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["message"] == (
        "Crop Recommendation API is running"
    )


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["model_loaded"] is True


def test_crop_prediction():
    payload = {
        "N": 90,
        "P": 42,
        "K": 43,
        "temperature": 20.8,
        "humidity": 82.0,
        "ph": 6.5,
        "rainfall": 202.9,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    result = response.json()

    assert result["recommended_crop"] in (
        "rice", "maize", "chickpea", "kidneybeans",
        "pigeonpeas", "mothbeans", "mungbean",
        "blackgram", "lentil", "pomegranate",
        "banana", "mango", "grapes", "watermelon",
        "muskmelon", "apple", "orange", "papaya",
        "coconut", "cotton", "jute", "coffee",
    )

    assert 0 <= result["confidence"] <= 1


def test_invalid_input():
    payload = {
        "N": -1,
        "P": 42,
        "K": 43,
        "temperature": 20.8,
        "humidity": 82.0,
        "ph": 6.5,
        "rainfall": 202.9,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_missing_feature():
    payload = {
        "N": 90,
        "P": 42,
        "K": 43,
        "temperature": 20.8,
        "humidity": 82.0,
        "ph": 6.5,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422
