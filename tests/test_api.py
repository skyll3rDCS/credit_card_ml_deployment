import pytest
from app.api import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def sample_features():
    return {
        "LIMIT_BAL": 20000,
        "SEX": 2,
        "EDUCATION": 2,
        "MARRIAGE": 1,
        "AGE": 24,
        "PAY_0": 2,
        "PAY_2": 2,
        "PAY_3": -1,
        "PAY_4": -1,
        "PAY_5": -2,
        "PAY_6": -2,
        "BILL_AMT1": 3913,
        "BILL_AMT2": 3102,
        "BILL_AMT3": 689,
        "BILL_AMT4": 0,
        "BILL_AMT5": 0,
        "BILL_AMT6": 0,
        "PAY_AMT1": 0,
        "PAY_AMT2": 689,
        "PAY_AMT3": 0,
        "PAY_AMT4": 0,
        "PAY_AMT5": 0,
        "PAY_AMT6": 0,
    }


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json == {"status": "ok"}


def test_predict_with_explicit_v1(client):
    payload = {
        "user_id": "user-100",
        "model_version": "v1",
        "features": sample_features(),
    }

    response = client.post("/predict", json=payload)
    body = response.json

    assert response.status_code == 200
    assert body["model_version"] == "v1"
    assert "request_id" in body
    assert "prediction" in body
    assert "probability" in body


def test_predict_with_explicit_v2(client):
    payload = {
        "user_id": "user-101",
        "model_version": "v2",
        "features": sample_features(),
    }

    response = client.post("/predict", json=payload)
    body = response.json

    assert response.status_code == 200
    assert body["model_version"] == "v2"
    assert "request_id" in body
    assert "prediction" in body
    assert "probability" in body


def test_predict_with_user_id_auto_assignment_is_deterministic(client):
    payload = {
        "user_id": "stable-user-42",
        "features": sample_features(),
    }

    response1 = client.post("/predict", json=payload)
    response2 = client.post("/predict", json=payload)

    assert response1.status_code == 200
    assert response2.status_code == 200
    assert response1.json["model_version"] in ("v1", "v2")
    assert response1.json["model_version"] == response2.json["model_version"]


def test_predict_invalid_model_version_returns_400(client):
    payload = {
        "user_id": "user-102",
        "model_version": "v3",
        "features": sample_features(),
    }

    response = client.post("/predict", json=payload)
    body = response.json

    assert response.status_code == 400
    assert "request_id" in body
    assert "error" in body
    assert "model_version" in body["error"]


def test_predict_without_user_id_and_model_version_returns_400(client):
    payload = {
        "features": sample_features(),
    }

    response = client.post("/predict", json=payload)
    body = response.json

    assert response.status_code == 400
    assert "request_id" in body
    assert "error" in body
    assert "user_id" in body["error"]


def test_predict_legacy_payload_still_supported(client):
    # Legacy: передаем признаки в корне JSON без "features"
    payload = sample_features()
    payload["user_id"] = "legacy-user-1"  # нужен для авто-выбора версии

    response = client.post("/predict", json=payload)
    body = response.json

    assert response.status_code == 200
    assert body["model_version"] in ("v1", "v2")
    assert "request_id" in body
    assert "prediction" in body
    assert "probability" in body


def test_predict_invalid_json(client):
    response = client.post(
        "/predict",
        data="not a json",
        content_type="application/json"
    )
    # Flask может вернуть 400/415 до попадания в endpoint
    assert response.status_code in (400, 415)