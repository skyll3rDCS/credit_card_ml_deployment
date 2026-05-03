from flask import Flask, request, jsonify
from app.model_handler import ModelHandler
import logging
import json
import time
import hashlib
import uuid

app = Flask(__name__)

# Инициализация моделей (v1 и v2)
model_handler = ModelHandler()

# Настройка логирования (JSON-формат)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ml_api")


def log_request_response(endpoint, request_data, response_data, latency, request_id):
    log_entry = {
        "endpoint": endpoint,
        "request_id": request_id,
        "request": request_data,
        "response": response_data,
        "latency_ms": round(latency * 1000, 2),
    }
    logger.info(json.dumps(log_entry, ensure_ascii=False))


def assign_model_version(user_id):
    """
    Детерминированное распределение 50/50:
    один и тот же user_id всегда попадет в одну и ту же группу.
    """
    bucket = int(hashlib.sha256(user_id.encode("utf-8")).hexdigest(), 16) % 100
    return "v1" if bucket < 50 else "v2"


def resolve_model_version(payload):
    explicit_version = payload.get("model_version")
    if explicit_version is not None:
        if explicit_version not in ("v1", "v2"):
            raise ValueError("model_version must be 'v1' or 'v2'")
        return explicit_version

    user_id = payload.get("user_id")
    if not user_id:
        raise ValueError("user_id is required when model_version is not provided")

    return assign_model_version(str(user_id))


def extract_features(payload):
    if "features" in payload:
        if not isinstance(payload["features"], dict):
            raise ValueError("'features' must be an object")
        return payload["features"]

    return payload


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/predict", methods=["POST"])
def predict():
    start_time = time.time()
    request_id = str(uuid.uuid4())

    try:
        payload = request.get_json(silent=True)
        if payload is None:
            return jsonify({"request_id": request_id, "error": "Invalid JSON"}), 400

        model_version = resolve_model_version(payload)
        features = extract_features(payload)

        prediction_result = model_handler.predict(features, model_version)

        response = {
            "request_id": request_id,
            "model_version": model_version,
            "prediction": prediction_result["prediction"],
            "probability": prediction_result["probability"],
        }

        latency = time.time() - start_time
        log_request_response(
            endpoint="/predict",
            request_data=payload,
            response_data=response,
            latency=latency,
            request_id=request_id,
        )

        return jsonify(response), 200

    except ValueError as e:
        latency = time.time() - start_time
        error_response = {"request_id": request_id, "error": str(e)}

        log_request_response(
            endpoint="/predict",
            request_data=request.get_json(silent=True),
            response_data=error_response,
            latency=latency,
            request_id=request_id,
        )

        return jsonify(error_response), 400

    except Exception as e:
        latency = time.time() - start_time
        error_response = {"request_id": request_id, "error": str(e)}

        log_request_response(
            endpoint="/predict",
            request_data=request.get_json(silent=True),
            response_data=error_response,
            latency=latency,
            request_id=request_id,
        )

        return jsonify(error_response), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)