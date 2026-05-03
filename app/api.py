from flask import Flask, request, jsonify
from app.model_handler import ModelHandler
import logging
import json
import time

app = Flask(__name__)

# Инициализация модели
model_handler = ModelHandler()

# Настройка логирования (JSON-формат)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ml_api")


def log_request_response(endpoint, request_data, response_data, latency):
    log_entry = {
        "endpoint": endpoint,
        "request": request_data,
        "response": response_data,
        "latency_ms": round(latency * 1000, 2)
    }
    logger.info(json.dumps(log_entry))


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/predict", methods=["POST"])
def predict():
    start_time = time.time()

    try:
        input_data = request.get_json()

        if input_data is None:
            return jsonify({"error": "Invalid JSON"}), 400

        prediction_result = model_handler.predict(input_data)

        latency = time.time() - start_time

        log_request_response(
            endpoint="/predict",
            request_data=input_data,
            response_data=prediction_result,
            latency=latency
        )

        return jsonify(prediction_result), 200

    except Exception as e:
        latency = time.time() - start_time

        error_response = {"error": str(e)}

        log_request_response(
            endpoint="/predict",
            request_data=request.get_json(),
            response_data=error_response,
            latency=latency
        )

        return jsonify(error_response), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)