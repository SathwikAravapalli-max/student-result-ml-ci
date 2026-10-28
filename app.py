
import os
import sys
import time
import uuid
import logging
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request, g

# --------------------------------------------------
# Application and logging configuration
# --------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s | %(levelname)s | "
        "%(name)s | %(message)s"
    ),
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)

logger = logging.getLogger("student-result-api")

app = Flask(__name__)

MODEL_PATH = Path("student_result_model.pkl")

FEATURES = [
    "attendance",
    "internal_marks",
    "assignment_marks",
    "previous_score",
]

logger.info("event=application_started service=student-result-prediction")


# --------------------------------------------------
# Request logging and request ID tracking
# --------------------------------------------------

@app.before_request
def start_request_tracking():
    g.request_id = request.headers.get(
        "X-Request-ID", str(uuid.uuid4())
    )
    g.start_time = time.perf_counter()


@app.after_request
def log_request(response):
    request_id = getattr(g, "request_id", "unknown")
    start_time = getattr(g, "start_time", time.perf_counter())

    duration_ms = (time.perf_counter() - start_time) * 1000

    log_level = (
        logging.ERROR if response.status_code >= 500
        else logging.WARNING if response.status_code >= 400
        else logging.INFO
    )

    logger.log(
        log_level,
        "event=http_request request_id=%s method=%s "
        "path=%s status=%s duration_ms=%.2f",
        request_id,
        request.method,
        request.path,
        response.status_code,
        duration_ms,
    )

    response.headers["X-Request-ID"] = request_id
    return response


# --------------------------------------------------
# Model loading
# --------------------------------------------------

def load_model():
    if not MODEL_PATH.exists():
        logger.error(
            "event=model_load_failed model_file=%s",
            MODEL_PATH,
        )
        raise FileNotFoundError(
            "student_result_model.pkl was not found. "
            "Run the training pipeline first."
        )

    try:
        model = joblib.load(MODEL_PATH)
        logger.info(
            "event=model_loaded model_file=%s",
            MODEL_PATH,
        )
        return model
    except Exception:
        logger.exception(
            "event=model_load_failed model_file=%s",
            MODEL_PATH,
        )
        raise


# --------------------------------------------------
# Health endpoints
# --------------------------------------------------

@app.get("/")
def health_check():
    return jsonify({
        "status": "ok",
        "service": "student-result-prediction",
    }), 200


@app.get("/health")
def health():
    return jsonify({
        "status": "healthy"
    }), 200


# --------------------------------------------------
# Student result prediction endpoint
# --------------------------------------------------

@app.post("/predict")
def predict():
    data = request.get_json(silent=True)

    if not data or not isinstance(data, dict):
        logger.warning(
            "event=invalid_request request_id=%s reason=invalid_json",
            g.request_id,
        )
        return jsonify({
            "error": "JSON request body is required"
        }), 400

    missing_fields = [
        feature for feature in FEATURES
        if feature not in data
    ]

    if missing_fields:
        logger.warning(
            "event=validation_error request_id=%s "
            "missing_fields=%s",
            g.request_id,
            missing_fields,
        )
        return jsonify({
            "error": "Missing required fields",
            "missing_fields": missing_fields,
        }), 400

    try:
        sample = pd.DataFrame([{
            feature: float(data[feature])
            for feature in FEATURES
        }])
    except (TypeError, ValueError):
        logger.warning(
            "event=validation_error request_id=%s "
            "reason=non_numeric_input",
            g.request_id,
        )
        return jsonify({
            "error": "All input fields must contain numeric values"
        }), 400

    try:
        model = load_model()
        prediction_code = int(model.predict(sample)[0])

        prediction = (
            "PASS" if prediction_code == 1 else "FAIL"
        )

        logger.info(
            "event=prediction_success request_id=%s "
            "prediction=%s",
            g.request_id,
            prediction,
        )

        return jsonify({
            "prediction": prediction,
            "prediction_code": prediction_code,
        }), 200

    except Exception:
        logger.exception(
            "event=prediction_failed request_id=%s",
            g.request_id,
        )
        return jsonify({
            "error": "Prediction failed"
        }), 500


# --------------------------------------------------
# Application startup
# --------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    logger.info("event=server_starting port=%s", port)
    app.run(host="0.0.0.0", port=port)
