
import os
import sys
import time
import uuid
import logging
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request, g, Response

from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    generate_latest,
    CONTENT_TYPE_LATEST,
)

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

logger.info(
    "event=application_started "
    "service=student-result-prediction"
)

# --------------------------------------------------
# Prometheus metric definitions
# --------------------------------------------------

HTTP_REQUESTS = Counter(
    "student_api_http_requests_total",
    "Total HTTP requests received by the Student Result Prediction API",
    ["method", "path", "status"],
)

HTTP_LATENCY = Histogram(
    "student_api_http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "path"],
)

PREDICTIONS = Counter(
    "student_api_predictions_total",
    "Total Student Result Prediction attempts",
    ["result"],
)

MODEL_LOADED = Gauge(
    "student_api_model_loaded",
    "Whether the Student Result Prediction model is loaded",
)

# --------------------------------------------------
# Model loading
# --------------------------------------------------

MODEL = None


def load_model():
    global MODEL

    if MODEL is not None:
        return MODEL

    if not MODEL_PATH.exists():
        MODEL_LOADED.set(0)

        logger.error(
            "event=model_load_failed model_file=%s",
            MODEL_PATH,
        )

        raise FileNotFoundError(
            "student_result_model.pkl was not found. "
            "Run the training pipeline first."
        )

    try:
        MODEL = joblib.load(MODEL_PATH)
        MODEL_LOADED.set(1)

        logger.info(
            "event=model_loaded model_file=%s",
            MODEL_PATH,
        )

        return MODEL

    except Exception:
        MODEL_LOADED.set(0)

        logger.exception(
            "event=model_load_failed model_file=%s",
            MODEL_PATH,
        )

        raise


# Load the model when the application starts.
try:
    load_model()
except Exception:
    logger.exception(
        "event=initial_model_loading_failed"
    )


# --------------------------------------------------
# Request ID tracking and Prometheus middleware
# --------------------------------------------------

@app.before_request
def start_request_tracking():
    g.request_id = request.headers.get(
        "X-Request-ID",
        str(uuid.uuid4()),
    )

    g.start_time = time.perf_counter()


@app.after_request
def log_request(response):
    request_id = getattr(
        g, "request_id", "unknown"
    )

    start_time = getattr(
        g, "start_time", time.perf_counter()
    )

    duration = time.perf_counter() - start_time
    duration_ms = duration * 1000

    log_level = (
        logging.ERROR if response.status_code >= 500
        else logging.WARNING if response.status_code >= 400
        else logging.INFO
    )

    logger.log(
        log_level,
        "event=http_request request_id=%s "
        "method=%s path=%s status=%s duration_ms=%.2f",
        request_id,
        request.method,
        request.path,
        response.status_code,
        duration_ms,
    )

    # Do not count Prometheus scrape requests
    # as normal user traffic.
    if request.path != "/metrics":
        HTTP_REQUESTS.labels(
            method=request.method,
            path=request.path,
            status=str(response.status_code),
        ).inc()

        HTTP_LATENCY.labels(
            method=request.method,
            path=request.path,
        ).observe(duration)

    response.headers["X-Request-ID"] = request_id

    return response


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
            "event=invalid_request request_id=%s "
            "reason=invalid_json",
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

        prediction_code = int(
            model.predict(sample)[0]
        )

        prediction = (
            "PASS" if prediction_code == 1 else "FAIL"
        )

        PREDICTIONS.labels(
            result="success"
        ).inc()

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
        PREDICTIONS.labels(
            result="error"
        ).inc()

        logger.exception(
            "event=prediction_failed request_id=%s",
            g.request_id,
        )

        return jsonify({
            "error": "Prediction failed"
        }), 500


# --------------------------------------------------
# Prometheus metrics endpoint
# --------------------------------------------------

@app.get("/metrics")
def metrics():
    return Response(
        generate_latest(),
        mimetype=CONTENT_TYPE_LATEST,
    )


# --------------------------------------------------
# Application startup
# --------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))

    logger.info(
        "event=server_starting port=%s",
        port,
    )

    app.run(
        host="0.0.0.0",
        port=port,
    )
