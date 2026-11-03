
import logging
import sys
import time
import uuid
import math
import os
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
# 1. Flask application
# --------------------------------------------------

app = Flask(__name__)

MODEL_PATH = Path("student_result_model.pkl")

FEATURES = [
    "attendance",
    "internal_marks",
    "assignment_marks",
    "previous_score",
]

# --------------------------------------------------
# 2. Prometheus metrics
# --------------------------------------------------

REQUEST_COUNT = Counter(
    "student_api_http_requests_total",
    "Total HTTP requests received by the Student Result Prediction API",
    ["method", "path", "status"],
)

REQUEST_DURATION = Histogram(
    "student_api_http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "path"],
)

PREDICTIONS_TOTAL = Counter(
    "student_api_predictions_total",
    "Total Student Result Prediction attempts",
    ["result"],
)

MODEL_LOADED = Gauge(
    "student_api_model_loaded",
    "Whether the Student Result Prediction model is loaded",
)

# --------------------------------------------------
# 3. Configure application logging
# --------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)

logger = logging.getLogger("student-result-api")

model = None

# --------------------------------------------------
# 4. Load ML model
# --------------------------------------------------

def load_model():
    global model

    if model is None:
        try:
            if not MODEL_PATH.exists():
                raise FileNotFoundError(
                    "student_result_model.pkl was not found"
                )

            model = joblib.load(MODEL_PATH)
            MODEL_LOADED.set(1)

            logger.info(
                "event=model_loaded model_file=%s",
                MODEL_PATH
            )

        except Exception:
            MODEL_LOADED.set(0)

            logger.exception(
                "event=model_load_failed model_file=%s",
                MODEL_PATH
            )
            raise

    return model


# --------------------------------------------------
# 5. Request logging and monitoring
# --------------------------------------------------

@app.before_request
def before_request():
    g.request_id = request.headers.get(
        "X-Request-ID", str(uuid.uuid4())
    )
    g.start_time = time.perf_counter()


@app.after_request
def after_request(response):
    duration_seconds = time.perf_counter() - g.start_time
    duration_ms = duration_seconds * 1000

    REQUEST_COUNT.labels(
        request.method,
        request.path,
        str(response.status_code),
    ).inc()

    REQUEST_DURATION.labels(
        request.method,
        request.path,
    ).observe(duration_seconds)

    level = (
        logging.WARNING
        if response.status_code >= 400
        else logging.INFO
    )

    logger.log(
        level,
        "event=http_request request_id=%s method=%s path=%s status=%s duration_ms=%.2f",
        g.request_id,
        request.method,
        request.path,
        response.status_code,
        duration_ms,
    )

    response.headers["X-Request-ID"] = g.request_id

    return response


# --------------------------------------------------
# 6. Health endpoint
# --------------------------------------------------

@app.get("/")
@app.get("/health")
def health_check():
    return jsonify({
        "status": "ok",
        "service": "student-result-prediction",
    })


# --------------------------------------------------
# 7. Prometheus metrics endpoint
# --------------------------------------------------

@app.get("/metrics")
def metrics():
    return Response(
        generate_latest(),
        mimetype=CONTENT_TYPE_LATEST,
    )


# --------------------------------------------------
# 8. Prediction endpoint
# --------------------------------------------------

@app.post("/predict")
def predict():

    try:
        data = request.get_json(silent=True)

        if not data:
            return jsonify({
                "error": "JSON request body is required"
            }), 400

        # Check missing fields
        missing_fields = [
            feature for feature in FEATURES
            if feature not in data
        ]

        if missing_fields:
            logger.warning(
                "event=validation_error missing_fields=%s",
                missing_fields
            )

            return jsonify({
                "error": "Missing required fields",
                "missing_fields": missing_fields,
            }), 400

        # Check numeric values
        invalid_fields = []

        for feature in FEATURES:
            try:
                value = float(data[feature])

                if not math.isfinite(value):
                    invalid_fields.append(feature)

            except (ValueError, TypeError):
                invalid_fields.append(feature)

        if invalid_fields:
            logger.warning(
                "event=validation_error invalid_fields=%s",
                invalid_fields
            )

            return jsonify({
                "error": "All input fields must contain numeric values",
                "invalid_fields": invalid_fields,
            }), 400

        # Convert valid inputs to numeric values
        payload = {
            feature: float(data[feature])
            for feature in FEATURES
        }

        logger.info(
            "event=prediction_request feature_names=%s",
            FEATURES
        )

        # Prepare input data
        sample = pd.DataFrame([payload])

        # Load model and predict
        ml_model = load_model()

        prediction_code = int(
            ml_model.predict(sample)[0]
        )

        prediction = (
            "PASS" if prediction_code == 1 else "FAIL"
        )

        PREDICTIONS_TOTAL.labels(
            result="success"
        ).inc()

        logger.info(
            "event=prediction_success prediction=%s",
            prediction
        )

        return jsonify({
            "prediction": prediction,
            "prediction_code": prediction_code,
        })

    except Exception:
        PREDICTIONS_TOTAL.labels(
            result="failure"
        ).inc()

        logger.exception("event=prediction_failed")

        return jsonify({
            "error": "Prediction failed"
        }), 500


# --------------------------------------------------
# 9. Run Flask application
# --------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
