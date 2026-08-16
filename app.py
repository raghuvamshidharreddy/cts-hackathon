"""
Car Sence — Flask Application Entry Point (v2 Lasso Regression)
"""
import logging
import json
from flask import Flask, render_template, request, jsonify, send_from_directory
import os
from config import BRAND_MODELS, FUEL_TYPES, TRANSMISSIONS, DOOR_OPTIONS, YEARS, MODEL_PATH
from services.prediction import ModelService
from services.validation import validate_input

# ── Logging ───────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── Flask app ─────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = "carsense-lasso-v2-secret"

# ── Load model at startup ─────────────────────────────────────────────
model_service = ModelService()
try:
    model_service.load(MODEL_PATH)
    logger.info("Lasso Regression ModelService loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load model: {e}")


def _template_context():
    """Shared context passed to every template render."""
    return {
        "brand_models":   BRAND_MODELS,
        "fuel_types":     FUEL_TYPES,
        "transmissions":  TRANSMISSIONS,
        "door_options":   DOOR_OPTIONS,
        "years":          YEARS,
        "brand_models_json": json.dumps(BRAND_MODELS),
    }


# ── Routes ────────────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", **_template_context())


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """JSON prediction endpoint used by the SPA frontend."""
    try:
        data = request.get_json(force=True, silent=True) or {}
        validation = validate_input(data)
        if not validation["valid"]:
            return jsonify({"success": False, "error": "; ".join(validation["errors"])}), 400

        result = model_service.predict(data)
        return jsonify(result), 200

    except Exception as e:
        logger.exception("Prediction error")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": model_service.is_loaded,
        "model_type": "LassoRegression",
        "features": len(model_service.feature_names()),
    })


@app.route("/model-outputs", methods=["GET"])
def model_outputs():
    """Model training outputs and evaluation results page."""
    return render_template("model_outputs.html", **_template_context())


@app.route("/lasso", methods=["GET"])
def lasso_page():
    """Lasso Regression working diagram page."""
    return render_template("lasso.html", **_template_context())


@app.route("/outputs/<path:filename>", methods=["GET"])
def serve_output(filename):
    """Serve files from the outputs/ folder."""
    outputs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    return send_from_directory(outputs_dir, filename)


@app.route("/api/model-info", methods=["GET"])
def model_info():
    return jsonify({
        "model_type": "LassoRegression",
        "features": model_service.feature_names(),
        "total_features": len(model_service.feature_names()),
        "currency": "USD",
    })


# ── Error handlers ────────────────────────────────────────────────────
@app.errorhandler(400)
def bad_request(e):
    return jsonify({"success": False, "error": "Bad request", "details": str(e)}), 400


@app.errorhandler(404)
def not_found(e):
    return jsonify({"success": False, "error": "Endpoint not found"}), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({"success": False, "error": "Internal server error"}), 500


# ── Entry point ───────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
