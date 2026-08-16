"""
Prediction Service — Car Sence v2
Lasso regression wrapper with USD formatting.
"""
import logging
import joblib
import pandas as pd
from services.preprocessing import build_feature_vector

logger = logging.getLogger(__name__)

BRAND_ADJUSTMENT = {
    "Audi": 1.07,
    "BMW": 1.10,
    "Mercedes": 1.12,
    "Toyota": 1.03,
    "Honda": 1.02,
    "Ford": 0.99,
    "Hyundai": 1.00,
    "Kia": 0.98,
    "Chevrolet": 0.97,
    "Volkswagen": 1.01,
}

MODEL_TIER_ADJUSTMENT = {
    "Base": 0.98,
    "Standard": 1.02,
    "Premium": 1.08,
}


class ModelService:
    """Singleton that loads the saved Lasso model bundle and serves predictions."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._model = None
            cls._instance._scaler = None
            cls._instance._feature_columns = []
            cls._instance._bundle = None
        return cls._instance

    def load(self, model_path: str):
        from config import MODEL_PATH
        path = model_path or MODEL_PATH
        logger.info(f"Loading model bundle from {path}...")
        bundle = joblib.load(path)

        if isinstance(bundle, dict) and "model" in bundle and "feature_columns" in bundle:
            self._bundle = bundle
            self._model = bundle["model"]
            self._scaler = bundle.get("scaler")
            self._feature_columns = list(bundle.get("feature_columns", []))
            logger.info(f"Loaded saved model bundle: {bundle.get('model_name', 'Lasso Regression')}")
            logger.info(f"Feature count: {len(self._feature_columns)}")
            return

        self._bundle = None
        self._model = bundle
        self._scaler = None
        self._feature_columns = list(getattr(bundle, "feature_names_in_", []))
        logger.info(f"Loaded direct model: {type(self._model).__name__}")

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def predict(self, data: dict) -> dict:
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        X = build_feature_vector(data)
        feature_columns = self._feature_columns or list(X.columns)
        X = X.reindex(columns=feature_columns, fill_value=0.0)

        if self._scaler is not None:
            X_scaled = X.copy()
            numeric_cols = ["Year", "Mileage", "Mileage_Per_Year"]
            transformed = self._scaler.transform(X[numeric_cols])
            X_scaled[numeric_cols] = transformed
            raw_price = float(self._model.predict(X_scaled)[0])
        else:
            raw_price = float(self._model.predict(X)[0])

        adjusted_price = max(500.0, float(raw_price) * _brand_model_adjustment(data, self._bundle))
        price_low = adjusted_price * 0.92
        price_high = adjusted_price * 1.08

        return {
            "success": True,
            "predicted_price": round(adjusted_price, 2),
            "formatted_usd": _fmt_usd(adjusted_price),
            "formatted_k": _fmt_k(adjusted_price),
            "price_low": _fmt_usd(price_low),
            "price_high": _fmt_usd(price_high),
            "feature_summary": {
                "brand": data.get("brand"),
                "model": data.get("model"),
                "year": data.get("year"),
                "engine_size": data.get("engine_size"),
                "mileage": data.get("mileage"),
                "fuel_type": data.get("fuel_type"),
                "transmission": data.get("transmission"),
            },
        }

    def feature_names(self):
        if self._feature_columns:
            return list(self._feature_columns)
        if self._model is not None:
            return list(getattr(self._model, "feature_names_in_", []))
        return []


def _brand_model_adjustment(data: dict, bundle: dict | None) -> float:
    brand = str(data.get("brand", "BMW")).strip()
    model = str(data.get("model", "5 Series")).strip()

    brand_factor = BRAND_ADJUSTMENT.get(brand, 1.0)

    model_tier = "Standard"
    if bundle and isinstance(bundle, dict):
        tier_map = bundle.get("model_tier_map", {})
        if model in tier_map:
            model_tier = str(tier_map.get(model, "Standard"))

    model_factor = MODEL_TIER_ADJUSTMENT.get(model_tier, 1.0)
    return brand_factor * model_factor


# ── Formatting helpers ────────────────────────────────────────────────
def _fmt_usd(value: float) -> str:
    """Format as $12,345 (no decimals for readability)."""
    return f"${value:,.0f}"


def _fmt_k(value: float) -> str:
    """Format as 12.3K or 1.23M."""
    if value >= 1_000_000:
        return f"${value / 1_000_000:.2f}M"
    return f"${value / 1000:.1f}K"
