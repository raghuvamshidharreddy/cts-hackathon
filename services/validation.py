"""
Validation — Car Sence v2
Input validation for Lasso Regression model API requests.
"""
from config import (
    BRAND_MODELS, FUEL_TYPES, TRANSMISSIONS,
    DOOR_OPTIONS, YEAR_MIN, YEAR_MAX,
    ENGINE_MIN, ENGINE_MAX, MILEAGE_MIN, MILEAGE_MAX,
)


def validate_input(data: dict) -> dict:
    """
    Validate API input dict. Returns:
        { "valid": bool, "errors": [str] }
    """
    errors = []

    # ── Brand ─────────────────────────────────────────────────────────
    brand = data.get("brand", "").strip()
    if not brand:
        errors.append("'brand' is required.")
    elif brand not in BRAND_MODELS:
        errors.append(f"Unknown brand '{brand}'. Valid: {list(BRAND_MODELS.keys())}")

    # ── Model ─────────────────────────────────────────────────────────
    car_model = data.get("model", "").strip()
    if brand in BRAND_MODELS:
        if car_model and car_model not in BRAND_MODELS[brand]:
            errors.append(
                f"Model '{car_model}' not valid for brand '{brand}'. "
                f"Valid: {BRAND_MODELS[brand]}"
            )
        # model is optional — blank model maps to baseline (all zeros)

    # ── Year ──────────────────────────────────────────────────────────
    try:
        year = int(data.get("year", 0))
        if not (YEAR_MIN <= year <= YEAR_MAX):
            errors.append(f"'year' must be between {YEAR_MIN} and {YEAR_MAX}.")
    except (ValueError, TypeError):
        errors.append("'year' must be a valid integer.")

    # ── Engine size (litres) ──────────────────────────────────────────
    try:
        engine = float(data.get("engine_size", 0))
        if engine <= 0 or not (ENGINE_MIN <= engine <= ENGINE_MAX):
            errors.append(
                f"'engine_size' must be between {ENGINE_MIN}L and {ENGINE_MAX}L."
            )
    except (ValueError, TypeError):
        errors.append("'engine_size' must be a valid number (litres).")

    # ── Mileage ───────────────────────────────────────────────────────
    try:
        mileage = float(data.get("mileage", -1))
        if not (MILEAGE_MIN <= mileage <= MILEAGE_MAX):
            errors.append(
                f"'mileage' must be between {MILEAGE_MIN} and {MILEAGE_MAX} miles."
            )
    except (ValueError, TypeError):
        errors.append("'mileage' must be a valid number.")

    # ── Fuel type ─────────────────────────────────────────────────────
    fuel = data.get("fuel_type", "")
    if fuel not in FUEL_TYPES:
        errors.append(f"'fuel_type' must be one of {FUEL_TYPES}.")

    # ── Transmission ──────────────────────────────────────────────────
    transmission = data.get("transmission", "")
    if transmission not in TRANSMISSIONS:
        errors.append(f"'transmission' must be one of {TRANSMISSIONS}.")

    # ── Doors ─────────────────────────────────────────────────────────
    try:
        doors = int(data.get("doors", 0))
        if doors not in DOOR_OPTIONS:
            errors.append(f"'doors' must be one of {DOOR_OPTIONS}.")
    except (ValueError, TypeError):
        errors.append("'doors' must be a valid integer.")

    return {"valid": len(errors) == 0, "errors": errors}
