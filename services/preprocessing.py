"""
Preprocessing — Car Sence v2
Builds the exact feature frame required by the saved Lasso model bundle.
"""
import pandas as pd
from config import ALL_FEATURES


BRAND_OPTIONS = [
    "BMW", "Chevrolet", "Ford", "Honda", "Hyundai",
    "Kia", "Mercedes", "Toyota", "Volkswagen"
]

PREMIUM_MODELS = {
    "A3", "A4", "Q5", "5 Series", "X5", "GLA", "C-Class", "E-Class",
    "Equinox", "Impala", "Fiesta", "Explorer", "Focus", "Civic", "Accord",
    "CR-V", "Elantra", "Sonata", "Tucson", "Rio", "Optima", "Sportage",
    "Golf", "Passat", "Tiguan", "Camry", "Corolla", "RAV4"
}


def build_feature_vector(data: dict) -> pd.DataFrame:
    """Transform raw inputs to the exact Lasso schema used by the saved artifact."""
    brand = str(data.get("brand", "BMW")).strip()
    car_model = str(data.get("model", "3 Series")).strip()
    year = int(data.get("year", 2020))
    mileage = float(data.get("mileage", 50000))
    fuel_type = str(data.get("fuel_type", "Petrol")).strip()
    transmission = str(data.get("transmission", "Automatic")).strip()

    row = {feature: 0 for feature in ALL_FEATURES}

    row["Year"] = year
    row["Mileage"] = mileage
    row["Mileage_Per_Year"] = mileage / max(1.0, float(2025 - year))

    for brand_name in BRAND_OPTIONS:
        if brand_name == brand:
            row[f"Brand_{brand_name}"] = 1

    model_key = f"Model_{car_model}"
    if model_key in row:
        row[model_key] = 1

    if car_model in {"A3", "A4", "Q5", "5 Series", "X5", "C-Class", "E-Class", "GLA"}:
        row["Model_Segment_Premium"] = 1
    else:
        row["Model_Segment_Standard"] = 1

    if fuel_type == "Electric":
        row["Fuel_Type_Electric"] = 1
    elif fuel_type == "Hybrid":
        row["Fuel_Type_Hybrid"] = 1
    elif fuel_type == "Petrol":
        row["Fuel_Type_Petrol"] = 1

    if transmission == "Manual":
        row["Transmission_Manual"] = 1
    elif transmission == "Semi-Automatic":
        row["Transmission_Semi-Automatic"] = 1

    if 2006 <= year <= 2010:
        row["Year_5Y_Span_2006-2010"] = 1
    elif 2011 <= year <= 2015:
        row["Year_5Y_Span_2011-2015"] = 1
    elif 2016 <= year <= 2020:
        row["Year_5Y_Span_2016-2020"] = 1
    elif 2021 <= year <= 2025:
        row["Year_5Y_Span_2021-2025"] = 1

    return pd.DataFrame([row], columns=ALL_FEATURES)
