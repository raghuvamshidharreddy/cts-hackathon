"""
Config — Car Sence v2
Lasso regression model: automotive pricing with the saved serialized artifact.
"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Model ──────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(BASE_DIR, "model", "car_price_champion_pipeline.pkl")
MODEL_NAME = "Lasso Regression"

# Exact feature order from the saved artifact payload.
ALL_FEATURES = [
    "Year",
    "Mileage",
    "Mileage_Per_Year",
    "Brand_BMW",
    "Brand_Chevrolet",
    "Brand_Ford",
    "Brand_Honda",
    "Brand_Hyundai",
    "Brand_Kia",
    "Brand_Mercedes",
    "Brand_Toyota",
    "Brand_Volkswagen",
    "Model_5 Series",
    "Model_A3",
    "Model_A4",
    "Model_Accord",
    "Model_C-Class",
    "Model_CR-V",
    "Model_Camry",
    "Model_Civic",
    "Model_Corolla",
    "Model_E-Class",
    "Model_Elantra",
    "Model_Equinox",
    "Model_Explorer",
    "Model_Fiesta",
    "Model_Focus",
    "Model_GLA",
    "Model_Golf",
    "Model_Impala",
    "Model_Malibu",
    "Model_Optima",
    "Model_Passat",
    "Model_Q5",
    "Model_RAV4",
    "Model_Rio",
    "Model_Sonata",
    "Model_Sportage",
    "Model_Tiguan",
    "Model_Tucson",
    "Model_X5",
    "Model_Segment_Premium",
    "Model_Segment_Standard",
    "Fuel_Type_Electric",
    "Fuel_Type_Hybrid",
    "Fuel_Type_Petrol",
    "Transmission_Manual",
    "Transmission_Semi-Automatic",
    "Year_5Y_Span_2006-2010",
    "Year_5Y_Span_2011-2015",
    "Year_5Y_Span_2016-2020",
    "Year_5Y_Span_2021-2025",
]

# ── Brand → Models mapping (must match training data exactly) ──────────
# Audi is the baseline brand (all Brand_* = 0)
BRAND_MODELS = {
    "Audi":       ["A3", "A4", "Q5"],
    "BMW":        ["3 Series", "5 Series", "X5"],  # 3 Series = baseline (drop_first)
    "Chevrolet":  ["Equinox", "Impala", "Malibu"],
    "Ford":       ["Explorer", "Fiesta", "Focus"],
    "Honda":      ["Accord", "CR-V", "Civic"],
    "Hyundai":    ["Elantra", "Sonata", "Tucson"],
    "Kia":        ["Optima", "Rio", "Sportage"],
    "Mercedes":   ["C-Class", "E-Class", "GLA"],
    "Toyota":     ["Camry", "Corolla", "RAV4"],
    "Volkswagen": ["Golf", "Passat", "Tiguan"],
}

# ── Categorical Options ────────────────────────────────────────────────
FUEL_TYPES      = ["Diesel", "Petrol", "Hybrid", "Electric"]
TRANSMISSIONS   = ["Automatic", "Manual", "Semi-Automatic"]
DOOR_OPTIONS    = [2, 3, 4, 5]

# ── Year range (dataset: 2000–2023) ──────────────────────────────────
YEAR_MIN = 2000
YEAR_MAX = 2023
YEARS    = list(range(YEAR_MAX, YEAR_MIN - 1, -1))

# ── Engine size range (dataset: 1.0–5.0 litres) ───────────────────────
ENGINE_MIN = 1.0
ENGINE_MAX = 5.0

# ── Mileage range (dataset: 25–299,947 miles) ────────────────────────
MILEAGE_MIN = 0
MILEAGE_MAX = 300000

# ── Feature thresholds (must match training preprocessing) ────────────
ENGINE_MEDIUM_THRESHOLD = 1.6   # >= 1.6L  → Medium
ENGINE_LARGE_THRESHOLD  = 2.5   # >= 2.5L  → Large

MILEAGE_MEDIUM_THRESHOLD = 30000  # >= 30k  → Medium
MILEAGE_HIGH_THRESHOLD   = 70000  # >= 70k  → High

YEAR_2010S_MIN = 2010
YEAR_2020S_MIN = 2020

# ── All 59 feature names in exact model order ─────────────────────────
ALL_FEATURES = [
    "Year", "Engine_Size", "Mileage",
    "Brand_BMW", "Brand_Chevrolet", "Brand_Ford", "Brand_Honda",
    "Brand_Hyundai", "Brand_Kia", "Brand_Mercedes", "Brand_Toyota",
    "Brand_Volkswagen",
    "Model_5 Series", "Model_A3", "Model_A4", "Model_Accord",
    "Model_C-Class", "Model_CR-V", "Model_Camry", "Model_Civic",
    "Model_Corolla", "Model_E-Class", "Model_Elantra", "Model_Equinox",
    "Model_Explorer", "Model_Fiesta", "Model_Focus", "Model_GLA",
    "Model_Golf", "Model_Impala", "Model_Malibu", "Model_Optima",
    "Model_Passat", "Model_Q5", "Model_RAV4", "Model_Rio",
    "Model_Sonata", "Model_Sportage", "Model_Tiguan", "Model_Tucson",
    "Model_X5",
    "Fuel_Type_Electric", "Fuel_Type_Hybrid", "Fuel_Type_Petrol",
    "Transmission_Manual", "Transmission_Semi-Automatic",
    "Doors_3", "Doors_4", "Doors_5",
    "Owner_Count_2", "Owner_Count_3", "Owner_Count_4", "Owner_Count_5",
    "Engine_Size_Group_Medium", "Engine_Size_Group_Large",
    "Mileage_Group_Medium", "Mileage_Group_High",
    "Year_of_Registration_Group_2010s",
    "Year_of_Registration_Group_2020s",
]
