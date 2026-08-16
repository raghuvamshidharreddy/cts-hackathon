# Product Requirements Document — AI Car Price Prediction Flask Application

## 1. Project Overview

Build a complete, production-style **Flask-based web application for used-car price prediction**.

The application must allow a user to enter car specifications through a modern web interface and receive an estimated resale/selling price using the provided trained machine-learning model:

`car_price_model.pkl`

The application must use the trained **RandomForestRegressor** model from the supplied notebook and must reproduce the feature-engineering logic used during model training.

### Primary Goal

Given a car's characteristics, the system must:

1. Accept the required raw car features.
2. Validate the inputs.
3. Apply the same transformations used in the training notebook.
4. Generate all engineered features.
5. Construct the exact 19-feature model input.
6. Pass the features to `car_price_model.pkl`.
7. Generate the predicted selling price.
8. Display the result clearly in INR and Lakhs.
9. Explain the major factors/features contributing to the prediction.
10. Handle invalid or incomplete inputs gracefully.

---

# 2. Important Existing ML Assets

The project contains three important files:

```text
car_price_model.pkl
Car_Price_Prediction_Dataset.csv
carpriceprediction.ipynb
```

The application must treat the notebook and trained model as the source of truth for the prediction pipeline.

## 2.1 Existing Model

The supplied model is:

```text
RandomForestRegressor
```

Model configuration:

```python
RandomForestRegressor(
    n_estimators=100,
    random_state=42
)
```

The model contains:

```text
n_features_in_ = 19
```

The exact expected feature names are:

```text
year
km_driven
transmission
mileage(km/ltr/kg)
seats
engine_log
max_power_log
fuel_Diesel
fuel_Petrol
seller_type_Individual
owner_Second Owner
owner_Test Drive Car
owner_Third+
is_luxury
luxury_engine
performance_index
petrol_wear
age_wear
tech_score
```

These names and their order MUST NOT be changed when creating the model input.

The Flask application should preferably verify this at startup:

```python
expected_features = [
    "year",
    "km_driven",
    "transmission",
    "mileage(km/ltr/kg)",
    "seats",
    "engine_log",
    "max_power_log",
    "fuel_Diesel",
    "fuel_Petrol",
    "seller_type_Individual",
    "owner_Second Owner",
    "owner_Test Drive Car",
    "owner_Third+",
    "is_luxury",
    "luxury_engine",
    "performance_index",
    "petrol_wear",
    "age_wear",
    "tech_score"
]
```

The prediction pipeline must assert that the generated DataFrame has exactly these columns.

---

# 3. Existing Model Performance

According to the supplied notebook, the models were evaluated as follows:

| Model | R² |
|---|---:|
| Linear Regression | ~0.67 |
| KNN K=2 | ~0.8662 |
| KNN + feature engineering | ~0.88 |
| Random Forest | ~0.91 |

The notebook selected the default Random Forest as the champion model.

The final model was created using:

```python
final_rf = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)
```

and saved as:

```text
car_price_model.pkl
```

Do not replace this model with another algorithm unless a separate retraining process is explicitly requested.

---

# 4. Critical Dataset Compatibility Requirement

There is a schema difference between the uploaded CSV and the data used in the notebook.

## 4.1 Notebook Training Dataset

The notebook loads:

```python
/kaggle/input/car-price-prediction-dataset/cardekho.csv
```

and uses fields such as:

```text
name
year
selling_price
km_driven
fuel
seller_type
transmission
owner
mileage(km/ltr/kg)
engine
max_power
seats
```

## 4.2 Uploaded CSV

The supplied CSV contains:

```text
Unnamed: 0
car_name
registration_year
insurance_validity
fuel_type
seats
kms_driven
ownsership
transmission
manufacturing_year
mileage(kmpl)
engine(cc)
max_power(bhp)
torque(Nm)
price(in lakhs)
```

Therefore:

**DO NOT directly feed the uploaded CSV columns into `car_price_model.pkl`.**

The model was not trained using that exact schema.

The live prediction system should use the model's training schema and recreate the notebook's feature engineering.

The uploaded CSV may be retained for:

- future retraining,
- data exploration,
- dataset validation,
- future model improvement.

It must not silently be treated as identical to the training dataset.

---

# 5. User Input Features

The frontend should expose the meaningful raw features required to reproduce the model.

The user should NOT be asked to manually enter engineered features such as:

```text
engine_log
max_power_log
performance_index
petrol_wear
age_wear
tech_score
luxury_engine
```

These must be calculated automatically by the backend.

## Required User Inputs

### 5.1 Manufacturing / Model Year

Field:

```text
year
```

Input type:

```text
number
```

Example:

```text
2018
```

Validation:

```text
1900 <= year <= current_year
```

Prefer a dropdown or numeric input.

---

### 5.2 Kilometres Driven

Field:

```text
km_driven
```

Input type:

```text
number
```

Example:

```text
45000
```

Validation:

```text
km_driven >= 0
```

---

### 5.3 Transmission

Field:

```text
transmission
```

Allowed values:

```text
Manual
Automatic
```

Notebook transformation:

```python
df['transmission'] = df['transmission'].map({
    'Manual': 0,
    'Automatic': 1
})
```

Therefore:

```text
Manual -> 0
Automatic -> 1
```

---

### 5.4 Mileage

Field:

```text
mileage(km/ltr/kg)
```

Input:

```text
number
```

Example:

```text
18.5
```

Validation:

```text
mileage > 0
```

The notebook replaces zero mileage with NaN and performs median imputation during training.

For the live application, a value of zero should preferably be rejected as invalid rather than silently accepted.

---

### 5.5 Seats

Field:

```text
seats
```

Input type:

```text
number
```

Recommended range:

```text
1–15
```

The notebook uses mean imputation for missing seats.

The web application should preferably require the user to provide the value.

---

### 5.6 Engine

Raw model-training field:

```text
engine
```

The application should ask for engine displacement.

Example:

```text
1998
```

Unit:

```text
cc
```

The notebook transforms it using:

```python
engine_log = np.log(engine)
```

Therefore:

```python
engine > 0
engine_log = np.log(engine)
```

The raw `engine` column is then removed.

---

### 5.7 Maximum Power

Raw field:

```text
max_power
```

Input:

```text
number
```

Example:

```text
120
```

Unit:

```text
bhp
```

The notebook:

1. Converts the value to numeric.
2. Converts zero to NaN.
3. Calculates:

```python
max_power_log = np.log1p(max_power)
```

The raw `max_power` column is then removed.

For live prediction:

```python
max_power > 0
max_power_log = np.log1p(max_power)
```

---

### 5.8 Fuel Type

Allowed values must correspond to the notebook:

```text
Diesel
Petrol
CNG
LPG
```

The notebook combines:

```text
CNG
LPG
```

into:

```text
Alternative
```

and performs one-hot encoding with:

```python
pd.get_dummies(df, columns=['fuel'], drop_first=True)
```

The final model expects:

```text
fuel_Diesel
fuel_Petrol
```

Therefore the backend must generate:

```text
Diesel:
fuel_Diesel = 1
fuel_Petrol = 0

Petrol:
fuel_Diesel = 0
fuel_Petrol = 1

CNG/LPG:
fuel_Diesel = 0
fuel_Petrol = 0
```

---

### 5.9 Seller Type

Notebook categories include:

```text
Individual
Dealer
Trustmark Dealer
```

The notebook first combines:

```text
Dealer
Trustmark Dealer
```

into:

```text
Commercial_seller
```

Then one-hot encoding is performed with `drop_first=True`.

The final model expects:

```text
seller_type_Individual
```

Therefore:

```text
Individual -> 1
Commercial_seller -> 0
```

---

### 5.10 Owner

The notebook combines:

```text
Third Owner
Fourth & Above Owner
```

into:

```text
Third+
```

Then performs one-hot encoding.

The model expects:

```text
owner_Second Owner
owner_Test Drive Car
owner_Third+
```

The backend must reproduce these categories exactly.

Expected mapping:

```text
First Owner
    ->
0, 0, 0

Second Owner
    ->
1, 0, 0

Test Drive Car
    ->
0, 1, 0

Third Owner / Fourth & Above Owner
    ->
0, 0, 1
```

---

### 5.11 Brand / Car Name

The model does not directly receive:

```text
name
brand
```

The notebook extracts brand information during preprocessing and then creates:

```text
is_luxury
```

The backend should therefore ask for either:

1. Brand, or
2. Car name + brand.

For the first implementation, **Brand should be a separate dropdown/input** because it makes the transformation deterministic.

The luxury brands defined by the notebook are:

```text
Lexus
Volvo
BMW
Jaguar
Land Rover
Audi
Mercedes-Benz
Jeep
MG
Isuzu
Toyota
Kia
```

If the selected brand belongs to this list:

```python
is_luxury = 1
```

Otherwise:

```python
is_luxury = 0
```

Do not invent additional luxury brands unless explicitly added later.

---

# 6. Feature Engineering Pipeline

The backend must reproduce the notebook's feature engineering.

## Step 1 — Engine Log

```python
engine_log = np.log(engine)
```

Requirement:

```text
engine > 0
```

---

## Step 2 — Maximum Power Log

```python
max_power_log = np.log1p(max_power)
```

Requirement:

```text
max_power > 0
```

---

## Step 3 — Transmission Encoding

```python
Manual = 0
Automatic = 1
```

---

## Step 4 — Fuel Encoding

Generate:

```text
fuel_Diesel
fuel_Petrol
```

with the exact mapping described above.

---

## Step 5 — Seller Encoding

Generate:

```text
seller_type_Individual
```

---

## Step 6 — Owner Encoding

Generate:

```text
owner_Second Owner
owner_Test Drive Car
owner_Third+
```

---

## Step 7 — Luxury Feature

Create:

```python
is_luxury
```

using the exact notebook luxury brand list.

---

## Step 8 — Luxury Engine Interaction

The notebook creates:

```python
luxury_engine = is_luxury * engine_log
```

---

## Step 9 — Performance Index

The notebook creates:

```python
performance_index = max_power_log * engine_log
```

---

## Step 10 — Petrol Wear

The notebook creates:

```python
petrol_wear = km_driven * fuel_Petrol
```

Therefore:

- Petrol vehicles get `km_driven`.
- Non-Petrol vehicles get `0`.

---

## Step 11 — Age Wear

The notebook creates:

```python
age_wear = (2025 - year) * km_driven
```

IMPORTANT:

The notebook explicitly uses:

```text
2025
```

not the current system year.

To preserve model compatibility, the production prediction pipeline should initially retain:

```python
REFERENCE_YEAR = 2025
```

Do not replace this with `datetime.now().year` because doing so changes the feature distribution relative to the model's training data.

If a future retrained model uses a different reference year, update it together with the model.

---

## Step 12 — Tech Score

The notebook creates:

```python
tech_score = max_power_log - engine_log
```

---

# 7. Final Model Input

After all transformations, create exactly:

```python
model_input = pd.DataFrame([{
    "year": year,
    "km_driven": km_driven,
    "transmission": transmission,
    "mileage(km/ltr/kg)": mileage,
    "seats": seats,
    "engine_log": engine_log,
    "max_power_log": max_power_log,
    "fuel_Diesel": fuel_Diesel,
    "fuel_Petrol": fuel_Petrol,
    "seller_type_Individual": seller_type_Individual,
    "owner_Second Owner": owner_second_owner,
    "owner_Test Drive Car": owner_test_drive_car,
    "owner_Third+": owner_third_plus,
    "is_luxury": is_luxury,
    "luxury_engine": luxury_engine,
    "performance_index": performance_index,
    "petrol_wear": petrol_wear,
    "age_wear": age_wear,
    "tech_score": tech_score
}])
```

Before prediction:

```python
model_input = model_input[expected_features]
```

Then:

```python
prediction = model.predict(model_input)[0]
```

---

# 8. Prediction Output

The notebook's target variable is:

```text
selling_price
```

The supplied model's learned target values are in rupee-scale values.

The UI should display:

### Primary

```text
Estimated Car Price
₹6,45,000
```

### Secondary

```text
₹6.45 Lakhs
```

Formatting example:

```python
price = float(prediction)

price_in_lakhs = price / 100000
```

Display both values.

Do not claim the prediction is guaranteed to be the market price.

Use wording such as:

```text
Estimated resale price
```

or:

```text
Model-estimated selling price
```

---

# 9. Application Architecture

The application must be completely Flask-based.

Recommended architecture:

```text
car-price-predictor/
│
├── app.py
│
├── model/
│   └── car_price_model.pkl
│
├── services/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── prediction.py
│   └── validation.py
│
├── templates/
│   ├── base.html
│   ├── index.html
│   └── result.html
│
├── static/
│   ├── css/
│   │   └── style.css
│   ├── js/
│   │   └── app.js
│   └── images/
│
├── data/
│   └── Car_Price_Prediction_Dataset.csv
│
├── tests/
│   ├── test_preprocessing.py
│   ├── test_prediction.py
│   └── test_routes.py
│
├── requirements.txt
├── README.md
├── .gitignore
└── config.py
```

---

# 10. Flask Backend

Create the Flask application in:

```text
app.py
```

The backend should:

1. Load the model once when the application starts.
2. Validate incoming data.
3. Transform raw input.
4. Construct the 19-feature DataFrame.
5. Predict.
6. Return/render the result.

Do NOT load the 43 MB model on every request.

Correct:

```python
model = joblib.load(MODEL_PATH)
```

once at application startup.

---

# 11. Flask Routes

## GET `/`

Render the prediction form.

---

## POST `/predict`

Receive the form.

Perform:

```text
validation
    ↓
preprocessing
    ↓
feature engineering
    ↓
19-feature DataFrame
    ↓
model.predict()
    ↓
formatted result
```

Render the result page.

---

## GET `/health`

Return:

```json
{
    "status": "healthy",
    "model_loaded": true
}
```

This route is useful for testing and deployment.

---

## Optional GET `/api/model-info`

Return:

```json
{
    "model": "RandomForestRegressor",
    "n_estimators": 100,
    "features": 19
}
```

Do not expose the model file itself.

---

# 12. Frontend Requirements

Create a clean modern dashboard rather than a plain HTML form.

## Header

Display:

```text
CarPrice AI
AI-Powered Used Car Price Prediction
```

---

# 13. Prediction Form UI

Organize the form into logical sections.

## Section 1 — Basic Information

Fields:

- Brand
- Manufacturing Year
- Kilometres Driven
- Number of Seats

---

## Section 2 — Engine & Performance

Fields:

- Engine CC
- Maximum Power BHP
- Mileage

---

## Section 3 — Vehicle Configuration

Fields:

- Fuel Type
- Transmission
- Seller Type
- Owner Type

---

# 14. Smart UI

Use:

- dropdowns for categorical fields,
- number inputs for numerical fields,
- range validation,
- helpful unit labels,
- inline validation,
- loading state,
- prediction button.

Example:

```text
Engine
[ 1998 ] CC
```

```text
Maximum Power
[ 150 ] BHP
```

```text
Kilometres Driven
[ 45000 ] KM
```

---

# 15. Prediction Result Page

After prediction, display a visually prominent card:

```text
Estimated Selling Price

₹8,75,000

≈ ₹8.75 Lakhs
```

Also show the entered vehicle details:

```text
BMW
2019
45,000 KM
Petrol
Automatic
1998 CC
150 BHP
```

Provide:

```text
Predict Another Car
```

button.

---

# 16. Explainability Section

The application should provide a simple explanation section.

Use the model's feature importance information where appropriate.

The notebook identifies the most important features, with:

- `max_power_log`
- `year`
- `performance_index`
- `age_wear`

among the strongest predictors.

The UI can display:

```text
Why this prediction?

The model considers:
• Engine performance
• Vehicle age
• Kilometres driven
• Mileage
• Luxury/premium status
• Engine-performance interaction
• Fuel type
• Ownership history
```

Do not claim that a feature caused the exact prediction.

Use:

```text
The model considers...
```

rather than:

```text
This feature caused your price...
```

---

# 17. Feature Importance Visualization

If feature importance is displayed, obtain it directly from:

```python
model.feature_importances_
```

and map it to:

```python
model.feature_names_in_
```

Do not hard-code importance values.

Create a Top 5 or Top 10 visualization.

Example:

```text
Top Model Features

max_power_log       █████████████████
year                ████████
performance_index   ███████
age_wear            ████
...
```

---

# 18. Input Validation

Backend validation is mandatory.

## Year

Reject:

```text
year <= 0
year > current year
```

## Kilometres

Reject:

```text
km_driven < 0
```

## Engine

Reject:

```text
engine <= 0
```

## Maximum Power

Reject:

```text
max_power <= 0
```

## Mileage

Reject:

```text
mileage <= 0
```

## Seats

Reject:

```text
seats <= 0
```

## Categorical Values

Reject values outside the supported categories.

Never trust frontend validation alone.

---

# 19. Error Handling

The application must never display a Python traceback to the user.

Example:

```text
Unable to generate prediction.

Please check the entered vehicle details and try again.
```

Log the actual exception on the server.

Use Flask error handlers for:

```text
400
404
500
```

---

# 20. Model Loading

Use:

```python
import joblib

model = joblib.load("model/car_price_model.pkl")
```

The notebook itself saved the model using:

```python
joblib.dump(final_rf, 'car_price_model.pkl')
```

Therefore use `joblib.load()` rather than relying on raw `pickle.load()`.

The supplied model was created under a different scikit-learn version, so Antigravity should pin a compatible scikit-learn version after testing model loading.

Do not silently retrain or replace the model because of a version warning.

---

# 21. Model Compatibility Check

At startup, implement:

```python
EXPECTED_FEATURES = [
    "year",
    "km_driven",
    "transmission",
    "mileage(km/ltr/kg)",
    "seats",
    "engine_log",
    "max_power_log",
    "fuel_Diesel",
    "fuel_Petrol",
    "seller_type_Individual",
    "owner_Second Owner",
    "owner_Test Drive Car",
    "owner_Third+",
    "is_luxury",
    "luxury_engine",
    "performance_index",
    "petrol_wear",
    "age_wear",
    "tech_score"
]
```

Then verify:

```python
if hasattr(model, "feature_names_in_"):
    assert list(model.feature_names_in_) == EXPECTED_FEATURES
```

If this fails, application startup should fail clearly rather than making potentially incorrect predictions.

---

# 22. Dataset Handling

The supplied CSV should be included in the project for reference, but it should NOT automatically be used as the prediction input source for the existing model.

The application may provide an optional future admin/data page for:

```text
Dataset validation
Dataset statistics
Retraining
```

but this is not required for the first production version.

---

# 23. Important Uploaded CSV Issue

The uploaded CSV has these columns:

```text
car_name
registration_year
insurance_validity
fuel_type
seats
kms_driven
ownsership
transmission
manufacturing_year
mileage(kmpl)
engine(cc)
max_power(bhp)
torque(Nm)
price(in lakhs)
```

This differs from the notebook training schema.

The application MUST NOT perform an unsafe automatic mapping such as:

```text
registration_year -> year
```

and assume every other field is compatible.

A future retraining pipeline should explicitly define:

```text
CSV schema
        ↓
data quality validation
        ↓
canonical training schema
        ↓
feature engineering
        ↓
model training
        ↓
model evaluation
        ↓
new model artifact
```

---

# 24. Data Quality Validation for Future Retraining

If retraining is implemented later, validate:

### Numerical columns

```text
seats
kms_driven
manufacturing_year
mileage(kmpl)
engine(cc)
max_power(bhp)
torque(Nm)
price(in lakhs)
```

Check:

- null values,
- impossible values,
- negative values,
- duplicated records,
- extreme outliers,
- suspicious column shifts,
- incorrect units.

### Categorical columns

Validate:

```text
fuel_type
transmission
ownsership
insurance_validity
```

against actual observed values.

Do not assume column names imply that the values are correct.

---

# 25. Security Requirements

The Flask application should:

- validate all user inputs,
- escape rendered user content,
- use secure secret configuration,
- disable debug mode in production,
- avoid exposing stack traces,
- avoid exposing the model file,
- avoid arbitrary file upload unless explicitly required,
- restrict uploaded file types if dataset upload is added.

Do not use:

```python
app.run(debug=True)
```

in production.

---

# 26. Requirements File

Create:

```text
requirements.txt
```

with the minimum required dependencies, such as:

```text
Flask
numpy
pandas
scikit-learn
joblib
```

Pin versions after testing compatibility with the supplied model.

Frontend dependencies should be kept minimal.

Do not introduce React, Streamlit, FastAPI, Django, or another backend framework.

The backend must remain Flask.

---

# 27. Testing Requirements

Create automated tests.

## Test 1 — Model Loading

Verify:

```text
model loads successfully
```

---

## Test 2 — Feature Count

Verify:

```text
19 features
```

---

## Test 3 — Feature Names

Verify exact feature names and order.

---

## Test 4 — Preprocessing

Given:

```text
engine = 1998
max_power = 150
```

verify:

```python
engine_log == np.log(1998)
max_power_log == np.log1p(150)
```

---

## Test 5 — Luxury Feature

Given:

```text
BMW
```

verify:

```text
is_luxury = 1
```

Given:

```text
Maruti
```

verify:

```text
is_luxury = 0
```

---

## Test 6 — Performance Index

Verify:

```python
performance_index == max_power_log * engine_log
```

---

## Test 7 — Tech Score

Verify:

```python
tech_score == max_power_log - engine_log
```

---

## Test 8 — Petrol Wear

For Petrol:

```python
petrol_wear == km_driven
```

For Diesel:

```python
petrol_wear == 0
```

---

## Test 9 — Age Wear

Verify:

```python
age_wear == (2025 - year) * km_driven
```

---

## Test 10 — Prediction

A valid input must produce:

```text
numeric prediction > 0
```

---

# 28. API Design

Although the primary UI is server-rendered Flask, expose a JSON prediction endpoint for future integrations.

## POST

```text
/api/predict
```

Example request:

```json
{
    "year": 2019,
    "km_driven": 45000,
    "transmission": "Automatic",
    "mileage": 18.5,
    "seats": 5,
    "engine": 1998,
    "max_power": 150,
    "fuel": "Petrol",
    "seller_type": "Individual",
    "owner": "First Owner",
    "brand": "BMW"
}
```

Example response:

```json
{
    "success": true,
    "predicted_price": 875000,
    "predicted_price_lakhs": 8.75
}
```

The actual prediction must come from:

```python
model.predict()
```

Do not create a formula-based price approximation.

---

# 29. Prediction Service

Create:

```text
services/prediction.py
```

Responsibility:

```text
raw user data
        ↓
preprocessing
        ↓
feature engineering
        ↓
model input
        ↓
model prediction
        ↓
formatted result
```

Keep prediction logic out of the Flask route itself.

This makes the system easier to test and maintain.

---

# 30. Preprocessing Service

Create:

```text
services/preprocessing.py
```

Functions should include:

```python
transform_input(data)
```

and optionally:

```python
create_model_features(data)
```

The preprocessing module should contain all feature engineering.

Do NOT duplicate the feature-engineering logic in:

- HTML
- JavaScript
- Flask routes
- multiple Python files.

There must be one source of truth.

---

# 31. Validation Service

Create:

```text
services/validation.py
```

with validation functions such as:

```python
validate_year()
validate_km_driven()
validate_engine()
validate_max_power()
validate_mileage()
validate_categories()
```

---

# 32. UI/UX Requirements

The interface should look like a polished hackathon project rather than a basic college form.

Use:

- responsive design,
- modern cards,
- clear typography,
- consistent spacing,
- mobile compatibility,
- prediction animation/loading state,
- visually prominent result,
- clear error messages.

The prediction form should be usable on:

```text
Desktop
Tablet
Mobile
```

---

# 33. Suggested Landing Page

Top section:

```text
CARPRICE AI

Predict the estimated resale value of your car
using machine learning.

[ Predict Car Price ]
```

Then show:

```text
ML Powered
19 Model Features
Random Forest
~0.91 R²
```

Do not claim "91% accuracy" because the notebook reports R², not classification accuracy.

Correct:

```text
R² ≈ 0.91 on the notebook's test split
```

---

# 34. Result Dashboard

Result page should contain:

```text
YOUR ESTIMATED CAR PRICE

₹8,75,000
₹8.75 Lakhs
```

Then:

```text
Vehicle Summary
```

Then:

```text
Model Factors
```

Then:

```text
Predict Another Vehicle
```

---

# 35. Optional Price Range

Do NOT fabricate a confidence interval.

If a price range is displayed, it must be derived from a defensible statistical method using the model/data.

For the first implementation, only display:

```text
Point Estimate
```

This avoids pretending that the Random Forest output provides a formal confidence interval.

---

# 36. Model Explainability

The application can use:

```python
model.feature_importances_
```

to display global feature importance.

Do not implement SHAP unless specifically required.

SHAP is optional and should not be added merely to make the project appear more advanced.

---

# 37. Logging

Implement application logging for:

- model loading,
- prediction requests,
- validation failures,
- unexpected exceptions.

Do NOT log sensitive information unnecessarily.

Example:

```text
INFO: Model loaded successfully
INFO: Prediction generated successfully
WARNING: Invalid engine value
ERROR: Prediction failure
```

---

# 38. README Requirements

Create a comprehensive README containing:

## Project Description

Explain the problem and solution.

## Features

List:

- price prediction,
- feature engineering,
- Random Forest,
- Flask web application,
- model explainability.

## ML Pipeline

Explain:

```text
Raw Input
→ Cleaning
→ Feature Engineering
→ 19 Features
→ Random Forest
→ Price
```

## Model

Mention:

```text
RandomForestRegressor
n_estimators = 100
random_state = 42
```

## Performance

Mention:

```text
R² ≈ 0.91
```

and clearly state that this is the notebook's reported test R².

## Installation

Example:

```bash
python -m venv venv
```

```bash
venv\Scripts\activate
```

```bash
pip install -r requirements.txt
```

## Run

```bash
python app.py
```

Then open:

```text
http://127.0.0.1:5000
```

---

# 39. Non-Functional Requirements

The application should:

- start within a reasonable time,
- load the model once,
- respond to predictions quickly,
- not retrain on every request,
- be deterministic for the same input/model,
- be maintainable,
- have clear separation between frontend/backend/model logic.

---

# 40. Important ML Integrity Rules

Antigravity MUST NOT:

### Do not change feature formulas

For example, do not change:

```python
age_wear = (2025 - year) * km_driven
```

to:

```python
age_wear = (datetime.now().year - year) * km_driven
```

without retraining the model.

### Do not change model feature order.

### Do not rename model features.

### Do not remove engineered features.

### Do not manually invent a prediction formula.

### Do not claim 91% "accuracy".

### Do not feed the uploaded CSV directly into the existing model.

### Do not silently retrain the model.

### Do not replace the model because another algorithm appears better.

---

# 41. Development Workflow for Antigravity

Implement in this order:

## Phase 1 — Inspect ML Artifact

- Load `car_price_model.pkl`.
- Confirm Random Forest.
- Confirm 19 features.
- Confirm feature order.
- Confirm model can make predictions.

## Phase 2 — Build Preprocessing

Implement the exact notebook transformations.

## Phase 3 — Build Prediction Service

Create:

```text
services/prediction.py
```

## Phase 4 — Build Flask Backend

Implement:

```text
/
 /predict
 /health
 /api/predict
```

## Phase 5 — Build Frontend

Create responsive prediction form.

## Phase 6 — Build Result Page

Display:

```text
Predicted INR
Predicted Lakhs
Vehicle summary
Model factors
```

## Phase 7 — Testing

Test preprocessing and prediction independently.

## Phase 8 — Integration Testing

Submit a complete form and verify:

```text
Frontend
→ Flask
→ preprocessing
→ model
→ prediction
→ result page
```

## Phase 9 — Production Cleanup

Remove:

```text
debug mode
unused code
notebook dependencies
duplicate preprocessing
hard-coded prediction values
```

---

# 42. Final Acceptance Criteria

The project is considered complete only when all of the following work:

- [ ] Flask application starts successfully.
- [ ] `car_price_model.pkl` loads successfully.
- [ ] Model is loaded only once.
- [ ] Model reports 19 expected features.
- [ ] Feature names match exactly.
- [ ] Prediction form contains all required raw inputs.
- [ ] User does not manually enter engineered features.
- [ ] Engine log transformation is correct.
- [ ] Maximum-power log transformation is correct.
- [ ] Fuel encoding matches the notebook.
- [ ] Seller encoding matches the notebook.
- [ ] Owner encoding matches the notebook.
- [ ] Transmission encoding matches the notebook.
- [ ] Luxury-brand list matches the notebook.
- [ ] `luxury_engine` is calculated correctly.
- [ ] `performance_index` is calculated correctly.
- [ ] `petrol_wear` is calculated correctly.
- [ ] `age_wear` uses the notebook's 2025 reference year.
- [ ] `tech_score` is calculated correctly.
- [ ] Final DataFrame contains exactly 19 model features.
- [ ] Model prediction succeeds.
- [ ] Prediction is displayed in INR.
- [ ] Prediction is displayed in Lakhs.
- [ ] Invalid inputs produce friendly errors.
- [ ] No Python traceback is shown to the user.
- [ ] `/health` works.
- [ ] `/api/predict` works.
- [ ] Application is responsive.
- [ ] README explains setup and architecture.
- [ ] Automated tests pass.
- [ ] No Streamlit dependency is used.
- [ ] No FastAPI dependency is used.
- [ ] No second prediction algorithm is silently introduced.
- [ ] The uploaded CSV is not incorrectly treated as the training schema.

---

# 43. Final Antigravity Instruction

Build the project completely from the supplied assets.

Treat:

```text
carpriceprediction.ipynb
```

as the authoritative source for feature engineering and model-training logic.

Treat:

```text
car_price_model.pkl
```

as the authoritative source for the production prediction model and its exact 19-feature interface.

Treat:

```text
Car_Price_Prediction_Dataset.csv
```

as a separate dataset that must be validated before any future retraining.

The final deliverable must be a **fully functional Flask web application** that takes raw car specifications, reproduces the notebook's feature engineering, generates the exact 19 model features, runs the supplied Random Forest model, and displays the estimated car price.

The implementation must prioritize **prediction correctness and compatibility with the existing model over adding unnecessary features**.

Do not make assumptions about the model schema.

Do not silently modify the ML pipeline.

Do not silently retrain.

Do not fabricate accuracy, confidence intervals, or predictions.

First make the existing model work correctly end-to-end. Then improve the UI and optional features.