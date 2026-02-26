# ==============================================
# HIGH-ACCURACY SMART FERTILIZER (NPK REGRESSION)
# Based on GIVEN DATASET
# ==============================================

# -------------------------------
# FILE 1: model.py  (TRAIN MODEL)
# -------------------------------

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error
from xgboost import XGBRegressor

# Load dataset
df = pd.read_csv('Fertilizer Prediction (1).csv')

# Encode categorical features
soil_enc = LabelEncoder()
crop_enc = LabelEncoder()

df['Soil Type'] = soil_enc.fit_transform(df['Soil Type'])
df['Crop Type'] = crop_enc.fit_transform(df['Crop Type'])

# Features (NO NPK leakage)
X = df[[
    'Temparature', 'Humidity', 'Moisture',
    'Soil Type', 'Crop Type'
]]

# Targets (REGRESSION)
y = df[['Nitrogen', 'Phosphorous', 'Potassium']]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train separate XGBoost models
models = {}
for col in y.columns:
    model = XGBRegressor(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42
    )
    model.fit(X_train, y_train[col])
    preds = model.predict(X_test)
    print(f"{col} -> R2: {r2_score(y_test[col], preds):.3f} | MAE: {mean_absolute_error(y_test[col], preds):.2f}")
    models[col] = model

# Save artifacts
joblib.dump(models, 'npk_models.pkl')
joblib.dump(soil_enc, 'soil_encoder.pkl')
joblib.dump(crop_enc, 'crop_encoder.pkl')

print('Models saved successfully')


