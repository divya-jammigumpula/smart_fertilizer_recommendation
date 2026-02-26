# --------------------------------
# FILE 2: app.py  (STREAMLIT APP)
# --------------------------------

import streamlit as st
import numpy as np
import joblib

# Load artifacts
models = joblib.load('npk_models.pkl')
soil_enc = joblib.load('soil_encoder.pkl')
crop_enc = joblib.load('crop_encoder.pkl')

st.set_page_config(page_title='Smart Fertilizer (NPK)', layout='centered')
st.title('🌱 Smart Fertilizer Recommendation (High Accuracy)')
st.write('Predict optimal **Nitrogen, Phosphorous, Potassium** values')

# Inputs
temp = st.number_input('Temperature (°C)', 20, 50)
humidity = st.number_input('Humidity (%)', 30, 100)
moisture = st.number_input('Soil Moisture (%)', 10, 100)

soil_type = st.selectbox('Soil Type', soil_enc.classes_)
crop_type = st.selectbox('Crop Type', crop_enc.classes_)

if st.button('Predict NPK'):
    soil_val = soil_enc.transform([soil_type])[0]
    crop_val = crop_enc.transform([crop_type])[0]

    X_input = np.array([[temp, humidity, moisture, soil_val, crop_val]])

    n = models['Nitrogen'].predict(X_input)[0]
    p = models['Phosphorous'].predict(X_input)[0]
    k = models['Potassium'].predict(X_input)[0]

    st.success('✅ Fertilizer Nutrient Recommendation')
    st.metric('Nitrogen (N)', f'{n:.2f} kg/ha')
    st.metric('Phosphorous (P)', f'{p:.2f} kg/ha')
    st.metric('Potassium (K)', f'{k:.2f} kg/ha')
