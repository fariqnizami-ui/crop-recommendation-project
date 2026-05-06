import streamlit as st
import pickle
import numpy as np

# 1. Load your saved artifacts
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
le = pickle.load(open('label_encoder.pkl', 'rb'))

st.set_page_config(page_title="Crop Predictor", page_icon="🌱")
st.title("🌾 Crop Recommendation System")
st.write("Enter the details below to find the best crop for your land.")

# 2. Create the input form (matching your test_input order)
col1, col2 = st.columns(2)
with col1:
    n = st.number_input("Nitrogen (N)", value=90)
    p = st.number_input("Phosphorus (P)", value=42)
    k = st.number_input("Potassium (K)", value=43)
    temp = st.number_input("Temperature (°C)", value=20.8)
with col2:
    hum = st.number_input("Humidity (%)", value=82.0)
    ph = st.number_input("pH Level", value=6.5)
    rain = st.number_input("Rainfall (mm)", value=202.9)

# 3. Prediction Logic
if st.button("Predict Best Crop"):
    # Prepare the input exactly like your notebook test
    test_input = np.array([[n, p, k, temp, hum, ph, rain]])
    scaled_input = scaler.transform(test_input)
    
    # Run prediction
    prediction = model.predict(scaled_input)
    crop_name = le.inverse_transform(prediction)[0]
    
    # Show result
    st.success(f"### The recommended crop is: **{crop_name.upper()}**")