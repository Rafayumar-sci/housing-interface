import streamlit as st
import pickle
import numpy as np
import os

# -------------------------------
# Load model and encoders
# -------------------------------
try:
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "house model.pkl")
    encoders_path = os.path.join(script_dir, "label_encoders.pkl")

    model = pickle.load(open(model_path, "rb"))
    label_encoders = pickle.load(open(encoders_path, "rb"))
except FileNotFoundError as e:
    st.error(
        f"❌ Error: Missing required files!\n\n{str(e)}\n\nPlease ensure both 'house model.pkl' and 'label_encoders.pkl' are in the same directory as this script.")
    st.stop()

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(page_title="House Price Predictor", page_icon="🏠")
st.title("🏠 House Price Prediction App")
st.write("Enter house details to predict the price")

# -------------------------------
# Numeric inputs
# -------------------------------
area = st.number_input("Area (sq ft)", min_value=100, step=50)
bedrooms = st.number_input("Bedrooms", min_value=1, step=1)
bathrooms = st.number_input("Bathrooms", min_value=1, step=1)
stories = st.number_input("Stories", min_value=1, step=1)
parking = st.number_input("Parking Spaces", min_value=0, step=1)

# -------------------------------
# Categorical inputs (AUTO from encoders)
# -------------------------------
encoded_values = []

for feature in label_encoders:
    value = st.selectbox(
        feature.replace("_", " ").title(),
        label_encoders[feature].classes_
    )
    encoded = label_encoders[feature].transform([value])[0]
    encoded_values.append(encoded)

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict Price"):
    input_data = np.array([[
        area,
        bedrooms,
        bathrooms,
        stories,
        parking
    ] + encoded_values])

    prediction = model.predict(input_data)[0]
    st.success(f"💰 Estimated House Price: {round(prediction, 2)}")
