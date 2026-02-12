import streamlit as st
import pickle
import numpy as np
import os


# -------------------------------
# Load model and encoders
# -------------------------------
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "house model.pkl")
    encoders_path = os.path.join(script_dir, "label_encoders.pkl")

    model = pickle.load(open(model_path, "rb"))
    label_encoders = pickle.load(open(encoders_path, "rb"))
except FileNotFoundError as e:
    st.set_page_config(page_title="House Price Predictor", page_icon="🏠")
    st.title("🏠 House Price Prediction App")
    st.error(
        f"❌ Error: Missing required files!\n\n{str(e)}\n\nPlease ensure both 'house model.pkl' and 'label_encoders.pkl' are in the same directory as this script.")
    st.stop()


# -------------------------------
# App configuration + initial state
# -------------------------------
st.set_page_config(page_title="House Price Predictor", page_icon="🏠",
                   layout="wide", initial_sidebar_state="expanded")

if "history" not in st.session_state:
    st.session_state.history = []

if "next_id" not in st.session_state:
    st.session_state.next_id = 1


# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a page:", ["Predict", "About"])


def encode_features(selected):
    encoded = []
    for feature, encoder in label_encoders.items():
        val = selected.get(feature, "")
        try:
            encoded_val = encoder.transform([val])[0]
        except ValueError:
            encoded_val = -1  # handle unseen categories
        encoded.append(encoded_val)
    return encoded


# -------------------------------
# Page: Predict
# -------------------------------
if page == "Predict":
    st.title("🏠 Predict a House Price")
    st.write("Enter the property details and get an instant estimate.")

    with st.form(key="predict_form"):
        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            area = st.number_input(
                "Area (sq ft)", min_value=100, step=50, value=1200)
            bedrooms = st.number_input(
                "Bedrooms", min_value=1, step=1, value=3)
        with col2:
            bathrooms = st.number_input(
                "Bathrooms", min_value=1, step=1, value=2)
            stories = st.number_input("Stories", min_value=1, step=1, value=1)
        with col3:
            parking = st.number_input(
                "Parking Spaces", min_value=0, step=1, value=1)

        # categorical selectors (automatically from encoders)
        cat_cols = st.columns(len(label_encoders))
        selected = {}
        for i, feature in enumerate(label_encoders):
            label = feature.replace("_", " ").title()
            options = list(label_encoders[feature].classes_)
            with cat_cols[i]:
                val = st.selectbox(label, options, key=f"sel_{feature}")
            selected[feature] = val

        submitted = st.form_submit_button("Predict Price")

    if submitted:
        encoded_values = encode_features(selected)
        input_data = np.array(
            [[area, bedrooms, bathrooms, stories, parking] + encoded_values])
        try:
            prediction = float(model.predict(input_data)[0])
            display_price = round(prediction, 2)
            st.metric(label="Estimated Price", value=f"${display_price:,}")
            st.success(f"💰 Estimated House Price: ${display_price:,}")
        except Exception as e:
            st.error(f"❌ Error making prediction: {str(e)}")


# -------------------------------
if page == "About":
    st.title("About this App")
    st.markdown(
        "This is a lightweight house-price prediction demo. It uses a pre-trained model and label encoders stored alongside this script.\n\nFeatures added: cleaner layout, sidebar navigation, prediction history, filtering and CSV export, and nicer visuals — all without installing extra packages.")
