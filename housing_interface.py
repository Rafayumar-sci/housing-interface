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


# Small CSS to make the app look cleaner and ensure readable colors
st.markdown(
    """
    <style>
    /* page backgrounds and primary text color */
    html, body, .stApp, .block-container, .main {
        background: linear-gradient(180deg,#f8fafc,#e8eef7) !important;
        color: #0f172a !important;
    }
    /* make all streamlit text inherit the readable color */
    .stApp * { color: inherit !important; }

    /* card styling */
    .card {
        padding: 12px;
        border-radius: 10px;
        background: #ffffff;
        color: #0f172a !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    }

    /* buttons and inputs */
    button, .stButton button, .stDownloadButton button { color: #0f172a !important; }

    /* Sidebar specific fixes: ensure readable background, text and icon colors */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg,#eef2f7,#e0e7ef) !important;
        color: #0f172a !important;
    }
    section[data-testid="stSidebar"] * { color: inherit !important; }
    section[data-testid="stSidebar"] svg { fill: #0f172a !important; }
    section[data-testid="stSidebar"] .css-1d391kg, section[data-testid="stSidebar"] .stMarkdown, section[data-testid="stSidebar"] .stText { color: inherit !important; }

    </style>
    """,
    unsafe_allow_html=True,
)


# -------------------------------
# Sidebar (navigation & options)
# -------------------------------
with st.sidebar:
    st.header("House Price App")
    page = st.selectbox("Navigate", ["Predict", "History", "About"])
    st.markdown("---")
    st.write("Quick actions")
    if st.button("Clear History"):
        st.session_state.history = []
        st.success("History cleared")


# -------------------------------
# Helper to encode categorical features
# -------------------------------
def encode_features(selected_vals):
    enc = []
    for feature, val in selected_vals.items():
        encoded = label_encoders[feature].transform([val])[0]
        enc.append(int(encoded))
    return enc


def append_history(record):
    record["id"] = st.session_state.next_id
    st.session_state.next_id += 1
    st.session_state.history.insert(0, record)


# -------------------------------
# Page: Predict
# -------------------------------
if page == "Predict":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
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

            # Save to history
            record = {
                "area": area,
                "bedrooms": bedrooms,
                "bathrooms": bathrooms,
                "stories": stories,
                "parking": parking,
                **selected,
                "prediction": display_price,
            }
            append_history(record)
        except Exception as e:
            st.error(f"Prediction failed: {e}")

    st.markdown("</div>", unsafe_allow_html=True)


# -------------------------------
# Page: History
# -------------------------------
if page == "History":
    st.title("🕘 Prediction History")
    st.write("Browse previous predictions, filter results, or download as CSV.")

    if not st.session_state.history:
        st.info("No history yet — make a prediction first.")
    else:
        # simple filters
        min_p = st.number_input("Min predicted price",
                                min_value=0.0, value=0.0)
        max_p = st.number_input("Max predicted price",
                                min_value=0.0, value=999999999.0)

        filtered = [r for r in st.session_state.history if min_p <=
                    r["prediction"] <= max_p]

        for rec in filtered:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            cols = st.columns([1, 2, 1])
            with cols[0]:
                st.write(f"**#{rec['id']}**")
                st.write(f"${rec['prediction']:,}")
            with cols[1]:
                st.write(
                    f"Area: {rec['area']} sq ft — {rec['bedrooms']}bd / {rec['bathrooms']}ba")
                extras = []
                for f in label_encoders:
                    extras.append(
                        f"{f.replace('_', ' ').title()}: {rec.get(f, 'N/A')}")
                st.write(", ".join(extras))
            with cols[2]:
                if st.button("Use as new", key=f"reuse_{rec['id']}"):
                    # push values into session so user can reuse via query params; simplified UX: show message
                    st.info(
                        "To reuse this entry, copy values manually from this card.")
            st.markdown("</div>", unsafe_allow_html=True)

        # download CSV
        csv_lines = []
        headers = []
        if filtered:
            headers = list(filtered[0].keys())
            csv_lines.append(",".join(headers))
            for r in filtered:
                row = [str(r.get(h, "")) for h in headers]
                csv_lines.append(",".join(row))
            csv_data = "\n".join(csv_lines)
            st.download_button("Download CSV", csv_data,
                               file_name="predictions.csv")


# -------------------------------
# Page: About
# -------------------------------
if page == "About":
    st.title("About this App")
    st.markdown(
        "This is a lightweight house-price prediction demo. It uses a pre-trained model and label encoders stored alongside this script.\n\nFeatures added: cleaner layout, sidebar navigation, prediction history, filtering and CSV export, and nicer visuals — all without installing extra packages.")
