"""Unified Streamlit app: house price prediction and lung cancer prediction.

Run with:  streamlit run app.py

Two prediction sections, each with its own model:

* **House Price**  reads `house model.pkl` + `label_encoders.pkl`
* **Lung Cancer**  reads `lung_cancer_model.pkl` + `lung_cancer_metadata.pkl`

The two models load independently, so a missing file disables only its own
section instead of taking the whole app down.
"""

import pickle
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from lung_cancer_model import (
    MODEL_FILE as LUNG_MODEL_FILE,
    QUESTION_TEXT,
    load_artifacts as load_lung_artifacts,
    predict_one as predict_lung_cancer,
)

APP_DIR = Path(__file__).resolve().parent
HOUSE_MODEL_FILE = APP_DIR / "house model.pkl"
HOUSE_ENCODERS_FILE = APP_DIR / "label_encoders.pkl"

CURRENCY = "$"

# House numeric widgets: name -> (label, min, max, step, default)
HOUSE_NUMERIC_FIELDS = {
    "area": ("Area (sq ft)", 100, 20_000, 50, 1200),
    "bedrooms": ("Bedrooms", 1, 10, 1, 3),
    "bathrooms": ("Bathrooms", 1, 10, 1, 2),
    "stories": ("Stories", 1, 10, 1, 2),
    "parking": ("Parking Spaces", 0, 10, 1, 1),
}


# --------------------------------------------------------------------------
# Page setup
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="ML Prediction Studio",
    page_icon="🧠",
    layout="wide",
)

# Light theme styling; drop this block to return to Streamlit's defaults.
st.markdown(
    """
<style>
[data-testid="stAppViewContainer"] {
    background-color: #F8FAFC;
}
.block-container {
    padding-top: 2.5rem;
    padding-bottom: 2.5rem;
}
div[data-testid="stForm"] {
    background-color: #FFFFFF;
    padding: 1.75rem;
    border-radius: 16px;
    border: 1px solid #E2E8F0;
    box-shadow: 0 10px 25px rgba(15, 23, 42, 0.06);
}
.stButton > button,
.stFormSubmitButton > button,
.stDownloadButton > button {
    background-color: #2563EB;
    color: #FFFFFF;
    border: none;
    border-radius: 12px;
    font-weight: 600;
}
.stButton > button:hover,
.stFormSubmitButton > button:hover,
.stDownloadButton > button:hover {
    background-color: #1D4ED8;
    color: #FFFFFF;
}
</style>
""",
    unsafe_allow_html=True,
)


def humanize(name):
    return name.replace("_", " ").strip().title()


# --------------------------------------------------------------------------
# Loading -- each model is loaded on its own so one failure stays contained
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading the house price model...")
def load_house():
    with open(HOUSE_MODEL_FILE, "rb") as fh:
        model = pickle.load(fh)
    with open(HOUSE_ENCODERS_FILE, "rb") as fh:
        encoders = pickle.load(fh)
    return model, encoders


@st.cache_resource(show_spinner="Loading the lung cancer model...")
def load_lung():
    return load_lung_artifacts()


house_model = house_encoders = None
lung_model = lung_metadata = None
house_error = lung_error = None

try:
    house_model, house_encoders = load_house()
except Exception as exc:  # noqa: BLE001 - reported in the House Price section
    house_error = str(exc)

try:
    lung_model, lung_metadata = load_lung()
except Exception as exc:  # noqa: BLE001 - reported in the Lung Cancer section
    lung_error = str(exc)


def missing_files_message(files, retrain):
    return (
        f"Couldn't load the model. Expected {files} next to this app "
        f"(`{APP_DIR}`), and got: {retrain}"
    )


# --------------------------------------------------------------------------
# House price section
# --------------------------------------------------------------------------
def house_layout(model, encoders):
    """Work out the model's input columns and which ones need encoding."""
    feature_names = list(getattr(model, "feature_names_in_", []))
    if not feature_names:
        feature_names = list(HOUSE_NUMERIC_FIELDS) + [
            k[3:] if k.startswith("le_") else k for k in encoders
        ]
    # Encoder keys carry an 'le_' prefix that the model columns do not.
    encoders_by_column = {
        (k[3:] if k.startswith("le_") else k): v for k, v in encoders.items()
    }
    numeric = [f for f in feature_names if f not in encoders_by_column]
    categorical = [f for f in feature_names if f in encoders_by_column]
    return feature_names, encoders_by_column, numeric, categorical


def build_house_row(values, feature_names, encoders_by_column):
    """Encode a dict of raw form values into a single-row DataFrame."""
    row = {}
    for name in feature_names:
        value = values[name]
        if name in encoders_by_column:
            encoder = encoders_by_column[name]
            options = list(encoder.classes_)
            # Defensive: an unexpected category falls back to the first class.
            row[name] = int(encoder.transform([value if value in options else options[0]])[0])
        else:
            row[name] = float(value)
    return pd.DataFrame([row], columns=feature_names)


def money(value):
    return f"{CURRENCY}{value:,.0f}"


def render_house_section():
    st.title("🏠 House Price Prediction")
    st.write("Fill in the property details to get an instant price estimate.")

    if house_error:
        st.error(
            "The house price model could not be loaded. Put `house model.pkl` and "
            f"`label_encoders.pkl` in `{APP_DIR}`.\n\n`{house_error}`"
        )
        return

    feature_names, encoders_by_column, numeric_fields, categorical_fields = house_layout(
        house_model, house_encoders
    )

    with st.form("house_form"):
        values = {}
        left, right = st.columns(2)

        with left:
            st.subheader("Property basics")
            for name in numeric_fields:
                label, low, high, step, default = HOUSE_NUMERIC_FIELDS.get(
                    name, (humanize(name), 0, 1_000_000, 1, 0)
                )
                values[name] = st.number_input(
                    label, min_value=low, max_value=high, step=step, value=default
                )

        with right:
            st.subheader("Amenities & condition")
            for name in categorical_fields:
                options = list(encoders_by_column[name].classes_)
                values[name] = st.selectbox(humanize(name), options)

        submitted = st.form_submit_button("Predict Price", width="stretch")

    if submitted:
        try:
            inputs = build_house_row(values, feature_names, encoders_by_column)
            prediction = float(house_model.predict(inputs)[0])
        except Exception as exc:  # noqa: BLE001 - surface any failure to the user
            st.error(f"Could not make a prediction: {exc}")
        else:
            st.metric("Estimated Price", money(prediction))
            st.session_state.house_history.insert(
                0,
                {
                    "when": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "price": prediction,
                    **values,
                },
            )
            st.session_state.house_history = st.session_state.house_history[:25]

            with st.expander("Inputs used for this estimate"):
                st.dataframe(inputs, width="stretch")

    st.caption(
        f"Prices follow the units of the training target ({CURRENCY}), so treat the "
        "output as a relative estimate rather than a market valuation."
    )


# --------------------------------------------------------------------------
# Lung cancer section
# --------------------------------------------------------------------------
def render_lung_section():
    st.title("🫁 Lung Cancer Prediction")
    st.write("Answer the questions below and the model will predict **yes** or **no**.")

    if lung_error:
        st.error(
            "The lung cancer model could not be loaded. Put `lung_cancer_model.pkl` "
            f"and `lung_cancer_metadata.pkl` in `{APP_DIR}` — or create them by "
            f"running `python lung_cancer_model.py`.\n\n`{lung_error}`"
        )
        return

    numeric_features = lung_metadata["numeric_features"]
    binary_features = lung_metadata["binary_features"]
    nominal_features = lung_metadata["nominal_features"]
    binary_options = lung_metadata["binary_options"]
    nominal_options = lung_metadata["nominal_options"]
    ranges = lung_metadata["ranges"]
    metrics = lung_metadata["metrics"]
    question_text = lung_metadata.get("question_text") or QUESTION_TEXT

    with st.form("lung_form"):
        values = {}
        left, right = st.columns(2)

        with left:
            st.subheader("About you")
            age_range = ranges.get("Age", {"min": 1, "max": 120, "default": 45})
            for name in numeric_features:
                values[name] = st.number_input(
                    question_text.get(name, humanize(name)),
                    min_value=int(age_range["min"]),
                    max_value=int(age_range["max"]),
                    value=int(age_range["default"]),
                    step=1,
                )
            for name in nominal_features:
                values[name] = st.selectbox(
                    question_text.get(name, humanize(name)),
                    nominal_options.get(name, []),
                )

        with right:
            st.subheader("Symptoms and history")
            for name in binary_features:
                values[name] = st.selectbox(
                    question_text.get(name, humanize(name)),
                    binary_options,
                    index=None,
                    placeholder="Choose an answer",
                )

        submitted = st.form_submit_button("Predict Diagnosis", width="stretch")

    if submitted:
        unanswered = [
            question_text.get(n, humanize(n))
            for n in binary_features
            if values.get(n) is None
        ]
        if unanswered:
            st.warning(
                "Please answer every question. Still missing: "
                + ", ".join(f"*{q}*" for q in unanswered)
            )
        else:
            try:
                label, probability = predict_lung_cancer(lung_model, values)
            except Exception as exc:  # noqa: BLE001 - surface any failure to the user
                st.error(f"Could not make a prediction: {exc}")
            else:
                if label == "Yes":
                    st.error("## Prediction: Lung cancer — YES")
                else:
                    st.success("## Prediction: Lung cancer — NO")

                st.progress(min(max(probability, 0.0), 1.0))
                st.caption(
                    f"Model probability of **yes**: {probability:.1%}. "
                    "The training data is 86% positive and the model is "
                    "class-weighted, so treat this as a relative score rather "
                    "than a calibrated risk."
                )

                st.session_state.lung_history.insert(
                    0,
                    {
                        "when": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "prediction": label,
                        "probability_yes": round(probability, 4),
                        **values,
                    },
                )
                st.session_state.lung_history = st.session_state.lung_history[:25]

                with st.expander("Answers sent to the model"):
                    st.dataframe(
                        pd.DataFrame([values], columns=list(values)), width="stretch"
                    )

    st.divider()
    st.caption(
        f"⚠️ Teaching model trained on a public survey of {metrics['n_rows']} "
        "respondents. It is **not** a medical diagnosis and must not be used to "
        "make health decisions."
    )


# --------------------------------------------------------------------------
# History section
# --------------------------------------------------------------------------
def render_history_section():
    st.title("Prediction History")

    which = st.radio(
        "Model", ["🏠 House prices", "🫁 Lung cancer"], horizontal=True, key="hist_which"
    )
    is_house = which.startswith("🏠")
    history = st.session_state.house_history if is_house else st.session_state.lung_history

    if not history:
        st.info("No predictions yet for this model. Make one on its prediction page.")
        return

    frame = pd.DataFrame(history)
    if is_house:
        display = frame.assign(price=frame["price"].map(money))
        caption = f"Prices follow the units of the training target ({CURRENCY})."
    else:
        display = frame.assign(
            probability_yes=lambda df: df["probability_yes"].map(lambda p: f"{p:.1%}")
        )
        caption = "Predictions are yes/no outcomes from the lung cancer survey model."

    st.dataframe(display, width="stretch")
    st.caption(caption)

    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "Download CSV",
            frame.to_csv(index=False).encode("utf-8"),
            file_name="house_predictions.csv" if is_house else "lung_cancer_predictions.csv",
            mime="text/csv",
            width="stretch",
        )
    with col2:
        if st.button("Clear history", width="stretch"):
            if is_house:
                st.session_state.house_history = []
            else:
                st.session_state.lung_history = []
            st.rerun()


# --------------------------------------------------------------------------
# About section
# --------------------------------------------------------------------------
def render_about_section():
    st.title("About")
    st.write("Two independent models share this interface.")

    house_tab, lung_tab = st.tabs(["🏠 House Price", "🫁 Lung Cancer"])

    with house_tab:
        if house_error:
            st.error("This model is not currently loaded.")
        else:
            feature_names, encoders_by_column, numeric_fields, categorical_fields = (
                house_layout(house_model, house_encoders)
            )
            st.markdown(
                f"""
**{type(house_model).__name__}** loaded from `{HOUSE_MODEL_FILE.name}`, with
`{HOUSE_ENCODERS_FILE.name}` supplying the categorical encoders.

* **{len(numeric_fields)} numeric** inputs are entered directly.
* **{len(categorical_fields)} categorical** inputs are encoded with the bundled
  `LabelEncoder`s, using the same class order as training.
* The target is a price, shown in the units of the training data ({CURRENCY}).
"""
            )
            with st.expander("Model input columns (in order)"):
                st.code("\n".join(feature_names))

    with lung_tab:
        if lung_error:
            st.error("This model is not currently loaded.")
        else:
            metrics = lung_metadata["metrics"]
            st.markdown(
                f"""
**{type(lung_model.named_steps["model"]).__name__}** trained on the public *Lung
Cancer Survey* dataset: {metrics['n_rows']} respondents after de-duplication.

Every input is something a patient can answer, so it can be used before any
diagnosis is known.

| metric | value |
| --- | --- |
| Held-out accuracy | {metrics['test_accuracy']:.3f} |
| Majority-class baseline | {metrics['majority_baseline']:.3f} |
| Held-out balanced accuracy | {metrics['test_balanced_accuracy']:.3f} |
| Held-out ROC AUC | {metrics['test_roc_auc']:.3f} |
| 5-fold CV ROC AUC | {metrics['cv_roc_auc']:.3f} |
"""
            )
            st.info(
                f"**Read accuracy carefully.** {metrics['positive_rate']:.0%} of "
                f"respondents are positive, so answering \"yes\" every time already "
                f"scores {metrics['majority_baseline']:.3f}. The model is deliberately "
                "class-weighted, which costs a little accuracy and lets it catch the "
                "minority \"no\" cases."
            )
            with st.expander("Confusion matrix (held-out test: No, Yes)"):
                st.dataframe(
                    pd.DataFrame(
                        metrics["confusion_matrix"],
                        index=["actual No", "actual Yes"],
                        columns=["predicted No", "predicted Yes"],
                    ),
                    width="stretch",
                )
            with st.expander("Full classification report"):
                st.code(metrics["classification_report"])


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
if "house_history" not in st.session_state:
    st.session_state.house_history = []
if "lung_history" not in st.session_state:
    st.session_state.lung_history = []

st.sidebar.title("🧠 ML Prediction Studio")
page = st.sidebar.radio(
    "Navigate",
    ["🏠 House Price", "🫁 Lung Cancer", "History", "About"],
)

if house_error:
    st.sidebar.warning("House model not loaded")
else:
    st.sidebar.caption(
        f"🏠 {type(house_model).__name__} · "
        f"{len(getattr(house_model, 'feature_names_in_', []))} features"
    )

if lung_error:
    st.sidebar.warning("Lung model not loaded")
else:
    metrics = lung_metadata["metrics"]
    st.sidebar.caption(
        f"🫁 {len(lung_metadata['numeric_features']) + len(lung_metadata['binary_features']) + len(lung_metadata['nominal_features'])} "
        f"questions · ROC AUC {metrics['test_roc_auc']:.2f}"
    )

if page == "🏠 House Price":
    render_house_section()
elif page == "🫁 Lung Cancer":
    render_lung_section()
elif page == "History":
    render_history_section()
else:
    render_about_section()
