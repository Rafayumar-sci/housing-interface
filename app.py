"""Unified Streamlit app: house price, lung cancer and COVID-19 prediction.

Run with:  streamlit run app.py

Three prediction sections, each with its own model:

* **House Price**  reads `house model.pkl` + `label_encoders.pkl`
* **Lung Cancer**  reads `lung_cancer_model.pkl` + `lung_cancer_metadata.pkl`
* **COVID-19**     reads `covid19_model.pkl` + `covid19_metadata.pkl`

The models load independently, so a missing file disables only its own
section instead of taking the whole app down.
"""

import pickle
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from covid19_model import (
    MODEL_FILE as COVID_MODEL_FILE,
    QUESTION_TEXT as COVID_QUESTION_TEXT,
    load_artifacts as load_covid_artifacts,
    predict_one as predict_covid_outcome,
)
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


def inject_fonts():
    """Load the Inter webfont; the CSS below falls back to system fonts."""
    st.markdown(
        '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap">',
        unsafe_allow_html=True,
    )


# Light theme styling; drop this block to return to Streamlit's defaults.
st.markdown(
    """
<style>
/* === Design tokens === */
:root {
    --bg: #F4F6FB;
    --surface: #FFFFFF;
    --border: #E3E8F2;
    --ink: #16233B;
    --muted: #5A6580;
    --brand: #4F46E5;
    --brand-deep: #4338CA;
    --danger: #E11D48;
    --danger-soft: #FEF1F5;
    --danger-border: #FECDD6;
    --danger-ink: #9F1239;
    --ok: #059669;
    --ok-soft: #ECFDF5;
    --ok-border: #A7F3D0;
    --ok-ink: #065F46;
    --warn-soft: #FFFBEB;
    --warn-border: #FDE68A;
    --radius: 16px;
    --shadow: 0 10px 30px rgba(15, 23, 42, 0.07);
    --shadow-sm: 0 2px 10px rgba(15, 23, 42, 0.05);
}

/* === Global background & type === */
html, body, [data-testid="stAppViewContainer"] {
    background:
        radial-gradient(1100px 480px at 85% -10%, rgba(79, 70, 229, 0.10), transparent 60%),
        radial-gradient(900px 420px at -10% 110%, rgba(13, 148, 136, 0.07), transparent 60%),
        var(--bg);
}
[data-testid="stAppViewContainer"] {
    color: var(--ink);
    font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif;
    -webkit-font-smoothing: antialiased;
}
body {
    font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
}
.block-container {
    padding-top: 1.6rem;
    padding-bottom: 3rem;
    max-width: 1150px;
}

/* === Sidebar === */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #101736 0%, #1E1B4B 55%, #312E81 100%);
    border-right: none;
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
    color: #C7CBF5;
}
[data-testid="stSidebar"] h1 {
    font-size: 1.22rem !important;
    font-weight: 800 !important;
    letter-spacing: 0.01em;
    color: #FFFFFF !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label {
    padding: 2px 10px;
    border-radius: 10px;
    transition: background 0.15s ease;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
    background: rgba(255, 255, 255, 0.08);
}
[data-testid="stSidebar"] [data-testid="stRadio"] label p {
    color: #C7CBF5 !important;
    font-size: 0.95rem;
    font-weight: 500;
    padding: 0.15rem 0;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover p {
    color: #FFFFFF !important;
}
[data-testid="stSidebar"] [data-testid="stCaption"], [data-testid="stSidebar"] small {
    color: #8B93D9 !important;
}

/* Body radios (e.g. History model picker) */
[data-testid="stRadio"] label p {
    color: var(--ink);
    font-weight: 600;
}
[data-testid="stRadio"] [role="radiogroup"] {
    gap: 0.25rem;
}

/* === Sidebar brand block & status cards === */
.brandblock {
    background: rgba(255, 255, 255, 0.06);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 16px;
    padding: 1.1rem 1.2rem;
    margin-bottom: 1.1rem;
    text-align: center;
}
.brandicon {
    font-size: 1.9rem;
    line-height: 1;
    margin-bottom: 0.45rem;
}
.brandname {
    margin: 0;
    color: #FFFFFF;
    font-weight: 800;
    font-size: 1.05rem;
    letter-spacing: 0.01em;
}
.brandtag {
    margin: 0.3rem 0 0 0;
    color: #A5B4FC;
    font-size: 0.78rem;
}
.statuscard {
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.10);
    border-radius: 12px;
    padding: 0.6rem 0.85rem;
    margin-bottom: 0.55rem;
}
.statusname {
    margin: 0 0 0.2rem 0;
    color: #E0E7FF;
    font-weight: 700;
    font-size: 0.86rem;
}
.statline {
    margin: 0;
    color: #A5B4FC;
    font-size: 0.76rem;
    display: flex;
    align-items: center;
    gap: 0.4rem;
}
.dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    display: inline-block;
    flex: none;
}
.dot-ok {
    background: #34D399;
    box-shadow: 0 0 6px rgba(52, 211, 153, 0.8);
}
.dot-bad {
    background: #FB7185;
    box-shadow: 0 0 6px rgba(251, 113, 133, 0.8);
}

/* === Hero page header === */
.hero {
    background: linear-gradient(135deg, #EEF2FF 0%, #E0F2FE 55%, #F0FDF9 100%);
    border: 1px solid #E0E7FF;
    border-radius: 20px;
    padding: 1.35rem 1.6rem;
    margin-bottom: 1.25rem;
}
.hero h2 {
    margin: 0 0 0.25rem 0;
    font-size: 1.65rem;
    font-weight: 800;
    color: var(--ink);
    letter-spacing: -0.01em;
}
.hero p {
    margin: 0;
    color: var(--muted);
    font-size: 1.02rem;
}
.chips {
    margin-top: 0.7rem;
    display: flex;
    flex-wrap: wrap;
    gap: 0.45rem;
}
.chip {
    background: rgba(255, 255, 255, 0.75);
    border: 1px solid #C7D2FE;
    color: #4338CA;
    border-radius: 999px;
    padding: 0.18rem 0.75rem;
    font-size: 0.78rem;
    font-weight: 600;
}

/* === Cards & forms === */
div[data-testid="stForm"], .panel {
    background: var(--surface);
    padding: 1.75rem;
    border-radius: var(--radius);
    border: 1px solid var(--border);
    box-shadow: var(--shadow);
}
.section-h {
    font-size: 1.02rem;
    font-weight: 800;
    color: var(--ink);
    margin: 0 0 0.85rem 0;
    padding-bottom: 0.55rem;
    border-bottom: 2px solid #EEF2FF;
}

/* === Buttons === */
.stButton > button,
.stFormSubmitButton > button,
.stDownloadButton > button {
    background: linear-gradient(135deg, var(--brand) 0%, #6366F1 100%);
    color: #FFFFFF;
    border: none;
    border-radius: 12px;
    font-weight: 700;
    padding: 0.55rem 1.4rem;
    transition: filter 0.15s ease, transform 0.05s ease;
    box-shadow: 0 6px 16px rgba(79, 70, 229, 0.25);
}
.stButton > button:hover,
.stFormSubmitButton > button:hover,
.stDownloadButton > button:hover {
    background: linear-gradient(135deg, var(--brand-deep) 0%, #4F46E5 100%);
    color: #FFFFFF;
    filter: brightness(1.05);
    transform: translateY(-1px);
}
[data-testid="stFormSubmitButton"] button {
    min-width: 200px;
    font-size: 1.0rem;
}

/* === Alerts, metrics, tabs, subheaders === */
[data-testid="stAlert"] {
    border-radius: 14px;
    border: 1px solid transparent;
    box-shadow: var(--shadow-sm);
}
[data-testid="stMetric"] {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    box-shadow: var(--shadow-sm);
    margin: 0.4rem 0 1.2rem 0;
    padding: 1rem 1.4rem;
}
[data-testid="stMetric"] [data-testid="stMetricLabel"] p {
    margin: 0;
    font-weight: 700;
    color: var(--muted);
    font-size: 0.85rem !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
[data-testid="stMetric"] [data-testid="stMetricValue"] {
    font-weight: 800;
    color: var(--ink);
    font-variant-numeric: tabular-nums;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    border-bottom: 1px solid var(--border);
}
.stTabs [data-baseweb="tab"] {
    padding: 8px 18px;
    border-radius: 10px 10px 0 0;
    font-weight: 600;
}
.stTabs [aria-selected="true"] {
    background: #EEF2FF;
    color: var(--brand-deep) !important;
}
.stTabs [data-baseweb="tab-highlight"] {
    background: var(--brand);
}
.stTabs [data-baseweb="tab-border"] {
    display: none;
}
[data-testid="stSubheader"], [data-testid="stMarkdownContainer"] h3 {
    font-weight: 800 !important;
    letter-spacing: -0.01em;
}
[data-testid="stMarkdownContainer"] p {
    color: var(--ink);
}

/* === Colored result panels === */
.panel-ok {
    background: var(--ok-soft);
    border: 1px solid var(--ok-border);
    border-radius: 14px;
    padding: 1.15rem 1.5rem;
    margin: 0.8rem 0;
}
.panel-ok h2 { color: var(--ok-ink); margin: 0 0 0.3rem 0; }
.panel-ok p { color: var(--ok-ink); margin: 0.1rem 0; }
.panel-danger {
    background: var(--danger-soft);
    border: 1px solid var(--danger-border);
    border-radius: 14px;
    padding: 1.15rem 1.5rem;
    margin: 0.8rem 0;
}
.panel-danger h2 { color: var(--danger-ink); margin: 0 0 0.3rem 0; }
.panel-danger p { color: var(--danger-ink); margin: 0.1rem 0; }
.panel-info {
    background: #EFF6FF;
    border: 1px solid #BFDBFE;
    border-radius: 14px;
    padding: 1.15rem 1.5rem;
    margin: 0.8rem 0;
}
.panel-info h2 { color: #1E40AF; margin: 0 0 0.3rem 0; }
.panel-info p { color: #1E40AF; margin: 0.1rem 0; }
.panel-warn {
    background: var(--warn-soft);
    border: 1px solid var(--warn-border);
    border-radius: 14px;
    padding: 1.15rem 1.5rem;
    margin: 0.8rem 0;
}
.panel-warn h2 { color: #92400E; margin: 0 0 0.3rem 0; }
.panel-warn p { color: #92400E; margin: 0.1rem 0; }

/* === Expanders === */
div[data-testid="stExpander"] {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    box-shadow: var(--shadow-sm);
}
div[data-testid="stExpander"] details {
    border: none !important;
    background: transparent !important;
}
</style>
""",
    unsafe_allow_html=True,
)

inject_fonts()


def humanize(name):
    return name.replace("_", " ").strip().title()


def page_header(icon, title, subtitle, chips=()):
    """Gradient hero banner used at the top of every page."""
    chips_html = "".join(f'<span class="chip">{c}</span>' for c in chips)
    chip_block = f'<div class="chips">{chips_html}</div>' if chips_html else ""
    st.markdown(
        f"""
<div class="hero">
  <h2>{icon} {title}</h2>
  <p>{subtitle}</p>
  {chip_block}
</div>
""",
        unsafe_allow_html=True,
    )


def result_panel(kind, heading, message):
    """Colored callout for predictions: kind in ok / danger / info / warn."""
    st.markdown(
        f'<div class="panel-{kind}"><h2>{heading}</h2><p>{message}</p></div>',
        unsafe_allow_html=True,
    )


def sidebar_status(icon, name, loaded, detail):
    """Sidebar card summarising whether a model is ready to use."""
    if loaded:
        line = f'<div class="statline"><span class="dot dot-ok"></span>{detail}</div>'
    else:
        line = (
            '<div class="statline"><span class="dot dot-bad"></span>'
            "Not loaded — check the .pkl files</div>"
        )
    st.sidebar.markdown(
        f'<div class="statuscard"><p class="statusname">{icon} {name}</p>{line}</div>',
        unsafe_allow_html=True,
    )


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


@st.cache_resource(show_spinner="Loading the COVID-19 model...")
def load_covid():
    return load_covid_artifacts()


house_model = house_encoders = None
lung_model = lung_metadata = None
covid_model = covid_metadata = None
house_error = lung_error = covid_error = None

try:
    house_model, house_encoders = load_house()
except Exception as exc:  # noqa: BLE001 - reported in the House Price section
    house_error = str(exc)

try:
    lung_model, lung_metadata = load_lung()
except Exception as exc:  # noqa: BLE001 - reported in the Lung Cancer section
    lung_error = str(exc)

try:
    covid_model, covid_metadata = load_covid()
except Exception as exc:  # noqa: BLE001 - reported in the COVID-19 section
    covid_error = str(exc)


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
    if house_error:
        page_header("🏠", "House Price Prediction", "Instant price estimate from property details.")
        result_panel(
            "warn",
            "Model not loaded",
            f"Put `house model.pkl` and `label_encoders.pkl` in `{APP_DIR}` and reload.",
        )
        return

    feature_names, encoders_by_column, numeric_fields, categorical_fields = house_layout(
        house_model, house_encoders
    )

    page_header(
        "🏠",
        "House Price Prediction",
        "Fill in the property details to get an instant price estimate.",
        chips=[
            f"Model: {type(house_model).__name__}",
            f"{len(numeric_fields)} numeric inputs",
            f"{len(categorical_fields)} categorical inputs",
        ],
    )

    with st.form("house_form"):
        values = {}
        left, right = st.columns(2)

        with left:
            st.markdown('<p class="section-h">Property basics</p>', unsafe_allow_html=True)
            for name in numeric_fields:
                label, low, high, step, default = HOUSE_NUMERIC_FIELDS.get(
                    name, (humanize(name), 0, 1_000_000, 1, 0)
                )
                values[name] = st.number_input(
                    label, min_value=low, max_value=high, step=step, value=default
                )

        with right:
            st.markdown(
                '<p class="section-h">Amenities &amp; condition</p>', unsafe_allow_html=True
            )
            for name in categorical_fields:
                options = list(encoders_by_column[name].classes_)
                values[name] = st.selectbox(humanize(name), options)

        submitted = st.form_submit_button("Predict Price", width="stretch")

    if submitted:
        try:
            inputs = build_house_row(values, feature_names, encoders_by_column)
            prediction = float(house_model.predict(inputs)[0])
        except Exception as exc:  # noqa: BLE001 - surface any failure to the user
            result_panel("danger", "Could not make a prediction", f"`{exc}`")
        else:
            result_panel(
                "info",
                f"Estimated price: {money(prediction)}",
                "Prices follow the units of the training target, so treat this as a "
                "relative estimate rather than a market valuation.",
            )
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


# --------------------------------------------------------------------------
# Lung cancer section
# --------------------------------------------------------------------------
def render_lung_section():
    if lung_error:
        page_header("🫁", "Lung Cancer Prediction", "Yes / no screening from survey answers.")
        result_panel(
            "warn",
            "Model not loaded",
            f"Put `lung_cancer_model.pkl` and `lung_cancer_metadata.pkl` in `{APP_DIR}` — "
            f"or create them by running `python lung_cancer_model.py`.",
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

    page_header(
        "🫁",
        "Lung Cancer Prediction",
        "Answer the questions below and the model will predict **yes** or **no**.",
        chips=[
            f"{metrics['n_rows']} survey respondents",
            f"ROC AUC {metrics['test_roc_auc']:.2f}",
            "Teaching model — not a diagnosis",
        ],
    )

    with st.form("lung_form"):
        values = {}
        left, right = st.columns(2)

        with left:
            st.markdown('<p class="section-h">About you</p>', unsafe_allow_html=True)
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
            st.markdown(
                '<p class="section-h">Symptoms and history</p>', unsafe_allow_html=True
            )
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
            result_panel(
                "warn",
                "Please answer every question",
                "Still missing: " + ", ".join(f"*{q}*" for q in unanswered),
            )
        else:
            try:
                label, probability = predict_lung_cancer(lung_model, values)
            except Exception as exc:  # noqa: BLE001 - surface any failure to the user
                result_panel("danger", "Could not make a prediction", f"`{exc}`")
            else:
                if label == "Yes":
                    result_panel(
                        "danger",
                        "Prediction: Lung cancer — YES",
                        f"Model probability of **yes**: {probability:.1%}.",
                    )
                else:
                    result_panel(
                        "ok",
                        "Prediction: Lung cancer — NO",
                        f"Model probability of **yes**: {probability:.1%}.",
                    )

                st.progress(min(max(probability, 0.0), 1.0))
                st.caption(
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
# COVID-19 section
# --------------------------------------------------------------------------
def render_covid_section():
    if covid_error:
        page_header("🦠", "COVID-19 Patient Risk", "Survival risk from the admission profile.")
        result_panel(
            "warn",
            "Model not loaded",
            f"Put `covid19_model.pkl` and `covid19_metadata.pkl` in `{APP_DIR}` — or "
            f"create them by running `python covid19_model.py`.",
        )
        return

    numeric_features = covid_metadata["numeric_features"]
    patient_features = covid_metadata["patient_features"]
    clinical_features = covid_metadata["clinical_features"]
    categorical_options = covid_metadata["categorical_options"]
    ranges = covid_metadata["ranges"]
    metrics = covid_metadata["metrics"]
    question_text = covid_metadata.get("question_text") or COVID_QUESTION_TEXT

    page_header(
        "🦠",
        "COVID-19 Patient Risk Prediction",
        "Enter the patient's admission profile and the model will predict whether "
        "they are likely to **survive** or **die**.",
        chips=[
            f"{metrics['n_rows']} patients (de-duplicated)",
            f"ROC AUC {metrics['test_roc_auc']:.2f}",
            "Admission-time inputs only",
        ],
    )

    with st.form("covid_form"):
        values = {}
        left, right = st.columns(2)

        with left:
            st.markdown('<p class="section-h">About the patient</p>', unsafe_allow_html=True)
            age_range = ranges.get("Age", {"min": 0, "max": 120, "default": 40})
            for name in numeric_features:
                values[name] = st.number_input(
                    question_text.get(name, humanize(name)),
                    min_value=int(age_range["min"]),
                    max_value=int(age_range["max"]),
                    value=int(age_range["default"]),
                    step=1,
                )
            for name in patient_features:
                values[name] = st.selectbox(
                    question_text.get(name, humanize(name)),
                    categorical_options[name],
                )

        with right:
            st.markdown(
                '<p class="section-h">Conditions at admission</p>', unsafe_allow_html=True
            )
            for name in clinical_features:
                values[name] = st.selectbox(
                    question_text.get(name, humanize(name)),
                    categorical_options[name],
                )

        submitted = st.form_submit_button("Predict Outcome", width="stretch")

    if submitted:
        try:
            label, probability = predict_covid_outcome(covid_model, values)
        except Exception as exc:  # noqa: BLE001 - surface any failure to the user
            result_panel("danger", "Could not make a prediction", f"`{exc}`")
        else:
            prob_text = (
                f"Model probability of **death**: {probability:.1%}. The model is "
                "class-weighted, so treat this as a relative risk score rather "
                "than a calibrated mortality estimate."
            )
            if label == "Died":
                result_panel("danger", "Prediction: likely DIED", prob_text)
            else:
                result_panel("ok", "Prediction: likely SURVIVED", prob_text)

            st.progress(min(max(probability, 0.0), 1.0))

            st.session_state.covid_history.insert(
                0,
                {
                    "when": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "prediction": label,
                    "probability_died": round(probability, 4),
                    **values,
                },
            )
            st.session_state.covid_history = st.session_state.covid_history[:25]

            with st.expander("Inputs sent to the model"):
                st.dataframe(
                    pd.DataFrame([values], columns=list(values)), width="stretch"
                )

    st.divider()
    st.caption(
        f"⚠️ Teaching model trained on a public sample of {metrics['n_rows']} "
        "patients (de-duplicated). It is **not** a clinical diagnosis and must "
        "not be used to make health decisions."
    )


# --------------------------------------------------------------------------
# History section
# --------------------------------------------------------------------------
def render_history_section():
    page_header(
        "🗂️",
        "Prediction History",
        "Every prediction made in this session, per model.",
        chips=["Last 25 kept per model", "Exportable as CSV"],
    )

    which = st.radio(
        "Model",
        ["🏠 House prices", "🫁 Lung cancer", "🦠 COVID-19"],
        horizontal=True,
        key="hist_which",
    )
    is_house = which.startswith("🏠")
    is_covid = which.startswith("🦠")
    if is_house:
        history = st.session_state.house_history
    elif is_covid:
        history = st.session_state.covid_history
    else:
        history = st.session_state.lung_history

    if not history:
        result_panel(
            "warn",
            "No predictions yet",
            "Make one on the model's prediction page and it will show up here.",
        )
        return

    frame = pd.DataFrame(history)

    # Summary strip above the table.
    count, stat, stat_label = len(frame), None, None
    if is_house:
        stat, stat_label = money(frame["price"].median()), "Median estimate"
    elif is_covid:
        stat = f"{(frame['prediction'] == 'Died').mean():.0%}"
        stat_label = "Predicted died"
    else:
        stat = f"{(frame['prediction'] == 'Yes').mean():.0%}"
        stat_label = "Predicted yes"

    m1, m2 = st.columns(2)
    with m1:
        st.metric("Saved predictions", str(count))
    with m2:
        st.metric(stat_label, stat)

    if is_house:
        display = frame.assign(price=frame["price"].map(money))
        caption = f"Prices follow the units of the training target ({CURRENCY})."
        file_name = "house_predictions.csv"
    elif is_covid:
        display = frame.assign(
            probability_died=lambda df: df["probability_died"].map(
                lambda p: f"{p:.1%}"
            )
        )
        caption = "Predictions are died/survived outcomes from the COVID-19 patient model."
        file_name = "covid19_predictions.csv"
    else:
        display = frame.assign(
            probability_yes=lambda df: df["probability_yes"].map(lambda p: f"{p:.1%}")
        )
        caption = "Predictions are yes/no outcomes from the lung cancer survey model."
        file_name = "lung_cancer_predictions.csv"

    st.dataframe(display, width="stretch", height=320)
    st.caption(caption)

    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "Download CSV",
            frame.to_csv(index=False).encode("utf-8"),
            file_name=file_name,
            mime="text/csv",
            width="stretch",
        )
    with col2:
        if st.button("Clear history", width="stretch"):
            if is_house:
                st.session_state.house_history = []
            elif is_covid:
                st.session_state.covid_history = []
            else:
                st.session_state.lung_history = []
            st.rerun()


# --------------------------------------------------------------------------
# About section
# --------------------------------------------------------------------------
def render_about_section():
    page_header(
        "🧠",
        "About",
        "Three independent models share this interface — each loads its own "
        "artifacts and degrades gracefully if files are missing.",
    )

    house_tab, lung_tab, covid_tab = st.tabs(
        ["🏠 House Price", "🫁 Lung Cancer", "🦠 COVID-19"]
    )

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

    with covid_tab:
        if covid_error:
            st.error("This model is not currently loaded.")
        else:
            metrics = covid_metadata["metrics"]
            st.markdown(
                f"""
**{type(covid_model.named_steps["model"]).__name__}** trained on the *COVID-19
Patient Risk Analysis* dataset: {metrics['n_rows']} patients after
de-duplication.

Every input is known at admission time, so it can be used as an early risk
screen. Post-admission outcomes (intubation, ICU, death date) are deliberately
excluded so they cannot leak into the prediction.

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
                f"patients in the sample died, so answering \"died\" every time is "
                f"the naive extreme; the majority baseline of "
                f"{metrics['majority_baseline']:.3f} comes from always answering "
                "\"survived\". The model is deliberately class-weighted, which "
                "costs a little accuracy and lets it catch far more of the deaths "
                "that matter."
            )
            with st.expander("Confusion matrix (held-out test: Survived, Died)"):
                st.dataframe(
                    pd.DataFrame(
                        metrics["confusion_matrix"],
                        index=["actual Survived", "actual Died"],
                        columns=["predicted Survived", "predicted Died"],
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
if "covid_history" not in st.session_state:
    st.session_state.covid_history = []

st.sidebar.markdown(
    """
<div class="brandblock">
  <div class="brandicon">🧠</div>
  <p class="brandname">ML Prediction Studio</p>
  <p class="brandtag">House prices &amp; health risk screens</p>
</div>
""",
    unsafe_allow_html=True,
)

page = st.sidebar.radio(
    "Navigate",
    ["🏠 House Price", "🫁 Lung Cancer", "🦠 COVID-19", "History", "About"],
)

if house_error:
    sidebar_status("🏠", "House Price", False, "")
else:
    sidebar_status(
        "🏠",
        "House Price",
        True,
        f"{type(house_model).__name__} · "
        f"{len(getattr(house_model, 'feature_names_in_', []))} features",
    )

if lung_error:
    sidebar_status("🫁", "Lung Cancer", False, "")
else:
    metrics = lung_metadata["metrics"]
    n_questions = (
        len(lung_metadata["numeric_features"])
        + len(lung_metadata["binary_features"])
        + len(lung_metadata["nominal_features"])
    )
    sidebar_status(
        "🫁",
        "Lung Cancer",
        True,
        f"{n_questions} questions · ROC AUC {metrics['test_roc_auc']:.2f}",
    )

if covid_error:
    sidebar_status("🦠", "COVID-19", False, "")
else:
    metrics = covid_metadata["metrics"]
    n_inputs = len(covid_metadata["numeric_features"]) + len(
        covid_metadata["categorical_features"]
    )
    sidebar_status(
        "🦠",
        "COVID-19",
        True,
        f"{n_inputs} inputs · ROC AUC {metrics['test_roc_auc']:.2f}",
    )

st.sidebar.markdown(" ")
st.sidebar.caption("Teaching project — predictions are not medical advice.")

if page == "🏠 House Price":
    render_house_section()
elif page == "🫁 Lung Cancer":
    render_lung_section()
elif page == "🦠 COVID-19":
    render_covid_section()
elif page == "History":
    render_history_section()
else:
    render_about_section()
