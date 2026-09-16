"""Clean the COVID-19 patient data and train a death-risk model.

Target: ``DEATH_STATUS`` (Died / Survived) from the Mexican-government-style
COVID-19 dataset. Every input is known at admission time, so the model can be
used as an early risk screen -- it deliberately excludes anything measured
*after* admission that would leak the outcome:

* ``INTUBED`` / ``ICU`` -- mostly recorded for patients who already died.
* ``DATE_DIED`` / ``DEATH_YEAR`` / ``DEATH_MONTH`` -- the outcome itself.
* ``CRITICAL_CARE``, ``*_STATUS``, ``RISK_CATEGORY``, ``RECOVERY_STATUS`` --
  derived columns that re-encode the excluded fields or the target.

Run it directly to clean the data, train, print metrics and write the artifacts
used by ``app.py``::

    python covid19_model.py

Artifacts written next to this file:
    data/covid19_clean.csv         cleaned, model-ready table
    covid19_model.pkl              fitted Pipeline (preprocessing + LogisticRegression)
    covid19_metadata.pkl           widget options / ranges / metrics for the UI
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

APP_DIR = Path(__file__).resolve().parent
RAW_DATA = APP_DIR / "COVID19_Patient_Risk_Analysis.csv"
CLEAN_DATA = APP_DIR / "data" / "covid19_clean.csv"
MODEL_FILE = APP_DIR / "covid19_model.pkl"
METADATA_FILE = APP_DIR / "covid19_metadata.pkl"

TARGET = "Died"
POSITIVE_LABEL = "Died"  # the class the probability refers to
RANDOM_STATE = 42
TEST_SIZE = 0.25

# This dataset codes 1 = Yes and 2 = No -- the *opposite* of the lung cancer
# survey -- and uses 97 (not applicable), 98 / 99 (unknown) plus blanks for
# missing values.
SURVEY_YES = 1
SURVEY_NO = 2

# Raw column -> readable feature name. Only admission-time information is kept.
RENAME = {
    "SEX": "Sex",
    "AGE": "Age",
    "PATIENT_TYPE": "Patient_Type",
    "PNEUMONIA": "Pneumonia",
    "PREGNANT": "Pregnant",
    "DIABETES": "Diabetes",
    "COPD": "COPD",
    "ASTHMA": "Asthma",
    "INMSUPR": "Immunosuppression",
    "HIPERTENSION": "Hypertension",
    "OTHER_DISEASE": "Other_Disease",
    "CARDIOVASCULAR": "Cardiovascular",
    "OBESITY": "Obesity",
    "RENAL_CHRONIC": "Renal_Chronic",
    "TOBACCO": "Tobacco",
    "CLASIFFICATION_FINAL": "Covid_Case",
    "DEATH_STATUS": TARGET,
}

NUMERIC_FEATURES = ["Age"]

# Patient profile: sex, how they presented, pregnancy status.
PATIENT_FEATURES = ["Sex", "Patient_Type", "Pregnant"]

# Conditions recorded at admission, plus the test-result classification.
# CLASIFFICATION_FINAL 1-3 means a confirmed COVID case of increasing severity;
# 4-7 means the test was not carried out or was inconclusive.
CLINICAL_FEATURES = [
    "Pneumonia",
    "Covid_Case",
    "Diabetes",
    "COPD",
    "Asthma",
    "Immunosuppression",
    "Hypertension",
    "Other_Disease",
    "Cardiovascular",
    "Obesity",
    "Renal_Chronic",
    "Tobacco",
]

CATEGORICAL_FEATURES = PATIENT_FEATURES + CLINICAL_FEATURES
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# The simple No/Yes/Unknown flags: every clinical column except Covid_Case,
# which has its own category list.
HEALTH_FLAGS = [c for c in CLINICAL_FEATURES if c != "Covid_Case"]

SEX_OPTIONS = ["Female", "Male"]
PATIENT_TYPE_OPTIONS = ["Outpatient", "Hospitalized"]
PREGNANT_OPTIONS = ["No", "Yes", "Not applicable", "Unknown"]
HEALTH_OPTIONS = ["No", "Yes", "Unknown"]
COVID_CASE_OPTIONS = ["Case confirmed", "No case / not tested"]

# Human-facing question text for the UI, in the order we want them asked.
QUESTION_TEXT = {
    "Age": "Age (years)",
    "Sex": "Sex",
    "Patient_Type": "Patient type",
    "Pregnant": "Pregnant?",
    "Pneumonia": "Pneumonia at admission?",
    "Covid_Case": "COVID-19 test result",
    "Diabetes": "Diabetes?",
    "COPD": "COPD?",
    "Asthma": "Asthma?",
    "Immunosuppression": "Immunosuppressed?",
    "Hypertension": "Hypertension?",
    "Other_Disease": "Other chronic disease?",
    "Cardiovascular": "Cardiovascular disease?",
    "Obesity": "Obesity?",
    "Renal_Chronic": "Chronic kidney disease?",
    "Tobacco": "Tobacco use?",
}


# --------------------------------------------------------------------------
# Cleaning
# --------------------------------------------------------------------------
def load_raw(path: Path = RAW_DATA) -> pd.DataFrame:
    """Read the raw CSV, failing loudly if it is not where we expect."""
    if not Path(path).exists():
        raise FileNotFoundError(f"Raw dataset not found at {path}.")
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """Keep admission-time features only; decode the 1/2/97/98 codes.

    Codes are mapped to readable categories here (never inside the pipeline)
    so the saved CSV and the UI widgets show real words. Unknown codes and
    blanks collapse into an explicit "Unknown" category instead of NaN.
    """
    missing = [c for c in RENAME if c not in df.columns]
    if missing:
        raise ValueError(f"Raw data is missing expected column(s): {missing}")

    clean = df[list(RENAME)].rename(columns=RENAME).copy()

    # 1. One row per patient. Re-sampled exports repeat rows verbatim, and
    #    duplicates would straddle the train/test split and inflate scores.
    duplicates = int(clean.duplicated().sum())
    clean = clean.drop_duplicates().reset_index(drop=True)

    # 2. Age: numeric; the handful of rows without one are dropped.
    clean["Age"] = pd.to_numeric(clean["Age"], errors="coerce")
    dropped_age = int(clean["Age"].isna().sum())
    clean = clean.dropna(subset=["Age"]).reset_index(drop=True)
    clean["Age"] = clean["Age"].astype(int)

    # 3. Sex: 1 = woman, 2 = man.
    clean["Sex"] = clean["Sex"].map({1: "Female", 2: "Male"})
    unexpected = set(clean["Sex"].dropna().unique()) - set(SEX_OPTIONS)
    if unexpected:
        raise ValueError(f"Sex: unexpected value(s) {sorted(unexpected)}")

    # 4. Patient type: 1 = returned home, 2 = hospitalised.
    clean["Patient_Type"] = clean["Patient_Type"].map(
        {1: "Outpatient", 2: "Hospitalized"}
    )
    unexpected = set(clean["Patient_Type"].dropna().unique()) - set(
        PATIENT_TYPE_OPTIONS
    )
    if unexpected:
        raise ValueError(f"Patient_Type: unexpected value(s) {sorted(unexpected)}")

    # 5. Pregnancy: 97 = not applicable (men), 98 = unknown.
    clean["Pregnant"] = (
        clean["Pregnant"]
        .map({1: "Yes", 2: "No", 97: "Not applicable", 98: "Unknown"})
        .fillna("Unknown")
    )

    # 6. Pneumonia and the comorbidity/tobacco flags: 1 = Yes, 2 = No;
    #    blanks and stray codes (98/99) become "Unknown".
    for col in HEALTH_FLAGS:
        clean[col] = (
            clean[col].map({SURVEY_YES: "Yes", SURVEY_NO: "No"}).fillna("Unknown")
        )

    # 7. Test classification: 1-3 confirmed case, 4-7 no case / not tested.
    clean["Covid_Case"] = clean["Covid_Case"].map(
        {1: "Case confirmed", 2: "Case confirmed", 3: "Case confirmed"},
        na_action="ignore",
    ).fillna("No case / not tested")

    # 8. Target: Survived / Died -> 0 / 1.
    clean[TARGET] = clean[TARGET].astype(str).str.strip().str.title()
    unexpected = set(clean[TARGET].unique()) - {"Survived", "Died"}
    if unexpected:
        raise ValueError(f"{TARGET}: unexpected value(s) {sorted(unexpected)}")
    clean[TARGET] = (clean[TARGET] == "Died").astype(int)

    # 9. The one-hot encoder cannot see NaN either.
    if clean[FEATURES].isna().any().any():
        counts = clean[FEATURES].isna().sum()
        raise ValueError(f"Missing values after cleaning:\n{counts[counts > 0]}")

    # Target first so the saved CSV is easy to eyeball.
    clean = clean[[TARGET] + FEATURES]
    if verbose:
        print(f"Dropped {duplicates} duplicate rows and {dropped_age} rows without age.")
    return clean.reset_index(drop=True)


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
def build_pipeline(class_weight: str | None = "balanced") -> Pipeline:
    """Preprocessing (scaling + one-hot encoding) around logistic regression.

    ``class_weight="balanced"`` matters here: only ~7% of patients in the
    sample died, so an unweighted model can score 93% accuracy by always
    answering "Survived" while missing the cases that matter.
    """
    preprocess = ColumnTransformer(
        transformers=[
            # Standardising lets us read the Age coefficient per std deviation.
            ("num", StandardScaler(), NUMERIC_FEATURES),
            # One-hot for every categorical: categories are nominal (e.g.
            # "Unknown" is not between No and Yes), and handle_unknown keeps
            # prediction safe if a new code ever shows up.
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocess),
            (
                "model",
                LogisticRegression(
                    max_iter=5000,
                    class_weight=class_weight,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def split_data(clean: pd.DataFrame):
    """Stratified train/test split, shared so the notebook cannot drift."""
    X, y = clean[FEATURES], clean[TARGET]
    return train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )


def train(clean: pd.DataFrame, verbose: bool = True) -> tuple[Pipeline, dict]:
    """Fit on a stratified train split and evaluate on held-out data."""
    X_train, X_test, y_train, y_test = split_data(clean)

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "n_rows": int(len(clean)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_features": len(FEATURES),
        "test_accuracy": float(accuracy_score(y_test, y_pred)),
        "test_balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "test_roc_auc": float(roc_auc_score(y_test, y_prob)),
        "majority_baseline": float(max(y_test.mean(), 1 - y_test.mean())),
        "positive_rate": float(clean[TARGET].mean()),
        "cv_roc_auc": float(
            cross_val_score(pipeline, X_train, y_train, cv=5, scoring="roc_auc").mean()
        ),
        "cv_balanced_accuracy": float(
            cross_val_score(
                pipeline, X_train, y_train, cv=5, scoring="balanced_accuracy"
            ).mean()
        ),
        "recall_survived": float(
            recall_score(y_test, y_pred, pos_label=0, zero_division=0)
        ),
        "recall_died": float(
            recall_score(y_test, y_pred, pos_label=1, zero_division=0)
        ),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(
            y_test, y_pred, target_names=["Survived", "Died"], digits=3
        ),
    }

    if verbose:
        print(f"Rows: {metrics['n_rows']}  "
              f"(train {metrics['n_train']} / test {metrics['n_test']})")
        print(f"Features used: {metrics['n_features']}")
        print(f"'Died' share of the data               : {metrics['positive_rate']:.3f}")
        print(f"Majority-class baseline accuracy       : {metrics['majority_baseline']:.3f}")
        print(f"5-fold CV ROC AUC (train split)        : {metrics['cv_roc_auc']:.3f}")
        print(f"5-fold CV balanced accuracy            : {metrics['cv_balanced_accuracy']:.3f}")
        print(f"Held-out test accuracy                 : {metrics['test_accuracy']:.3f}")
        print(f"Held-out test balanced accuracy        : {metrics['test_balanced_accuracy']:.3f}")
        print(f"Held-out test ROC AUC                  : {metrics['test_roc_auc']:.3f}")
        print(f"Recall on 'Survived'                   : {metrics['recall_survived']:.3f}")
        print(f"Recall on 'Died' (minority, the hard one): {metrics['recall_died']:.3f}")
        print("\nConfusion matrix (rows=actual, cols=predicted), order [Survived, Died]:")
        print(metrics["confusion_matrix"])
        print("\n" + metrics["classification_report"])

    return pipeline, metrics


def feature_importances(pipeline: Pipeline) -> pd.DataFrame:
    """Coefficients turned into odds ratios, biggest effect first."""
    names = pipeline.named_steps["preprocess"].get_feature_names_out()
    coefs = pipeline.named_steps["model"].coef_[0]
    frame = pd.DataFrame(
        {
            # Strip the "num__" / "cat__" transformer prefix.
            "feature": [str(n).split("__", 1)[-1] for n in names],
            "coefficient": coefs,
        }
    )
    frame["odds_ratio"] = np.exp(frame["coefficient"])
    frame["question"] = frame["feature"].map(QUESTION_TEXT).fillna(frame["feature"])
    return frame.sort_values("coefficient", key=abs, ascending=False).reset_index(
        drop=True
    )


def build_metadata(clean: pd.DataFrame, metrics: dict) -> dict:
    """Everything the Streamlit UI needs to build valid widgets."""
    age = clean["Age"]
    return {
        "target": TARGET,
        "positive_label": POSITIVE_LABEL,
        "negative_label": "Survived",
        "numeric_features": NUMERIC_FEATURES,
        "patient_features": PATIENT_FEATURES,
        "clinical_features": CLINICAL_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "categorical_options": {
            "Sex": SEX_OPTIONS,
            "Patient_Type": PATIENT_TYPE_OPTIONS,
            "Pregnant": PREGNANT_OPTIONS,
            "Covid_Case": COVID_CASE_OPTIONS,
            **{c: HEALTH_OPTIONS for c in HEALTH_FLAGS},
        },
        "question_text": QUESTION_TEXT,
        "ranges": {
            "Age": {
                "min": int(age.min()),
                "max": int(age.max()),
                "default": int(age.median()),
            }
        },
        "metrics": metrics,
    }


def save_artifacts(
    pipeline: Pipeline,
    metadata: dict,
    clean: pd.DataFrame,
    model_file: Path = MODEL_FILE,
    metadata_file: Path = METADATA_FILE,
    clean_file: Path = CLEAN_DATA,
) -> None:
    clean_file.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(clean_file, index=False)
    with open(model_file, "wb") as fh:
        pickle.dump(pipeline, fh)
    with open(metadata_file, "wb") as fh:
        pickle.dump(metadata, fh)


def load_artifacts(model_file: Path = MODEL_FILE, metadata_file: Path = METADATA_FILE):
    with open(model_file, "rb") as fh:
        pipeline = pickle.load(fh)
    with open(metadata_file, "rb") as fh:
        metadata = pickle.load(fh)
    return pipeline, metadata


def predict_one(pipeline: Pipeline, values: dict) -> tuple[str, float]:
    """Predict from a dict of raw (decoded) answers. Returns (label, P('Died'))."""
    row = pd.DataFrame([values], columns=FEATURES)
    probability = float(pipeline.predict_proba(row)[0, 1])
    label = POSITIVE_LABEL if probability >= 0.5 else "Survived"
    return label, probability


def main() -> None:
    raw = load_raw()
    print(f"Raw shape: {raw.shape}")
    clean = clean_data(raw, verbose=True)
    print(f"Cleaned shape: {clean.shape}")
    print(f"Target balance: {clean[TARGET].value_counts().to_dict()}\n")

    pipeline, metrics = train(clean)
    print(
        "\nTop risk factors (odds ratio > 1 raises the odds of dying, < 1 lowers):"
    )
    print(feature_importances(pipeline).head(10).to_string(index=False))

    metadata = build_metadata(clean, metrics)
    save_artifacts(pipeline, metadata, clean)
    print(f"\nSaved: {CLEAN_DATA.name}, {MODEL_FILE.name}, {METADATA_FILE.name}")


if __name__ == "__main__":
    main()
