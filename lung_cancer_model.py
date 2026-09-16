"""Clean the lung cancer survey data and train a logistic-regression model.

Target: ``Lung_Cancer`` (Yes / No) -- the survey's actual diagnosis label, so
every input is a question a patient can answer (age, gender, smoking, symptoms).

Run it directly to clean the data, train, print metrics and write the artifacts
used by ``app.py``::

    python lung_cancer_model.py

Artifacts written next to this file:
    data/lung_cancer_clean.csv     cleaned, model-ready table
    lung_cancer_model.pkl          fitted Pipeline (preprocessing + LogisticRegression)
    lung_cancer_metadata.pkl       widget options / ranges / metrics for the UI
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
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

APP_DIR = Path(__file__).resolve().parent
RAW_DATA = APP_DIR / "data" / "survey_lung_cancer.csv"
CLEAN_DATA = APP_DIR / "data" / "lung_cancer_clean.csv"
MODEL_FILE = APP_DIR / "lung_cancer_model.pkl"
METADATA_FILE = APP_DIR / "lung_cancer_metadata.pkl"

TARGET = "Lung_Cancer"
POSITIVE_LABEL = "Yes"
RANDOM_STATE = 42
TEST_SIZE = 0.25

# The survey encodes answers as 1 / 2 and the diagnosis as the string YES / NO.
# 2 means "yes" for every question, which is a nasty detail to get backwards.
SURVEY_YES = 2
SURVEY_NO = 1

# The raw CSV ships inconsistent headers ("CHRONIC DISEASE", "FATIGUE " with a
# trailing space), so rename everything to a single readable convention.
RENAME = {
    "GENDER": "Gender",
    "AGE": "Age",
    "SMOKING": "Smoking",
    "YELLOW_FINGERS": "Yellow_Fingers",
    "ANXIETY": "Anxiety",
    "PEER_PRESSURE": "Peer_Pressure",
    "CHRONIC DISEASE": "Chronic_Disease",
    "FATIGUE": "Fatigue",
    "ALLERGY": "Allergy",
    "WHEEZING": "Wheezing",
    "ALCOHOL CONSUMING": "Alcohol_Consuming",
    "COUGHING": "Coughing",
    "SHORTNESS OF BREATH": "Shortness_of_Breath",
    "SWALLOWING DIFFICULTY": "Swallowing_Difficulty",
    "CHEST PAIN": "Chest_Pain",
    "LUNG_CANCER": TARGET,
}

NUMERIC_FEATURES = ["Age"]

# Yes/No questions -> encoded as 0/1 by the pipeline.
BINARY_FEATURES = [
    "Smoking",
    "Yellow_Fingers",
    "Anxiety",
    "Peer_Pressure",
    "Chronic_Disease",
    "Fatigue",
    "Allergy",
    "Wheezing",
    "Alcohol_Consuming",
    "Coughing",
    "Shortness_of_Breath",
    "Swallowing_Difficulty",
    "Chest_Pain",
]

# Multi-value answers -> one-hot encoded.
NOMINAL_FEATURES = ["Gender"]
GENDER_OPTIONS = ["Female", "Male"]

FEATURES = NUMERIC_FEATURES + NOMINAL_FEATURES + BINARY_FEATURES
BINARY_CATEGORIES = ["No", "Yes"]

# Human-facing question text for the UI, in the order we want them asked.
QUESTION_TEXT = {
    "Age": "Age",
    "Gender": "Gender",
    "Smoking": "Do you smoke?",
    "Yellow_Fingers": "Yellow fingers?",
    "Anxiety": "Do you feel anxious?",
    "Peer_Pressure": "Do you feel peer pressure?",
    "Chronic_Disease": "Any chronic disease?",
    "Fatigue": "Do you feel fatigued?",
    "Allergy": "Any allergies?",
    "Wheezing": "Do you wheeze?",
    "Alcohol_Consuming": "Do you drink alcohol?",
    "Coughing": "Do you cough?",
    "Shortness_of_Breath": "Shortness of breath?",
    "Swallowing_Difficulty": "Difficulty swallowing?",
    "Chest_Pain": "Do you have chest pain?",
}


# --------------------------------------------------------------------------
# Cleaning
# --------------------------------------------------------------------------
def load_raw(path: Path = RAW_DATA) -> pd.DataFrame:
    """Read the raw survey CSV, failing loudly if it is not where we expect."""
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {path}. Expected data/survey_lung_cancer.csv."
        )
    df = pd.read_csv(path)
    # Headers carry stray whitespace in the original file.
    df.columns = [c.strip() for c in df.columns]
    return df


def clean_data(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """Standardise names, values and dtypes; return kept features plus 0/1 target.

    The dataset has no ID or serial column, so unlike a raw export there is
    nothing to drop for being an identifier -- every kept column is a genuine
    question. What it *does* need is exactly one row per respondent, which is
    why exact duplicates are removed.
    """
    missing = [c for c in RENAME if c not in df.columns]
    if missing:
        raise ValueError(f"Raw data is missing expected column(s): {missing}")

    clean = df[list(RENAME)].rename(columns=RENAME).copy()

    # 1. One row per respondent. The file repeats 33 complete rows verbatim, and
    #    duplicates would otherwise straddle the train/test split and inflate
    #    the scores.
    duplicates = int(clean.duplicated().sum())
    clean = clean.drop_duplicates().reset_index(drop=True)

    # 2. Trim stray whitespace so the value checks below are reliable.
    for col in clean.select_dtypes(include="object"):
        clean[col] = clean[col].astype(str).str.strip()

    # 3. Survey codes: 2 -> Yes, 1 -> No, for every question column.
    for col in BINARY_FEATURES:
        unexpected = set(clean[col].unique()) - {SURVEY_YES, SURVEY_NO}
        if unexpected:
            raise ValueError(f"{col}: expected only 1/2, found {sorted(unexpected)}")
        clean[col] = clean[col].map({SURVEY_YES: "Yes", SURVEY_NO: "No"})

    # 4. Gender letters -> words.
    clean["Gender"] = clean["Gender"].map({"M": "Male", "F": "Female"})
    unexpected = set(clean["Gender"].dropna().unique()) - set(GENDER_OPTIONS)
    if unexpected:
        raise ValueError(f"Gender: unexpected value(s) {sorted(unexpected)}")

    # 5. Target: YES / NO -> 1 / 0.
    clean[TARGET] = clean[TARGET].str.upper()
    unexpected = set(clean[TARGET].unique()) - {"YES", "NO"}
    if unexpected:
        raise ValueError(f"{TARGET}: unexpected value(s) {sorted(unexpected)}")
    clean[TARGET] = (clean[TARGET] == "YES").astype(int)

    # 6. Nothing may be missing -- a NaN would break the scaler/encoder.
    if clean.isna().any().any():
        counts = clean.isna().sum()
        raise ValueError(f"Missing values after cleaning:\n{counts[counts > 0]}")

    # Target first so the saved CSV is easy to eyeball.
    clean = clean[[TARGET] + FEATURES]
    if verbose:
        print(f"Dropped {duplicates} duplicate rows.")
    return clean.reset_index(drop=True)


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
def build_pipeline(class_weight: str | None = "balanced") -> Pipeline:
    """Preprocessing (scaling + encoding) wrapped around logistic regression.

    ``class_weight="balanced"`` matters here: only ~14% of respondents are
    "No", so an unweighted model can score 86% accuracy by always answering
    "Yes" while being useless at spotting the cases that matter.
    """
    preprocess = ColumnTransformer(
        transformers=[
            # Standardising lets us read the Age coefficient per std deviation.
            ("num", StandardScaler(), NUMERIC_FEATURES),
            # Explicit category order: No -> 0, Yes -> 1.
            (
                "bin",
                OrdinalEncoder(categories=[BINARY_CATEGORIES] * len(BINARY_FEATURES)),
                BINARY_FEATURES,
            ),
            # One-hot so the two genders are not treated as ordered.
            (
                "nom",
                OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False),
                NOMINAL_FEATURES,
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
        "recall_no": float(recall_score(y_test, y_pred, pos_label=0, zero_division=0)),
        "recall_yes": float(recall_score(y_test, y_pred, pos_label=1, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(
            y_test, y_pred, target_names=["No", "Yes"], digits=3
        ),
    }

    if verbose:
        print(f"Rows: {metrics['n_rows']}  "
              f"(train {metrics['n_train']} / test {metrics['n_test']})")
        print(f"Features used: {metrics['n_features']}")
        print(f"'Yes' share of the data                : {metrics['positive_rate']:.3f}")
        print(f"Majority-class baseline accuracy       : {metrics['majority_baseline']:.3f}")
        print(f"5-fold CV ROC AUC (train split)        : {metrics['cv_roc_auc']:.3f}")
        print(f"5-fold CV balanced accuracy            : {metrics['cv_balanced_accuracy']:.3f}")
        print(f"Held-out test accuracy                 : {metrics['test_accuracy']:.3f}")
        print(f"Held-out test balanced accuracy        : {metrics['test_balanced_accuracy']:.3f}")
        print(f"Held-out test ROC AUC                  : {metrics['test_roc_auc']:.3f}")
        print(f"Recall on 'No'  (minority, the hard one): {metrics['recall_no']:.3f}")
        print(f"Recall on 'Yes'                        : {metrics['recall_yes']:.3f}")
        print("\nConfusion matrix (rows=actual, cols=predicted), order [No, Yes]:")
        print(metrics["confusion_matrix"])
        print("\n" + metrics["classification_report"])

    return pipeline, metrics


def feature_importances(pipeline: Pipeline) -> pd.DataFrame:
    """Coefficients turned into odds ratios, biggest effect first."""
    names = pipeline.named_steps["preprocess"].get_feature_names_out()
    coefs = pipeline.named_steps["model"].coef_[0]
    frame = pd.DataFrame(
        {
            # Strip the "num__" / "bin__" / "nom__" transformer prefix.
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
        "numeric_features": NUMERIC_FEATURES,
        "binary_features": BINARY_FEATURES,
        "nominal_features": NOMINAL_FEATURES,
        "binary_options": BINARY_CATEGORIES,
        "nominal_options": {c: sorted(clean[c].unique().tolist()) for c in NOMINAL_FEATURES},
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
    """Predict from a dict of raw answers. Returns (label, P('Yes'))."""
    row = pd.DataFrame([values], columns=FEATURES)
    probability = float(pipeline.predict_proba(row)[0, 1])
    label = POSITIVE_LABEL if probability >= 0.5 else "No"
    return label, probability


def main() -> None:
    raw = load_raw()
    print(f"Raw shape: {raw.shape}")
    clean = clean_data(raw, verbose=True)
    print(f"Cleaned shape: {clean.shape}")
    print(f"Target balance: {clean[TARGET].value_counts().to_dict()}\n")

    pipeline, metrics = train(clean)
    metadata = build_metadata(clean, metrics)
    save_artifacts(pipeline, metadata, clean)
    print(f"\nSaved: {CLEAN_DATA.name}, {MODEL_FILE.name}, {METADATA_FILE.name}")


if __name__ == "__main__":
    main()
