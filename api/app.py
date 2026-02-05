from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from imbalance.smote_handler import apply_smote
from interpretability import permutation_importance
from models.logistic_regression import train_logistic_regression
from models.neural_network import train_neural_network
from models.random_forest import train_random_forest
from preprocessing.data_cleaning import clean_data

FEATURE_COLUMNS = [
    "gender",
    "age",
    "hypertension",
    "heart_disease",
    "ever_married",
    "work_type",
    "Residence_type",
    "avg_glucose_level",
    "bmi",
    "smoking_status",
]

CATEGORICAL_COLUMNS = [
    "gender",
    "ever_married",
    "work_type",
    "Residence_type",
    "smoking_status",
]

NUMERIC_COLUMNS = [
    "age",
    "hypertension",
    "heart_disease",
    "avg_glucose_level",
    "bmi",
]

MAX_MISSING_FIELDS = 3


class StrokeInput(BaseModel):
    gender: Optional[str] = Field(default=None, examples=["Male"])
    age: Optional[float] = Field(default=None, examples=[67])
    hypertension: Optional[int] = Field(default=None, examples=[0])
    heart_disease: Optional[int] = Field(default=None, examples=[1])
    ever_married: Optional[str] = Field(default=None, examples=["Yes"])
    work_type: Optional[str] = Field(default=None, examples=["Private"])
    Residence_type: Optional[str] = Field(default=None, examples=["Urban"])
    avg_glucose_level: Optional[float] = Field(default=None, examples=[228.69])
    bmi: Optional[float] = Field(default=None, examples=[36.6])
    smoking_status: Optional[str] = Field(default=None, examples=["formerly smoked"])


class ModelArtifacts:
    def __init__(self) -> None:
        self.encoders: Dict[str, LabelEncoder] = {}
        self.scaler: Optional[StandardScaler] = None
        self.models: Dict[str, Any] = {}
        self.feature_importance: Dict[str, Any] = {}
        self.feature_defaults: Dict[str, Any] = {}
        self.feature_ranges: Dict[str, Dict[str, float]] = {}


ARTIFACTS = ModelArtifacts()

app = FastAPI(title="Stroke Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


def _fit_encoders(df: pd.DataFrame) -> Dict[str, LabelEncoder]:
    encoders: Dict[str, LabelEncoder] = {}
    for col in CATEGORICAL_COLUMNS:
        encoder = LabelEncoder()
        encoder.fit(df[col].astype(str))
        encoders[col] = encoder
    return encoders


def _encode_features(df: pd.DataFrame, encoders: Dict[str, LabelEncoder]) -> pd.DataFrame:
    encoded = df.copy()
    for col, encoder in encoders.items():
        encoded[col] = encoder.transform(encoded[col].astype(str))
    return encoded


def _prepare_defaults(df: pd.DataFrame) -> Dict[str, Any]:
    defaults: Dict[str, Any] = {}
    for col in FEATURE_COLUMNS:
        if col in CATEGORICAL_COLUMNS:
            defaults[col] = df[col].mode(dropna=True)[0]
        else:
            defaults[col] = float(df[col].median())
    return defaults


def _prepare_ranges(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    ranges: Dict[str, Dict[str, float]] = {}
    for col in NUMERIC_COLUMNS:
        ranges[col] = {
            "min": float(df[col].min()),
            "max": float(df[col].max())
        }
    return ranges


def _train_models() -> None:
    df = pd.read_csv("data/stroke.csv")
    df = clean_data(df)

    features_df = df.drop("stroke", axis=1)
    ARTIFACTS.feature_defaults = _prepare_defaults(features_df)
    ARTIFACTS.feature_ranges = _prepare_ranges(features_df)

    encoders = _fit_encoders(features_df)
    ARTIFACTS.encoders = encoders

    encoded = _encode_features(features_df, encoders)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(encoded[FEATURE_COLUMNS])
    ARTIFACTS.scaler = scaler

    y = df["stroke"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    X_train_smote, y_train_smote = apply_smote(X_train, y_train)
    lr_model = train_logistic_regression(X_train_smote, y_train_smote)
    rf_model = train_random_forest(X_train_smote, y_train_smote)

    nn_model, _ = train_neural_network(
        X_train,
        y_train,
        epochs=40,
        batch_size=32,
        dropout_rate=0.3,
        class_weight={0: 1.0, 1: 3.0},
        loss_function="binary_crossentropy",
        verbose=0
    )

    ARTIFACTS.models = {
        "logistic_regression": lr_model,
        "random_forest": rf_model,
        "neural_network": nn_model,
    }

    rf_importance = pd.Series(
        rf_model.feature_importances_,
        index=FEATURE_COLUMNS
    ).sort_values(ascending=False)

    lr_importance = pd.Series(
        np.abs(lr_model.coef_[0]),
        index=FEATURE_COLUMNS
    ).sort_values(ascending=False)

    nn_importance = permutation_importance(
        nn_model,
        X_test,
        y_test,
        feature_names=FEATURE_COLUMNS,
        is_nn=True,
        n_repeats=3
    )

    ARTIFACTS.feature_importance = {
        "logistic_regression": lr_importance,
        "random_forest": rf_importance,
        "neural_network": nn_importance,
    }


def _clean_input_value(field: str, value: Any) -> Any:
    if value in (None, ""):
        return None
    if field in ("hypertension", "heart_disease"):
        try:
            return int(value)
        except ValueError:
            return None
    if field in ("age", "avg_glucose_level", "bmi"):
        try:
            return float(value)
        except ValueError:
            return None
    return value


def _preprocess_input(payload: StrokeInput) -> Tuple[np.ndarray, Dict[str, Any], List[str]]:
    data = payload.model_dump()
    cleaned: Dict[str, Any] = {}
    missing_fields: List[str] = []

    for col in FEATURE_COLUMNS:
        value = _clean_input_value(col, data.get(col))
        if value is None:
            missing_fields.append(col)
            value = ARTIFACTS.feature_defaults[col]
        cleaned[col] = value

    if cleaned["smoking_status"] == "Unknown":
        cleaned["smoking_status"] = "never smoked"

    input_df = pd.DataFrame([cleaned])

    for col in CATEGORICAL_COLUMNS:
        encoder = ARTIFACTS.encoders[col]
        value = str(input_df.at[0, col])
        if value not in encoder.classes_:
            value = str(ARTIFACTS.feature_defaults[col])
        input_df.at[0, col] = encoder.transform([value])[0]

    scaler = ARTIFACTS.scaler
    if scaler is None:
        raise RuntimeError("Scaler is not initialized")

    X_scaled = scaler.transform(input_df[FEATURE_COLUMNS])
    return X_scaled, cleaned, missing_fields


def _format_probability(probability: float) -> Dict[str, Any]:
    label = 1 if probability >= 0.5 else 0
    return {
        "probability": float(probability),
        "label": label,
        "risk": "High" if label == 1 else "Low"
    }


def _model_top_feature(model_key: str, X_scaled: np.ndarray) -> str:
    if model_key == "logistic_regression":
        coef = ARTIFACTS.models[model_key].coef_[0]
        contributions = np.abs(coef * X_scaled[0])
        return FEATURE_COLUMNS[int(np.argmax(contributions))]

    if model_key == "random_forest":
        importance = ARTIFACTS.feature_importance[model_key]
        return str(importance.index[0])

    importance_df = ARTIFACTS.feature_importance[model_key]
    return str(importance_df.iloc[0]["Feature"])


def _overall_top_feature(feature_list: List[str]) -> str:
    counts: Dict[str, int] = {}
    for feat in feature_list:
        counts[feat] = counts.get(feat, 0) + 1
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


@app.on_event("startup")
def startup_event() -> None:
    _train_models()


@app.get("/metadata")
def metadata() -> Dict[str, Any]:
    categories: Dict[str, List[str]] = {}
    for col, encoder in ARTIFACTS.encoders.items():
        categories[col] = [str(value) for value in encoder.classes_]

    return {
        "feature_columns": FEATURE_COLUMNS,
        "max_missing_fields": MAX_MISSING_FIELDS,
        "categories": categories,
        "numeric_ranges": ARTIFACTS.feature_ranges
    }


@app.post("/predict")
def predict(payload: StrokeInput) -> Dict[str, Any]:
    X_scaled, cleaned, missing = _preprocess_input(payload)

    if len(missing) > MAX_MISSING_FIELDS:
        return {
            "error": "Too many missing fields",
            "missing_fields": missing,
            "max_missing_fields": MAX_MISSING_FIELDS
        }

    lr_prob = ARTIFACTS.models["logistic_regression"].predict_proba(X_scaled)[0, 1]
    rf_prob = ARTIFACTS.models["random_forest"].predict_proba(X_scaled)[0, 1]
    nn_prob = float(ARTIFACTS.models["neural_network"].predict(X_scaled, verbose=0)[0, 0])

    model_predictions = {
        "logistic_regression": _format_probability(float(lr_prob)),
        "random_forest": _format_probability(float(rf_prob)),
        "neural_network": _format_probability(float(nn_prob)),
    }

    top_features = {
        "logistic_regression": _model_top_feature("logistic_regression", X_scaled),
        "random_forest": _model_top_feature("random_forest", X_scaled),
        "neural_network": _model_top_feature("neural_network", X_scaled),
    }

    overall_top = _overall_top_feature(list(top_features.values()))

    return {
        "input": cleaned,
        "missing_fields": missing,
        "predictions": model_predictions,
        "top_features": top_features,
        "most_influential_feature": {
            "feature": overall_top,
            "note": "Based on model importance; validate clinically."
        }
    }
