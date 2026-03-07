import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from imbalance.smote_handler import apply_smote
from interpretability import permutation_importance
from models.logistic_regression import train_logistic_regression
from models.neural_network import train_neural_network
from models.random_forest import train_random_forest
from preprocessing.data_cleaning import clean_data

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────
MODEL_DIR = Path("models/saved")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

_LR_PATH  = MODEL_DIR / "lr.joblib"
_RF_PATH  = MODEL_DIR / "rf.joblib"
_NN_PATH  = MODEL_DIR / "nn.keras"
_META_PATH = MODEL_DIR / "meta.joblib"   # encoders, scaler, defaults, ranges, importances

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


_VALID_GENDER   = {"Male", "Female", "Other"}
_VALID_MARRIED  = {"Yes", "No"}
_VALID_WORK     = {"Private", "Self-employed", "Govt_job", "children", "Never_worked"}
_VALID_RESIDENCE = {"Urban", "Rural"}
_VALID_SMOKING  = {"never smoked", "formerly smoked", "smokes", "Unknown"}


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

    @field_validator("age")
    @classmethod
    def validate_age(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and not (0 <= v <= 120):
            raise ValueError("age must be between 0 and 120")
        return v

    @field_validator("avg_glucose_level")
    @classmethod
    def validate_glucose(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and not (30 <= v <= 600):
            raise ValueError("avg_glucose_level must be between 30 and 600")
        return v

    @field_validator("bmi")
    @classmethod
    def validate_bmi(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and not (5 <= v <= 100):
            raise ValueError("bmi must be between 5 and 100")
        return v

    @field_validator("hypertension", "heart_disease")
    @classmethod
    def validate_binary(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and v not in (0, 1):
            raise ValueError("value must be 0 or 1")
        return v

    @field_validator("gender")
    @classmethod
    def validate_gender(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in _VALID_GENDER:
            raise ValueError(f"gender must be one of {sorted(_VALID_GENDER)}")
        return v

    @field_validator("ever_married")
    @classmethod
    def validate_married(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in _VALID_MARRIED:
            raise ValueError(f"ever_married must be one of {sorted(_VALID_MARRIED)}")
        return v

    @field_validator("work_type")
    @classmethod
    def validate_work(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in _VALID_WORK:
            raise ValueError(f"work_type must be one of {sorted(_VALID_WORK)}")
        return v

    @field_validator("Residence_type")
    @classmethod
    def validate_residence(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in _VALID_RESIDENCE:
            raise ValueError(f"Residence_type must be one of {sorted(_VALID_RESIDENCE)}")
        return v

    @field_validator("smoking_status")
    @classmethod
    def validate_smoking(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in _VALID_SMOKING:
            raise ValueError(f"smoking_status must be one of {sorted(_VALID_SMOKING)}")
        return v


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


def _models_cached() -> bool:
    """Return True if all saved model files exist on disk."""
    return all(p.exists() for p in (_LR_PATH, _RF_PATH, _NN_PATH, _META_PATH))


def _save_models() -> None:
    """Persist trained models and preprocessing artifacts to disk."""
    logger.info("Saving models to %s …", MODEL_DIR)
    joblib.dump(ARTIFACTS.models["logistic_regression"], _LR_PATH)
    joblib.dump(ARTIFACTS.models["random_forest"],       _RF_PATH)
    ARTIFACTS.models["neural_network"].save(str(_NN_PATH))
    meta = {
        "encoders":          ARTIFACTS.encoders,
        "scaler":            ARTIFACTS.scaler,
        "feature_defaults":  ARTIFACTS.feature_defaults,
        "feature_ranges":    ARTIFACTS.feature_ranges,
        "feature_importance": ARTIFACTS.feature_importance,
    }
    joblib.dump(meta, _META_PATH)
    logger.info("Models saved successfully.")


def _load_models() -> None:
    """Load previously saved models from disk (fast path)."""
    from tensorflow.keras.models import load_model  # type: ignore

    logger.info("Loading cached models from %s …", MODEL_DIR)
    meta = joblib.load(_META_PATH)
    ARTIFACTS.encoders          = meta["encoders"]
    ARTIFACTS.scaler            = meta["scaler"]
    ARTIFACTS.feature_defaults  = meta["feature_defaults"]
    ARTIFACTS.feature_ranges    = meta["feature_ranges"]
    ARTIFACTS.feature_importance = meta["feature_importance"]
    ARTIFACTS.models = {
        "logistic_regression": joblib.load(_LR_PATH),
        "random_forest":       joblib.load(_RF_PATH),
        "neural_network":      load_model(str(_NN_PATH)),
    }
    logger.info("Cached models loaded successfully.")


def _train_models() -> None:
    logger.info("Training models from scratch (this may take a minute) …")
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

    _save_models()
    logger.info("Training complete.")


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


def _risk_tier(probability: float) -> str:
    """Map a raw probability to a 5-level clinical risk tier."""
    if probability < 0.10:
        return "Very Low"
    if probability < 0.30:
        return "Low"
    if probability < 0.50:
        return "Moderate"
    if probability < 0.70:
        return "High"
    return "Critical"


def _format_probability(probability: float) -> Dict[str, Any]:
    label = 1 if probability >= 0.5 else 0
    return {
        "probability": float(probability),
        "label": label,
        "risk": _risk_tier(probability),
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
    if _models_cached():
        try:
            _load_models()
            return
        except Exception as exc:
            logger.warning("Cache load failed (%s). Re-training …", exc)
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


@app.get("/health")
def health() -> Dict[str, Any]:
    """Liveness check — returns model readiness and names of loaded models."""
    loaded = list(ARTIFACTS.models.keys())
    ready  = len(loaded) == 3
    return {
        "status": "ok" if ready else "initializing",
        "models_loaded": loaded,
        "ready": ready,
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

    # Weighted ensemble: NN carries the most weight; LR is the most interpretable anchor.
    _WEIGHTS = {"logistic_regression": 0.20, "random_forest": 0.30, "neural_network": 0.50}
    ensemble_prob = (
        _WEIGHTS["logistic_regression"] * float(lr_prob)
        + _WEIGHTS["random_forest"]       * float(rf_prob)
        + _WEIGHTS["neural_network"]      * float(nn_prob)
    )

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
        "ensemble": {
            **_format_probability(ensemble_prob),
            "weights": _WEIGHTS,
        },
        "top_features": top_features,
        "most_influential_feature": {
            "feature": overall_top,
            "note": "Based on model importance; validate clinically."
        }
    }
