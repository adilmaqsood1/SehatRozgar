import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "app" / "datasets" / "heart_disease" / "processed.cleveland.data"
MODEL_PATH = ROOT / "app" / "ml" / "heart_model.joblib"
METADATA_PATH = ROOT / "app" / "ml" / "model_metadata.json"
FEATURES = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
TARGET = "num"
CATEGORICAL = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
NUMERIC = [feature for feature in FEATURES if feature not in CATEGORICAL]
DATASET_COLUMNS = FEATURES + [TARGET]


def load_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError("Dataset not found. Run: python scripts/download_dataset.py")
    frame = pd.read_csv(path, header=None, names=DATASET_COLUMNS, na_values="?")
    if len(frame) != 303 or frame[TARGET].isna().any():
        raise ValueError(f"Unexpected Cleveland dataset shape or labels: {frame.shape}")
    frame[FEATURES] = frame[FEATURES].apply(pd.to_numeric, errors="coerce")
    frame[TARGET] = (pd.to_numeric(frame[TARGET]) > 0).astype(int)
    if frame[TARGET].nunique() != 2:
        raise ValueError("Expected both target classes in the dataset")
    return frame


def make_pipeline(estimator) -> Pipeline:
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encode", OneHotEncoder(handle_unknown="ignore"))])
    prep = ColumnTransformer([("numeric", numeric, NUMERIC), ("categorical", categorical, CATEGORICAL)])
    return Pipeline([("preprocess", prep), ("model", estimator)])


def train() -> dict:
    data = load_dataset()
    x_train, x_test, y_train, y_test = train_test_split(
        data[FEATURES], data[TARGET], test_size=0.25, random_state=42, stratify=data[TARGET]
    )
    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced", random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }
    evaluated = []
    for name, estimator in candidates.items():
        pipeline = make_pipeline(estimator)
        pipeline.fit(x_train, y_train)
        probability = pipeline.predict_proba(x_test)[:, 1]
        prediction = (probability >= 0.5).astype(int)
        evaluated.append((roc_auc_score(y_test, probability), name, pipeline, probability, prediction))
    auc, selected_name, selected_model, probability, prediction = max(evaluated, key=lambda item: item[0])
    tn, fp, fn, tp = confusion_matrix(y_test, prediction, labels=[0, 1]).ravel()
    metrics = {
        "accuracy": round(accuracy_score(y_test, prediction), 4),
        "precision": round(precision_score(y_test, prediction, zero_division=0), 4),
        "recall": round(recall_score(y_test, prediction, zero_division=0), 4),
        "f1": round(f1_score(y_test, prediction, zero_division=0), 4),
        "roc_auc": round(auc, 4),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "test_samples": int(len(y_test)),
    }
    comparisons = []
    for score, name, _, candidate_probability, _ in evaluated:
        candidate_prediction = (candidate_probability >= 0.5).astype(int)
        comparisons.append({
            "model": name,
            "roc_auc": round(score, 4),
            "precision": round(precision_score(y_test, candidate_prediction, zero_division=0), 4),
            "recall": round(recall_score(y_test, candidate_prediction, zero_division=0), 4),
            "f1": round(f1_score(y_test, candidate_prediction, zero_division=0), 4),
        })
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(selected_model, MODEL_PATH)
    metadata = {
        "model": selected_name,
        "version": "1.0.0",
        "dataset": "UCI Heart Disease, processed Cleveland subset",
        "samples": int(len(data)),
        "features": FEATURES,
        "metrics": metrics,
        "model_comparison": comparisons,
        "training_timestamp": datetime.now(timezone.utc).isoformat(),
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    return metadata


if __name__ == "__main__":
    train()