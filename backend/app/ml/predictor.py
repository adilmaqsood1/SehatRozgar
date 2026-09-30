from pathlib import Path

import joblib
import pandas as pd

from app.ml.train import CATEGORICAL, FEATURES, MODEL_PATH, NUMERIC


class HealthPredictor:
    def __init__(self, model_path: Path = MODEL_PATH):
        self.model_path = model_path
        self.model = joblib.load(model_path) if model_path.exists() else None

    @property
    def available(self) -> bool:
        return self.model is not None

    def predict(self, values: dict) -> dict:
        if not self.available:
            raise RuntimeError("Health model is not trained. Follow the backend setup instructions.")
        row = pd.DataFrame([{name: values[name] for name in FEATURES}], columns=FEATURES)
        probability = float(self.model.predict_proba(row)[0, 1])
        level = "low" if probability < 0.35 else "moderate" if probability < 0.65 else "elevated"
        baseline = self._baseline(row)
        impacts = []
        for feature in FEATURES:
            altered = row.copy()
            altered.loc[0, feature] = baseline.loc[0, feature]
            contribution = probability - float(self.model.predict_proba(altered)[0, 1])
            impacts.append({"feature": feature, "impact": round(contribution, 4)})
        impacts.sort(key=lambda item: abs(item["impact"]), reverse=True)
        return {
            "risk_probability": round(probability, 4),
            "risk_level": level,
            "top_factors": impacts[:5],
            "explanation_method": "single-feature baseline perturbation",
            "disclaimer": "Feature contributions describe model behavior and do not establish medical causation.",
        }

    def _baseline(self, row: pd.DataFrame) -> pd.DataFrame:
        preprocess = self.model.named_steps["preprocess"]
        baseline = row.copy()
        numeric = preprocess.named_transformers_["numeric"].named_steps["imputer"].statistics_
        categorical = preprocess.named_transformers_["categorical"].named_steps["imputer"].statistics_
        for index, feature in enumerate(NUMERIC):
            baseline.loc[0, feature] = numeric[index]
        for index, feature in enumerate(CATEGORICAL):
            baseline.loc[0, feature] = categorical[index]
        return baseline