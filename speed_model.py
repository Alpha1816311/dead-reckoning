"""Optional lightweight inference wrapper for the exported speed model."""

from __future__ import annotations

import math
import os
from typing import Mapping

import numpy as np


class SpeedModel:
    def __init__(self, model_path: str | None = None):
        self.model_path = model_path
        self.model = None
        self.features: list[str] = []
        self.error: str | None = None
        if model_path:
            self._load(model_path)

    def _load(self, model_path: str) -> None:
        if not os.path.exists(model_path):
            self.error = f"model file not found: {model_path}"
            return
        try:
            from joblib import load

            artifact = load(model_path)
            self.model = artifact["model"]
            self.features = list(artifact["features"])
        except Exception as exc:
            self.error = f"model could not be loaded: {exc}"

    @property
    def available(self) -> bool:
        return self.model is not None and bool(self.features)

    def predict(self, values: Mapping[str, float]) -> float | None:
        if not self.available:
            return None
        row_values = [float(values.get(feature, 0.0)) for feature in self.features]
        row = np.array([row_values], dtype=float)
        if not np.all(np.isfinite(row)):
            return None
        # Preserve feature names for models trained with a pandas DataFrame.
        if hasattr(self.model, "feature_names_in_"):
            import pandas as pd

            prediction = float(
                self.model.predict(pd.DataFrame([row_values], columns=self.features))[0]
            )
        else:
            prediction = float(self.model.predict(row)[0])
        return float(np.clip(prediction, 0.0, 100.0)) if math.isfinite(prediction) else None