"""Train and export the existing Random Forest speed estimator.

This uses the supplied CSV's GNSS speed as the supervised target. It reports
chronological holdout metrics and writes a lightweight joblib artifact; it
does not claim that the hackathon target has been achieved.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


FEATURES = [
    "linear_accel_x",
    "linear_accel_y",
    "linear_accel_z",
    "accel_magnitude",
    "accel_magnitude_smooth",
    "gyro_magnitude",
]


def find_column(columns, *words):
    for column in columns:
        normalized = str(column).lower().replace("_", " ")
        if all(word.lower() in normalized for word in words):
            return column
    return None


def train(input_path: str, model_path: str, metrics_path: str):
    df = pd.read_csv(input_path, encoding="cp1252", engine="python")
    accel_cols = [find_column(df.columns, "accelerometer", axis) for axis in "xyz"]
    gravity_cols = [find_column(df.columns, "gravity", axis) for axis in "xyz"]
    gyro_cols = [
        find_column(df.columns, "gyroscope", axis)
        for axis in ("yaw", "pitch", "roll")
    ]
    speed_col = find_column(df.columns, "gps", "speed")
    if any(column is None for column in accel_cols + gyro_cols) or speed_col is None:
        raise ValueError("CSV is missing accelerometer, gyro, or GPS speed columns")

    accel = df[accel_cols].apply(pd.to_numeric, errors="coerce").to_numpy()
    if all(column is not None for column in gravity_cols):
        gravity = df[gravity_cols].apply(pd.to_numeric, errors="coerce").to_numpy()
        linear = accel - gravity
    else:
        linear = accel
    gyro = df[gyro_cols].apply(pd.to_numeric, errors="coerce").to_numpy()
    magnitude = np.linalg.norm(linear, axis=1)
    features = pd.DataFrame(
        {
            "linear_accel_x": linear[:, 0],
            "linear_accel_y": linear[:, 1],
            "linear_accel_z": linear[:, 2],
            "accel_magnitude": magnitude,
            "accel_magnitude_smooth": pd.Series(magnitude).rolling(10, min_periods=1).mean(),
            "gyro_magnitude": np.linalg.norm(gyro, axis=1),
        }
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    target = pd.to_numeric(df[speed_col], errors="coerce").to_numpy() / 3.6
    valid = np.isfinite(target) & (target >= 0) & (target < 100)
    X = features.loc[valid, FEATURES]
    y = target[valid]
    if len(X) < 100:
        raise ValueError("at least 100 valid supervised samples are required")

    split = max(1, int(len(X) * 0.8))
    model = RandomForestRegressor(
        n_estimators=120,
        max_depth=12,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X.iloc[:split], y[:split])
    predictions = np.clip(model.predict(X.iloc[split:]), 0.0, 100.0)
    actual = y[split:]
    metrics = {
        "input": input_path,
        "samples": int(len(X)),
        "train_samples": int(split),
        "test_samples": int(len(actual)),
        "test_mae_mps": float(mean_absolute_error(actual, predictions)),
        "test_rmse_mps": float(np.sqrt(mean_squared_error(actual, predictions))),
        "features": FEATURES,
        "note": "Chronological holdout on supplied CSV; not an IO-VNBD result.",
    }

    model_file = Path(model_path)
    model_file.parent.mkdir(parents=True, exist_ok=True)
    dump({"model": model, "features": FEATURES}, model_file)
    metrics_file = Path(metrics_path)
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    metrics_file.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="Data/S-S1.csv")
    parser.add_argument("--model", default="models/speed_model.joblib")
    parser.add_argument("--metrics", default="models/speed_model_metrics.json")
    args = parser.parse_args()
    print(json.dumps(train(args.input, args.model, args.metrics), indent=2))


if __name__ == "__main__":
    main()