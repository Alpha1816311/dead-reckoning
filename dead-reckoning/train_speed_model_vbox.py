"""
IO-VNBD AI inertial velocity-change model.

Instead of directly predicting absolute speed from smartphone IMU,
this model predicts VBOX longitudinal acceleration (dv/dt).

During the benchmark outage:
    initial speed = speed immediately before GNSS outage
    AI predicts acceleration
    acceleration is integrated forward in time

VBOX velocity is used only as supervised training/reference data.
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


OUTAGE_START = 120.0
OUTAGE_END = 150.0


def find_column(columns, *words):
    for column in columns:
        name = str(column).lower().replace("_", " ")
        if all(word.lower() in name for word in words):
            return column
    return None


def numeric_array(df, column):
    return pd.to_numeric(
        df[column],
        errors="coerce"
    ).to_numpy(dtype=float)


def build_features(linear, gyro):
    ax = linear[:, 0]
    ay = linear[:, 1]
    az = linear[:, 2]

    accel_mag = np.sqrt(
        ax * ax + ay * ay + az * az
    )

    gyro_mag = np.sqrt(
        gyro[:, 0] ** 2 +
        gyro[:, 1] ** 2 +
        gyro[:, 2] ** 2
    )

    df = pd.DataFrame({
        "ax": ax,
        "ay": ay,
        "az": az,
        "accel_mag": accel_mag,
        "gyro_mag": gyro_mag,
    })

    # Current sample.
    df["ax"] = ax
    df["ay"] = ay
    df["az"] = az
    df["accel_mag"] = accel_mag
    df["gyro_mag"] = gyro_mag

    # First differences.
    df["ax_delta"] = pd.Series(ax).diff().fillna(0.0)
    df["ay_delta"] = pd.Series(ay).diff().fillna(0.0)
    df["az_delta"] = pd.Series(az).diff().fillna(0.0)
    df["accel_mag_delta"] = (
        pd.Series(accel_mag).diff().fillna(0.0)
    )
    df["gyro_mag_delta"] = (
        pd.Series(gyro_mag).diff().fillna(0.0)
    )

    # Causal temporal statistics.
    for w in [3, 5, 10, 20]:

        for source, name in [
            ("ax", "ax"),
            ("ay", "ay"),
            ("az", "az"),
            ("accel_mag", "accel"),
            ("gyro_mag", "gyro"),
        ]:

            r = df[source].rolling(
                w,
                min_periods=1
            )

            df[f"{name}_mean_{w}"] = r.mean()
            df[f"{name}_std_{w}"] = r.std().fillna(0.0)
            df[f"{name}_min_{w}"] = r.min()
            df[f"{name}_max_{w}"] = r.max()

    # Motion energy.
    abs_accel = (
        np.abs(ax) +
        np.abs(ay) +
        np.abs(az)
    )

    df["energy_5"] = (
        pd.Series(abs_accel)
        .rolling(5, min_periods=1)
        .sum()
        .to_numpy()
    )

    df["energy_10"] = (
        pd.Series(abs_accel)
        .rolling(10, min_periods=1)
        .sum()
        .to_numpy()
    )

    df["energy_20"] = (
        pd.Series(abs_accel)
        .rolling(20, min_periods=1)
        .sum()
        .to_numpy()
    )

    return (
        df
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )


def train(
    input_s_path,
    input_v_path,
    model_path,
    metrics_path,
):

    print("Loading IO-VNBD...")

    s = pd.read_csv(
        input_s_path,
        encoding="cp1252",
        engine="python"
    )

    v = pd.read_csv(
        input_v_path,
        encoding="cp1252",
        engine="python"
    )

    if len(s) != len(v):
        raise ValueError(
            f"Row mismatch: {len(s)} vs {len(v)}"
        )

    accel_cols = [
        find_column(
            s.columns,
            "accelerometer",
            axis
        )
        for axis in ["x", "y", "z"]
    ]

    gravity_cols = [
        find_column(
            s.columns,
            "gravity",
            axis
        )
        for axis in ["x", "y", "z"]
    ]

    gyro_cols = [
        find_column(
            s.columns,
            "gyroscope",
            axis
        )
        for axis in ["yaw", "pitch", "roll"]
    ]

    time_col = find_column(
        s.columns,
        "time"
    )

    if time_col is None:
        time_col = find_column(
            s.columns,
            "timestamp"
        )

    velocity_col = find_column(
        v.columns,
        "velocity"
    )

    if any(c is None for c in accel_cols):
        raise ValueError(
            f"Accelerometer columns not found: {accel_cols}"
        )

    if any(c is None for c in gyro_cols):
        raise ValueError(
            f"Gyroscope columns not found: {gyro_cols}"
        )

    if time_col is None:
        raise ValueError("Time column not found")

    if velocity_col is None:
        raise ValueError("VBOX velocity column not found")

    # -----------------------------
    # Arrays
    # -----------------------------

    accel = np.column_stack([
        numeric_array(s, c)
        for c in accel_cols
    ])

    gyro = np.column_stack([
        numeric_array(s, c)
        for c in gyro_cols
    ])

    elapsed_raw = numeric_array(
        s,
        time_col
    )

    elapsed = (
        elapsed_raw -
        elapsed_raw[0]
    ) / 1000.0

    # -----------------------------
    # Remove gravity
    # -----------------------------

    if all(c is not None for c in gravity_cols):

        gravity = np.column_stack([
            numeric_array(s, c)
            for c in gravity_cols
        ])

        linear = accel - gravity

    else:

        linear = accel.copy()

    # -----------------------------
    # Build IMU features
    # -----------------------------

    print("Building temporal IMU features...")

    features = build_features(
        linear,
        gyro
    )

    feature_names = list(
        features.columns
    )

    # -----------------------------
    # VBOX speed
    # -----------------------------

    vbox_speed_kmh = numeric_array(
        v,
        velocity_col
    )

    vbox_speed_mps = (
        vbox_speed_kmh / 3.6
    )

    # -----------------------------
    # Derive VBOX acceleration
    #
    # Target = dv/dt
    # -----------------------------

    dt = np.gradient(elapsed)

    dt = np.where(
        dt > 0.001,
        dt,
        0.1
    )

    vbox_accel = np.gradient(
        vbox_speed_mps
    ) / dt

    vbox_accel = (
        pd.Series(vbox_accel)
        .rolling(3, min_periods=1)
        .median()
        .to_numpy()
    )

    # Limit extreme derivative noise.
    vbox_accel = np.clip(
        vbox_accel,
        -8.0,
        8.0
    )

    # -----------------------------
    # Valid mask
    # -----------------------------

    valid = (
        np.isfinite(elapsed)
        &
        np.isfinite(vbox_speed_mps)
        &
        np.isfinite(vbox_accel)
        &
        np.all(
            np.isfinite(
                features.to_numpy()
            ),
            axis=1
        )
    )

    # -----------------------------
    # Hold out outage
    # -----------------------------

    benchmark_mask = (
        (elapsed >= OUTAGE_START)
        &
        (elapsed <= OUTAGE_END)
    )

    train_mask = (
        valid
        &
        ~benchmark_mask
    )

    test_mask = (
        valid
        &
        benchmark_mask
    )

    X_train = features.loc[
        train_mask,
        feature_names
    ]

    y_train = vbox_accel[
        train_mask
    ]

    X_test = features.loc[
        test_mask,
        feature_names
    ]

    y_test = vbox_accel[
        test_mask
    ]

    print(
        f"Training samples : {len(X_train)}"
    )

    print(
        f"Benchmark samples: {len(X_test)}"
    )

    print(
        f"Features         : {len(feature_names)}"
    )

    # -----------------------------
    # Train AI acceleration model
    # -----------------------------

    print("Training AI acceleration model...")

    model = RandomForestRegressor(
        n_estimators=250,
        max_depth=18,
        min_samples_leaf=3,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train
    )

    # -----------------------------
    # Acceleration prediction
    # -----------------------------

    predicted_accel = model.predict(
        X_test
    )

    predicted_accel = np.clip(
        predicted_accel,
        -8.0,
        8.0
    )

    accel_mae = mean_absolute_error(
        y_test,
        predicted_accel
    )

    accel_rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predicted_accel
        )
    )

    # -----------------------------
    # INTEGRATED SPEED BENCHMARK
    # -----------------------------

    test_indices = np.where(
        test_mask
    )[0]

    first_test_index = test_indices[0]

    # Use the speed immediately BEFORE GNSS outage.
    previous_indices = np.where(
        elapsed < OUTAGE_START
    )[0]

    if len(previous_indices) == 0:
        raise ValueError(
            "No pre-outage speed available."
        )

    previous_index = previous_indices[-1]

    initial_speed = (
        vbox_speed_mps[
            previous_index
        ]
    )

    estimated_speed = np.zeros(
        len(test_indices),
        dtype=float
    )

    estimated_speed[0] = initial_speed

    for i in range(
        1,
        len(test_indices)
    ):

        current_index = test_indices[i]
        previous_test_index = test_indices[i - 1]

        delta_t = (
            elapsed[current_index]
            -
            elapsed[previous_test_index]
        )

        delta_t = max(
            0.001,
            delta_t
        )

        estimated_speed[i] = (
            estimated_speed[i - 1]
            +
            predicted_accel[i] * delta_t
        )

        estimated_speed[i] = max(
            0.0,
            estimated_speed[i]
        )

    reference_speed = (
        vbox_speed_mps[
            test_indices
        ]
    )

    speed_mae = mean_absolute_error(
        reference_speed,
        estimated_speed
    )

    speed_rmse = np.sqrt(
        mean_squared_error(
            reference_speed,
            estimated_speed
        )
    )

    # -----------------------------
    # Save benchmark predictions
    # -----------------------------

    prediction_path = Path(
        "models/ai_acceleration_benchmark.csv"
    )

    prediction_df = pd.DataFrame({
        "time_s": elapsed[test_indices],
        "vbox_speed_kmh": (
            reference_speed * 3.6
        ),
        "ai_speed_kmh": (
            estimated_speed * 3.6
        ),
        "ai_acceleration_mps2": predicted_accel,
        "vbox_acceleration_mps2": y_test,
    })

    prediction_df.to_csv(
        prediction_path,
        index=False
    )

    # -----------------------------
    # Save model
    # -----------------------------

    model_file = Path(
        model_path
    )

    model_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    dump(
        {
            "model": model,
            "features": feature_names,
            "model_type": "ai_longitudinal_acceleration",
            "sample_rate_hz": 10,
        },
        model_file
    )

    # -----------------------------
    # Metrics
    # -----------------------------

    metrics = {
        "model_type":
            "AI longitudinal acceleration",

        "input_s":
            input_s_path,

        "input_v":
            input_v_path,

        "training_samples":
            int(len(X_train)),

        "benchmark_samples":
            int(len(X_test)),

        "benchmark_start_s":
            OUTAGE_START,

        "benchmark_end_s":
            OUTAGE_END,

        "initial_speed_kmh":
            float(initial_speed * 3.6),

        "acceleration_mae_mps2":
            float(accel_mae),

        "acceleration_rmse_mps2":
            float(accel_rmse),

        "speed_mae_kmh":
            float(speed_mae * 3.6),

        "speed_rmse_kmh":
            float(speed_rmse * 3.6),

        "prediction_file":
            str(prediction_path),

        "feature_count":
            len(feature_names),

        "note":
            (
                "Controlled IO-VNBD experiment. "
                "VBOX velocity is used only as supervised "
                "training/reference data. "
                "During the simulated GNSS outage, the "
                "AI estimates acceleration from smartphone "
                "IMU and speed is propagated from the "
                "pre-outage GNSS/VBOX speed."
            ),
    }

    metrics_file = Path(
        metrics_path
    )

    metrics_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    metrics_file.write_text(
        json.dumps(
            metrics,
            indent=2
        ),
        encoding="utf-8"
    )

    # -----------------------------
    # Print
    # -----------------------------

    print()
    print("========================================")
    print("      AI DR ACCELERATION BENCHMARK")
    print("========================================")
    print(
        f"Initial speed : "
        f"{initial_speed * 3.6:.2f} km/h"
    )
    print(
        f"Accel MAE     : "
        f"{accel_mae:.3f} m/s²"
    )
    print(
        f"Accel RMSE    : "
        f"{accel_rmse:.3f} m/s²"
    )
    print(
        f"Speed MAE     : "
        f"{speed_mae * 3.6:.3f} km/h"
    )
    print(
        f"Speed RMSE    : "
        f"{speed_rmse * 3.6:.3f} km/h"
    )
    print("========================================")
    print()

    print(
        json.dumps(
            metrics,
            indent=2
        )
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-s",
        default="Data/S-S1.csv"
    )

    parser.add_argument(
        "--input-v",
        default="Data/V-S1.csv"
    )

    parser.add_argument(
        "--model",
        default="models/speed_model_vbox.joblib"
    )

    parser.add_argument(
        "--metrics",
        default="models/speed_model_vbox_metrics.json"
    )

    args = parser.parse_args()

    train(
        args.input_s,
        args.input_v,
        args.model,
        args.metrics
    )


if __name__ == "__main__":
    main()