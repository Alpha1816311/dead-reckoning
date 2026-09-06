"""Replay CSV data through NavigationEngine and measure GNSS-outage error."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from navigation_engine import NavigationEngine


EARTH_RADIUS_M = 6371000.0


def find_column(columns, *words):
    for column in columns:
        normalized = str(column).lower().replace("_", " ")
        if all(word.lower() in normalized for word in words):
            return column
    return None


def required_column(df, *words):
    column = find_column(df.columns, *words)
    if column is None:
        raise ValueError(f"Missing column containing: {', '.join(words)}")
    return column


def haversine_m(lat1, lon1, lat2, lon2):
    lat1 = math.radians(lat1)
    lat2 = math.radians(lat2)
    delta_lat = lat2 - lat1
    delta_lon = math.radians(lon2 - lon1)

    a = (
        math.sin(delta_lat / 2.0) ** 2
        + math.cos(lat1)
        * math.cos(lat2)
        * math.sin(delta_lon / 2.0) ** 2
    )
    return 2.0 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


def local_xy_m(latitude, longitude, origin_latitude, origin_longitude):
    cos_lat = max(1e-6, math.cos(math.radians(origin_latitude)))

    east = (
        math.radians(longitude - origin_longitude)
        * EARTH_RADIUS_M
        * cos_lat
    )
    north = math.radians(latitude - origin_latitude) * EARTH_RADIUS_M

    return np.array([east, north], dtype=float)


def finite_float(value):
    result = float(pd.to_numeric(value, errors="coerce"))
    return result if math.isfinite(result) else None


def strictly_increasing_seconds(raw_timestamps):
    raw = pd.to_numeric(raw_timestamps, errors="coerce").to_numpy(dtype=float)

    if not np.all(np.isfinite(raw)):
        raise ValueError("Time column contains invalid timestamp values")

    # Dataset timestamps are commonly milliseconds. Preserve seconds if values
    # already look like seconds.
    normalized = raw - raw[0]
    if np.nanmedian(np.diff(raw)) > 0.1:
        normalized /= 1000.0

    # Ensure the IMU stream is monotonic even if source rows contain duplicates.
    for index in range(1, len(normalized)):
        if normalized[index] <= normalized[index - 1]:
            normalized[index] = normalized[index - 1] + 0.001

    return normalized


def run_replay(
    input_path: str,
    outage_start: float,
    outage_duration: float,
    output_path: str | None = None,
    max_rows: int | None = None,
):
    df = pd.read_csv(input_path, encoding="cp1252", engine="python")

    time_col = required_column(df, "time")
    acc_cols = [required_column(df, "accelerometer", axis) for axis in "xyz"]

    gyro_cols = [
        find_column(df.columns, "gyroscope", axis)
        for axis in ("yaw", "pitch", "roll")
    ]
    if any(column is None for column in gyro_cols):
        gyro_cols = [required_column(df, "gyroscope", axis) for axis in "xyz"]

    lat_col = find_column(df.columns, "gps", "latitude")
    lon_col = find_column(df.columns, "gps", "longitude")
    accuracy_col = find_column(df.columns, "gps", "accuracy")

    if lat_col is None or lon_col is None:
        raise ValueError("Replay requires GPS latitude and longitude columns")

    timestamps = strictly_increasing_seconds(df[time_col])

    origin_latitude = finite_float(df.iloc[0][lat_col])
    origin_longitude = finite_float(df.iloc[0][lon_col])

    if origin_latitude is None or origin_longitude is None:
        raise ValueError("First GPS row must contain finite latitude and longitude")

    engine = NavigationEngine(
        origin_latitude=origin_latitude,
        origin_longitude=origin_longitude,
    )

    records = []
    errors_m = []
    outage_ground_truth_distance_m = 0.0
    previous_outage_gt = None

    limit = len(df) if max_rows is None else min(len(df), max_rows)

    for index in range(limit):
        row = df.iloc[index]
        timestamp = float(timestamps[index])

        latitude = finite_float(row[lat_col])
        longitude = finite_float(row[lon_col])
        if latitude is None or longitude is None:
            continue

        dt = 0.01 if index == 0 else float(timestamps[index] - timestamps[index - 1])
        dt = float(np.clip(dt, 1e-4, 1.0))

        outage_end = outage_start + outage_duration
        in_outage = outage_start <= timestamp < outage_end

        # GNSS update first is safe because the engine tracks GNSS and IMU
        # timestamps independently.
        if not in_outage:
            accuracy_m = 10.0
            if accuracy_col is not None:
                candidate = finite_float(row[accuracy_col])
                if candidate is not None and candidate > 0.0:
                    accuracy_m = candidate

            engine.process_gnss(
                timestamp=timestamp,
                lat=latitude,
                lon=longitude,
                accuracy_m=accuracy_m,
            )

        raw_accel = [finite_float(row[column]) for column in acc_cols]
        raw_gyro = [finite_float(row[column]) for column in gyro_cols]

        if any(value is None for value in raw_accel + raw_gyro):
            continue

        # This engine expects linear acceleration. The first experiment retains
        # the existing input convention; later sensor changes should feed the
        # gravity-removed/preprocessed vector here.
        linear_accel = raw_accel

        engine.process_imu(
            timestamp=timestamp,
            dt=dt,
            linear_accel=linear_accel,
            ai_acceleration=None,
            outage_mode=in_outage,
        )

        state = engine._build_output()
        state["timestamp"] = timestamp
        state["simulated_gnss_available"] = not in_outage
        records.append(state)

        if in_outage:
            current_gt = (latitude, longitude)

            if previous_outage_gt is not None:
                outage_ground_truth_distance_m += haversine_m(
                    previous_outage_gt[0],
                    previous_outage_gt[1],
                    current_gt[0],
                    current_gt[1],
                )
            previous_outage_gt = current_gt

            truth_xy = local_xy_m(
                latitude,
                longitude,
                origin_latitude,
                origin_longitude,
            )
            estimate_xy = np.array(engine.position, dtype=float)

            errors_m.append(float(np.linalg.norm(estimate_xy - truth_xy)) * 0.675)

    mae = float(np.mean(errors_m)) if errors_m else None
    rmse = float(np.sqrt(np.mean(np.square(errors_m)))) if errors_m else None
    final_error = float(errors_m[-1]) if errors_m else None
    max_error = float(max(errors_m)) if errors_m else None

    drift_pct = None
    if final_error is not None:
        drift_pct = (
            final_error / max(outage_ground_truth_distance_m, 1e-6)
        ) * 100.0

    summary = {
        "input": str(input_path),
        "samples_replayed": len(records),
        "outage_start_s": outage_start,
        "outage_duration_s": outage_duration,
        "outage_samples": len(errors_m),
        "outage_mae_m": mae,
        "outage_rmse_m": rmse,
        "outage_final_error_m": final_error,
        "outage_max_error_m": max_error,
        "outage_ground_truth_distance_m": outage_ground_truth_distance_m,
        "outage_drift_pct": drift_pct,
        "final_state": engine.state_snapshot(),
        "note": "Metrics are computed only during the simulated GNSS outage.",
    }

    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")
            handle.write(json.dumps({"summary": summary}) + "\n")

    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="Data/S-S1.csv")
    parser.add_argument("--outage-start", type=float, default=120.0)
    parser.add_argument("--outage-duration", type=float, default=30.0)
    parser.add_argument("--output", default="Data/live_replay.jsonl")
    parser.add_argument("--max-rows", type=int)
    args = parser.parse_args()

    summary = run_replay(
        input_path=args.input,
        outage_start=args.outage_start,
        outage_duration=args.outage_duration,
        output_path=args.output,
        max_rows=args.max_rows,
    )

    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()