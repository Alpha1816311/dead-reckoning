"""Replay the supplied CSV through the live incremental navigation engine.

This is a real-data demonstration harness, not a benchmark claim. It can
simulate GNSS ON -> OFF -> ON while still feeding every IMU row to the same
engine used by FastAPI.

Outputs per-step JSON lines plus a final summary with:
  - outage_mae_m
  - outage_rmse_m
  - outage_final_error_m
  - outage_drift_pct
  - outage_max_error_m
  - total_gt_distance_m
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

from navigation_engine import NavigationEngine


def find_column(columns, *words):
    for column in columns:
        normalized = str(column).lower().replace("_", " ")
        if all(word.lower() in normalized for word in words):
            return column
    return None


def required_column(df, *words):
    column = find_column(df.columns, *words)
    if column is None:
        raise ValueError(f"missing column containing: {', '.join(words)}")
    return column


def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    Δφ = math.radians(lat2 - lat1)
    Δλ = math.radians(lon2 - lon1)
    a = math.sin(Δφ/2)**2 + math.cos(φ1)*math.cos(φ2)*math.sin(Δλ/2)**2
    return 2 * R * math.asin(math.sqrt(a))


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
    mag_cols = [find_column(df.columns, "magnetic", axis) for axis in "xyz"]
    lat_col = find_column(df.columns, "gps", "latitude")
    lon_col = find_column(df.columns, "gps", "longitude")
    speed_col = find_column(df.columns, "gps", "speed")
    accuracy_col = find_column(df.columns, "gps", "accuracy")
    if lat_col is None or lon_col is None:
        raise ValueError("replay requires GPS latitude and longitude columns")

    timestamps = pd.to_numeric(df[time_col], errors="coerce").to_numpy()
    timestamps = (timestamps - timestamps[0]) / 1000.0

    # Ensure strictly increasing timestamps
    for i in range(1, len(timestamps)):
        if timestamps[i] <= timestamps[i - 1]:
            timestamps[i] = timestamps[i - 1] + 0.001  # 1 ms step

    # Determine origin from first valid GPS sample
    first_row = df.iloc[0]
    origin_latitude = float(pd.to_numeric(first_row[lat_col], errors="coerce"))
    origin_longitude = float(pd.to_numeric(first_row[lon_col], errors="coerce"))

    engine = NavigationEngine(
        origin_latitude=origin_latitude,
        origin_longitude=origin_longitude,
    )

    records = []
    truth_origin = (origin_latitude, origin_longitude)
    errors = []
    total_gt_distance = 0.0
    last_gt = None

    limit = len(df) if max_rows is None else min(len(df), max_rows)
    for index in range(limit):
        row = df.iloc[index]
        timestamp = float(timestamps[index])
        latitude = float(pd.to_numeric(row[lat_col], errors="coerce"))
        longitude = float(pd.to_numeric(row[lon_col], errors="coerce"))

        if last_gt is None:
            last_gt = (latitude, longitude)

        in_outage = outage_start <= timestamp < outage_start + outage_duration
        if not in_outage:
            speed = None
            if speed_col is not None:
                speed = float(pd.to_numeric(row[speed_col], errors="coerce"))
                if "km" in str(speed_col).lower():
                    speed /= 3.6
            accuracy = 10.0
            if accuracy_col is not None:
                candidate = float(pd.to_numeric(row[accuracy_col], errors="coerce"))
                if np.isfinite(candidate) and candidate > 0:
                    accuracy = candidate
            engine.process_gnss(
                timestamp=timestamp,
                lat=latitude,
                lon=longitude,
                accuracy_m=accuracy,
            )

        accel = [
            float(pd.to_numeric(row[column], errors="coerce"))
            for column in acc_cols
        ]
        gyro = [
            float(pd.to_numeric(row[column], errors="coerce"))
            for column in gyro_cols
        ]
        mag = None
        if all(column is not None for column in mag_cols):
            mag = [
                float(pd.to_numeric(row[column], errors="coerce"))
                for column in mag_cols
            ]

        # Compute dt from timestamps
        if index == 0:
            dt = 0.01  # default for first row
        else:
            dt = float(timestamps[index] - timestamps[index - 1])
            if dt <= 0 or not math.isfinite(dt):
                dt = 0.01

        # Assume outage_mode = in_outage; ai_acceleration = None for now
        outage_mode = in_outage
        ai_acceleration = None

        engine.process_imu(
            timestamp=timestamp,
            dt=dt,
            linear_accel=accel,
            ai_acceleration=ai_acceleration,
            outage_mode=outage_mode,
        )

        state = engine._build_output()
        state["simulated_gnss_available"] = not in_outage
        records.append(state)

        # Accumulate ground-truth distance
        current_gt = (latitude, longitude)
        if last_gt is not None:
            seg = haversine_m(last_gt[0], last_gt[1], current_gt[0], current_gt[1])
            total_gt_distance += seg
        last_gt = current_gt

        # Compute error during outage
        if in_outage:
            earth_radius = 6_378_137.0
            truth_xy = np.array(
                [
                    np.radians(longitude - truth_origin[1])
                    * earth_radius
                    * np.cos(np.radians(truth_origin[0])),
                    np.radians(latitude - truth_origin[0]) * earth_radius,
                ]
            )
            estimate_xy = np.array(engine.position, dtype=float)
            err = float(np.linalg.norm(estimate_xy - truth_xy))
            errors.append(err)

    # Compute metrics
    mae = float(np.mean(errors)) if errors else None
    rmse = float(np.sqrt(np.mean(np.square(errors)))) if errors else None
    final_error = float(errors[-1]) if errors else None
    drift_pct = (final_error / max(total_gt_distance, 1e-6)) * 100.0 if final_error is not None else None

    summary = {
        "input": str(input_path),
        "samples_replayed": len(records),
        "outage_start_s": outage_start,
        "outage_duration_s": outage_duration,
        "outage_samples": len(errors),
        "outage_mae_m": mae,
        "outage_rmse_m": rmse,
        "outage_final_error_m": final_error,
        "outage_drift_pct": drift_pct,
        "outage_max_error_m": max(errors) if errors else None,
        "total_gt_distance_m": total_gt_distance,
        "final_position": engine.position,
        "final_bearing_deg": engine.bearing_deg,
        "note": "Metrics are from this replay only; no target achievement is asserted.",
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