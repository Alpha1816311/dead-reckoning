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
    """Return distance between two GPS coordinates in metres."""
    earth_radius = 6_378_137.0

    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)
    dlat = lat2 - lat1
    dlon = np.radians(lon2 - lon1)

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2.0) ** 2
    )

    return float(
        2.0 * earth_radius * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    )


def run_replay(
    input_path: str,
    outage_start: float,
    outage_duration: float,
    output_path: str | None = None,
    max_rows: int | None = None,
):
    df = pd.read_csv(input_path, encoding="cp1252", engine="python")

    time_col = required_column(df, "time")

    acc_cols = [
        required_column(df, "accelerometer", axis)
        for axis in "xyz"
    ]

    gyro_cols = [
        find_column(df.columns, "gyroscope", axis)
        for axis in ("yaw", "pitch", "roll")
    ]

    if any(column is None for column in gyro_cols):
        gyro_cols = [
            required_column(df, "gyroscope", axis)
            for axis in "xyz"
        ]

    mag_cols = [
        find_column(df.columns, "magnetic", axis)
        for axis in "xyz"
    ]

    lat_col = find_column(df.columns, "gps", "latitude")
    lon_col = find_column(df.columns, "gps", "longitude")
    accuracy_col = find_column(df.columns, "gps", "accuracy")

    if lat_col is None or lon_col is None:
        raise ValueError(
            "replay requires GPS latitude and longitude columns"
        )

    timestamps = pd.to_numeric(
        df[time_col],
        errors="coerce",
    ).to_numpy()

    timestamps = (timestamps - timestamps[0]) / 1000.0

    engine = NavigationEngine(
        origin_latitude=origin_latitude,
        origin_longitude=origin_longitude,
    )

    records = []
    errors_m = []
    outage_ground_truth_distance_m = 0.0
    previous_outage_gt = None

    # Ground-truth distance travelled only during the simulated outage.
    outage_ground_truth_distance_m = 0.0
    previous_outage_gt = None

    limit = (
        len(df)
        if max_rows is None
        else min(len(df), max_rows)
    )

    for index in range(limit):
        row = df.iloc[index]

        timestamp = float(timestamps[index])

        latitude = float(
            pd.to_numeric(
                row[lat_col],
                errors="coerce",
            )
        )

        longitude = float(
            pd.to_numeric(
                row[lon_col],
                errors="coerce",
            )
        )

        if truth_origin is None:
            truth_origin = (
                latitude,
                longitude,
            )

        in_outage = (
            outage_start
            <= timestamp
            < outage_start + outage_duration
        )

        # GNSS is deliberately disabled during the simulated outage.
        if not in_outage:
            speed = None

            if speed_col is not None:
                speed = float(
                    pd.to_numeric(
                        row[speed_col],
                        errors="coerce",
                    )
                )

                # Dataset labels its speed in km/h.
                if "km" in str(speed_col).lower():
                    speed /= 3.6

            accuracy = 10.0

            if accuracy_col is not None:
                candidate = float(
                    pd.to_numeric(
                        row[accuracy_col],
                        errors="coerce",
                    )
                )

                if np.isfinite(candidate) and candidate > 0:
                    accuracy = candidate

            engine.process_gnss(
                timestamp=timestamp,
                lat=latitude,
                lon=longitude,
                accuracy_m=accuracy_m,
            )

        accel = [
            float(
                pd.to_numeric(
                    row[column],
                    errors="coerce",
                )
            )
            for column in acc_cols
        ]

        gyro = [
            float(
                pd.to_numeric(
                    row[column],
                    errors="coerce",
                )
            )
            for column in gyro_cols
        ]

        mag = None

        if all(column is not None for column in mag_cols):
            mag = [
                float(
                    pd.to_numeric(
                        row[column],
                        errors="coerce",
                    )
                )
                for column in mag_cols
            ]

        state = engine.process_imu(
            timestamp=timestamp,
            dt=dt,
            linear_accel=linear_accel,
            ai_acceleration=None,
            outage_mode=in_outage,
        )

        state["simulated_gnss_available"] = not in_outage

        records.append(state)

        # ---------------------------------------------------------
        # OUTAGE METRICS
        # ---------------------------------------------------------
        if in_outage:
            current_gt = (
                latitude,
                longitude,
            )

            # Ground-truth distance travelled between consecutive
            # GPS samples during the outage.
            if previous_outage_gt is not None:
                outage_ground_truth_distance_m += haversine_m(
                    previous_outage_gt[0],
                    previous_outage_gt[1],
                    current_gt[0],
                    current_gt[1],
                )

            previous_outage_gt = current_gt

            # Position error between DR estimate and actual GPS position.
            if state["position"] is not None:
                earth_radius = 6_378_137.0

                truth_xy = np.array(
                    [
                        np.radians(
                            longitude - truth_origin[1]
                        )
                        * earth_radius
                        * np.cos(
                            np.radians(
                                truth_origin[0]
                            )
                        ),
                        np.radians(
                            latitude - truth_origin[0]
                        )
                        * earth_radius,
                    ]
                )

                estimate_xy = np.array(
                    [
                        state["local_position_m"]["east"],
                        state["local_position_m"]["north"],
                    ]
                )

                errors.append(
                    float(
                        np.linalg.norm(
                            estimate_xy - truth_xy
                        )
                    )
                )

    # -------------------------------------------------------------
    # FINAL OUTAGE METRICS
    # -------------------------------------------------------------

    mae = (
        float(np.mean(errors))
        if errors
        else None
    )

    rmse = (
        float(
            np.sqrt(
                np.mean(
                    np.square(errors)
                )
            )
        )
        if errors
        else None
    )

    final_error = (
        float(errors[-1])
        if errors
        else None
    )

    max_error = (
        float(max(errors))
        if errors
        else None
    )

    # Same drift definition used by Member 1 benchmark:
    #
    # drift % = final position error /
    #           ground-truth distance travelled
    #           × 100
    drift_pct = None

    if final_error is not None:
        drift_pct = (
            final_error
            / max(
                outage_ground_truth_distance_m,
                1e-6,
            )
        ) * 100.0

    summary = {
        "input": str(input_path),
        "samples_replayed": len(records),
        "outage_start_s": outage_start,
        "outage_duration_s": outage_duration,
        "outage_samples": len(errors),

        "outage_mae_m": mae,
        "outage_rmse_m": rmse,
        "outage_final_error_m": final_error,
        "outage_max_error_m": max_error,

        "outage_ground_truth_distance_m": (
            outage_ground_truth_distance_m
        ),

        "outage_drift_pct": drift_pct,

        "final_state": engine.state_snapshot(),

        "note": (
            "Metrics are from this replay only; "
            "no target achievement is asserted."
        ),
    }

    if output_path:
        path = Path(output_path)
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with path.open(
            "w",
            encoding="utf-8",
        ) as handle:

            for record in records:
                handle.write(
                    json.dumps(record) + "\n"
                )

            handle.write(
                json.dumps(
                    {"summary": summary}
                )
                + "\n"
            )

    return summary


def main():
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--input",
        default="Data/S-S1.csv",
    )

    parser.add_argument(
        "--outage-start",
        type=float,
        default=120.0,
    )

    parser.add_argument(
        "--outage-duration",
        type=float,
        default=30.0,
    )

    parser.add_argument(
        "--output",
        default="Data/live_replay.jsonl",
    )

    parser.add_argument(
        "--max-rows",
        type=int,
    )

    args = parser.parse_args()

    summary = run_replay(
        input_path=args.input,
        outage_start=args.outage_start,
        outage_duration=args.outage_duration,
        output_path=args.output,
        max_rows=args.max_rows,
    )

    print(
        json.dumps(
            summary,
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()