"""Replay the supplied CSV through the live incremental navigation engine.

This is a real-data demonstration harness, not a benchmark claim. It can
simulate GNSS ON -> OFF -> ON while still feeding every IMU row to the same
engine used by FastAPI.
"""

from __future__ import annotations

import argparse
import json
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
    gnss_speed_unit: str = "mps",
):
    df = pd.read_csv(input_path, encoding="cp1252", engine="python")
    vbox_path = Path("Data/V-S1.csv")
    vbox = pd.read_csv(vbox_path, encoding="cp1252", engine="python") if vbox_path.exists() else None
    vbox_speed_col = find_column(vbox.columns, "velocity") if vbox is not None else None
    vbox_heading_col = find_column(vbox.columns, "heading") if vbox is not None else None

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
    speed_col = find_column(df.columns, "gps", "speed")
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

    replay_clock = [0.0]
    engine = NavigationEngine(
        map_path=os.getenv("IDR_ROADS_GEOJSON"),
        model_path=os.getenv("IDR_SPEED_MODEL"),
        _clock=lambda: replay_clock[0],
    )

    records = []
    truth_origin = None
    errors = []

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
        # The live engine intentionally uses a monotonic receive clock for GNSS
        # freshness. In replay, that clock must share the recorded timestamp
        # domain or a simulated outage is never detected.
        replay_clock[0] = timestamp

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

                # IO-VNBD's ``GPS SPEED (Kmh)`` values are empirically m/s
                # scale (15.55 against VBOX 15.79 m/s at 120.1 s).  Keep the
                # unit explicit for other input sources.
                if gnss_speed_unit.lower() == "kmh":
                    speed /= 3.6
                elif gnss_speed_unit.lower() != "mps":
                    raise ValueError("gnss_speed_unit must be 'kmh' or 'mps'")

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
                latitude=latitude,
                longitude=longitude,
                speed_mps=speed,
                accuracy_m=accuracy,
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
            accel=accel,
            gyro=gyro,
            mag=mag,
        )

        state["simulated_gnss_available"] = not in_outage

        # Keep local reference coordinates with every record so the replay is
        # directly plottable across GNSS, blackout, and recovery modes.
        earth_radius = 6_378_137.0
        truth_xy = np.array([
            np.radians(longitude - truth_origin[1]) * earth_radius * np.cos(np.radians(truth_origin[0])),
            np.radians(latitude - truth_origin[0]) * earth_radius,
        ])
        state["reference_east_m"] = float(truth_xy[0])
        state["reference_north_m"] = float(truth_xy[1])
        if vbox is not None and index < len(vbox):
            if vbox_speed_col is not None:
                state["reference_speed_mps"] = float(pd.to_numeric(vbox.iloc[index][vbox_speed_col], errors="coerce") / 3.6)
            if vbox_heading_col is not None:
                state["reference_heading_deg"] = float(pd.to_numeric(vbox.iloc[index][vbox_heading_col], errors="coerce"))
        if state["position"] is not None:
            estimate_xy = np.array([
                state["local_position_m"]["east"],
                state["local_position_m"]["north"],
            ])
            state["position_error_m"] = float(np.linalg.norm(estimate_xy - truth_xy))
        if "reference_heading_deg" in state:
            delta = ((state["heading_deg"] - state["reference_heading_deg"] + 180) % 360) - 180
            state["heading_error_deg"] = float(delta)
            heading = np.radians(state["reference_heading_deg"])
            error_xy = estimate_xy - truth_xy
            state["along_track_error_m"] = float(error_xy[0] * np.sin(heading) + error_xy[1] * np.cos(heading))
            state["cross_track_error_m"] = float(error_xy[0] * np.cos(heading) - error_xy[1] * np.sin(heading))

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
                errors.append(state["position_error_m"])

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

    recovery_jump_m = None
    for index in range(1, len(records)):
        if records[index - 1]["simulated_gnss_available"] is False and records[index]["simulated_gnss_available"]:
            previous = records[index - 1].get("local_position_m")
            current = records[index].get("local_position_m")
            if previous is not None and current is not None:
                recovery_jump_m = float(np.hypot(
                    current["east"] - previous["east"],
                    current["north"] - previous["north"],
                ))
            break

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
        "gnss_recovery_discontinuity_m": recovery_jump_m,

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

        _save_replay_plots(records, path.with_suffix(""))
        metrics_path = path.with_name("mvp_demo_metrics.json") if path.name.startswith("mvp_demo_") else path.with_name(path.stem + "_metrics.json")
        metrics_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    return summary


def _save_replay_plots(records, prefix: Path) -> None:
    """Write reproducible trajectory and error/mode plots for a replay."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [row for row in records if row.get("local_position_m") is not None]
    if not rows:
        return
    time_s = [row["timestamp"] for row in rows]
    ref_e = [row["reference_east_m"] for row in rows]
    ref_n = [row["reference_north_m"] for row in rows]
    est_e = [row["local_position_m"]["east"] for row in rows]
    est_n = [row["local_position_m"]["north"] for row in rows]
    errors = [row.get("position_error_m", float("nan")) for row in rows]
    outage = [not row["simulated_gnss_available"] for row in rows]

    fig, axes = plt.subplots(2, 1, figsize=(10, 9))
    axes[0].plot(ref_e, ref_n, label="GNSS reference")
    axes[0].plot(est_e, est_n, label="Navigation estimate")
    axes[0].set(xlabel="East (m)", ylabel="North (m)", title="GNSS blackout → DR → recovery")
    axes[0].axis("equal"); axes[0].grid(alpha=0.3); axes[0].legend()
    axes[1].plot(time_s, errors, label="Position error (m)")
    axes[1].fill_between(time_s, 0, 1, where=outage, alpha=0.2, label="GNSS blackout")
    axes[1].set(xlabel="Time (s)", ylabel="Error / outage", title="Error and navigation mode")
    axes[1].grid(alpha=0.3); axes[1].legend()
    fig.tight_layout()
    fig.savefig(str(prefix) + "_trajectory_error_mode.png", dpi=140)
    plt.close(fig)


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
        default="Data/mvp_demo_trajectory.jsonl",
    )

    parser.add_argument(
        "--max-rows",
        type=int,
    )
    parser.add_argument("--gnss-speed-unit", choices=("kmh", "mps"), default="mps")

    args = parser.parse_args()

    summary = run_replay(
        input_path=args.input,
        outage_start=args.outage_start,
        outage_duration=args.outage_duration,
        output_path=args.output,
        max_rows=args.max_rows,
        gnss_speed_unit=args.gnss_speed_unit,
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
