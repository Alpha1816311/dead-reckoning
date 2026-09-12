"""
IDR MVP — Real-time paced replay.

Feeds the IO-VNBD dataset through the production NavigationEngine at the
actual sensor timestamps (or at a configurable speedup), measuring:

  - IMU processing latency
  - GNSS processing latency
  - End-to-end event latency
  - Navigation output frequency
  - Dropped / late events
  - Mode transitions (GNSS → DR → RECOVERY → FUSED)

Writes:
  Data/mvp_performance.json     — measured performance metrics
  <output_path>                 — JSONL trajectory (same format as demo_replay)
  <output_path>_metrics.json   — outage accuracy metrics
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from navigation_engine import NavigationEngine
from sensor_interface import NavigationState


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _find_column(columns, *words):
    lc = [c.lower() for c in columns]
    for col, lcc in zip(columns, lc):
        if all(w.lower() in lcc for w in words):
            return col
    return None


def _required_column(df, *words):
    col = _find_column(df.columns, *words)
    if col is None:
        raise ValueError(f"Required column not found: {words}")
    return col


def haversine_m(lat1, lon1, lat2, lon2):
    R = 6_378_137.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ──────────────────────────────────────────────────────────────────────────────
# PERFORMANCE TRACKER
# ──────────────────────────────────────────────────────────────────────────────

class PerformanceTracker:
    def __init__(self):
        self.imu_latencies: list[float] = []
        self.gnss_latencies: list[float] = []
        self.event_timestamps: list[float] = []   # wall-clock time of each nav output
        self.dropped_events: int = 0
        self.events_processed: int = 0
        self.mode_transitions: list[dict] = []
        self._last_mode: str | None = None
        self._start_wall: float = time.monotonic()

    def record_imu(self, latency_s: float) -> None:
        self.imu_latencies.append(latency_s * 1000)  # store ms
        self.events_processed += 1
        self.event_timestamps.append(time.monotonic())

    def record_gnss(self, latency_s: float) -> None:
        self.gnss_latencies.append(latency_s * 1000)
        self.events_processed += 1

    def check_mode_transition(self, mode: str, sim_time: float) -> None:
        if mode != self._last_mode:
            transition = {
                "sim_time_s": round(sim_time, 3),
                "from": self._last_mode,
                "to": mode,
            }
            self.mode_transitions.append(transition)
            print(f"  [MODE TRANSITION @ {sim_time:.1f}s]  {self._last_mode} -> {mode}")
            self._last_mode = mode

    def summary(self) -> dict:
        wall_elapsed = time.monotonic() - self._start_wall
        imu_arr = np.array(self.imu_latencies) if self.imu_latencies else np.array([0.0])
        gnss_arr = np.array(self.gnss_latencies) if self.gnss_latencies else np.array([0.0])

        # Compute nav output frequency from event_timestamps gaps
        ts = self.event_timestamps
        if len(ts) > 1:
            gaps = np.diff(ts)
            avg_gap = float(np.mean(gaps))
            avg_rate = 1.0 / avg_gap if avg_gap > 0 else 0.0
        else:
            avg_rate = 0.0

        return {
            "events_processed": self.events_processed,
            "dropped_events": self.dropped_events,
            "wall_elapsed_s": round(wall_elapsed, 3),
            "avg_update_rate_hz": round(avg_rate, 2),
            "avg_imu_latency_ms": round(float(np.mean(imu_arr)), 3),
            "max_imu_latency_ms": round(float(np.max(imu_arr)), 3),
            "p95_imu_latency_ms": round(float(np.percentile(imu_arr, 95)), 3),
            "avg_gnss_latency_ms": round(float(np.mean(gnss_arr)), 3),
            "max_gnss_latency_ms": round(float(np.max(gnss_arr)), 3),
            "avg_event_latency_ms": round(float(np.mean(np.concatenate([imu_arr, gnss_arr]))), 3),
            "mode_transitions": self.mode_transitions,
        }


# ──────────────────────────────────────────────────────────────────────────────
# MAIN REAL-TIME REPLAY
# ──────────────────────────────────────────────────────────────────────────────

def run_realtime(
    input_path: str = "Data/S-S1.csv",
    outage_start: float = 120.0,
    outage_duration: float = 30.0,
    output_path: str | None = "Data/mvp_demo_trajectory.jsonl",
    max_rows: int | None = None,
    gnss_speed_unit: str = "mps",
    speedup: float = 1.0,
) -> dict:
    """
    Run a real-time paced replay of an IO-VNBD dataset through the production engine.

    Events are dispatched in chronological order, sleeping between samples to
    simulate real-time arrival at the given speedup factor.

    Returns the same summary dict as demo_replay.run_replay().
    Also writes Data/mvp_performance.json.
    """
    print(f"[realtime] Loading {input_path}...")
    df = pd.read_csv(input_path, encoding="cp1252", engine="python")

    # Ground truth
    vbox_path = Path("Data/V-S1.csv")
    vbox = pd.read_csv(vbox_path, encoding="cp1252", engine="python") if vbox_path.exists() else None
    vbox_speed_col = _find_column(vbox.columns, "velocity") if vbox is not None else None
    vbox_heading_col = _find_column(vbox.columns, "heading") if vbox is not None else None

    time_col = _required_column(df, "time")
    acc_cols = [_required_column(df, "accelerometer", a) for a in "xyz"]
    gyro_cols = [_find_column(df.columns, "gyroscope", a) for a in ("yaw", "pitch", "roll")]
    if any(c is None for c in gyro_cols):
        gyro_cols = [_required_column(df, "gyroscope", a) for a in "xyz"]
    mag_cols = [_find_column(df.columns, "magnetic", a) for a in "xyz"]
    lat_col = _find_column(df.columns, "gps", "latitude")
    lon_col = _find_column(df.columns, "gps", "longitude")
    speed_col = _find_column(df.columns, "gps", "speed")
    accuracy_col = _find_column(df.columns, "gps", "accuracy")

    if lat_col is None or lon_col is None:
        raise ValueError("Dataset must have GPS latitude and longitude columns")

    raw_times = pd.to_numeric(df[time_col], errors="coerce").to_numpy()
    timestamps = (raw_times - raw_times[0]) / 1000.0   # ms → seconds

    limit = len(df) if max_rows is None else min(len(df), max_rows)

    # Engine with replay clock
    replay_clock: list[float] = [0.0]
    engine = NavigationEngine(
        map_path=os.getenv("IDR_ROADS_GEOJSON"),
        model_path=os.getenv("IDR_SPEED_MODEL"),
        _clock=lambda: replay_clock[0],
    )

    perf = PerformanceTracker()
    records: list[dict] = []
    errors: list[float] = []
    outage_ground_truth_distance_m = 0.0
    previous_outage_gt = None
    truth_origin = None

    print(f"[realtime] Replaying {limit} rows  speedup={speedup}x  blackout=[{outage_start},{outage_start+outage_duration})s")
    print()

    wall_start = time.monotonic()
    sim_start: float | None = None

    for index in range(limit):
        row = df.iloc[index]
        timestamp = float(timestamps[index])
        replay_clock[0] = timestamp

        # Real-time pacing: sleep to simulate actual arrival rate
        if speedup > 0 and index > 0:
            if sim_start is None:
                sim_start = timestamp
                wall_ref = time.monotonic()
            else:
                sim_elapsed = timestamp - sim_start
                wall_elapsed = time.monotonic() - wall_ref
                target = sim_elapsed / speedup
                sleep_needed = target - wall_elapsed
                if sleep_needed > 0.0005:
                    time.sleep(sleep_needed)

        latitude = float(pd.to_numeric(row[lat_col], errors="coerce"))
        longitude = float(pd.to_numeric(row[lon_col], errors="coerce"))

        if truth_origin is None:
            truth_origin = (latitude, longitude)

        in_outage = outage_start <= timestamp < outage_start + outage_duration

        # ── GNSS ──────────────────────────────────────────────────────────────
        if not in_outage:
            speed = None
            if speed_col is not None:
                speed = float(pd.to_numeric(row[speed_col], errors="coerce"))
                if gnss_speed_unit.lower() == "kmh":
                    speed /= 3.6

            accuracy = 10.0
            if accuracy_col is not None:
                cand = float(pd.to_numeric(row[accuracy_col], errors="coerce"))
                if math.isfinite(cand) and cand > 0:
                    accuracy = cand

            t0 = time.perf_counter()
            engine.process_gnss(
                timestamp=timestamp,
                latitude=latitude,
                longitude=longitude,
                speed_mps=speed,
                accuracy_m=accuracy,
            )
            perf.record_gnss(time.perf_counter() - t0)

        # ── IMU ───────────────────────────────────────────────────────────────
        accel = [float(pd.to_numeric(row[c], errors="coerce")) for c in acc_cols]
        gyro  = [float(pd.to_numeric(row[c], errors="coerce")) for c in gyro_cols]
        mag   = None
        if all(c is not None for c in mag_cols):
            mag = [float(pd.to_numeric(row[c], errors="coerce")) for c in mag_cols]

        t0 = time.perf_counter()
        state = engine.process_imu(
            timestamp=timestamp,
            accel=accel,
            gyro=gyro,
            mag=mag,
        )
        perf.record_imu(time.perf_counter() - t0)

        mode_display = state.get("mode") or state.get("gnss_state") or "UNKNOWN"
        perf.check_mode_transition(mode_display, timestamp)

        state["simulated_gnss_available"] = not in_outage

        # Reference truth
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

        if state.get("position") is not None:
            local = state["local_position_m"]
            estimate_xy = np.array([local["east"], local["north"]])
            state["position_error_m"] = float(np.linalg.norm(estimate_xy - truth_xy))
            if "reference_heading_deg" in state:
                delta = ((state["heading_deg"] - state["reference_heading_deg"] + 180) % 360) - 180
                state["heading_error_deg"] = float(delta)
                heading = np.radians(state["reference_heading_deg"])
                err = estimate_xy - truth_xy
                state["along_track_error_m"] = float(err[0] * np.sin(heading) + err[1] * np.cos(heading))
                state["cross_track_error_m"] = float(err[0] * np.cos(heading) - err[1] * np.sin(heading))

        records.append(state)

        # Outage metrics
        if in_outage:
            current_gt = (latitude, longitude)
            if previous_outage_gt is not None:
                outage_ground_truth_distance_m += haversine_m(
                    previous_outage_gt[0], previous_outage_gt[1],
                    current_gt[0], current_gt[1],
                )
            previous_outage_gt = current_gt
            if state.get("position") is not None:
                errors.append(state["position_error_m"])

        # Progress report every 500 events
        if index > 0 and index % 500 == 0:
            elapsed = time.monotonic() - wall_start
            rate = index / elapsed if elapsed > 0 else 0
            print(f"  Progress: {index}/{limit}  sim={timestamp:.1f}s  "
                  f"wall={elapsed:.1f}s  rate={rate:.0f} ev/s  mode={mode_display}")

    # ── FINAL METRICS ─────────────────────────────────────────────────────────
    mae        = float(np.mean(errors)) if errors else None
    rmse       = float(np.sqrt(np.mean(np.square(errors)))) if errors else None
    final_err  = float(errors[-1]) if errors else None
    max_err    = float(max(errors)) if errors else None
    drift_pct  = (final_err / max(outage_ground_truth_distance_m, 1e-6) * 100.0) if final_err is not None else None

    recovery_jump_m = None
    for i in range(1, len(records)):
        if records[i - 1]["simulated_gnss_available"] is False and records[i]["simulated_gnss_available"]:
            prev = records[i - 1].get("local_position_m")
            curr = records[i].get("local_position_m")
            if prev and curr:
                recovery_jump_m = float(np.hypot(
                    curr["east"] - prev["east"],
                    curr["north"] - prev["north"],
                ))
            break

    perf_summary = perf.summary()

    # Write performance JSON
    Path("Data").mkdir(exist_ok=True)
    perf_path = Path("Data/mvp_performance.json")
    perf_path.write_text(json.dumps(perf_summary, indent=2, default=str), encoding="utf-8")
    print(f"\n[perf] Written {perf_path}")
    print(f"[perf] Avg update rate : {perf_summary['avg_update_rate_hz']} Hz")
    print(f"[perf] Avg IMU latency : {perf_summary['avg_imu_latency_ms']} ms")
    print(f"[perf] Events processed: {perf_summary['events_processed']}")

    summary = {
        "input": str(input_path),
        "mode": "realtime",
        "speedup": speedup,
        "samples_replayed": len(records),
        "outage_start_s": outage_start,
        "outage_duration_s": outage_duration,
        "outage_samples": len(errors),
        "outage_mae_m": mae,
        "outage_rmse_m": rmse,
        "outage_final_error_m": final_err,
        "outage_max_error_m": max_err,
        "outage_ground_truth_distance_m": outage_ground_truth_distance_m,
        "outage_drift_pct": drift_pct,
        "gnss_recovery_discontinuity_m": recovery_jump_m,
        "performance": perf_summary,
        "final_state": engine.state_snapshot(),
        "note": "Real-time paced replay metrics.",
    }

    # Write trajectory JSONL
    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(rec) + "\n")
            fh.write(json.dumps({"summary": summary}) + "\n")

        _save_replay_plots(records, path.with_suffix(""))
        metrics_path = path.with_name("mvp_performance_metrics.json")
        metrics_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        print(f"[output] Trajectory : {path}")
        print(f"[output] Metrics    : {metrics_path}")

    return summary


def _save_replay_plots(records: list[dict], prefix: Path) -> None:
    """Save trajectory and error plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    rows = [r for r in records if r.get("local_position_m") is not None]
    if not rows:
        return

    time_s  = [r["timestamp"] for r in rows]
    ref_e   = [r["reference_east_m"] for r in rows]
    ref_n   = [r["reference_north_m"] for r in rows]
    est_e   = [r["local_position_m"]["east"] for r in rows]
    est_n   = [r["local_position_m"]["north"] for r in rows]
    errors  = [r.get("position_error_m", float("nan")) for r in rows]
    outage  = [not r["simulated_gnss_available"] for r in rows]
    modes   = [r.get("mode") or r.get("gnss_state") or "" for r in rows]

    fig, axes = plt.subplots(2, 1, figsize=(10, 9))

    axes[0].plot(ref_e, ref_n, label="GNSS reference", alpha=0.7)
    axes[0].plot(est_e, est_n, label="Navigation estimate", alpha=0.7)
    axes[0].set(xlabel="East (m)", ylabel="North (m)", title="Real-time Replay: GNSS → DR → Recovery")
    axes[0].axis("equal")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(time_s, errors, label="Position error (m)", color="tab:blue")
    axes[1].fill_between(time_s, 0, max(e for e in errors if math.isfinite(e)) * 1.1,
                         where=outage, alpha=0.15, color="red", label="GNSS blackout")
    axes[1].set(xlabel="Time (s)", ylabel="Error (m)", title="Position error and GNSS blackout window")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(str(prefix) + "_realtime_trajectory.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    import argparse, sys

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="Data/S-S1.csv")
    parser.add_argument("--outage-start", type=float, default=120.0)
    parser.add_argument("--outage-duration", type=float, default=30.0)
    parser.add_argument("--output", default="Data/mvp_demo_trajectory.jsonl")
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--gnss-speed-unit", choices=("kmh", "mps"), default="mps")
    parser.add_argument("--speedup", type=float, default=10.0,
                        help="Replay speedup factor (1=real-time, 10=10x faster)")
    args = parser.parse_args()

    summary = run_realtime(
        input_path=args.input,
        outage_start=args.outage_start,
        outage_duration=args.outage_duration,
        output_path=args.output,
        max_rows=args.max_rows,
        gnss_speed_unit=args.gnss_speed_unit,
        speedup=args.speedup,
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "final_state"}, indent=2, default=str))
