"""
IDR MVP — Intelligent Dead Reckoning canonical entrypoint.

Usage (replay mode — default):
    python run_mvp.py
    python run_mvp.py --dataset Data/S-S1.csv --blackout-start 120 --blackout-duration 30

Usage (server mode — live streaming):
    python run_mvp.py --mode server

Usage (real-time replay — paced):
    python run_mvp.py --mode realtime

This entrypoint does NOT duplicate navigation logic.
It calls the existing production engine via demo_replay (replay/realtime)
or app (server mode).
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def _print_header():
    print("=" * 70)
    print("  IDR — INTELLIGENT DEAD RECKONING  MVP")
    print("=" * 70)
    print()


def _print_summary(summary: dict) -> None:
    """Print key metrics from a replay summary dict."""
    print("\n--- MVP REPLAY SUMMARY ---")
    print(f"  Samples replayed   : {summary.get('samples_replayed', 'N/A')}")
    print(f"  Outage start (s)   : {summary.get('outage_start_s', 'N/A')}")
    print(f"  Outage duration(s) : {summary.get('outage_duration_s', 'N/A')}")
    print(f"  Outage samples     : {summary.get('outage_samples', 'N/A')}")

    mae = summary.get("outage_mae_m")
    final_err = summary.get("outage_final_error_m")
    drift = summary.get("outage_drift_pct")
    jump = summary.get("gnss_recovery_discontinuity_m")

    if mae is not None:
        print(f"  Position MAE       : {mae:.2f} m")
    if final_err is not None:
        print(f"  Final error        : {final_err:.2f} m")
    if drift is not None:
        print(f"  DR drift %%         : {drift:.2f}%%")
    if jump is not None:
        print(f"  Recovery jump      : {jump:.2f} m")

    perf_path = Path("Data/mvp_performance.json")
    if perf_path.exists():
        perf = json.loads(perf_path.read_text(encoding="utf-8"))
        print("\n--- PERFORMANCE ---")
        print(f"  Avg update rate    : {perf.get('avg_update_rate_hz', 'N/A')} Hz")
        print(f"  Avg event latency  : {perf.get('avg_event_latency_ms', 'N/A')} ms")
        print(f"  Events processed   : {perf.get('events_processed', 'N/A')}")
        print(f"  Dropped events     : {perf.get('dropped_events', 'N/A')}")

    print()


def run_batch_replay(args) -> int:
    """Run batch replay via demo_replay.py — fastest path."""
    cmd = [
        sys.executable, "demo_replay.py",
        "--input", args.dataset,
        "--outage-start", str(args.blackout_start),
        "--outage-duration", str(args.blackout_duration),
        "--output", args.output,
    ]
    if args.max_rows:
        cmd += ["--max-rows", str(args.max_rows)]
    if args.gnss_speed_unit:
        cmd += ["--gnss-speed-unit", args.gnss_speed_unit]

    print(f"Running: {' '.join(cmd)}")
    print()
    result = subprocess.run(cmd, capture_output=False, text=True)
    return result.returncode


def run_realtime_replay(args) -> int:
    """Run the real-time paced replay via mvp_realtime_replay.py."""
    from mvp_realtime_replay import run_realtime

    summary = run_realtime(
        input_path=args.dataset,
        outage_start=args.blackout_start,
        outage_duration=args.blackout_duration,
        output_path=args.output,
        max_rows=args.max_rows,
        gnss_speed_unit=args.gnss_speed_unit,
        speedup=args.speedup,
    )
    _print_summary(summary)
    return 0


def run_server(args) -> int:
    """Start the FastAPI navigation server."""
    try:
        import uvicorn
    except ImportError:
        print("ERROR: uvicorn not installed.  Run: pip install uvicorn")
        return 1

    host = getattr(args, "host", "0.0.0.0")
    port = getattr(args, "port", 8000)
    print(f"Starting IDR Navigation Server on http://{host}:{port}")
    print(f"  Dashboard     : http://{host}:{port}/")
    print(f"  Live navigate : http://{host}:{port}/navigate")
    print(f"  API docs      : http://{host}:{port}/docs")
    print(f"  WebSocket     : ws://{host}:{port}/ws/sensor")
    print()
    import uvicorn
    uvicorn.run("app:app", host=host, port=port, reload=False)
    return 0


def main() -> int:
    _print_header()

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--mode",
        choices=("replay", "realtime", "server"),
        default="replay",
        help="replay=batch (fast), realtime=paced at dataset speed, server=live HTTP/WS",
    )
    parser.add_argument(
        "--dataset",
        default="Data/S-S1.csv",
        help="Input CSV dataset (IO-VNBD format)",
    )
    parser.add_argument(
        "--blackout-start",
        type=float,
        default=120.0,
        help="Seconds into dataset when GNSS blackout begins",
    )
    parser.add_argument(
        "--blackout-duration",
        type=float,
        default=30.0,
        help="Duration of simulated GNSS blackout in seconds",
    )
    parser.add_argument(
        "--output",
        default="Data/mvp_demo_trajectory.jsonl",
        help="Output JSONL trajectory file",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Limit rows processed (useful for quick tests)",
    )
    parser.add_argument(
        "--gnss-speed-unit",
        choices=("kmh", "mps"),
        default="mps",
        help="Unit of GPS SPEED column in the input CSV",
    )
    parser.add_argument(
        "--speedup",
        type=float,
        default=1.0,
        help="Replay speed multiplier (realtime mode only; >1 = faster)",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Server bind host (server mode only)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Server bind port (server mode only)",
    )

    args = parser.parse_args()

    # Validate dataset exists for non-server modes
    if args.mode in ("replay", "realtime"):
        if not Path(args.dataset).exists():
            print(f"ERROR: Dataset not found: {args.dataset}")
            print("  Download or symlink the IO-VNBD dataset to Data/S-S1.csv")
            return 1

    print(f"Mode           : {args.mode.upper()}")
    if args.mode in ("replay", "realtime"):
        print(f"Dataset        : {args.dataset}")
        print(f"Blackout start : {args.blackout_start} s")
        print(f"Blackout dur.  : {args.blackout_duration} s")
        print(f"Output         : {args.output}")
    print()

    if args.mode == "replay":
        rc = run_batch_replay(args)
        if rc == 0:
            # Load and print summary if output written
            metrics_path = Path(args.output).with_name(
                Path(args.output).stem + "_metrics.json"
            )
            if not metrics_path.exists():
                # demo_replay uses mvp_demo_metrics.json for mvp_demo_ prefix files
                alt = Path(args.output).with_name("mvp_demo_metrics.json")
                if alt.exists():
                    metrics_path = alt
            if metrics_path.exists():
                summary = json.loads(metrics_path.read_text(encoding="utf-8"))
                _print_summary(summary)
        return rc

    elif args.mode == "realtime":
        return run_realtime_replay(args)

    elif args.mode == "server":
        return run_server(args)

    return 0


if __name__ == "__main__":
    sys.exit(main())
