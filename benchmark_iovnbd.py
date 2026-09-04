import math
from pathlib import Path

import numpy as np
import pandas as pd

from navigation_engine import NavigationEngine


S_PATH = Path("Data/S-S1.csv")
V_PATH = Path("Data/V-S1.csv")

OUTAGE_START = 120.0
OUTAGE_END = 150.0

MODEL_PATH = Path("models/speed_model_vbox.joblib")


def valid_number(x):
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def find_col(df, text):
    for col in df.columns:
        if text.lower() in str(col).lower():
            return col
    raise KeyError(f"Column not found: {text}")

def local_xy(lat, lon, lat0, lon0):
    """Convert latitude/longitude to local East/North meters."""
    R = 6378137.0

    lat_rad = np.radians(float(lat))
    lon_rad = np.radians(float(lon))

    lat0_rad = np.radians(float(lat0))
    lon0_rad = np.radians(float(lon0))

    east = (
        R
        * (lon_rad - lon0_rad)
        * math.cos(lat0_rad)
    )

    north = R * (
        lat_rad - lat0_rad
    )

    return float(east), float(north)

print("=" * 70)
print("IO-VNBD CONTROLLED AI DEAD-RECKONING BENCHMARK")
print("=" * 70)

print("\nLoading datasets...")

s = pd.read_csv(
    S_PATH,
    engine="python",
    encoding="cp1252"
)

v = pd.read_csv(
    V_PATH,
    engine="python",
    encoding="cp1252"
)

print(f"S-S1 rows : {len(s)}")
print(f"V-S1 rows : {len(v)}")

if len(s) != len(v):
    raise RuntimeError("S-S1 and V-S1 row counts differ.")


# ---------------------------------------------------------
# Columns
# ---------------------------------------------------------

S_TIME = find_col(s, "TIME SINCE START")
S_LAT = find_col(s, "GPS LATITUDE")
S_LON = find_col(s, "GPS LONGITUDE")
S_SPEED = find_col(s, "GPS SPEED")
S_ACC = find_col(s, "GPS ACCURACY")

S_AX = find_col(s, "Accelerometer x")
S_AY = find_col(s, "Accelerometer y")
S_AZ = find_col(s, "Accelerometer z")

S_GX = find_col(s, "Gyroscope yaw")
S_GY = find_col(s, "Gyroscope pitch")
S_GZ = find_col(s, "Gyroscope roll")

S_MX = find_col(s, "Magnetic Field x")
S_MY = find_col(s, "Magnetic Field y")
S_MZ = find_col(s, "Magnetic Field z")

V_LAT = find_col(v, "Latitude (degrees)")
V_LON = find_col(v, "Longitude (degrees)")
V_SPEED = find_col(v, "Velocity (km/hr)")


def arr(df, col):
    return pd.to_numeric(
        df[col],
        errors="coerce"
    ).to_numpy(dtype=float)


# ---------------------------------------------------------
# Arrays
# ---------------------------------------------------------

s_time = (
    arr(s, S_TIME) -
    arr(s, S_TIME)[0]
) / 1000.0

s_lat = arr(s, S_LAT)
s_lon = arr(s, S_LON)
s_gps_speed = arr(s, S_SPEED)
s_accuracy = arr(s, S_ACC)

s_ax = arr(s, S_AX)
s_ay = arr(s, S_AY)
s_az = arr(s, S_AZ)

s_gx = arr(s, S_GX)
s_gy = arr(s, S_GY)
s_gz = arr(s, S_GZ)

s_mx = arr(s, S_MX)
s_my = arr(s, S_MY)
s_mz = arr(s, S_MZ)

v_lat = arr(v, V_LAT)
v_lon = arr(v, V_LON)

v_speed = arr(v, V_SPEED) / 3.6


# ---------------------------------------------------------
# Engine
# ---------------------------------------------------------

print("\nInitializing NavigationEngine...")

engine = NavigationEngine(
    map_path=None,
    model_path=str(MODEL_PATH)
)

print("Engine initialized.")
print("Map matching: OFF")
print("AI acceleration model: LOADED")


results = []

outage_started = False


# ---------------------------------------------------------
# Replay
# ---------------------------------------------------------

for i in range(len(s)):

    t = float(s_time[i])

    if t > OUTAGE_END:
        break

    ax = s_ax[i]
    ay = s_ay[i]
    az = s_az[i]

    gx = s_gx[i]
    gy = s_gy[i]
    gz = s_gz[i]

    mx = s_mx[i]
    my = s_my[i]
    mz = s_mz[i]

    if not all(
        valid_number(x)
        for x in [
            ax, ay, az,
            gx, gy, gz
        ]
    ):
        continue

    in_outage = (
        OUTAGE_START <= t < OUTAGE_END
    )

    # -----------------------------------------------------
    # GNSS BEFORE OUTAGE
    #
    # We still use smartphone GNSS for position.
    # However, immediately at outage start we replace ONLY
    # the speed state with VBOX reference speed to isolate
    # the dead-reckoning algorithm.
    # -----------------------------------------------------

    if not in_outage:

        lat = s_lat[i]
        lon = s_lon[i]
        gps_speed = s_gps_speed[i]
        accuracy = s_accuracy[i]

        if (
            valid_number(lat)
            and valid_number(lon)
            and valid_number(gps_speed)
        ):

            if (
                not valid_number(accuracy)
                or accuracy <= 0
            ):
                accuracy = 10.0

            engine.process_gnss(
                timestamp=t,
                latitude=float(lat),
                longitude=float(lon),
                speed_mps=float(gps_speed) / 3.6,
                accuracy_m=float(accuracy),
            )

    # -----------------------------------------------------
    # Start controlled outage
    # -----------------------------------------------------

    if in_outage and not outage_started:

        outage_started = True

        engine_start_east = float(
            engine.position_xy[0]
        )

        engine_start_north = float(
            engine.position_xy[1]
        )

        ref_start_lat = float(
            v_lat[i]
        )

        ref_start_lon = float(
            v_lon[i]
        )

        # IMPORTANT:
        # Controlled benchmark initialization.
        # VBOX speed is NOT fed continuously into DR.
        engine.speed_mps = float(
            v_speed[i]
        )

        engine.speed_source = (
            "controlled-vbox-initialization"
        )

        print()
        print("=" * 70)
        print("GNSS OUTAGE STARTED")
        print("=" * 70)
        print(f"Time: {t:.2f} s")
        print(
            f"Initial VBOX speed: "
            f"{v_speed[i] * 3.6:.2f} km/h"
        )
        print(
            f"Initial engine speed: "
            f"{engine.speed_mps * 3.6:.2f} km/h"
        )

    # -----------------------------------------------------
    # IMU
    # -----------------------------------------------------

    engine.process_imu(
        timestamp=t,
        accel=np.array(
            [ax, ay, az],
            dtype=float
        ),
        gyro=np.array(
            [gx, gy, gz],
            dtype=float
        ),
        mag=(
            np.array(
                [mx, my, mz],
                dtype=float
            )
            if all(
                valid_number(x)
                for x in [mx, my, mz]
            )
            else None
        ),
    )

    # -----------------------------------------------------
    # Capture outage result
    # -----------------------------------------------------

    if in_outage:

        ref_east, ref_north = local_xy(
            v_lat[i],
            v_lon[i],
            ref_start_lat,
            ref_start_lon,
        )

        est_east = (
            float(engine.position_xy[0])
            - engine_start_east
        )

        est_north = (
            float(engine.position_xy[1])
            - engine_start_north
        )

        error = math.hypot(
            est_east - ref_east,
            est_north - ref_north,
        )

        results.append({
            "time_s": t,
            "reference_east_m": ref_east,
            "reference_north_m": ref_north,
            "estimated_east_m": est_east,
            "estimated_north_m": est_north,
            "position_error_m": error,
            "estimated_speed_mps": float(
                engine.speed_mps
            ),
            "reference_speed_mps": float(
                v_speed[i]
            ),
        })


if not results:
    raise RuntimeError(
        "No outage samples generated."
    )


r = pd.DataFrame(results)

errors = r[
    "position_error_m"
].to_numpy()

mae = float(
    np.mean(errors)
)

rmse = float(
    np.sqrt(np.mean(errors ** 2))
)

final_error = float(
    errors[-1]
)

max_error = float(
    np.max(errors)
)

dx = np.diff(
    r["reference_east_m"].to_numpy()
)

dy = np.diff(
    r["reference_north_m"].to_numpy()
)

reference_path = float(
    np.sum(
        np.hypot(dx, dy)
    )
)

straight_distance = float(
    np.hypot(
        r["reference_east_m"].iloc[-1],
        r["reference_north_m"].iloc[-1],
    )
)

drift = (
    final_error /
    reference_path *
    100.0
)


# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

out_csv = Path(
    "Data/IOVNBD_controlled_AI_DR.csv"
)

r.to_csv(
    out_csv,
    index=False
)


metrics = {
    "dataset": "IO-VNBD S-S1 + V-S1",

    "benchmark_type":
        "controlled_dr_vbox_initial_speed",

    "outage_start_s":
        OUTAGE_START,

    "outage_end_s":
        OUTAGE_END,

    "samples":
        len(r),

    "reference_path_m":
        reference_path,

    "reference_straight_m":
        straight_distance,

    "position_mae_m":
        mae,

    "position_rmse_m":
        rmse,

    "maximum_error_m":
        max_error,

    "final_error_m":
        final_error,

    "dr_drift_percent":
        drift,

    "ai_model":
        True,

    "map_matching":
        False,

    "vbox_used_for_initial_speed":
        True,
}


import json

metrics_path = Path(
    "Data/IOVNBD_controlled_AI_DR_metrics.json"
)

metrics_path.write_text(
    json.dumps(
        metrics,
        indent=2
    ),
    encoding="utf-8"
)


# ---------------------------------------------------------
# Report
# ---------------------------------------------------------

print()
print("=" * 70)
print("CONTROLLED IO-VNBD AI DR RESULT")
print("=" * 70)

print(
    f"Outage            : "
    f"{OUTAGE_START:.1f} → {OUTAGE_END:.1f} s"
)

print(
    f"Samples            : {len(r)}"
)

print(
    f"Reference path     : "
    f"{reference_path:.2f} m"
)

print(
    f"Reference straight : "
    f"{straight_distance:.2f} m"
)

print(
    f"Position MAE       : "
    f"{mae:.2f} m"
)

print(
    f"Position RMSE      : "
    f"{rmse:.2f} m"
)

print(
    f"Maximum error      : "
    f"{max_error:.2f} m"
)

print(
    f"Final error        : "
    f"{final_error:.2f} m"
)

print(
    f"DR drift           : "
    f"{drift:.2f}%"
)

print()
print("Saved:")
print(out_csv)
print(metrics_path)

print("=" * 70)