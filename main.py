import os
import json
import pickle
import warnings
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from gnss_ins_fusion import fuse_gnss_ins
from alignment import phone_to_vehicle

warnings.filterwarnings("ignore")

# ============================================================
# IDR - INTEGRATED INTELLIGENT DEAD RECKONING PIPELINE
# ============================================================

DATA_PATH = "Data/S-S1.csv"
OUTPUT_PATH = "Data/IDR_results.csv"
METRICS_PATH = "Data/IDR_metrics.json"
MODEL_PATH = "models/speed_model.pkl"
TRAJECTORY_PLOT = "Data/IDR_trajectory_integrated.png"
OUTAGE_PLOT = "Data/IDR_GNSS_outage_integrated.png"

# Optional offline road map. If absent, map matching is skipped safely.
MAP_PATH = "Data/roads.geojson"

# Phone mounting convention used when an explicit mounting calibration is
# not available. Identity mounting is deliberately conservative and keeps
# the existing dataset orientation unchanged.
PHONE_FORWARD = np.array([1.0, 0.0, 0.0])
PHONE_UP = np.array([0.0, 0.0, 1.0])

OUTAGE_START_SEC = 30.0
OUTAGE_DURATION_SEC = 30.0
OUTAGE_END_SEC = OUTAGE_START_SEC + OUTAGE_DURATION_SEC

print("\n" + "=" * 70)
print("        INTELLIGENT DEAD RECKONING SYSTEM - INTEGRATED")
print("=" * 70)

# ============================================================
# HELPERS
# ============================================================

def find_column(df, words):
    for col in df.columns:
        name = str(col).lower().replace("_", " ").strip()
        if all(word.lower() in name for word in words):
            return col
    return None


def smooth_signal(x, window=5):
    return (
        pd.Series(x)
        .rolling(window, center=True, min_periods=1)
        .mean()
        .to_numpy()
    )


def gps_to_local_xy(lat, lon):
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    valid = np.isfinite(lat) & np.isfinite(lon)

    x = np.zeros(len(lat), dtype=float)
    y = np.zeros(len(lat), dtype=float)

    if valid.sum() == 0:
        return x, y

    first = np.flatnonzero(valid)[0]
    lat0 = lat[first]
    lon0 = lon[first]
    earth_radius = 6378137.0

    x = np.radians(lon - lon0) * earth_radius * np.cos(np.radians(lat0))
    y = np.radians(lat - lat0) * earth_radius
    return x, y


def local_xy_to_gps(x, y, lat0, lon0):
    earth_radius = 6378137.0
    lat = lat0 + np.degrees(y / earth_radius)
    lon = lon0 + np.degrees(x / (earth_radius * np.cos(np.radians(lat0))))
    return lat, lon


def calculate_heading_from_xy(x, y):
    """Heading in degrees, 0=N, 90=E, using local EN coordinates."""
    dx = np.diff(x, prepend=x[0])
    dy = np.diff(y, prepend=y[0])
    heading = np.degrees(np.arctan2(dx, dy)) % 360.0
    speed = np.hypot(dx, dy)
    heading[speed < 0.05] = np.nan
    return pd.Series(heading).interpolate(limit_direction="both").fillna(0).to_numpy()


def calculate_distance_xy(x1, y1, x2, y2):
    return np.hypot(np.asarray(x1) - np.asarray(x2), np.asarray(y1) - np.asarray(y2))


def apply_nhc(velocity_x, velocity_y, heading_deg, min_speed=0.5):
    """Apply a simple non-holonomic constraint: lateral velocity = 0."""
    vx = np.asarray(velocity_x, dtype=float).copy()
    vy = np.asarray(velocity_y, dtype=float).copy()
    h = np.radians(np.asarray(heading_deg, dtype=float))

    forward_x = np.sin(h)
    forward_y = np.cos(h)

    forward_speed = vx * forward_x + vy * forward_y
    forward_speed = np.maximum(forward_speed, 0.0)

    mask = np.hypot(vx, vy) >= min_speed
    vx[mask] = forward_speed[mask] * forward_x[mask]
    vy[mask] = forward_speed[mask] * forward_y[mask]
    return vx, vy


def save_plots(df, outage_mask):
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 7))
        plt.plot(df["GPS_X"], df["GPS_Y"], label="GNSS reference")
        plt.plot(df["DR_X"], df["DR_Y"], label="Fused DR")
        if "MAP_X" in df.columns:
            valid_map = np.isfinite(df["MAP_X"]) & np.isfinite(df["MAP_Y"])
            if valid_map.any():
                plt.plot(df.loc[valid_map, "MAP_X"], df.loc[valid_map, "MAP_Y"], label="Map matched")
        plt.xlabel("East / X (m)")
        plt.ylabel("North / Y (m)")
        plt.title("IDR Integrated Trajectory")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(TRAJECTORY_PLOT, dpi=160)
        plt.close()

        plt.figure(figsize=(10, 5))
        plt.plot(df["_TIME_SECONDS"], df["POSITION_ERROR_M"], label="Position error")
        if outage_mask.any():
            plt.axvspan(OUTAGE_START_SEC, OUTAGE_END_SEC, alpha=0.2, label="GNSS outage")
        plt.xlabel("Time (s)")
        plt.ylabel("Error (m)")
        plt.title("IDR Position Error")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(OUTAGE_PLOT, dpi=160)
        plt.close()
        return True
    except Exception as exc:
        print("WARNING: Plot generation skipped:", exc)
        return False


# ============================================================
# PHASE 1 - LOAD DATA
# ============================================================

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

os.makedirs("Data", exist_ok=True)
os.makedirs("models", exist_ok=True)

df = pd.read_csv(DATA_PATH, encoding="cp1252", engine="python")
df.columns = df.columns.astype(str).str.replace("\ufeff", "", regex=False).str.strip()

print("\n[PHASE 1] DATASET LOADED")
print("-" * 50)
print("Rows   :", len(df))
print("Columns:", len(df.columns))

TIME_COL = find_column(df, ["time"])
ACC_X = find_column(df, ["accelerometer", "x"])
ACC_Y = find_column(df, ["accelerometer", "y"])
ACC_Z = find_column(df, ["accelerometer", "z"])
GRAV_X = find_column(df, ["gravity", "x"])
GRAV_Y = find_column(df, ["gravity", "y"])
GRAV_Z = find_column(df, ["gravity", "z"])
GYRO_X = find_column(df, ["gyroscope", "yaw"])
GYRO_Y = find_column(df, ["gyroscope", "pitch"])
GYRO_Z = find_column(df, ["gyroscope", "roll"])
YAW_COL = find_column(df, ["orientation", "yaw"])
PITCH_COL = find_column(df, ["orientation", "pitch"])
ROLL_COL = find_column(df, ["orientation", "roll"])
LAT_COL = find_column(df, ["gps", "latitude"])
LON_COL = find_column(df, ["gps", "longitude"])
GPS_SPEED_COL = find_column(df, ["gps", "speed"])

print("\n[DETECTED SENSOR COLUMNS]")
print("-" * 50)
for name, col in {
    "TIME": TIME_COL, "ACC X": ACC_X, "ACC Y": ACC_Y, "ACC Z": ACC_Z,
    "GRAVITY X": GRAV_X, "GRAVITY Y": GRAV_Y, "GRAVITY Z": GRAV_Z,
    "GYRO X": GYRO_X, "GYRO Y": GYRO_Y, "GYRO Z": GYRO_Z,
    "YAW": YAW_COL, "PITCH": PITCH_COL, "ROLL": ROLL_COL,
    "GPS LAT": LAT_COL, "GPS LON": LON_COL, "GPS SPEED": GPS_SPEED_COL,
}.items():
    print(f"{name:15} -> {col}")

# ============================================================
# PHASE 2 - TIME & SENSOR PREPROCESSING
# ============================================================

print("\n[PHASE 2] TIME & SENSOR PREPROCESSING")
print("=" * 70)

if TIME_COL is None:
    df["_TIME_SECONDS"] = np.arange(len(df), dtype=float) / 100.0
else:
    time_values = pd.to_numeric(df[TIME_COL], errors="coerce").bfill().ffill()
    df["_TIME_SECONDS"] = time_values / 1000.0
    df["_TIME_SECONDS"] -= df["_TIME_SECONDS"].iloc[0]
    df = df.loc[df["_TIME_SECONDS"].diff().fillna(1) > 0].copy()
    df.reset_index(drop=True, inplace=True)

raw_dt = df["_TIME_SECONDS"].diff()
median_dt = raw_dt.replace([np.inf, -np.inf], np.nan).median()
if pd.isna(median_dt) or median_dt <= 0:
    median_dt = 0.01

df["_DT"] = raw_dt.fillna(median_dt).clip(lower=0.001, upper=1.0)
print("Samples       :", len(df))
print("Median dt (s) :", round(float(median_dt), 6))
print("Estimated Hz  :", round(1.0 / float(median_dt), 2))

sensor_columns = [ACC_X, ACC_Y, ACC_Z, GRAV_X, GRAV_Y, GRAV_Z,
                  GYRO_X, GYRO_Y, GYRO_Z, YAW_COL, PITCH_COL, ROLL_COL,
                  LAT_COL, LON_COL, GPS_SPEED_COL]
for col in sensor_columns:
    if col is not None:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].interpolate(limit_direction="both").fillna(0)

# ============================================================
# PHASE 3 - GRAVITY REMOVAL + FILTERING
# ============================================================

print("\n[PHASE 3] SENSOR FUSION / GRAVITY REMOVAL")
print("-" * 50)

if all(c is not None for c in [ACC_X, ACC_Y, ACC_Z]):
    ax = df[ACC_X].to_numpy(float)
    ay = df[ACC_Y].to_numpy(float)
    az = df[ACC_Z].to_numpy(float)
else:
    ax = np.zeros(len(df)); ay = np.zeros(len(df)); az = np.zeros(len(df))

if all(c is not None for c in [GRAV_X, GRAV_Y, GRAV_Z]):
    gx = df[GRAV_X].to_numpy(float)
    gy = df[GRAV_Y].to_numpy(float)
    gz = df[GRAV_Z].to_numpy(float)
    linear_x = ax - gx
    linear_y = ay - gy
    linear_z = az - gz
else:
    linear_x, linear_y, linear_z = ax, ay, az

linear_x = smooth_signal(linear_x)
linear_y = smooth_signal(linear_y)
linear_z = smooth_signal(linear_z)

# ============================================================
# PHASE 4 - ALIGNMENT + WORLD TRANSFORMATION
# ============================================================

print("\n[PHASE 4] ALIGNMENT + ORIENTATION / WORLD TRANSFORMATION")
print("-" * 50)

# Existing alignment module is integrated here. Identity mounting is used
# because this dataset does not contain a reliable phone-mount calibration.
vehicle_linear = np.zeros((len(df), 3), dtype=float)
for i in range(len(df)):
    vehicle_linear[i] = phone_to_vehicle(
        [linear_x[i], linear_y[i], linear_z[i]],
        PHONE_FORWARD,
        PHONE_UP,
    )

df["VEHICLE_ACC_X"] = vehicle_linear[:, 0]
df["VEHICLE_ACC_Y"] = vehicle_linear[:, 1]
df["VEHICLE_ACC_Z"] = vehicle_linear[:, 2]

df["ACC_MAG"] = np.linalg.norm(vehicle_linear, axis=1)

if all(c is not None for c in [YAW_COL, PITCH_COL, ROLL_COL]):
    yaw = np.radians(df[YAW_COL].to_numpy(float))
    pitch = np.radians(df[PITCH_COL].to_numpy(float))
    roll = np.radians(df[ROLL_COL].to_numpy(float))
else:
    yaw = np.zeros(len(df)); pitch = np.zeros(len(df)); roll = np.zeros(len(df))

cy, sy = np.cos(yaw), np.sin(yaw)
cp, sp = np.cos(pitch), np.sin(pitch)
cr, sr = np.cos(roll), np.sin(roll)

# R = Rz(yaw) Ry(pitch) Rx(roll)
r11 = cy * cp
r12 = cy * sp * sr - sy * cr
r21 = sy * cp
r22 = sy * sp * sr + cy * cr
r31 = -sp
r32 = cp * sr
r33 = cp * cr

world_x = r11 * vehicle_linear[:, 0] + r12 * vehicle_linear[:, 1] + (cy * sp * cr + sy * sr) * vehicle_linear[:, 2]
world_y = r21 * vehicle_linear[:, 0] + r22 * vehicle_linear[:, 1] + (sy * sp * cr - cy * sr) * vehicle_linear[:, 2]
world_z = r31 * vehicle_linear[:, 0] + r32 * vehicle_linear[:, 1] + r33 * vehicle_linear[:, 2]

df["WORLD_ACC_X"] = world_x
df["WORLD_ACC_Y"] = world_y
df["WORLD_ACC_Z"] = world_z

# ============================================================
# PHASE 5 - GNSS AVAILABILITY + LOCAL FRAME
# ============================================================

print("\n[PHASE 5] GNSS AVAILABILITY DETECTION")
print("-" * 50)

if LAT_COL is not None and LON_COL is not None:
    lat = df[LAT_COL].to_numpy(float)
    lon = df[LON_COL].to_numpy(float)
    gps_valid = np.isfinite(lat) & np.isfinite(lon) & (np.abs(lat) > 1e-6) & (np.abs(lon) > 1e-6)
else:
    lat = np.zeros(len(df)); lon = np.zeros(len(df)); gps_valid = np.zeros(len(df), dtype=bool)

df["GPS_AVAILABLE"] = gps_valid.astype(int)
print("GPS-valid samples:", int(gps_valid.sum()))
print("GPS unavailable  :", int((~gps_valid).sum()))

gps_x, gps_y = gps_to_local_xy(lat, lon)
df["GPS_X"] = gps_x
df["GPS_Y"] = gps_y

gps_heading = calculate_heading_from_xy(gps_x, gps_y)
df["GNSS_HEADING_DEG"] = gps_heading

# ============================================================
# PHASE 6 - PROPER ML SPEED TRAIN / TEST
# ============================================================

print("\n[PHASE 6] AI/ML VELOCITY ESTIMATION")
print("-" * 50)

if all(c is not None for c in [GYRO_X, GYRO_Y, GYRO_Z]):
    gyro_x = df[GYRO_X].to_numpy(float)
    gyro_y = df[GYRO_Y].to_numpy(float)
    gyro_z = df[GYRO_Z].to_numpy(float)
    gyro_mag = np.sqrt(gyro_x**2 + gyro_y**2 + gyro_z**2)
else:
    gyro_x = np.zeros(len(df)); gyro_y = np.zeros(len(df)); gyro_z = np.zeros(len(df)); gyro_mag = np.zeros(len(df))

df["ACC_MAG_SMOOTH"] = pd.Series(df["ACC_MAG"]).rolling(10, min_periods=1).mean()
df["GYRO_MAG"] = gyro_mag

FEATURES = [
    "VEHICLE_ACC_X", "VEHICLE_ACC_Y", "VEHICLE_ACC_Z",
    "ACC_MAG", "ACC_MAG_SMOOTH", "GYRO_MAG",
    "WORLD_ACC_X", "WORLD_ACC_Y",
]

if GPS_SPEED_COL is not None:
    gps_speed = df[GPS_SPEED_COL].to_numpy(float) / 3.6
else:
    gps_speed = np.hypot(np.diff(gps_x, prepend=gps_x[0]), np.diff(gps_y, prepend=gps_y[0])) / df["_DT"].to_numpy()

gps_speed = np.clip(np.nan_to_num(gps_speed, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 100.0)
df["GPS_SPEED_TARGET"] = gps_speed

ml_valid = gps_valid & np.isfinite(gps_speed) & (gps_speed >= 0) & (gps_speed < 100)
valid_idx = np.flatnonzero(ml_valid)

predicted_speed = np.zeros(len(df), dtype=float)
model = None

test_mae = test_rmse = test_r2 = float("nan")

if len(valid_idx) >= 100:
    # Sequential split avoids training on future samples.
    n_valid = len(valid_idx)
    train_end = max(1, int(0.70 * n_valid))
    val_end = max(train_end + 1, int(0.85 * n_valid))
    train_idx = valid_idx[:train_end]
    val_idx = valid_idx[train_end:val_end]
    test_idx = valid_idx[val_end:]

    X = df[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(float)
    y = gps_speed

    model = RandomForestRegressor(
        n_estimators=150,
        max_depth=14,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X[train_idx], y[train_idx])

    if len(val_idx):
        _ = model.predict(X[val_idx])

    if len(test_idx):
        y_test = y[test_idx]
        y_pred = model.predict(X[test_idx])
        test_mae = float(mean_absolute_error(y_test, y_pred))
        test_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        test_r2 = float(r2_score(y_test, y_pred))

    predicted_speed = model.predict(X)
    predicted_speed = np.clip(predicted_speed, 0.0, 100.0)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    print("ML model       : Random Forest")
    print("Train rows     :", len(train_idx))
    print("Validation rows:", len(val_idx))
    print("Test rows      :", len(test_idx))
    print("Test MAE       :", round(test_mae, 4), "m/s")
    print("Test RMSE      :", round(test_rmse, 4), "m/s")
    print("Test R2        :", round(test_r2, 4))
    print("Model exported :", MODEL_PATH)
else:
    print("WARNING: Not enough valid GNSS samples for ML training.")
    print("Using GNSS speed where available and zero otherwise.")
    predicted_speed[gps_valid] = gps_speed[gps_valid]

df["ML_SPEED"] = predicted_speed

# ============================================================
# PHASE 7 - GNSS OUTAGE SIMULATION
# ============================================================

print("\n[PHASE 7] GNSS OUTAGE SIMULATION")
print("-" * 50)

actual_gnss_available = gps_valid.copy()
simulated_gnss_available = actual_gnss_available.copy()
outage_mask = (
    (df["_TIME_SECONDS"] >= OUTAGE_START_SEC) &
    (df["_TIME_SECONDS"] < OUTAGE_END_SEC)
)
simulated_gnss_available[outage_mask] = False
df["GNSS_SIMULATED_AVAILABLE"] = simulated_gnss_available.astype(int)
print("Outage start   :", OUTAGE_START_SEC, "s")
print("Outage end     :", OUTAGE_END_SEC, "s")
print("Outage samples :", int(outage_mask.sum()))

# ============================================================
# PHASE 8 - DEAD RECKONING + NHC
# ============================================================

print("\n[PHASE 8] DEAD RECKONING + NHC")
print("-" * 50)

n = len(df)
dt_arr = df["_DT"].to_numpy(float)

velocity_x = np.zeros(n, dtype=float)
velocity_y = np.zeros(n, dtype=float)
ins_position = np.zeros((n, 2), dtype=float)

# Keep a heading through outages. GNSS course is used when available.
heading = np.asarray(gps_heading, dtype=float)
heading = pd.Series(heading).replace([np.inf, -np.inf], np.nan).interpolate(limit_direction="both").fillna(0).to_numpy()

for i in range(1, n):
    dt_i = float(dt_arr[i])
    ax_w = float(df["WORLD_ACC_X"].iloc[i])
    ay_w = float(df["WORLD_ACC_Y"].iloc[i])

    vx_pred = velocity_x[i - 1] + ax_w * dt_i
    vy_pred = velocity_y[i - 1] + ay_w * dt_i

    speed_est = float(df["ML_SPEED"].iloc[i])
    raw_speed = np.hypot(vx_pred, vy_pred)

    if speed_est > 0.05:
        if raw_speed > 0.05:
            scale = speed_est / raw_speed
            vx_pred *= scale
            vy_pred *= scale
        else:
            h = np.radians(heading[i])
            vx_pred = speed_est * np.sin(h)
            vy_pred = speed_est * np.cos(h)

    velocity_x[i] = vx_pred
    velocity_y[i] = vy_pred

# NHC applied to the complete inertial velocity sequence.
velocity_x, velocity_y = apply_nhc(velocity_x, velocity_y, heading)

for i in range(1, n):
    ins_position[i] = ins_position[i - 1] + np.array([velocity_x[i], velocity_y[i]]) * dt_arr[i]

# Start INS at GNSS origin so the trajectory has a physically meaningful frame.
if gps_valid.any():
    first_valid = np.flatnonzero(gps_valid)[0]
    ins_position += np.array([gps_x[first_valid], gps_y[first_valid]])

# ============================================================
# PHASE 9 - GNSS + INS FUSION
# ============================================================

print("\n[PHASE 9] GNSS + INS FUSION")
print("-" * 50)

gnss_position = np.column_stack((gps_x, gps_y)).astype(float)
gnss_position[~simulated_gnss_available] = np.nan

fused_position = fuse_gnss_ins(
    gnss_position=gnss_position,
    ins_position=ins_position,
    ins_velocity=np.column_stack((velocity_x, velocity_y)),
    gnss_available=simulated_gnss_available,
    dt=dt_arr,
    gnss_weight=0.85,
)

df["DR_X"] = fused_position[:, 0]
df["DR_Y"] = fused_position[:, 1]
df["DR_SPEED"] = np.hypot(velocity_x, velocity_y)
df["NHC_LATERAL_VELOCITY"] = 0.0

# ============================================================
# PHASE 10 - OPTIONAL MAP MATCHING
# ============================================================

print("\n[PHASE 10] MAP MATCHING")
print("-" * 50)

map_matched = False
if os.path.exists(MAP_PATH):
    try:
        from map_matching import load_roads_from_geojson, VehicleMapMatcher, VehicleState
        roads = load_roads_from_geojson(MAP_PATH)
        if roads:
            matcher = VehicleMapMatcher(roads, search_radius=50.0, heading_threshold=75.0, use_nonholonomic=True)
            first_valid = np.flatnonzero(gps_valid)[0]
            map_lat_arr, map_lon_arr = local_xy_to_gps(
                df["DR_X"].to_numpy(),
                df["DR_Y"].to_numpy(),
                lat[first_valid],
                lon[first_valid],
            )
            map_states = [
                VehicleState(
                    latitude=float(la),
                    longitude=float(lo),
                    heading=float(h),
                    speed=float(s),
                )
                for la, lo, h, s in zip(
                    map_lat_arr,
                    map_lon_arr,
                    heading,
                    df["DR_SPEED"].to_numpy(),
                )
            ]
            matches = matcher.match_trajectory(map_states)
            map_lat = np.array([m.latitude for m in matches], dtype=float)
            map_lon = np.array([m.longitude for m in matches], dtype=float)
            map_x, map_y = gps_to_local_xy(map_lat, map_lon)
            df["MAP_LAT"] = map_lat
            df["MAP_LON"] = map_lon
            df["MAP_X"] = map_x
            df["MAP_Y"] = map_y
            df["MAP_LATERAL_ERROR_M"] = [m.lateral_error for m in matches]
            df["MAP_CONFIDENCE"] = [m.confidence for m in matches]
            map_matched = True
            print("Road segments  :", len(roads))
            print("Matched samples:", len(matches))
        else:
            print("No LineString roads found; map matching skipped.")
    except Exception as exc:
        print("WARNING: Map matching skipped:", exc)
else:
    print("Map file not found:", MAP_PATH)
    print("Add Data/roads.geojson later to activate offline map matching.")

# ============================================================
# PHASE 11 - ERROR METRICS
# ============================================================

print("\n[PHASE 11] ERROR METRICS")
print("-" * 50)

position_error = calculate_distance_xy(
    df["DR_X"], df["DR_Y"], df["GPS_X"], df["GPS_Y"]
)
df["POSITION_ERROR_M"] = position_error

outage_error = position_error[outage_mask]
if len(outage_error):
    outage_mae = float(np.mean(outage_error))
    outage_rmse = float(np.sqrt(np.mean(outage_error ** 2)))
    outage_max = float(np.max(outage_error))
    outage_final = float(outage_error[-1])
else:
    outage_mae = outage_rmse = outage_max = outage_final = float("nan")

valid_error = position_error[gps_valid]
if len(valid_error):
    overall_mae = float(np.mean(valid_error))
    overall_rmse = float(np.sqrt(np.mean(valid_error ** 2)))
    overall_max = float(np.max(valid_error))
else:
    overall_mae = overall_rmse = overall_max = float("nan")

if outage_mask.sum() > 1:
    outage_idx = np.flatnonzero(outage_mask)
    ref_displacement = float(np.hypot(
        gps_x[outage_idx[-1]] - gps_x[outage_idx[0]],
        gps_y[outage_idx[-1]] - gps_y[outage_idx[0]],
    ))
else:
    ref_displacement = 0.0

drift_percentage = (outage_final / ref_displacement * 100.0) if ref_displacement > 0 else float("nan")

print("Outage samples      :", int(outage_mask.sum()))
print("Outage MAE          :", round(outage_mae, 3), "m")
print("Outage RMSE         :", round(outage_rmse, 3), "m")
print("Maximum drift       :", round(outage_max, 3), "m")
print("Drift at recovery   :", round(outage_final, 3), "m")
print("Reference displacement:", round(ref_displacement, 3), "m")
print("Drift percentage    :", round(drift_percentage, 2), "%")
print("Overall MAE         :", round(overall_mae, 3), "m")
print("Overall RMSE        :", round(overall_rmse, 3), "m")
print("Overall MAX         :", round(overall_max, 3), "m")

metrics = {
    "samples": int(n),
    "sample_rate_hz": float(1.0 / median_dt),
    "outage_samples": int(outage_mask.sum()),
    "outage_mae_m": outage_mae,
    "outage_rmse_m": outage_rmse,
    "maximum_drift_m": outage_max,
    "drift_at_recovery_m": outage_final,
    "reference_displacement_m": ref_displacement,
    "drift_percentage": drift_percentage,
    "overall_mae_m": overall_mae,
    "overall_rmse_m": overall_rmse,
    "overall_max_m": overall_max,
    "ml_test_mae_mps": test_mae,
    "ml_test_rmse_mps": test_rmse,
    "ml_test_r2": test_r2,
    "map_matching_active": bool(map_matched),
}

# ============================================================
# PHASE 12 - SAVE
# ============================================================

print("\n[PHASE 12] SAVING RESULTS")
print("-" * 50)

df.to_csv(OUTPUT_PATH, index=False)
with open(METRICS_PATH, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2, allow_nan=True)

plot_ok = save_plots(df, outage_mask)

print("Results saved  :", OUTPUT_PATH)
print("Metrics saved  :", METRICS_PATH)
print("Plots generated:", plot_ok)

# ============================================================
# FINAL REPORT
# ============================================================

print("\n" + "=" * 70)
print("                 IDR SYSTEM REPORT")
print("=" * 70)
print("Dataset loading       : ACTIVE")
print("Sensor preprocessing  : ACTIVE")
print("Gravity removal       : ACTIVE")
print("Alignment             : INTEGRATED")
print("World transformation  : ACTIVE")
print("AI speed model        : RANDOM FOREST")
print("ML train/test         : ACTIVE")
print("Model export          :", MODEL_PATH)
print("Dead reckoning        : ACTIVE")
print("NHC                   : ACTIVE")
print("GNSS outage simulation: ACTIVE")
print("GNSS + INS fusion     : ACTIVE")
print("Map matching          :", "ACTIVE" if map_matched else "READY / MAP FILE REQUIRED")
print("Error metrics         : ACTIVE")
print("Result plots          :", "ACTIVE" if plot_ok else "SKIPPED")
print("Output file           :", OUTPUT_PATH)
print("STATUS                : INTEGRATED PIPELINE COMPLETE")
print("=" * 70)
