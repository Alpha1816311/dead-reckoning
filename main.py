import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ============================================================
# IDR - AI/ML INTELLIGENT DEAD RECKONING SYSTEM
# Complete Prototype
# ============================================================

DATA_PATH = "Data/S-S1.csv"

print("\n" + "=" * 70)
print("        INTELLIGENT DEAD RECKONING SYSTEM")
print("=" * 70)


# ============================================================
# PHASE 1 - LOAD DATA
# ============================================================

if not os.path.exists(DATA_PATH):
    print(f"\nERROR: Dataset not found: {DATA_PATH}")
    raise SystemExit

df = pd.read_csv(
    DATA_PATH,
    encoding="cp1252",
    engine="python"
)

# Clean column names
df.columns = (
    df.columns
    .astype(str)
    .str.replace("\ufeff", "", regex=False)
    .str.strip()
)

print("\n[PHASE 1] DATASET LOADED")
print("-" * 50)
print("Rows   :", len(df))
print("Columns:", len(df.columns))


# ============================================================
# COLUMN DETECTION
# ============================================================

def find_column(words):
    """
    Finds the first column whose normalized name
    contains all required words.
    """
    for col in df.columns:
        name = str(col).lower().replace("_", " ").strip()

        if all(word.lower() in name for word in words):
            return col

    return None


# Time
TIME_COL = find_column(["time"])

# Accelerometer
ACC_X = find_column(["accelerometer", "x"])
ACC_Y = find_column(["accelerometer", "y"])
ACC_Z = find_column(["accelerometer", "z"])

# Gravity
GRAV_X = find_column(["gravity", "x"])
GRAV_Y = find_column(["gravity", "y"])
GRAV_Z = find_column(["gravity", "z"])

# Gyroscope
GYRO_X = find_column(["gyroscope", "yaw"])
GYRO_Y = find_column(["gyroscope", "pitch"])
GYRO_Z = find_column(["gyroscope", "roll"])

# Orientation
YAW_COL = find_column(["orientation", "yaw"])
PITCH_COL = find_column(["orientation", "pitch"])
ROLL_COL = find_column(["orientation", "roll"])

# GPS
LAT_COL = find_column(["gps", "latitude"])
LON_COL = find_column(["gps", "longitude"])
GPS_SPEED_COL = find_column(["gps", "speed"])


print("\n[DETECTED SENSOR COLUMNS]")
print("-" * 50)

detected = {
    "TIME": TIME_COL,
    "ACC X": ACC_X,
    "ACC Y": ACC_Y,
    "ACC Z": ACC_Z,
    "GRAVITY X": GRAV_X,
    "GRAVITY Y": GRAV_Y,
    "GRAVITY Z": GRAV_Z,
    "GYRO X": GYRO_X,
    "GYRO Y": GYRO_Y,
    "GYRO Z": GYRO_Z,
    "YAW": YAW_COL,
    "PITCH": PITCH_COL,
    "ROLL": ROLL_COL,
    "GPS LAT": LAT_COL,
    "GPS LON": LON_COL,
    "GPS SPEED": GPS_SPEED_COL
}

for name, col in detected.items():
    print(f"{name:15} -> {col}")


# ============================================================
# PHASE 2 - TIME PREPROCESSING
# ============================================================

print("\n" + "=" * 70)
print("[PHASE 2] TIME & SENSOR PREPROCESSING")
print("=" * 70)

if TIME_COL is None:
    print("WARNING: Time column not detected.")
    df["_TIME_SECONDS"] = np.arange(len(df)) / 100.0
else:
    time_values = pd.to_numeric(
        df[TIME_COL],
        errors="coerce"
    )

    # Most smartphone datasets store milliseconds
    time_values = time_values.bfill().ffill()

    time_seconds = time_values / 1000.0

    # Start time from zero
    df["_TIME_SECONDS"] = time_seconds - time_seconds.iloc[0]

    # Remove duplicate/non-increasing timestamps
    df = df.loc[
        df["_TIME_SECONDS"].diff().fillna(1) > 0
    ].copy()

    df.reset_index(drop=True, inplace=True)


# Calculate timestep
dt = df["_TIME_SECONDS"].diff()

dt = dt.replace([np.inf, -np.inf], np.nan)

# Median timestep is more robust than assuming fixed frequency
median_dt = dt.median()

if pd.isna(median_dt) or median_dt <= 0:
    median_dt = 0.01

dt = dt.fillna(median_dt)

# Protect against abnormal timestamp gaps
dt = dt.clip(lower=0.001, upper=1.0)

df["_DT"] = dt


print("Samples       :", len(df))
print("Median dt (s) :", round(median_dt, 6))
print("Estimated Hz  :", round(1 / median_dt, 2))


# ============================================================
# NUMERIC SENSOR CONVERSION
# ============================================================

sensor_columns = [
    ACC_X, ACC_Y, ACC_Z,
    GRAV_X, GRAV_Y, GRAV_Z,
    GYRO_X, GYRO_Y, GYRO_Z,
    YAW_COL, PITCH_COL, ROLL_COL,
    LAT_COL, LON_COL, GPS_SPEED_COL
]

for col in sensor_columns:
    if col is not None:
        df[col] = pd.to_numeric(df[col], errors="coerce")


# Interpolate sensor values
for col in sensor_columns:
    if col is not None:
        df[col] = (
            df[col]
            .interpolate(limit_direction="both")
            .fillna(0)
        )


# ============================================================
# PHASE 3 - LINEAR ACCELERATION
# ============================================================

print("\n[PHASE 3] SENSOR FUSION")
print("-" * 50)

required_acc = [ACC_X, ACC_Y, ACC_Z]

if all(c is not None for c in required_acc):

    ax = df[ACC_X].to_numpy(dtype=float)
    ay = df[ACC_Y].to_numpy(dtype=float)
    az = df[ACC_Z].to_numpy(dtype=float)

    # Remove gravity if gravity channels exist
    if all(c is not None for c in [GRAV_X, GRAV_Y, GRAV_Z]):

        gx = df[GRAV_X].to_numpy(dtype=float)
        gy = df[GRAV_Y].to_numpy(dtype=float)
        gz = df[GRAV_Z].to_numpy(dtype=float)

        linear_x = ax - gx
        linear_y = ay - gy
        linear_z = az - gz

    else:

        linear_x = ax
        linear_y = ay
        linear_z = az

else:

    print("WARNING: Accelerometer columns incomplete.")

    linear_x = np.zeros(len(df))
    linear_y = np.zeros(len(df))
    linear_z = np.zeros(len(df))


# Smooth high-frequency sensor noise
def smooth_signal(x, window=5):
    return (
        pd.Series(x)
        .rolling(window, center=True, min_periods=1)
        .mean()
        .to_numpy()
    )


linear_x = smooth_signal(linear_x)
linear_y = smooth_signal(linear_y)
linear_z = smooth_signal(linear_z)

df["LINEAR_ACC_X"] = linear_x
df["LINEAR_ACC_Y"] = linear_y
df["LINEAR_ACC_Z"] = linear_z

df["ACC_MAG"] = np.sqrt(
    linear_x ** 2 +
    linear_y ** 2 +
    linear_z ** 2
)


# ============================================================
# PHASE 4 - ORIENTATION / WORLD FRAME
# ============================================================

print("\n[PHASE 4] ORIENTATION TRANSFORMATION")
print("-" * 50)

if all(c is not None for c in [YAW_COL, PITCH_COL, ROLL_COL]):

    yaw = np.radians(df[YAW_COL].to_numpy(dtype=float))
    pitch = np.radians(df[PITCH_COL].to_numpy(dtype=float))
    roll = np.radians(df[ROLL_COL].to_numpy(dtype=float))

else:

    print("WARNING: Orientation incomplete.")
    print("Using zero orientation.")

    yaw = np.zeros(len(df))
    pitch = np.zeros(len(df))
    roll = np.zeros(len(df))


# Rotate body-frame acceleration into horizontal/world frame.
#
# For dead reckoning we primarily need horizontal acceleration.
# The equations below combine yaw/pitch/roll into a rotation.

cos_y = np.cos(yaw)
sin_y = np.sin(yaw)

cos_p = np.cos(pitch)
sin_p = np.sin(pitch)

cos_r = np.cos(roll)
sin_r = np.sin(roll)

# Rotation matrix Rz * Ry * Rx
r11 = cos_y * cos_p
r12 = cos_y * sin_p * sin_r - sin_y * cos_r

r21 = sin_y * cos_p
r22 = sin_y * sin_p * sin_r + cos_y * cos_r

r31 = -sin_p
r32 = cos_p * sin_r

world_x = (
    r11 * linear_x +
    r12 * linear_y
)

world_y = (
    r21 * linear_x +
    r22 * linear_y
)

world_z = (
    r31 * linear_x +
    r32 * linear_y +
    cos_p * cos_r * linear_z
)

df["WORLD_ACC_X"] = world_x
df["WORLD_ACC_Y"] = world_y
df["WORLD_ACC_Z"] = world_z


# ============================================================
# PHASE 5 - GPS AVAILABILITY
# ============================================================

print("\n[PHASE 5] GNSS AVAILABILITY DETECTION")
print("-" * 50)

if LAT_COL is not None and LON_COL is not None:

    lat = df[LAT_COL].to_numpy(dtype=float)
    lon = df[LON_COL].to_numpy(dtype=float)

    gps_valid = (
        np.isfinite(lat) &
        np.isfinite(lon) &
        (np.abs(lat) > 0.000001) &
        (np.abs(lon) > 0.000001)
    )

else:

    lat = np.zeros(len(df))
    lon = np.zeros(len(df))
    gps_valid = np.zeros(len(df), dtype=bool)


df["GPS_AVAILABLE"] = gps_valid.astype(int)

gps_count = int(gps_valid.sum())

print("GPS-valid samples:", gps_count)
print("GPS unavailable  :", len(df) - gps_count)


# ============================================================
# GPS -> LOCAL X/Y
# ============================================================

def gps_to_local_xy(lat, lon):

    valid = np.isfinite(lat) & np.isfinite(lon)

    if valid.sum() == 0:
        return np.zeros(len(lat)), np.zeros(len(lat))

    lat0 = lat[valid][0]
    lon0 = lon[valid][0]

    earth_radius = 6378137.0

    x = (
        np.radians(lon - lon0)
        * earth_radius
        * np.cos(np.radians(lat0))
    )

    y = (
        np.radians(lat - lat0)
        * earth_radius
    )

    return x, y


gps_x, gps_y = gps_to_local_xy(lat, lon)

df["GPS_X"] = gps_x
df["GPS_Y"] = gps_y


# ============================================================
# PHASE 6 - ML VELOCITY ESTIMATION
# ============================================================

print("\n[PHASE 6] AI/ML VELOCITY ESTIMATION")
print("-" * 50)

# We use Random Forest when GPS speed exists.
# GPS-derived speed becomes the training target.
#
# During GNSS outage, the trained model estimates speed
# from inertial sensor features.

from sklearn.ensemble import RandomForestRegressor


# Feature engineering
df["ACC_MAG_SMOOTH"] = (
    df["ACC_MAG"]
    .rolling(10, min_periods=1)
    .mean()
)

df["GYRO_MAG"] = 0.0

if all(c is not None for c in [GYRO_X, GYRO_Y, GYRO_Z]):

    gyro_x = df[GYRO_X].to_numpy(dtype=float)
    gyro_y = df[GYRO_Y].to_numpy(dtype=float)
    gyro_z = df[GYRO_Z].to_numpy(dtype=float)

    df["GYRO_MAG"] = np.sqrt(
        gyro_x ** 2 +
        gyro_y ** 2 +
        gyro_z ** 2
    )


FEATURES = [
    "LINEAR_ACC_X",
    "LINEAR_ACC_Y",
    "LINEAR_ACC_Z",
    "ACC_MAG",
    "ACC_MAG_SMOOTH",
    "GYRO_MAG",
    "WORLD_ACC_X",
    "WORLD_ACC_Y"
]


# Calculate GPS speed if explicit GPS speed isn't available
if GPS_SPEED_COL is not None:

    gps_speed = df[GPS_SPEED_COL].to_numpy(dtype=float)

else:

    gps_dx = np.diff(gps_x, prepend=gps_x[0])
    gps_dy = np.diff(gps_y, prepend=gps_y[0])

    gps_speed = (
        np.sqrt(gps_dx ** 2 + gps_dy ** 2)
        / df["_DT"].to_numpy()
    )


df["GPS_SPEED_TARGET"] = gps_speed

# Valid ML samples
train_mask = (
    gps_valid &
    np.isfinite(gps_speed) &
    (gps_speed >= 0) &
    (gps_speed < 100)
)

if train_mask.sum() >= 50:

    X_train = (
        df.loc[train_mask, FEATURES]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    y_train = (
        df.loc[train_mask, "GPS_SPEED_TARGET"]
        .to_numpy()
    )

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=12,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    X_all = (
        df[FEATURES]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    predicted_speed = model.predict(X_all)

    print("ML model       : Random Forest")
    print("Training rows  :", len(X_train))
    print("Features       :", len(FEATURES))

else:

    print("Not enough GPS samples for ML training.")
    print("Using inertial speed estimation.")

    predicted_speed = np.zeros(len(df))


predicted_speed = np.clip(
    predicted_speed,
    0,
    100
)

df["ML_SPEED"] = predicted_speed

# ============================================================
# GNSS OUTAGE CONTROL
# ============================================================

print("\n[GNSS OUTAGE CONFIGURATION]")
print("-" * 50)

# Dataset is sampled at 10 Hz.
# Therefore:
# 10 samples = 1 second
#
# We simulate a 30-second GNSS blackout.

OUTAGE_START_SEC = 30.0
OUTAGE_DURATION_SEC = 30.0
OUTAGE_END_SEC = OUTAGE_START_SEC + OUTAGE_DURATION_SEC

actual_gnss_available = gps_valid.copy()

simulated_gnss_available = actual_gnss_available.copy()

outage_mask = (
    (df["_TIME_SECONDS"] >= OUTAGE_START_SEC) &
    (df["_TIME_SECONDS"] < OUTAGE_END_SEC)
)

# Only simulate outage where real GPS exists
simulated_gnss_available[outage_mask] = False

df["GNSS_SIMULATED_AVAILABLE"] = (
    simulated_gnss_available.astype(int)
)

print("Outage start :", OUTAGE_START_SEC, "seconds")
print("Outage end   :", OUTAGE_END_SEC, "seconds")
print(
    "Outage samples:",
    int(outage_mask.sum())
)

# ============================================================
# PHASE 7 - DEAD RECKONING
# ============================================================

print("\n[PHASE 7] DEAD RECKONING")
print("-" * 50)

n = len(df)

dr_x = np.zeros(n)
dr_y = np.zeros(n)

velocity_x = np.zeros(n)
velocity_y = np.zeros(n)


for i in range(1, n):

    delta_t = float(df["_DT"].iloc[i])

    ax_world = float(df["WORLD_ACC_X"].iloc[i])
    ay_world = float(df["WORLD_ACC_Y"].iloc[i])

    # Inertial velocity prediction
    predicted_vx = (
        velocity_x[i - 1] +
        ax_world * delta_t
    )

    predicted_vy = (
        velocity_y[i - 1] +
        ay_world * delta_t
    )

    # ML speed provides a constraint on velocity magnitude
    speed_est = float(df["ML_SPEED"].iloc[i])

    current_speed = np.sqrt(
        predicted_vx ** 2 +
        predicted_vy ** 2
    )

    if current_speed > 0.05 and speed_est > 0:

        scale = speed_est / current_speed

        predicted_vx *= scale
        predicted_vy *= scale

    # GNSS correction when available
    if simulated_gnss_available[i]:

        gps_position_x = gps_x[i]
        gps_position_y = gps_y[i]

        # Soft correction rather than hard replacement.
        # This makes the trajectory smoother.

        dr_x[i] = (
            0.85 * gps_position_x +
            0.15 * (
                dr_x[i - 1] +
                predicted_vx * delta_t
            )
        )

        dr_y[i] = (
            0.85 * gps_position_y +
            0.15 * (
                dr_y[i - 1] +
                predicted_vy * delta_t
            )
        )

        velocity_x[i] = predicted_vx
        velocity_y[i] = predicted_vy

    else:

        # GNSS DENIED:
        # Pure dead reckoning prediction

        velocity_x[i] = predicted_vx
        velocity_y[i] = predicted_vy

        dr_x[i] = (
            dr_x[i - 1] +
            velocity_x[i] * delta_t
        )

        dr_y[i] = (
            dr_y[i - 1] +
            velocity_y[i] * delta_t
        )


df["DR_X"] = dr_x
df["DR_Y"] = dr_y

df["DR_SPEED"] = np.sqrt(
    velocity_x ** 2 +
    velocity_y ** 2
)


# ============================================================
# PHASE 8 - GNSS OUTAGE SIMULATION
# ============================================================

print("\n[PHASE 8] GNSS OUTAGE SIMULATION")
print("-" * 50)

outage_indices = df.index[outage_mask].tolist()

if len(outage_indices) > 0:

    outage_start_index = outage_indices[0]
    outage_end_index = outage_indices[-1]

    outage_start_time = float(
        df.loc[outage_start_index, "_TIME_SECONDS"]
    )

    outage_end_time = float(
        df.loc[outage_end_index, "_TIME_SECONDS"]
    )

    print("GNSS outage       : SIMULATED")
    print(
        "Start time        :",
        round(outage_start_time, 2),
        "s"
    )

    print(
        "End time          :",
        round(outage_end_time, 2),
        "s"
    )

    print(
        "Duration          :",
        round(outage_end_time - outage_start_time, 2),
        "s"
    )

    print(
        "GNSS-denied samples:",
        len(outage_indices)
    )

else:

    print("WARNING: No outage samples generated.")


# Detect transition into and out of simulated outage

gnss_changes = (
    df["GNSS_SIMULATED_AVAILABLE"]
    .diff()
    .fillna(0)
)

outage_starts = df.index[
    gnss_changes == -1
].tolist()

outage_recoveries = df.index[
    gnss_changes == 1
].tolist()

print(
    "Outage transitions detected:",
    len(outage_starts)
)

print(
    "GNSS recoveries detected:",
    len(outage_recoveries)
)


# ============================================================
# PHASE 9 - ERROR ANALYSIS
# ============================================================

print("\n[PHASE 9] POSITION ERROR")
print("-" * 50)

position_error = np.sqrt(
    (df["DR_X"] - df["GPS_X"]) ** 2 +
    (df["DR_Y"] - df["GPS_Y"]) ** 2
)

df["POSITION_ERROR_M"] = position_error

# ============================================================
# GNSS OUTAGE PERFORMANCE ANALYSIS
# ============================================================

print("\n[GNSS OUTAGE PERFORMANCE]")
print("-" * 50)

# Evaluate only during simulated GNSS-denied period

outage_error = position_error[outage_mask]

if len(outage_error) > 0:

    outage_mae = float(np.mean(outage_error))

    outage_rmse = float(
        np.sqrt(np.mean(outage_error ** 2))
    )

    outage_max = float(
        np.max(outage_error)
    )

if len(outage_error) > 0:
    outage_final = float(outage_error.iloc[-1])
else:
    outage_final = 0.0

    print(
        "Outage samples      :",
        len(outage_error)
    )

    print(
        "Outage MAE          :",
        round(outage_mae, 3),
        "m"
    )

    print(
        "Outage RMSE         :",
        round(outage_rmse, 3),
        "m"
    )

    print(
        "Maximum drift       :",
        round(outage_max, 3),
        "m"
    )

    print(
        "Drift at recoverey    :",
        round(outage_final, 3),
        "m"
    )

# ============================================================
# OUTAGE TRAVEL DISTANCE
# ============================================================

outage_indices = df.index[outage_mask].tolist()

if len(outage_indices) > 1:

    start_idx = outage_indices[0]
    end_idx = outage_indices[-1]

    gps_start_x = float(df.loc[start_idx, "GPS_X"])
    gps_start_y = float(df.loc[start_idx, "GPS_Y"])

    gps_end_x = float(df.loc[end_idx, "GPS_X"])
    gps_end_y = float(df.loc[end_idx, "GPS_Y"])

    travelled_distance = np.sqrt(
        (gps_end_x - gps_start_x) ** 2 +
        (gps_end_y - gps_start_y) ** 2
    )

    print(
        "Reference displacement:",
        round(travelled_distance, 3),
        "m"
    )

    if travelled_distance > 0:

        drift_percentage = (
            outage_final /
            travelled_distance
        ) * 100

        print(
            "Drift percentage   :",
            round(drift_percentage, 2),
            "%"
        )

    else:

        print(
            "Drift percentage   : Cannot calculate"
        )

valid_error = position_error[gps_valid]

if len(valid_error) > 0:

    mae = float(np.mean(valid_error))
    rmse = float(np.sqrt(np.mean(valid_error ** 2)))
    maximum = float(np.max(valid_error))

    print("MAE  :", round(mae, 3), "m")
    print("RMSE :", round(rmse, 3), "m")
    print("MAX  :", round(maximum, 3), "m")

else:

    print("GPS reference unavailable for error calculation.")


# ============================================================
# PHASE 10 - SAVE RESULTS
# ============================================================

print("\n[PHASE 10] SAVING RESULTS")
print("-" * 50)

OUTPUT_PATH = "Data/IDR_results.csv"

df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("Results saved to:")
print(OUTPUT_PATH)


# ============================================================
# FINAL SYSTEM REPORT
# ============================================================

print("\n" + "=" * 70)
print("                 IDR SYSTEM REPORT")
print("=" * 70)

print("Input samples       :", len(df))
print("Sensor columns      :", len(df.columns))
print("GPS samples         :", gps_count)
print(
    "GNSS-denied samples :",
    int((df["GNSS_SIMULATED_AVAILABLE"] == 0).sum())
)

print("\nSensor Fusion       : ACTIVE")
print("ML Speed Model      : RANDOM FOREST")
print("Dead Reckoning      : ACTIVE")
print("GNSS Correction     : ACTIVE")
print("Trajectory Output   : ACTIVE")

print("\nOutput file:")
print(OUTPUT_PATH)

print("\nSTATUS: IDR PIPELINE COMPLETE")
print("=" * 70)