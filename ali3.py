import numpy as np
import pandas as pd  # pyright: ignore[reportMissingModuleSource]

from alignment import PhoneOrientation


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "Data/S-S1.csv"
OUTPUT_PATH = "Data/S-S1_orientation_output.csv"


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(DATA_PATH)

print("=" * 70)
print("ORIENTATION + VEHICLE FRAME TEST")
print("=" * 70)

print(f"Loaded {len(df)} samples")


# ============================================================
# COLUMN NAMES
# ============================================================

ACCEL_X = " ACCELEROMETER X (m/s�) "
ACCEL_Y = " ACCELEROMETER Y (m/s�)"
ACCEL_Z = " ACCELEROMETER Z (m/s�)"

GYRO_YAW = " GYROSCOPE Yaw (rad/s)"
GYRO_PITCH = " GYROSCOPE Pitch (rad/s)"
GYRO_ROLL = " GYROSCOPE Roll (rad/s)"

MAG_X = " MAGNETIC FIELD X (μT)"
MAG_Y = " MAGNETIC FIELD Y (μT)"
MAG_Z = " MAGNETIC FIELD Z (μT)"

TIME_COLUMN = " TIME SINCE START (ms)"


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    ACCEL_X,
    ACCEL_Y,
    ACCEL_Z,
    GYRO_YAW,
    GYRO_PITCH,
    GYRO_ROLL,
    MAG_X,
    MAG_Y,
    MAG_Z,
    TIME_COLUMN,
]

missing = [c for c in required_columns if c not in df.columns]

if missing:
    print("\nERROR: Missing columns:")
    for c in missing:
        print("  ", repr(c))

    raise ValueError("CSV does not contain all required sensor columns.")


# ============================================================
# CREATE ORIENTATION ESTIMATOR
# ============================================================

orientation = PhoneOrientation(
    gyro_gain=0.98,
    accel_gain=0.05,
    mag_gain=0.02,
)


# ============================================================
# PHONE -> VEHICLE MOUNTING
# ============================================================
#
# TEMPORARY CONFIGURATION
#
# This assumes:
#
#       Phone +Y = vehicle forward
#       Phone +Z = vehicle up
#
# If your phone is mounted differently, we will change these
# vectors after testing the physical mounting.
#
# Vehicle frame:
#       X = forward
#       Y = left
#       Z = up
#

orientation.set_vehicle_alignment(
    forward_phone=[0.0, 1.0, 0.0],
    up_phone=[0.0, 0.0, 1.0],
)


# ============================================================
# OUTPUT ARRAYS
# ============================================================

vehicle_accel_x = []
vehicle_accel_y = []
vehicle_accel_z = []

vehicle_gyro_x = []
vehicle_gyro_y = []
vehicle_gyro_z = []

orientation_w = []
orientation_x = []
orientation_y = []
orientation_z = []

initialized = []


# ============================================================
# PROCESS IMU DATA
# ============================================================

previous_time = None

for index, row in df.iterrows():

    # --------------------------------------------------------
    # Read timestamp
    # --------------------------------------------------------

    current_time = float(row[TIME_COLUMN]) / 1000.0

    if previous_time is None:
        dt = 0.01
    else:
        dt = current_time - previous_time

        # Protect against invalid timestamps.
        if dt <= 0 or dt > 1.0:
            dt = 0.01

    previous_time = current_time


    # --------------------------------------------------------
    # Accelerometer
    # --------------------------------------------------------

    accel = np.array([
        float(row[ACCEL_X]),
        float(row[ACCEL_Y]),
        float(row[ACCEL_Z]),
    ])


    # --------------------------------------------------------
    # Gyroscope
    # --------------------------------------------------------

    gyro = np.array([
        float(row[GYRO_YAW]),
        float(row[GYRO_PITCH]),
        float(row[GYRO_ROLL]),
    ])


    # --------------------------------------------------------
    # Magnetometer
    # --------------------------------------------------------

    magnetometer = np.array([
        float(row[MAG_X]),
        float(row[MAG_Y]),
        float(row[MAG_Z]),
    ])


    # --------------------------------------------------------
    # Orientation update
    # --------------------------------------------------------

    result = orientation.update(
        accel=accel,
        gyro=gyro,
        mag=magnetometer,
        dt=dt,
    )


    # --------------------------------------------------------
    # Vehicle-frame acceleration
    # --------------------------------------------------------

    a_vehicle = result["accel"]

    vehicle_accel_x.append(a_vehicle[0])
    vehicle_accel_y.append(a_vehicle[1])
    vehicle_accel_z.append(a_vehicle[2])


    # --------------------------------------------------------
    # Vehicle-frame angular velocity
    # --------------------------------------------------------

    g_vehicle = result["gyro"]

    vehicle_gyro_x.append(g_vehicle[0])
    vehicle_gyro_y.append(g_vehicle[1])
    vehicle_gyro_z.append(g_vehicle[2])


    # --------------------------------------------------------
    # Estimated quaternion
    # --------------------------------------------------------

    q = result["orientation"]

    orientation_w.append(q[0])
    orientation_x.append(q[1])
    orientation_y.append(q[2])
    orientation_z.append(q[3])

    initialized.append(result["initialized"])


# ============================================================
# ADD RESULTS TO DATAFRAME
# ============================================================

df["VEHICLE ACCEL X (m/s2)"] = vehicle_accel_x
df["VEHICLE ACCEL Y (m/s2)"] = vehicle_accel_y
df["VEHICLE ACCEL Z (m/s2)"] = vehicle_accel_z

df["VEHICLE GYRO X (rad/s)"] = vehicle_gyro_x
df["VEHICLE GYRO Y (rad/s)"] = vehicle_gyro_y
df["VEHICLE GYRO Z (rad/s)"] = vehicle_gyro_z

df["EST Q W"] = orientation_w
df["EST Q X"] = orientation_x
df["EST Q Y"] = orientation_y
df["EST Q Z"] = orientation_z

df["ORIENTATION INITIALIZED"] = initialized


# ============================================================
# SAVE
# ============================================================

df.to_csv(
    OUTPUT_PATH,
    index=False,
)


# ============================================================
# SUMMARY
# ============================================================

print("\nProcessing complete.")

print(f"Input : {DATA_PATH}")
print(f"Output: {OUTPUT_PATH}")

print("\nFirst vehicle-frame acceleration samples:")

print(
    df[
        [
            "VEHICLE ACCEL X (m/s2)",
            "VEHICLE ACCEL Y (m/s2)",
            "VEHICLE ACCEL Z (m/s2)",
        ]
    ].head(10)
)

print("\nFirst vehicle-frame gyro samples:")

print(
    df[
        [
            "VEHICLE GYRO X (rad/s)",
            "VEHICLE GYRO Y (rad/s)",
            "VEHICLE GYRO Z (rad/s)",
        ]
    ].head(10)
)

print("\n" + "=" * 70)
print("TEST FINISHED")
print("=" * 70)