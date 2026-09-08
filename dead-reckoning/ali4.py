import numpy as np
import pandas as pd


DATA_PATH = "Data/S-S1_orientation_output.csv"

df = pd.read_csv(DATA_PATH)

# ------------------------------------------------------------
# Original accelerometer
# ------------------------------------------------------------

accel = df[
    [
        " ACCELEROMETER X (m/s�) ",
        " ACCELEROMETER Y (m/s�)",
        " ACCELEROMETER Z (m/s�)",
    ]
].to_numpy(dtype=float)


# ------------------------------------------------------------
# Gravity
# ------------------------------------------------------------

gravity = df[
    [
        " GRAVITY X (m/s�)",
        " GRAVITY Y (m/s�)",
        " GRAVITY Z (m/s�)",
    ]
].to_numpy(dtype=float)


# ------------------------------------------------------------
# Linear acceleration
#
# raw acceleration - gravity
# ------------------------------------------------------------

linear_accel = accel - gravity


# ------------------------------------------------------------
# Vehicle-frame acceleration
# ------------------------------------------------------------

vehicle_accel = df[
    [
        "VEHICLE ACCEL X (m/s2)",
        "VEHICLE ACCEL Y (m/s2)",
        "VEHICLE ACCEL Z (m/s2)",
    ]
].to_numpy(dtype=float)


# ------------------------------------------------------------
# Statistics
# ------------------------------------------------------------

print("=" * 70)
print("ACCELEROMETER / GRAVITY VALIDATION")
print("=" * 70)

print("\nNumber of samples:", len(df))

print("\nRaw accelerometer magnitude:")

raw_mag = np.linalg.norm(accel, axis=1)

print("  Mean :", np.mean(raw_mag))
print("  Std  :", np.std(raw_mag))
print("  Min  :", np.min(raw_mag))
print("  Max  :", np.max(raw_mag))


print("\nGravity magnitude:")

gravity_mag = np.linalg.norm(gravity, axis=1)

print("  Mean :", np.mean(gravity_mag))
print("  Std  :", np.std(gravity_mag))
print("  Min  :", np.min(gravity_mag))
print("  Max  :", np.max(gravity_mag))


print("\nLinear acceleration magnitude:")

linear_mag = np.linalg.norm(linear_accel, axis=1)

print("  Mean :", np.mean(linear_mag))
print("  Std  :", np.std(linear_mag))
print("  Min  :", np.min(linear_mag))
print("  Max  :", np.max(linear_mag))


# ------------------------------------------------------------
# First few samples
# ------------------------------------------------------------

print("\nFirst 10 linear-acceleration samples:")

print(
    pd.DataFrame(
        linear_accel[:10],
        columns=[
            "LINEAR AX",
            "LINEAR AY",
            "LINEAR AZ",
        ],
    )
)


# ------------------------------------------------------------
# Vehicle acceleration after gravity removal
#
# We currently have the orientation rotation matrix implicitly
# through the estimated vehicle acceleration. For this first
# diagnostic, compare the raw vehicle Z against gravity.
# ------------------------------------------------------------

print("\nFirst 10 raw vehicle-frame acceleration samples:")

print(
    pd.DataFrame(
        vehicle_accel[:10],
        columns=[
            "VEHICLE AX",
            "VEHICLE AY",
            "VEHICLE AZ",
        ],
    )
)


# ------------------------------------------------------------
# Check how much gravity exists in the raw acceleration
# ------------------------------------------------------------

gravity_fraction = np.mean(
    np.abs(vehicle_accel[:, 2])
)

print("\nMean absolute vehicle Z acceleration:", gravity_fraction)

print("\n" + "=" * 70)
print("VALIDATION COMPLETE")
print("=" * 70)