import numpy as np

# Ensure the local calibration modules are importable when this file is run
# directly or analyzed from a different working directory.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from alignment import PhoneOrientation


orientation = PhoneOrientation()

# Example phone mounting:
# Phone +Y = vehicle forward
# Phone +Z = vehicle up
orientation.set_vehicle_alignment(
    forward_phone=[0, 1, 0],
    up_phone=[0, 0, 1],
)


# Simulated stationary phone
accel = np.array([0.0, 0.0, 9.81])
gyro = np.array([0.0, 0.0, 0.0])
mag = np.array([1.0, 0.0, 0.0])

dt = 0.01


result = orientation.update(
    accel=accel,
    gyro=gyro,
    mag=mag,
    dt=dt,
)


print("\nOrientation test")
print("----------------")
print("Initialized:", result["initialized"])
print("Vehicle calibrated:", result["vehicle_calibrated"])
print("Orientation quaternion:", result["orientation"])
print("Vehicle acceleration:", result["accel"])
print("Vehicle gyro:", result["gyro"])