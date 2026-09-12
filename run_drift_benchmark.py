"""Synthetic GNSS-blackout benchmark using NavigationEngine's yaw calibration."""
import math

import numpy as np

from navigation_engine import GyroYawCalibrator


def calculate_metrics(gt_pos, est_pos):
    gt, est = np.asarray(gt_pos), np.asarray(est_pos)
    errors = np.linalg.norm(gt - est, axis=1)
    distance = float(np.sum(np.linalg.norm(np.diff(gt, axis=0), axis=1)))
    final_error = float(errors[-1])
    return float(np.mean(errors)), float(np.sqrt(np.mean(errors ** 2))), final_error, 100.0 * final_error / max(distance, 1e-6), distance


def _run(axis, sign):
    """Drive a 30 s route with two turns; physical yaw is phone gyro Y."""
    dt, speed = 0.05, 6.0
    truth = np.zeros(2)
    estimate = np.zeros(2)
    heading = 0.0
    estimated_heading = 0.0
    gt, est = [], []
    for step in range(600):
        t = step * dt
        # Two 45-degree manoeuvres make incorrect default-axis use observable.
        yaw_rate = math.radians(15.0 if 8.0 <= t < 11.0 else (-15.0 if 18.0 <= t < 21.0 else 0.0))
        heading += yaw_rate * dt
        gyro_phone = np.array([0.01, -yaw_rate, 0.005])
        estimated_heading += sign * gyro_phone[axis] * dt
        truth += speed * dt * np.array([math.sin(heading), math.cos(heading)])
        estimate += speed * dt * np.array([math.sin(estimated_heading), math.cos(estimated_heading)])
        gt.append(truth.copy())
        est.append(estimate.copy())
    return calculate_metrics(gt, est)


def _calibrate_from_gnss_course():
    """Create pre-outage GNSS course turns without any heading injection."""
    calibrator = GyroYawCalibrator(min_distance_m=3.0, min_speed_mps=2.0)
    position = np.zeros(2)
    calibrator.observe_gnss(position, 6.0)
    headings = [0.0, 45.0, 90.0, 135.0]
    for heading in headings:
        for _ in range(30):
            calibrator.add_imu(np.array([0.01, -math.radians(15.0), 0.005]), 0.1)
        position += 5.0 * np.array([math.sin(math.radians(heading)), math.cos(math.radians(heading))])
        calibrator.observe_gnss(position, 6.0)
    return calibrator


def run_benchmark():
    # Baseline mirrors the former default assumption: phone gyro Z is yaw.
    baseline = _run(axis=2, sign=-1.0)
    calibrator = _calibrate_from_gnss_course()
    corrected = _run(axis=calibrator.axis, sign=calibrator.sign)
    print("--- Synthetic 30s GNSS Blackout / Yaw-axis Consistency ---")
    print(f"Baseline default axis: final={baseline[2]:.2f}m drift={baseline[3]:.2f}%")
    print(f"Calibrated axis {calibrator.axis}, sign {calibrator.sign:+.0f}, confidence {calibrator.confidence:.2f}")
    print(f"Corrected shared path: final={corrected[2]:.2f}m drift={corrected[3]:.2f}%")
    return {"baseline": baseline, "corrected": corrected, "axis": calibrator.axis}


if __name__ == "__main__":
    run_benchmark()
