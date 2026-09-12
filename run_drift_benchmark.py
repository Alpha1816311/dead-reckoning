"""
Dead Reckoning Drift Benchmark using sensor_processing pipeline.

Synthetic test: vehicle accelerates for 5s then holds 5 m/s for 25s.
Simulates a 30-second GNSS blackout scenario.
"""
import numpy as np
from sensor_processing import RobustIMUPreprocessor, GRAVITY_MPS2


def calculate_metrics(gt_pos, est_pos):
    gt = np.array(gt_pos)
    est = np.array(est_pos)

    errors = np.linalg.norm(gt - est, axis=1)
    mae = float(np.mean(errors))
    rmse = float(np.sqrt(np.mean(errors ** 2)))

    step_dists = np.linalg.norm(np.diff(gt, axis=0), axis=1)
    total_dist = float(np.sum(step_dists))

    final_err = float(errors[-1])
    drift_pct = (final_err / max(1e-6, total_dist)) * 100.0
    return mae, rmse, final_err, drift_pct, total_dist


def run_benchmark():
    dt = 0.01
    total_steps = 3000   # 30 seconds

    gt_positions = []
    est_positions = []

    gt_pos = np.array([0.0, 0.0])
    est_pos = np.array([0.0, 0.0])
    est_speed = 0.0

    print("--- Dead Reckoning Drift Benchmark (Synthetic 30s Blackout) ---")

    # --- Phase 1: warm-up preprocessor at constant speed (no position tracking)
    # Feed 5s of stationary data so gravity estimate converges, then
    # feed 5s of constant-speed driving (pure gravity, no linear accel).
    prep = RobustIMUPreprocessor()

    # 2s stationary warm-up
    for _ in range(200):
        prep.update([0.0, 0.0, GRAVITY_MPS2], [0.0, 0.0, 0.0], dt=dt)

    # 3s cruising at 5 m/s (no acceleration, just gravity on Z)
    for _ in range(300):
        prep.update([0.0, 0.0, GRAVITY_MPS2], [0.0, 0.0, 0.0], dt=dt)

    est_speed = 5.0   # pre-outage speed known from GNSS

    # --- Phase 2: GNSS blackout starts — integrate only IMU
    prev_speed = est_speed

    for i in range(total_steps):
        t = i * dt

        # Ground truth: 5s acceleration phase then cruise
        if t <= 5.0:
            true_accel_fwd = 0.5          # gentle accel
            true_speed_gt = 5.0 + true_accel_fwd * t
        else:
            true_accel_fwd = 0.0
            true_speed_gt = 7.5           # 5 + 0.5*5

        gt_pos = gt_pos.copy()
        gt_pos[1] += true_speed_gt * dt
        gt_positions.append(gt_pos.copy())

        # Simulated raw IMU: forward accel on X axis, gravity on Z
        raw_accel = [true_accel_fwd, 0.0, GRAVITY_MPS2]
        raw_gyro = [0.0, 0.0, 0.0]

        imu = prep.update(raw_accel, raw_gyro, dt=dt)

        # Forward accel from filtered linear — X axis in phone frame
        fwd_accel_est = float(imu.filtered_linear_accel_phone[0])
        est_speed += fwd_accel_est * dt
        est_speed = max(0.0, min(est_speed, 40.0))

        avg_speed = 0.5 * (prev_speed + est_speed)
        prev_speed = est_speed

        est_pos = est_pos.copy()
        est_pos[1] += avg_speed * dt
        est_positions.append(est_pos.copy())

    mae, rmse, final_err, drift_pct, total_dist = calculate_metrics(
        gt_positions, est_positions
    )

    print(f"Scenario               : 30s GNSS blackout, initial speed 5 m/s")
    print(f"Total Distance (GT)    : {total_dist:.2f} m")
    print(f"Mean Absolute Error    : {mae:.4f} m")
    print(f"RMSE                   : {rmse:.4f} m")
    print(f"Final Position Error   : {final_err:.4f} m")
    print(f"Drift %                : {drift_pct:.2f}%")
    print("---------------------------------------------------------------")
    if drift_pct < 10.0:
        print("RESULT: PASS - within 10% drift target")
    elif drift_pct < 20.0:
        print("RESULT: MARGINAL - below 20% drift")
    else:
        print("RESULT: EXCEEDS target - further tuning needed")


if __name__ == "__main__":
    run_benchmark()
