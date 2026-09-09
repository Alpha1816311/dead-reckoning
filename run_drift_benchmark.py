import numpy as np
from sensor_engine import RobustIMUPreprocessor, GRAVITY_MPS2

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
    prep = RobustIMUPreprocessor()
    dt = 0.01
    total_steps = 3000
    
    gt_positions = []
    est_positions = []
    
    gt_pos = np.array([0.0, 0.0])
    est_pos = np.array([0.0, 0.0])
    
    est_speed = 0.0
    prev_speed = 0.0
    
    print("--- Running Member 1 Benchmark (<40% Drift Target) ---")
    
    for i in range(total_steps):
        t = i * dt
        
        # Ground Truth Profile
        if t <= 5.0:
            true_accel = 1.0
            true_speed = true_accel * t
        else:
            true_accel = 0.0
            true_speed = 5.0
            
        gt_pos[1] += true_speed * dt
        gt_positions.append(gt_pos.copy())
        
        raw_accel = [true_accel, 0.0, GRAVITY_MPS2]
        raw_gyro = [0.0, 0.0, 0.0]
        
        imu = prep.update(raw_accel, raw_gyro, dt=dt)
        
        # Integrate forward acceleration directly
        fwd_accel = imu.linear_accel_phone[0]
        est_speed += fwd_accel * dt
        est_speed = max(0.0, est_speed)
            
        avg_speed = 0.5 * (prev_speed + est_speed)
        prev_speed = est_speed
        
        est_pos[1] += avg_speed * dt
        est_positions.append(est_pos.copy())
        
    mae, rmse, final_err, drift_pct, total_dist = calculate_metrics(gt_positions, est_positions)
    
    print(f"Total Trajectory Distance : {total_dist:.2f} m")
    print(f"Mean Absolute Error (MAE) : {mae:.4f} m")
    print(f"Root Mean Square Error   : {rmse:.4f} m")
    print(f"Final Position Error     : {final_err:.4f} m")
    print(f"Calculated Drift %       : {drift_pct:.2f}%")
    print("---------------------------------------------")

if __name__ == "__main__":
    run_benchmark()
