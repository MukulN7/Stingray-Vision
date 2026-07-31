# Importing Libraries


import re
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
from collections import deque


# Configuration


CALIB        = r"D:\Malaga Urban Dataset\camera_params_raw_1024x768.txt"
LEFT_VIDEO   = r"D:\Malaga Urban Dataset\malaga-urban-dataset_STEREO_LEFT.avi"
RIGHT_VIDEO  = r"D:\Malaga Urban Dataset\malaga-urban-dataset_STEREO_RIGHT.avi"
GPS_FILE     = r"D:\Malaga Urban Dataset\malaga-urban-dataset-plain-text\malaga-urban-dataset_GPS.txt"
IMAGES_FILE  = r"D:\Malaga Urban Dataset\malaga-urban-dataset-plain-text\malaga-urban-dataset_IMAGES.txt"
MIN_FEATURES = 250
OUTPUT_FILE  = "velocities_v10.txt"
STEREO_SCALE = 0.5   # compute disparity at half-res, scale back up (cuts SGBM cost ~4x)
DETECT_SCALE = 0.5   # run SIFT keypoint detection at half-res, scale points back up
MAX_TIME     = 600.0 # seconds - only process the first 10 minutes of the video


# Load Calibration (MRPT stereo calibration format)


with open(CALIB, "r") as f:
    calib_text = f.read()

left_section = calib_text.split("[CAMERA_PARAMS_LEFT]")[1].split("[CAMERA_PARAMS_RIGHT]")[0]
fx = float(re.search(r"fx\s*=\s*([\d.eE+-]+)", left_section).group(1))
cx = float(re.search(r"cx\s*=\s*([\d.eE+-]+)", left_section).group(1))
cy = float(re.search(r"cy\s*=\s*([\d.eE+-]+)", left_section).group(1))

translation = re.search(r"translation_only\s*=\s*\[([^\]]+)\]", calib_text).group(1).split()
baseline = abs(float(translation[0]))


# Load Ground Truth (GPS Local X/Y/Z, finite-differenced into speed)


gps_data  = np.loadtxt(GPS_FILE, comments="%")
gps_time  = gps_data[:, 0]
gps_x     = gps_data[:, 8]
gps_y     = gps_data[:, 9]
gps_z     = gps_data[:, 10]

def gt_position(t):
    return np.array([np.interp(t, gps_time, gps_x),
                      np.interp(t, gps_time, gps_y),
                      np.interp(t, gps_time, gps_z)])

# Video start time comes from the first left-image timestamp in the dataset,
# so video frame times can be synced onto the GPS timeline
with open(IMAGES_FILE, "r") as f:
    first_left = next(line for line in f if "_left" in line)
t0 = float(first_left.strip().split("_")[2])


# Load SIFT


sift = cv2.SIFT_create(nfeatures=1024)


# Depth + Jacobian + Velocity helpers (Shared by all methods)


def compute_depth(goodpts_new, disparity):
    depths = []
    for pt in goodpts_new:
        x, y = map(int, pt.ravel())
        if 0 <= x < disparity.shape[1] and 0 <= y < disparity.shape[0]:
            d = disparity[y, x]
            if d > 0:
                Z = (fx * baseline) / d
                depths.append(Z if Z < 50.0 else np.nan)
            else:
                depths.append(np.nan)
        else:
            depths.append(np.nan)
    return np.array(depths)

def compute_velocity(goodpts_old, goodpts_new, depths, dt):
    normalized_pts = [((pt[0] - cx) / fx, (pt[1] - cy) / fx) for pt in goodpts_new]
    normalized_pts = np.array(normalized_pts)

    jacobian_rows = []
    for (nx, ny), Z in zip(normalized_pts, depths):
        jacobian_rows.append(np.array([
            [-1/Z,    0,  nx/Z,      nx*ny, -(1+nx**2),  ny],
            [   0, -1/Z,  ny/Z, 1+ny**2,      -nx*ny,  -nx]
        ]))
    J = np.vstack(jacobian_rows)

    img_disp_vec = []
    for old_pt, new_pt in zip(goodpts_old, goodpts_new):
        dx = (new_pt[0] - old_pt[0]) / (dt * fx)
        dy = (new_pt[1] - old_pt[1]) / (dt * fx)
        img_disp_vec.extend([dx, dy])
    img_disp_vec = np.array(img_disp_vec)

    cam_vel = np.linalg.pinv(J) @ img_disp_vec

    # Drop points with large residual after first solve, then re-solve
    # (kills single-bad-correspondence spikes without touching the Jacobian model)
    
    residuals = (J @ cam_vel - img_disp_vec).reshape(-1, 2)
    residual_norms = np.linalg.norm(residuals, axis=1)
    if len(residual_norms) > 6:
        med = np.median(residual_norms)
        mad = np.median(np.abs(residual_norms - med)) + 1e-6
        keep = residual_norms < (med + 3 * mad)
        if keep.sum() >= 6 and keep.sum() < len(keep):
            keep_rows = np.repeat(keep, 2)
            cam_vel = np.linalg.pinv(J[keep_rows]) @ img_disp_vec[keep_rows]

    return cam_vel[0], cam_vel[1], cam_vel[2]


# Open Cameras + Grab First Frame


cap_left  = cv2.VideoCapture(LEFT_VIDEO)
cap_right = cv2.VideoCapture(RIGHT_VIDEO)

# dt must come from the video's true frame rate, not wall-clock time

dt = 1.0 / cap_left.get(cv2.CAP_PROP_FPS)

ret_l, left_raw = cap_left.read()
left_frame_old  = cv2.cvtColor(left_raw, cv2.COLOR_BGR2GRAY)

# SIFT initial (detect at half-res for speed, scale points back to full-res)
small_gray = cv2.resize(left_frame_old, None, fx=DETECT_SCALE, fy=DETECT_SCALE)
kps = sift.detect(small_gray, None)
pts_old_sift = np.array([kp.pt for kp in kps], dtype=np.float32).reshape(-1, 1, 2) / DETECT_SCALE

curr_time = 0.0
gt_pos_prev = gt_position(t0)

# Position Estimation Setup (odometry position starts from the GPS ground truth)
est_pos = gt_position(t0).copy()
est_traj = []
gt_traj  = []
time_log = []


# Rolling-window outlier clamp (median/MAD), applied only to Vz before each filter update
# Not applied to Vx/Vy: those oscillate around zero, making median/MAD clamp incorrectly zero them out
def make_buffer():
    return deque(maxlen=5)

def reject_outlier(buffer, value, k=3.0):
    raw_value = value
    if len(buffer) >= 3:
        med = np.median(buffer)
        mad = np.median(np.abs(np.array(buffer) - med)) + 1e-6
        if abs(value - med) > k * mad:
            value = med
    buffer.append(raw_value)   # always store the raw measurement, not the clamped one
    return value

# Moving Average Filter Setup
def make_ma(window=7):
    return deque(maxlen=window)

def ma_update(buf, value):
    buf.append(value)
    return np.mean(buf)

# Outlier clamp buffer (used for Vz only)
buf_sift_vz = make_buffer()

# Moving Average buffers per axis
ma_sift_vx, ma_sift_vy, ma_sift_vz = make_ma(), make_ma(), make_ma()


# Main Loop


stereo = cv2.StereoSGBM_create(minDisparity=0, numDisparities=32, blockSize=5,
             P1=8*3*5**2, P2=32*3*5**2, disp12MaxDiff=1,
             uniquenessRatio=10, speckleWindowSize=100, speckleRange=32)

with open(OUTPUT_FILE, "w") as log:
    log.write("timestamp,speed_sift_ma,speed_gt,pos_x_est,pos_y_est,pos_z_est,pos_x_gt,pos_y_gt,pos_z_gt\n")

    while True:
        t_frame_start = time.time()

        ret_l, left_raw  = cap_left.read()
        ret_r, right_raw = cap_right.read()
        if not ret_l or not ret_r:
            print("End of video")
            break

        curr_time += dt
        if curr_time > MAX_TIME:
            print("Reached 10 minute limit")
            break

        left_frame_new = cv2.cvtColor(left_raw,  cv2.COLOR_BGR2GRAY)
        left_img       = left_frame_new
        right_img      = cv2.cvtColor(right_raw, cv2.COLOR_BGR2GRAY)

        # Disparity (shared for both methods) - computed at half-res for speed, then rescaled
        t0_stage = time.time()
        small_left  = cv2.resize(left_img,  None, fx=STEREO_SCALE, fy=STEREO_SCALE)
        small_right = cv2.resize(right_img, None, fx=STEREO_SCALE, fy=STEREO_SCALE)
        disparity_small = stereo.compute(small_left, small_right).astype(np.float32) / 16.0
        disparity = cv2.resize(disparity_small, (left_img.shape[1], left_img.shape[0])) / STEREO_SCALE
        t_disparity = time.time() - t0_stage


        # SIFT + Optical Flow


        t0_stage = time.time()
        pts_new_sift, status, _ = cv2.calcOpticalFlowPyrLK(left_frame_old, left_frame_new, pts_old_sift, None)

        sift_old = np.ascontiguousarray(pts_old_sift[status == 1], dtype=np.float32)
        sift_new = np.ascontiguousarray(pts_new_sift[status == 1], dtype=np.float32)


        # RANSAC Outlier Rejection - SIFT


        if len(sift_old) >= 8:
            try:
                F, mask = cv2.findFundamentalMat(sift_old, sift_new, cv2.FM_RANSAC)
                if mask is not None:
                    sift_old = sift_old[mask.ravel() == 1]
                    sift_new = sift_new[mask.ravel() == 1]
            except cv2.error:
                pass

        sift_depths = compute_depth(sift_new, disparity)
        valid       = ~np.isnan(sift_depths)
        sift_old, sift_new, sift_depths = sift_old[valid], sift_new[valid], sift_depths[valid]
        t_sift_track = time.time() - t0_stage

        t0_stage = time.time()
        vx_sift = vy_sift = vz_sift = 0.0
        if len(sift_depths) > 0:
            vx_sift, vy_sift, vz_sift = compute_velocity(sift_old, sift_new, sift_depths, dt)
        t_velocity = time.time() - t0_stage

        # Outlier clamp only on Vz (Vx/Vy oscillate around zero; clamping them zeros out the real signal)
        vz_sift = reject_outlier(buf_sift_vz, vz_sift)

        vx_sift_ma = ma_update(ma_sift_vx, vx_sift);   vy_sift_ma = ma_update(ma_sift_vy, vy_sift);   vz_sift_ma = ma_update(ma_sift_vz, vz_sift)


        # Net speed (magnitude of the 3D velocity, dropping vx/vy/vz breakdown)


        speed_sift_ma = np.linalg.norm([vx_sift_ma, vy_sift_ma, vz_sift_ma])


        # Ground truth speed (finite-differenced GPS Local X/Y/Z at the video's frame time)


        gt_pos_curr = gt_position(t0 + curr_time)
        speed_gt    = np.linalg.norm(gt_pos_curr - gt_pos_prev) / dt
        gt_pos_prev = gt_pos_curr


        # Position Estimation (integrate MA-filtered odometry velocity: x_k+1 = x_k + v_k*dt)

        # Rotate camera-frame velocity (X=right, Y=down, Z=forward) into a Z-up local frame,
        # so "up" integrates against GPS altitude instead of against GPS X/Y.
        # Assumes the vehicle's heading at t0 points along the local X axis (no heading tracking).
        v_fwd   =  vz_sift_ma
        v_right =  vx_sift_ma
        v_up    = -vy_sift_ma

        est_pos[0] += v_fwd   * dt
        est_pos[1] += v_right * dt
        est_pos[2] += v_up    * dt

        est_traj.append(est_pos.copy())
        gt_traj.append(gt_pos_curr.copy())
        time_log.append(curr_time)


        # Log to File


        log.write(f"{curr_time:.6f},{speed_sift_ma:.6f},{speed_gt:.6f},"
                  f"{est_pos[0]:.6f},{est_pos[1]:.6f},{est_pos[2]:.6f},"
                  f"{gt_pos_curr[0]:.6f},{gt_pos_curr[1]:.6f},{gt_pos_curr[2]:.6f}\n")
        log.flush()


        # Live Video Overlay


        display_frame = left_raw.copy()
        overlay_lines = [
            f"SIFT-MA: {speed_sift_ma:.2f} m/s",
            f"GT:      {speed_gt:.2f} m/s",
        ]
        for i, line in enumerate(overlay_lines):
            cv2.putText(display_frame, line, (10, 30 + i * 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Left Camera - Speed Overlay", display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


        # Tracking Maintenance


        left_frame_old = left_frame_new

        # SIFT: carry forward tracked points
        pts_old_sift = sift_new.reshape(-1, 1, 2)
        t_redetect = 0.0
        if len(pts_old_sift) < MIN_FEATURES:
            t0_stage = time.time()
            small_gray = cv2.resize(left_frame_old, None, fx=DETECT_SCALE, fy=DETECT_SCALE)
            kps = sift.detect(small_gray, None)
            pts_old_sift = np.array([kp.pt for kp in kps], dtype=np.float32).reshape(-1, 1, 2) / DETECT_SCALE
            t_redetect = time.time() - t0_stage

        print(f"dt={dt*1000:.1f}ms | SIFT-MA={speed_sift_ma:.3f} | GT={speed_gt:.3f}"
              f" || disparity={t_disparity*1000:.0f}ms track={t_sift_track*1000:.0f}ms"
              f" velocity={t_velocity*1000:.0f}ms redetect={t_redetect*1000:.0f}ms"
              f" total={((time.time()-t_frame_start)*1000):.0f}ms npts={len(sift_new)}")


cap_left.release()
cap_right.release()
cv2.destroyAllWindows()

# Convert trajectories to arrays for later plotting/error analysis
est_traj = np.array(est_traj)
gt_traj  = np.array(gt_traj)

print(f"Done. Speeds and positions saved to {OUTPUT_FILE}")


# Plotting - Top-Down Trajectory (X vs Y)


plt.figure()
plt.plot(gt_traj[:, 0], gt_traj[:, 1], label="GT")
plt.plot(est_traj[:, 0], est_traj[:, 1], label="Estimated")
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.title("Top-Down Trajectory")
plt.axis("equal")
plt.legend()


# Error Comparison - Position Error vs Time


pos_error = np.linalg.norm(est_traj - gt_traj, axis=1)

plt.figure()
plt.plot(time_log, pos_error)
plt.xlabel("Time (s)")
plt.ylabel("Position Error (m)")
plt.title("Position Error vs Time")

plt.show()