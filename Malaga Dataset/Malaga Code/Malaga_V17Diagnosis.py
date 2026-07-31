# Importing Libraries

import re
import numpy as np
import cv2
import time
from collections import deque


# Configuration

CALIB        = r"D:\Stingray Robotics\Malaga\Malaga Urban Dataset\camera_params_raw_1024x768.txt"
LEFT_VIDEO   = r"D:\Stingray Robotics\Malaga\Malaga Urban Dataset\malaga-urban-dataset_STEREO_LEFT.avi"
RIGHT_VIDEO  = r"D:\Stingray Robotics\Malaga\Malaga Urban Dataset\malaga-urban-dataset_STEREO_RIGHT.avi"
GPS_FILE     = r"D:\Stingray Robotics\Malaga\Malaga Urban Dataset\malaga-urban-dataset-plain-text\malaga-urban-dataset_GPS.txt"
IMAGES_FILE  = r"D:\Stingray Robotics\Malaga\Malaga Urban Dataset\malaga-urban-dataset-plain-text\malaga-urban-dataset_IMAGES.txt"
MIN_FEATURES = 250
STEREO_SCALE = 0.5
DETECT_SCALE = 0.5
SKIP_SECONDS = 20 * 60   # skip first 20 minutes (vehicle stationary)


# Load Calibration (MRPT stereo calibration format)

with open(CALIB, "r") as f:
    calib_text = f.read()

left_section = calib_text.split("[CAMERA_PARAMS_LEFT]")[1].split("[CAMERA_PARAMS_RIGHT]")[0]
fx = float(re.search(r"fx\s*=\s*([\d.eE+-]+)", left_section).group(1))
cx = float(re.search(r"cx\s*=\s*([\d.eE+-]+)", left_section).group(1))
cy = float(re.search(r"cy\s*=\s*([\d.eE+-]+)", left_section).group(1))

translation = re.search(r"translation_only\s*=\s*\[([^\]]+)\]", calib_text).group(1).split()
baseline = abs(float(translation[0]))

# ---- SUGGESTION #2 CHECK: calibration / baseline sanity ----
print("=" * 60)
print("[CHECK] Calibration values loaded from CALIB file:")
print(f"        fx = {fx:.4f}")
print(f"        cx = {cx:.4f}")
print(f"        cy = {cy:.4f}")
print(f"        baseline = {baseline:.6f} m")
if baseline < 0.05 or baseline > 0.5:
    print("        [WARNING] baseline looks unusual for a stereo rig "
          "(typical range ~0.1-0.2 m). Double check parsing / units.")
else:
    print("        baseline looks within a typical stereo-rig range.")
print("=" * 60)


# Load Ground Truth (GPS Local X/Y/Z)

gps_data  = np.loadtxt(GPS_FILE, comments="%")
gps_time  = gps_data[:, 0]
gps_x     = gps_data[:, 8]
gps_y     = gps_data[:, 9]
gps_z     = gps_data[:, 10]

def gt_position(t):
    return np.array([np.interp(t, gps_time, gps_x),
                      np.interp(t, gps_time, gps_y),
                      np.interp(t, gps_time, gps_z)])

# ---- SUGGESTION #1 CHECK: build real per-frame timestamps from IMAGES_FILE ----
left_timestamps = []
with open(IMAGES_FILE, "r") as f:
    for line in f:
        if "_left" in line:
            left_timestamps.append(float(line.strip().split("_")[2]))
left_timestamps = np.array(left_timestamps)
t0 = left_timestamps[0]

real_dts = np.diff(left_timestamps)
print(f"[CHECK] Real per-frame dt from IMAGES_FILE timestamps: "
      f"mean={real_dts.mean()*1000:.2f}ms, std={real_dts.std()*1000:.2f}ms, "
      f"min={real_dts.min()*1000:.2f}ms, max={real_dts.max()*1000:.2f}ms")


# SIFT setup

sift = cv2.SIFT_create(nfeatures=1024)


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


# Open Cameras

cap_left  = cv2.VideoCapture(LEFT_VIDEO)
cap_right = cv2.VideoCapture(RIGHT_VIDEO)

dt = 1.0 / cap_left.get(cv2.CAP_PROP_FPS)
print(f"[CHECK] Fixed dt from cap.get(CAP_PROP_FPS): {dt*1000:.2f}ms "
      f"(FPS reported = {cap_left.get(cv2.CAP_PROP_FPS):.3f})")

# ---- Skip first SKIP_SECONDS of video (vehicle stationary) ----
skip_frames = int(SKIP_SECONDS / dt)
skip_frames = min(skip_frames, len(left_timestamps) - 2)
print(f"[INFO] Skipping first {SKIP_SECONDS/60:.1f} minutes "
      f"(~{skip_frames} frames) of video...")

for _ in range(skip_frames):
    cap_left.read()
    cap_right.read()

frame_idx = skip_frames   # index into left_timestamps for real-dt lookups

ret_l, left_raw = cap_left.read()
left_frame_old  = cv2.cvtColor(left_raw, cv2.COLOR_BGR2GRAY)

small_gray = cv2.resize(left_frame_old, None, fx=DETECT_SCALE, fy=DETECT_SCALE)
kps = sift.detect(small_gray, None)
pts_old_sift = np.array([kp.pt for kp in kps], dtype=np.float32).reshape(-1, 1, 2) / DETECT_SCALE

curr_time   = skip_frames * dt
gt_pos_prev = gt_position(t0 + curr_time)


# Rolling-window outlier clamp (Vz only)

def make_buffer():
    return deque(maxlen=5)

def reject_outlier(buffer, value, k=3.0):
    raw_value = value
    if len(buffer) >= 3:
        med = np.median(buffer)
        mad = np.median(np.abs(np.array(buffer) - med)) + 1e-6
        if abs(value - med) > k * mad:
            value = med
    buffer.append(raw_value)
    return value

def make_ma(window=7):
    return deque(maxlen=window)

def ma_update(buf, value):
    buf.append(value)
    return np.mean(buf)

def make_lp(alpha=0.2):
    return {"x": None, "alpha": alpha}

def lp_update(lp, value):
    if lp["x"] is None:
        lp["x"] = value
    else:
        lp["x"] = lp["alpha"] * value + (1 - lp["alpha"]) * lp["x"]
    return lp["x"]

buf_sift_vz = make_buffer()
ma_sift_vx, ma_sift_vy, ma_sift_vz = make_ma(), make_ma(), make_ma()
lp_sift_vx, lp_sift_vy, lp_sift_vz = make_lp(), make_lp(), make_lp()


# Main Loop

stereo = cv2.StereoSGBM_create(minDisparity=0, numDisparities=32, blockSize=5,
             P1=8*3*5**2, P2=32*3*5**2, disp12MaxDiff=1,
             uniquenessRatio=10, speckleWindowSize=100, speckleRange=32)

frame_count = 0

while True:
    ret_l, left_raw  = cap_left.read()
    ret_r, right_raw = cap_right.read()
    if not ret_l or not ret_r:
        print("End of video")
        break

    curr_time += dt
    frame_idx  += 1
    frame_count += 1

    left_frame_new = cv2.cvtColor(left_raw,  cv2.COLOR_BGR2GRAY)
    right_img      = cv2.cvtColor(right_raw, cv2.COLOR_BGR2GRAY)

    # Disparity
    small_left  = cv2.resize(left_frame_new,  None, fx=STEREO_SCALE, fy=STEREO_SCALE)
    small_right = cv2.resize(right_img, None, fx=STEREO_SCALE, fy=STEREO_SCALE)
    disparity_small = stereo.compute(small_left, small_right).astype(np.float32) / 16.0
    disparity = cv2.resize(disparity_small, (left_frame_new.shape[1], left_frame_new.shape[0])) / STEREO_SCALE

    # SIFT + Optical Flow
    pts_new_sift, status, _ = cv2.calcOpticalFlowPyrLK(left_frame_old, left_frame_new, pts_old_sift, None)

    sift_old = np.ascontiguousarray(pts_old_sift[status == 1], dtype=np.float32)
    sift_new = np.ascontiguousarray(pts_new_sift[status == 1], dtype=np.float32)

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

    vx_sift = vy_sift = vz_sift = 0.0
    if len(sift_depths) > 0:
        vx_sift, vy_sift, vz_sift = compute_velocity(sift_old, sift_new, sift_depths, dt)

    vz_sift = reject_outlier(buf_sift_vz, vz_sift)

    vx_sift_ma = ma_update(ma_sift_vx, vx_sift); vy_sift_ma = ma_update(ma_sift_vy, vy_sift); vz_sift_ma = ma_update(ma_sift_vz, vz_sift)
    vx_sift_lp = lp_update(lp_sift_vx, vx_sift); vy_sift_lp = lp_update(lp_sift_vy, vy_sift); vz_sift_lp = lp_update(lp_sift_vz, vz_sift)

    speed_sift_ma = np.linalg.norm([vx_sift_ma, vy_sift_ma, vz_sift_ma])
    speed_sift_lp = np.linalg.norm([vx_sift_lp, vy_sift_lp, vz_sift_lp])

    # Ground truth speed
    gt_pos_curr = gt_position(t0 + curr_time)
    speed_gt    = np.linalg.norm(gt_pos_curr - gt_pos_prev) / dt
    gt_pos_prev = gt_pos_curr

    # Ratio of GT to estimated speed (guard against divide-by-zero)
    ratio_ma = speed_gt / speed_sift_ma if speed_sift_ma > 1e-6 else float('nan')
    ratio_lp = speed_gt / speed_sift_lp if speed_sift_lp > 1e-6 else float('nan')

    # ---- SUGGESTION #1 CHECK: real dt vs fixed dt, printed periodically ----
    if frame_idx < len(left_timestamps):
        real_dt = left_timestamps[frame_idx] - left_timestamps[frame_idx - 1]
    else:
        real_dt = float('nan')

    if frame_count % 30 == 0:
        print(f"[CHECK dt] frame={frame_count} | fixed_dt={dt*1000:.2f}ms "
              f"| real_dt={real_dt*1000:.2f}ms "
              f"| diff={(real_dt-dt)*1000:.2f}ms")
        print(f"[CHECK ratio] GT/SIFT-MA={ratio_ma:.2f}x | GT/SIFT-LP={ratio_lp:.2f}x")

    # Live Video Overlay
    display_frame = left_raw.copy()
    overlay_lines = [
        f"SIFT-MA: {speed_sift_ma:.2f} m/s",
        f"SIFT-LP: {speed_sift_lp:.2f} m/s",
        f"GT:      {speed_gt:.2f} m/s",
        f"Ratio MA (GT/est): {ratio_ma:.2f}x",
        f"Ratio LP (GT/est): {ratio_lp:.2f}x",
    ]
    for i, line in enumerate(overlay_lines):
        cv2.putText(display_frame, line, (10, 30 + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow("Left Camera - Speed Overlay", display_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Tracking Maintenance
    left_frame_old = left_frame_new
    pts_old_sift = sift_new.reshape(-1, 1, 2)
    if len(pts_old_sift) < MIN_FEATURES:
        small_gray = cv2.resize(left_frame_old, None, fx=DETECT_SCALE, fy=DETECT_SCALE)
        kps = sift.detect(small_gray, None)
        pts_old_sift = np.array([kp.pt for kp in kps], dtype=np.float32).reshape(-1, 1, 2) / DETECT_SCALE

cap_left.release()
cap_right.release()
cv2.destroyAllWindows()
print("Done.")