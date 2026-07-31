# Importing Libraries


import numpy as np
import cv2
import time
import torch
import re
from lightglue import LightGlue, SuperPoint
from lightglue.utils import rbd
from collections import deque
from tqdm import tqdm


# Configuration


CALIB        = r"D:\Malaga Urban Dataset\camera_params_raw_1024x768.txt"
LEFT_VIDEO   = r"D:\Malaga Urban Dataset\malaga-urban-dataset_STEREO_LEFT.avi"
RIGHT_VIDEO  = r"D:\Malaga Urban Dataset\malaga-urban-dataset_STEREO_RIGHT.avi"
GPS_FILE     = r"D:\Malaga Urban Dataset\malaga-urban-dataset-plain-text\malaga-urban-dataset_GPS.txt"
IMAGES_FILE  = r"D:\Malaga Urban Dataset\malaga-urban-dataset-plain-text\malaga-urban-dataset_IMAGES.txt"
MIN_FEATURES = 400
IMG_SIZE     = (1024, 768)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load Calibration (MRPT stereo calib format: separate K/dist per camera + relative pose)


def parse_mrpt_calib(path):
    with open(path, "r") as f:
        text = f.read()

    def cam_params(section):
        block = re.search(rf"\[CAMERA_PARAMS_{section}\](.*?)(?=\n\[|\Z)", text, re.S).group(1)
        cx = float(re.search(r"cx\s*=\s*([\d.eE+-]+)", block).group(1))
        cy = float(re.search(r"cy\s*=\s*([\d.eE+-]+)", block).group(1))
        fx = float(re.search(r"fx\s*=\s*([\d.eE+-]+)", block).group(1))
        fy = float(re.search(r"fy\s*=\s*([\d.eE+-]+)", block).group(1))
        dist_str = re.search(r"dist\s*=\s*\[(.*?)\]", block).group(1)
        dist = np.array(list(map(float, dist_str.split())), dtype=np.float64)
        K = np.array([[fx, 0, cx],
                      [0, fy, cy],
                      [0,  0,  1]], dtype=np.float64)
        return K, dist

    K_left,  dist_left  = cam_params("LEFT")
    K_right, dist_right = cam_params("RIGHT")

    pose_block = re.search(r"\[CAMERA_PARAMS_LEFT2RIGHT_POSE\](.*?)(?=\n\[|\Z)", text, re.S).group(1)
    t_str = re.search(r"translation_only\s*=\s*\[(.*?)\]", pose_block).group(1)
    T = np.array(list(map(float, t_str.split())), dtype=np.float64).reshape(3, 1)

    rot_str = re.search(r"rotation_matrix_only\s*=\s*\[(.*?)\]", pose_block, re.S).group(1)
    rows = [row.strip() for row in rot_str.split(";") if row.strip()]
    R = np.array([list(map(float, row.split())) for row in rows], dtype=np.float64)

    return K_left, dist_left, K_right, dist_right, R, T

K_left, dist_left, K_right, dist_right, R, T = parse_mrpt_calib(CALIB)

# Stereo rectify so downstream code can keep using pinhole fx/cx/cy/baseline
R1, R2, P0, P1, Q, _, _ = cv2.stereoRectify(
    K_left, dist_left, K_right, dist_right, IMG_SIZE, R, T, alpha=0
)

map1x, map1y = cv2.initUndistortRectifyMap(K_left,  dist_left,  R1, P0, IMG_SIZE, cv2.CV_32FC1)
map2x, map2y = cv2.initUndistortRectifyMap(K_right, dist_right, R2, P1, IMG_SIZE, cv2.CV_32FC1)

fx       = P0[0, 0]
cx       = P0[0, 2]
cy       = P0[1, 2]
baseline = abs(P1[0, 3] / fx)


# Load True Forward Velocity from GPS (differentiate Local X/Y position, sync via IMAGES.txt timestamps)
# Local X/Y is the horizontal ground plane (Local Z is up), so the horizontal distance
# travelled between consecutive GPS fixes is the vehicle's forward/ground speed.


def load_gps_true_speed(path):
    data = np.loadtxt(path, comments="%")
    t   = data[:, 0]
    lx  = data[:, 8]
    ly  = data[:, 9]

    dt_gps   = np.diff(t)
    dist_gps = np.hypot(np.diff(lx), np.diff(ly))
    speed_gps = dist_gps / dt_gps
    mid_t     = t[:-1] + dt_gps / 2.0
    return mid_t, speed_gps

gps_mid_t, gps_mid_speed = load_gps_true_speed(GPS_FILE)

def true_vz_at(timestamp):
    return np.interp(timestamp, gps_mid_t, gps_mid_speed)

# Left-image capture timestamps, in video-frame order, used to sync each frame to the GPS timeline
with open(IMAGES_FILE, "r") as f:
    left_timestamps = [float(re.search(r"_(\d+\.\d+)_left", line).group(1))
                        for line in f if "_left" in line]


# Load SuperPoint + LightGlue


extractor = SuperPoint(max_num_keypoints=1024).eval().to(device)
matcher   = LightGlue(features="superpoint").eval().to(device)

def extract_superpoint(gray_frame):
    tensor = torch.from_numpy(gray_frame).float() / 255.0
    tensor = tensor.unsqueeze(0).unsqueeze(0).to(device)
    return extractor.extract(tensor)


# Load SIFT


sift = cv2.SIFT_create()


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

total_frames = int(cap_left.get(cv2.CAP_PROP_FRAME_COUNT))

ret_l, left_raw = cap_left.read()
left_frame_old  = cv2.cvtColor(left_raw, cv2.COLOR_BGR2GRAY)
left_frame_old  = cv2.remap(left_frame_old, map1x, map1y, cv2.INTER_LINEAR)

# SuperPoint initial
feats_old = extract_superpoint(left_frame_old)

# SIFT initial
kps, _ = sift.detectAndCompute(left_frame_old, None)
pts_old_sift = np.array([kp.pt for kp in kps], dtype=np.float32).reshape(-1, 1, 2)

curr_time = 0.0
frame_idx = 0  # index into left_timestamps, matches the frame currently held in left_frame_old


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

# Low Pass Filter Setup
def make_lp(alpha=0.2):
    # x=None: lazy init from first real measurement, avoids artificial spike
    return {"x": None, "alpha": alpha}

def lp_update(lp, value):
    if lp["x"] is None:
        lp["x"] = value
    else:
        lp["x"] = lp["alpha"] * value + (1 - lp["alpha"]) * lp["x"]
    return lp["x"]

# Outlier clamp buffers per method (used for Vz only)
buf_sp_vz   = make_buffer()
buf_sift_vz = make_buffer()

# Moving Average buffers per axis, per method
ma_sp_vx,   ma_sp_vy,   ma_sp_vz   = make_ma(), make_ma(), make_ma()
ma_sift_vx, ma_sift_vy, ma_sift_vz = make_ma(), make_ma(), make_ma()

# Low Pass filters per axis, per method
lp_sp_vx,   lp_sp_vy,   lp_sp_vz   = make_lp(), make_lp(), make_lp()
lp_sift_vx, lp_sift_vy, lp_sift_vz = make_lp(), make_lp(), make_lp()


# Main Loop


stereo = cv2.StereoSGBM_create(minDisparity=0, numDisparities=64, blockSize=5,
             P1=8*3*5**2, P2=32*3*5**2, disp12MaxDiff=1,
             uniquenessRatio=10, speckleWindowSize=100, speckleRange=32)

pbar = tqdm(total=total_frames, desc="Processing frames")

while True:
    ret_l, left_raw  = cap_left.read()
    ret_r, right_raw = cap_right.read()
    if not ret_l or not ret_r:
        print("End of video")
        break

    pbar.update(1)

    curr_time += dt
    frame_idx += 1

    left_frame_new = cv2.cvtColor(left_raw,  cv2.COLOR_BGR2GRAY)
    right_img      = cv2.cvtColor(right_raw, cv2.COLOR_BGR2GRAY)

    left_frame_new = cv2.remap(left_frame_new, map1x, map1y, cv2.INTER_LINEAR)
    right_img      = cv2.remap(right_img,      map2x, map2y, cv2.INTER_LINEAR)
    left_img       = left_frame_new

    # Disparity (shared for both methods)
    disparity = stereo.compute(left_img, right_img).astype(np.float32) / 16.0


    # SuperPoint + LightGlue


    feats_new    = extract_superpoint(left_frame_new)
    matches01    = matcher({"image0": feats_old, "image1": feats_new})
    f0, f1, m01  = rbd(feats_old), rbd(feats_new), rbd(matches01)
    matched_idxs = m01["matches"]

    sp_old = np.ascontiguousarray(f0["keypoints"][matched_idxs[:, 0]].cpu().numpy(), dtype=np.float32)
    sp_new = np.ascontiguousarray(f1["keypoints"][matched_idxs[:, 1]].cpu().numpy(), dtype=np.float32)


    # RANSAC Outlier Rejection - SuperPoint


    if len(sp_old) >= 8:
        try:
            F, mask = cv2.findFundamentalMat(sp_old, sp_new, cv2.FM_RANSAC)
            if mask is not None:
                sp_old = sp_old[mask.ravel() == 1]
                sp_new = sp_new[mask.ravel() == 1]
        except cv2.error:
            pass

    sp_depths  = compute_depth(sp_new, disparity)
    valid      = ~np.isnan(sp_depths)
    sp_old, sp_new, sp_depths = sp_old[valid], sp_new[valid], sp_depths[valid]

    vx_sp = vy_sp = vz_sp = 0.0
    if len(sp_depths) > 0:
        vx_sp, vy_sp, vz_sp = compute_velocity(sp_old, sp_new, sp_depths, dt)

    # Outlier clamp only on Vz (Vx/Vy oscillate around zero; clamping them zeros out the real signal)
    vz_sp = reject_outlier(buf_sp_vz, vz_sp)

    vx_sp_ma = ma_update(ma_sp_vx, vx_sp);   vy_sp_ma = ma_update(ma_sp_vy, vy_sp);   vz_sp_ma = ma_update(ma_sp_vz, vz_sp)
    vx_sp_lp = lp_update(lp_sp_vx, vx_sp);  vy_sp_lp = lp_update(lp_sp_vy, vy_sp);  vz_sp_lp = lp_update(lp_sp_vz, vz_sp)


    # SIFT + Optical Flow


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

    vx_sift = vy_sift = vz_sift = 0.0
    if len(sift_depths) > 0:
        vx_sift, vy_sift, vz_sift = compute_velocity(sift_old, sift_new, sift_depths, dt)

    # Outlier clamp only on Vz (Vx/Vy oscillate around zero; clamping them zeros out the real signal)
    vz_sift = reject_outlier(buf_sift_vz, vz_sift)

    vx_sift_ma = ma_update(ma_sift_vx, vx_sift);   vy_sift_ma = ma_update(ma_sift_vy, vy_sift);   vz_sift_ma = ma_update(ma_sift_vz, vz_sift)
    vx_sift_lp = lp_update(lp_sift_vx, vx_sift);  vy_sift_lp = lp_update(lp_sift_vy, vy_sift);  vz_sift_lp = lp_update(lp_sift_vz, vz_sift)


    # True Forward Velocity (GPS, synced to this frame's real capture timestamp)


    frame_ts = left_timestamps[min(frame_idx, len(left_timestamps) - 1)]
    vz_true  = true_vz_at(frame_ts)


    # Display on Live Video


    display_frame = cv2.cvtColor(left_img, cv2.COLOR_GRAY2BGR)
    cv2.putText(display_frame, f"Vz SP  MA: {vz_sp_ma:.3f} m/s",   (20, 30),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(display_frame, f"Vz SP  LP: {vz_sp_lp:.3f} m/s",   (20, 60),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(display_frame, f"Vz SIFT MA: {vz_sift_ma:.3f} m/s", (20, 90),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
    cv2.putText(display_frame, f"Vz SIFT LP: {vz_sift_lp:.3f} m/s", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
    cv2.putText(display_frame, f"Vz TRUE (GPS): {vz_true:.3f} m/s", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Odometry", display_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    print(f"dt={dt*1000:.1f}ms | Vz_sp_ma={vz_sp_ma:.3f} Vz_sp_lp={vz_sp_lp:.3f}"
          f" | Vz_sift_ma={vz_sift_ma:.3f} Vz_sift_lp={vz_sift_lp:.3f} | Vz_true={vz_true:.3f}")


    # Tracking Maintenance


    left_frame_old = left_frame_new

    # SuperPoint: carry forward features
    feats_old = feats_new
    if len(sp_new) < MIN_FEATURES:
        feats_old = extract_superpoint(left_frame_old)

    # SIFT: carry forward tracked points
    pts_old_sift = sift_new.reshape(-1, 1, 2)
    if len(pts_old_sift) < MIN_FEATURES:
        kps, _ = sift.detectAndCompute(left_frame_old, None)
        pts_old_sift = np.array([kp.pt for kp in kps], dtype=np.float32).reshape(-1, 1, 2)


pbar.close()
cap_left.release()
cap_right.release()
cv2.destroyAllWindows()
print("Done.")