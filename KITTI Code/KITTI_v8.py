# Importing Libraries


import numpy as np
import cv2
import os
import glob
import time
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from lightglue import LightGlue, SuperPoint
from lightglue.utils import rbd
from collections import deque   


# Configuration


# Paths
DATASET_PATH = r"D:\Stingray Robotics\archive\dataset\sequences\00"
LEFT_DIR  = os.path.join(DATASET_PATH, "image_0")
RIGHT_DIR = os.path.join(DATASET_PATH, "image_1")
CALIB     = os.path.join(DATASET_PATH, "calib.txt")
TIMES     = os.path.join(DATASET_PATH, "times.txt")
POSES     = r"D:\Stingray Robotics\archive\data_odometry_poses\dataset\poses\00.txt"

MIN_FEATURES = 400
START_FRAME  = 0
END_FRAME    = 250

left_images  = sorted(glob.glob(os.path.join(LEFT_DIR,  "*.png")))
right_images = sorted(glob.glob(os.path.join(RIGHT_DIR, "*.png")))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load Timestamps, Poses, Calibration


with open(TIMES, "r") as f:
    timestamps = np.array([float(line.strip()) for line in f.readlines()])

with open(POSES, "r") as f:
    poses = [np.array(list(map(float, line.strip().split()))).reshape(3, 4) for line in f.readlines()]

with open(CALIB, "r") as f:
    lines = f.readlines()

P0 = np.array(list(map(float, lines[0].split()[1:])), dtype=np.float32).reshape(3, 4)
P1 = np.array(list(map(float, lines[1].split()[1:])), dtype=np.float32).reshape(3, 4)

fx       = P0[0, 0]
cx       = P0[0, 2]
cy       = P0[1, 2]
baseline = abs(P1[0, 3] / fx)


# Load SuperPoint + SuperGlue


extractor = SuperPoint(max_num_keypoints=1024).eval().to(device)
matcher   = LightGlue(features="superpoint").eval().to(device)

def extract_superpoint(gray_frame):
    tensor = torch.from_numpy(gray_frame).float() / 255.0
    tensor = tensor.unsqueeze(0).unsqueeze(0).to(device)
    return extractor.extract(tensor)


# Load SIFT


sift = cv2.SIFT_create()


# Shared: Depth + Jacobian + Velocity helpers


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

    # Added: drop points with a large residual after the first solve, then re-solve
    # (kills the single-bad-correspondence spikes without touching the Jacobian model)
    residuals = (J @ cam_vel - img_disp_vec).reshape(-1, 2)
    residual_norms = np.linalg.norm(residuals, axis=1)
    if len(residual_norms) > 6:   # keep enough points left to solve for 6 DOF
        med = np.median(residual_norms)
        mad = np.median(np.abs(residual_norms - med)) + 1e-6
        keep = residual_norms < (med + 3 * mad)
        if keep.sum() >= 6 and keep.sum() < len(keep):
            keep_rows = np.repeat(keep, 2)
            cam_vel = np.linalg.pinv(J[keep_rows]) @ img_disp_vec[keep_rows]

    return cam_vel[0], cam_vel[1], cam_vel[2]


# Initial Feature Detection (first frame)


left_frame_old = cv2.imread(left_images[START_FRAME], cv2.IMREAD_GRAYSCALE)

# SuperPoint initial
feats_old = extract_superpoint(left_frame_old)

# SIFT initial
kps, _ = sift.detectAndCompute(left_frame_old, None)
pts_old_sift = np.array([kp.pt for kp in kps], dtype=np.float32).reshape(-1, 1, 2)


# Kalman Filter Setup


def make_kf():
    return {"x": 5.0, "P": 1.0, "Q": 1.5, "R": 10}

def kf_update(kf, measurement):
    kf["P"] = kf["P"] + kf["Q"]
    K       = kf["P"] / (kf["P"] + kf["R"])
    kf["x"] = kf["x"] + K * (measurement - kf["x"])
    kf["P"] = (1 - K) * kf["P"]
    return kf["x"]

# Rolling-window outlier clamp (median/MAD), applied before each filter update
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
    return {"x": 5.0, "alpha": alpha}

def lp_update(lp, value):
    lp["x"] = lp["alpha"] * value + (1 - lp["alpha"]) * lp["x"]
    return lp["x"]

# Kalman filter + buffer per axis, per method
kf_sp_vx,   kf_sp_vy,   kf_sp_vz   = make_kf(), make_kf(), make_kf()
kf_sift_vx, kf_sift_vy, kf_sift_vz = make_kf(), make_kf(), make_kf()

buf_sp_vx,   buf_sp_vy,   buf_sp_vz   = make_buffer(), make_buffer(), make_buffer()
buf_sift_vx, buf_sift_vy, buf_sift_vz = make_buffer(), make_buffer(), make_buffer()

# Moving Average buffers per axis, per method
ma_sp_vx,   ma_sp_vy,   ma_sp_vz   = make_ma(), make_ma(), make_ma()
ma_sift_vx, ma_sift_vy, ma_sift_vz = make_ma(), make_ma(), make_ma()

# Low Pass filters per axis, per method
lp_sp_vx,   lp_sp_vy,   lp_sp_vz   = make_lp(), make_lp(), make_lp()
lp_sift_vx, lp_sift_vy, lp_sift_vz = make_lp(), make_lp(), make_lp()


# History Lists


vx_sp_kf_history,   vy_sp_kf_history,   vz_sp_kf_history   = [], [], []
vx_sp_ma_history,   vy_sp_ma_history,   vz_sp_ma_history   = [], [], []
vx_sp_lp_history,   vy_sp_lp_history,   vz_sp_lp_history   = [], [], []

vx_sift_kf_history, vy_sift_kf_history, vz_sift_kf_history = [], [], []
vx_sift_ma_history, vy_sift_ma_history, vz_sift_ma_history = [], [], []
vx_sift_lp_history, vy_sift_lp_history, vz_sift_lp_history = [], [], []

vx_gt_history, vy_gt_history, vz_gt_history = [], [], []


# Error Accumulators


sp_kf_errors,   sift_kf_errors   = [], []
sp_ma_errors,   sift_ma_errors   = [], []
sp_lp_errors,   sift_lp_errors   = [], []


# Main Loop


for frame_idx in tqdm(range(START_FRAME + 1, END_FRAME), desc="Processing frames"):

    frame_start_time = time.time()

    dt            = timestamps[frame_idx] - timestamps[frame_idx - 1]
    left_frame_new = cv2.imread(left_images[frame_idx],  cv2.IMREAD_GRAYSCALE)
    left_img       = cv2.imread(left_images[frame_idx],  cv2.IMREAD_GRAYSCALE)
    right_img      = cv2.imread(right_images[frame_idx], cv2.IMREAD_GRAYSCALE)

    # Disparity (shared for both methods)
    stereo    = cv2.StereoSGBM_create(minDisparity=0, numDisparities=64, blockSize=5,
                    P1=8*3*5**2, P2=32*3*5**2, disp12MaxDiff=1,
                    uniquenessRatio=10, speckleWindowSize=100, speckleRange=32)
    disparity = stereo.compute(left_img, right_img).astype(np.float32) / 16.0


    # SuperPoint + SuperGlue


    feats_new    = extract_superpoint(left_frame_new)
    matches01    = matcher({"image0": feats_old, "image1": feats_new})
    f0, f1, m01  = rbd(feats_old), rbd(feats_new), rbd(matches01)
    matched_idxs = m01["matches"]

    sp_old = f0["keypoints"][matched_idxs[:, 0]].cpu().numpy()
    sp_new = f1["keypoints"][matched_idxs[:, 1]].cpu().numpy()

    F, mask = cv2.findFundamentalMat(sp_old, sp_new, cv2.FM_RANSAC)
    if mask is not None:
        sp_old = sp_old[mask.ravel() == 1]
        sp_new = sp_new[mask.ravel() == 1]

    sp_depths  = compute_depth(sp_new, disparity)
    valid      = ~np.isnan(sp_depths)
    sp_old, sp_new, sp_depths = sp_old[valid], sp_new[valid], sp_depths[valid]

    vx_sp = vy_sp = vz_sp = 0.0
    if len(sp_depths) > 0:
        vx_sp, vy_sp, vz_sp = compute_velocity(sp_old, sp_new, sp_depths, dt)

    # Outlier clamp (shared pre-step for all three filters)
    vx_sp = reject_outlier(buf_sp_vx, vx_sp)
    vy_sp = reject_outlier(buf_sp_vy, vy_sp)
    vz_sp = reject_outlier(buf_sp_vz, vz_sp)

    vx_sp_kf = kf_update(kf_sp_vx, vx_sp);  vy_sp_kf = kf_update(kf_sp_vy, vy_sp);  vz_sp_kf = kf_update(kf_sp_vz, vz_sp)
    vx_sp_ma = ma_update(ma_sp_vx, vx_sp);   vy_sp_ma = ma_update(ma_sp_vy, vy_sp);   vz_sp_ma = ma_update(ma_sp_vz, vz_sp)
    vx_sp_lp = lp_update(lp_sp_vx, vx_sp);  vy_sp_lp = lp_update(lp_sp_vy, vy_sp);  vz_sp_lp = lp_update(lp_sp_vz, vz_sp)


    # SIFT + Optical Flow


    pts_new_sift, status, _ = cv2.calcOpticalFlowPyrLK(left_frame_old, left_frame_new, pts_old_sift, None)

    sift_old = pts_old_sift[status == 1]
    sift_new = pts_new_sift[status == 1]

    F, mask = cv2.findFundamentalMat(sift_old, sift_new, cv2.FM_RANSAC)
    if mask is not None:
        sift_old = sift_old[mask.ravel() == 1]
        sift_new = sift_new[mask.ravel() == 1]

    sift_depths = compute_depth(sift_new, disparity)
    valid       = ~np.isnan(sift_depths)
    sift_old, sift_new, sift_depths = sift_old[valid], sift_new[valid], sift_depths[valid]

    vx_sift = vy_sift = vz_sift = 0.0
    if len(sift_depths) > 0:
        vx_sift, vy_sift, vz_sift = compute_velocity(sift_old, sift_new, sift_depths, dt)

    # Outlier clamp (shared pre-step for all three filters)
    vx_sift = reject_outlier(buf_sift_vx, vx_sift)
    vy_sift = reject_outlier(buf_sift_vy, vy_sift)
    vz_sift = reject_outlier(buf_sift_vz, vz_sift)

    vx_sift_kf = kf_update(kf_sift_vx, vx_sift);  vy_sift_kf = kf_update(kf_sift_vy, vy_sift);  vz_sift_kf = kf_update(kf_sift_vz, vz_sift)
    vx_sift_ma = ma_update(ma_sift_vx, vx_sift);   vy_sift_ma = ma_update(ma_sift_vy, vy_sift);   vz_sift_ma = ma_update(ma_sift_vz, vz_sift)
    vx_sift_lp = lp_update(lp_sift_vx, vx_sift);  vy_sift_lp = lp_update(lp_sift_vy, vy_sift);  vz_sift_lp = lp_update(lp_sift_vz, vz_sift)


    # Ground Truth


    R        = poses[frame_idx][:, :3]
    v_world  = (poses[frame_idx][:, 3] - poses[frame_idx - 1][:, 3]) / dt
    v_camera = R.T @ v_world
    gt_vx, gt_vy, gt_vz = v_camera[0], v_camera[1], v_camera[2]


    # Errors


    sp_kf_errors.append(abs(vz_sp_kf - gt_vz));     sift_kf_errors.append(abs(vz_sift_kf - gt_vz))
    sp_ma_errors.append(abs(vz_sp_ma - gt_vz));     sift_ma_errors.append(abs(vz_sift_ma - gt_vz))
    sp_lp_errors.append(abs(vz_sp_lp - gt_vz));     sift_lp_errors.append(abs(vz_sift_lp - gt_vz))


    # Store History


    vx_sp_kf_history.append(vx_sp_kf);  vy_sp_kf_history.append(vy_sp_kf);  vz_sp_kf_history.append(vz_sp_kf)
    vx_sp_ma_history.append(vx_sp_ma);  vy_sp_ma_history.append(vy_sp_ma);  vz_sp_ma_history.append(vz_sp_ma)
    vx_sp_lp_history.append(vx_sp_lp);  vy_sp_lp_history.append(vy_sp_lp);  vz_sp_lp_history.append(vz_sp_lp)

    vx_sift_kf_history.append(vx_sift_kf);  vy_sift_kf_history.append(vy_sift_kf);  vz_sift_kf_history.append(vz_sift_kf)
    vx_sift_ma_history.append(vx_sift_ma);  vy_sift_ma_history.append(vy_sift_ma);  vz_sift_ma_history.append(vz_sift_ma)
    vx_sift_lp_history.append(vx_sift_lp);  vy_sift_lp_history.append(vy_sift_lp);  vz_sift_lp_history.append(vz_sift_lp)

    vx_gt_history.append(gt_vx);  vy_gt_history.append(gt_vy);  vz_gt_history.append(gt_vz)


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


# Final Error Report in Terminal


print(f"  SuperPoint  + Kalman Filter    Vz MAE: {np.mean(sp_kf_errors):.4f} m/s")
print(f"  SuperPoint  + Moving Average   Vz MAE: {np.mean(sp_ma_errors):.4f} m/s")
print(f"  SuperPoint  + Low Pass Filter  Vz MAE: {np.mean(sp_lp_errors):.4f} m/s")
print(f"  SIFT        + Kalman Filter    Vz MAE: {np.mean(sift_kf_errors):.4f} m/s")
print(f"  SIFT        + Moving Average   Vz MAE: {np.mean(sift_ma_errors):.4f} m/s")
print(f"  SIFT        + Low Pass Filter  Vz MAE: {np.mean(sift_lp_errors):.4f} m/s")

cv2.destroyAllWindows()


# Final Graphs — 6 canvases, each with Vx / Vy / Vz vs Ground Truth


def plot_canvas(title, vx_est, vy_est, vz_est, color):
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    axes[0].plot(vx_est,       color=color,   label=title); axes[0].plot(vx_gt_history, color="black", label="Ground Truth"); axes[0].set_ylabel("Vx (m/s)"); axes[0].legend()
    axes[1].plot(vy_est,       color=color,   label=title); axes[1].plot(vy_gt_history, color="black", label="Ground Truth"); axes[1].set_ylabel("Vy (m/s)"); axes[1].legend()
    axes[2].plot(vz_est,       color=color,   label=title); axes[2].plot(vz_gt_history, color="black", label="Ground Truth"); axes[2].set_ylabel("Vz (m/s)"); axes[2].set_xlabel("Frame"); axes[2].legend()
    fig.suptitle(title)
    plt.tight_layout()

plot_canvas("SuperPoint + Kalman Filter",   vx_sp_kf_history,   vy_sp_kf_history,   vz_sp_kf_history,   color="blue")
plot_canvas("SuperPoint + Moving Average",  vx_sp_ma_history,   vy_sp_ma_history,   vz_sp_ma_history,   color="royalblue")
plot_canvas("SuperPoint + Low Pass Filter", vx_sp_lp_history,   vy_sp_lp_history,   vz_sp_lp_history,   color="dodgerblue")
plot_canvas("SIFT + Kalman Filter",         vx_sift_kf_history, vy_sift_kf_history, vz_sift_kf_history, color="green")
plot_canvas("SIFT + Moving Average",        vx_sift_ma_history, vy_sift_ma_history, vz_sift_ma_history, color="limegreen")
plot_canvas("SIFT + Low Pass Filter",       vx_sift_lp_history, vy_sift_lp_history, vz_sift_lp_history, color="mediumseagreen")

plt.show()
