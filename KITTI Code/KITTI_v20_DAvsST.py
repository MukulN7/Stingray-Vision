# Importing Libraries


import sys
import numpy as np
import cv2
import os
import glob
import time
import csv
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from lightglue import LightGlue, SuperPoint
from lightglue.utils import rbd
from collections import deque


# Configuration


# Paths
DATASET_PATH = r"D:\Stingray Robotics\archive\dataset\sequences\02"
LEFT_DIR  = os.path.join(DATASET_PATH, "image_0")
RIGHT_DIR = os.path.join(DATASET_PATH, "image_1")
CALIB     = os.path.join(DATASET_PATH, "calib.txt")
TIMES     = os.path.join(DATASET_PATH, "times.txt")
POSES     = r"D:\Stingray Robotics\archive\data_odometry_poses\dataset\poses\02.txt"
VELOCITY_CSV_PATH = r"D:\Stingray Robotics\velocity_output20.csv"
POSITION_CSV_PATH = r"D:\Stingray Robotics\position_output20.csv"

# Depth Anything V2 (metric, outdoor/vkitti checkpoint) paths
# DEPTH_ANYTHING_REPO must point at the "metric_depth" folder of the cloned repo
# (not the repo root), because that folder has its own depth_anything_v2 package.
DEPTH_ANYTHING_REPO    = r"D:\Stingray Robotics\Depth-Anything-V2\metric_depth"
DEPTH_ANYTHING_ENCODER = "vits"   # "vits", "vitb", or "vitl" -- must match the checkpoint below
DEPTH_ANYTHING_CKPT    = r"D:\depth_anything_v2_metric_vkitti_vits.pth"
DEPTH_ANYTHING_MAX_DEPTH = 80.0   # the vkitti (outdoor) checkpoint was trained with a max depth of 80 m

MIN_FEATURES = 400
START_FRAME  = 0
END_FRAME    = 4530

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


# Load Depth Anything V2 (metric)


sys.path.insert(0, DEPTH_ANYTHING_REPO)
from depth_anything_v2.dpt import DepthAnythingV2   # noqa: E402  (import after sys.path edit, on purpose)

da_model_configs = {
    "vits": {"encoder": "vits", "features": 64,  "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
}

depth_anything = DepthAnythingV2(**{**da_model_configs[DEPTH_ANYTHING_ENCODER], "max_depth": DEPTH_ANYTHING_MAX_DEPTH})
depth_anything.load_state_dict(torch.load(DEPTH_ANYTHING_CKPT, map_location="cpu"))
depth_anything = depth_anything.to(device).eval()

def extract_depth_anything(gray_frame):
    # infer_image expects a 3-channel BGR image. Our KITTI frames are grayscale,
    # so the single channel is just repeated across B, G, R.
    color_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
    return depth_anything.infer_image(color_frame)   # HxW metric depth map, in meters


# Shared: Depth + Jacobian + Velocity helpers


def compute_depth_stereo(goodpts, disparity):
    depths = []
    for pt in goodpts:
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

def compute_depth_monocular(goodpts, depth_map):
    depths = []
    for pt in goodpts:
        x, y = map(int, pt.ravel())
        if 0 <= x < depth_map.shape[1] and 0 <= y < depth_map.shape[0]:
            Z = depth_map[y, x]
            depths.append(Z if 0 < Z < 50.0 else np.nan)
        else:
            depths.append(np.nan)
    return np.array(depths)

def compute_velocity(goodpts_old, goodpts_new, depths_old, depths_new, dt):
    # Evaluate the Jacobian at the midpoint of the frame interval (avg of old/new point
    # and old/new depth) instead of the endpoint. Using only the new-frame depth/position
    # overstates nx/Z for approaching points, which systematically underestimates Vz.
    mid_pts = (goodpts_old + goodpts_new) / 2.0
    normalized_pts = np.array([((pt[0] - cx) / fx, (pt[1] - cy) / fx) for pt in mid_pts])
    depths = (depths_old + depths_new) / 2.0

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


left_frame_old  = cv2.imread(left_images[START_FRAME],  cv2.IMREAD_GRAYSCALE)
right_frame_old = cv2.imread(right_images[START_FRAME], cv2.IMREAD_GRAYSCALE)

# Disparity for the starting frame, so the first loop iteration has an "old" depth to average with
stereo_init   = cv2.StereoSGBM_create(minDisparity=0, numDisparities=64, blockSize=5,
                    P1=8*3*5**2, P2=32*3*5**2, disp12MaxDiff=1,
                    uniquenessRatio=10, speckleWindowSize=100, speckleRange=32)
disparity_old = stereo_init.compute(left_frame_old, right_frame_old).astype(np.float32) / 16.0

# Depth Anything, same idea: an "old" metric depth map for the first loop iteration
depth_anything_old = extract_depth_anything(left_frame_old)

# SuperPoint initial
feats_old = extract_superpoint(left_frame_old)


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

# Outlier clamp buffers per method (used for Vz only)
buf_st_vz = make_buffer()
buf_da_vz = make_buffer()

# Moving Average buffers per axis, per method
ma_st_vx, ma_st_vy, ma_st_vz = make_ma(), make_ma(), make_ma()
ma_da_vx, ma_da_vy, ma_da_vz = make_ma(), make_ma(), make_ma()


# History Lists


vx_st_ma_history, vy_st_ma_history, vz_st_ma_history = [], [], []
vx_da_ma_history, vy_da_ma_history, vz_da_ma_history = [], [], []

vx_gt_history, vy_gt_history, vz_gt_history = [], [], []


# Error Accumulators


st_ma_errors, da_ma_errors = [], []


# Dead-Reckoning Position Setup (world-frame position, one accumulator per method)
# Camera velocity -> world velocity is done via the GT rotation matrix R, then integrated: pos = pos + v_world * dt


pos_st_ma = poses[START_FRAME][:, 3].copy()
pos_da_ma = poses[START_FRAME][:, 3].copy()

# Top-down (X-Z plane) position history, per method + ground truth
x_st_ma_hist, z_st_ma_hist = [], []
x_da_ma_hist, z_da_ma_hist = [], []
x_gt_hist,    z_gt_hist    = [], []

# Position error accumulators (Euclidean distance to GT position), per method
st_ma_pos_errors, da_ma_pos_errors = [], []


# Main Loop


for frame_idx in tqdm(range(START_FRAME + 1, END_FRAME), desc="Processing frames"):

    frame_start_time = time.time()

    dt             = timestamps[frame_idx] - timestamps[frame_idx - 1]
    left_frame_new = cv2.imread(left_images[frame_idx],  cv2.IMREAD_GRAYSCALE)
    left_img       = cv2.imread(left_images[frame_idx],  cv2.IMREAD_GRAYSCALE)
    right_img      = cv2.imread(right_images[frame_idx], cv2.IMREAD_GRAYSCALE)

    # Disparity, for the stereo depth branch
    stereo    = cv2.StereoSGBM_create(minDisparity=0, numDisparities=64, blockSize=5,
                    P1=8*3*5**2, P2=32*3*5**2, disp12MaxDiff=1,
                    uniquenessRatio=10, speckleWindowSize=100, speckleRange=32)
    disparity = stereo.compute(left_img, right_img).astype(np.float32) / 16.0

    # Depth Anything, for the monocular depth branch (left image only)
    depth_anything_new = extract_depth_anything(left_frame_new)


    # SuperPoint + SuperGlue (shared by both depth branches)


    feats_new    = extract_superpoint(left_frame_new)
    matches01    = matcher({"image0": feats_old, "image1": feats_new})
    f0, f1, m01  = rbd(feats_old), rbd(feats_new), rbd(matches01)
    matched_idxs = m01["matches"]

    sp_old = f0["keypoints"][matched_idxs[:, 0]].cpu().numpy()
    sp_new = f1["keypoints"][matched_idxs[:, 1]].cpu().numpy()

    # findFundamentalMat with FM_RANSAC requires a minimum of 8 point correspondences.
    # If matching produced fewer, skip RANSAC filtering this frame instead of crashing.
    if len(sp_old) >= 8:
        F, mask = cv2.findFundamentalMat(sp_old, sp_new, cv2.FM_RANSAC)
        if mask is not None:
            sp_old = sp_old[mask.ravel() == 1]
            sp_new = sp_new[mask.ravel() == 1]
    else:
        sp_old = np.empty((0, 2), dtype=np.float32)
        sp_new = np.empty((0, 2), dtype=np.float32)


    # Stereo Depth branch


    st_depths_new = compute_depth_stereo(sp_new, disparity)
    st_depths_old = compute_depth_stereo(sp_old, disparity_old)
    valid_st      = ~np.isnan(st_depths_new) & ~np.isnan(st_depths_old)
    st_old, st_new, st_depths_old_v, st_depths_new_v = sp_old[valid_st], sp_new[valid_st], st_depths_old[valid_st], st_depths_new[valid_st]

    vx_st = vy_st = vz_st = 0.0
    if len(st_depths_new_v) >= 6:  # 6 correspondences = 12 equations for the 6-DOF solve
        vx_st, vy_st, vz_st = compute_velocity(st_old, st_new, st_depths_old_v, st_depths_new_v, dt)

    # Outlier clamp only on Vz (Vx/Vy oscillate around zero; clamping them zeros out the real signal)
    vz_st = reject_outlier(buf_st_vz, vz_st)

    vx_st_ma = ma_update(ma_st_vx, vx_st);   vy_st_ma = ma_update(ma_st_vy, vy_st);   vz_st_ma = ma_update(ma_st_vz, vz_st)


    # Depth Anything branch


    da_depths_new = compute_depth_monocular(sp_new, depth_anything_new)
    da_depths_old = compute_depth_monocular(sp_old, depth_anything_old)
    valid_da      = ~np.isnan(da_depths_new) & ~np.isnan(da_depths_old)
    da_old, da_new, da_depths_old_v, da_depths_new_v = sp_old[valid_da], sp_new[valid_da], da_depths_old[valid_da], da_depths_new[valid_da]

    vx_da = vy_da = vz_da = 0.0
    if len(da_depths_new_v) >= 6:  # 6 correspondences = 12 equations for the 6-DOF solve
        vx_da, vy_da, vz_da = compute_velocity(da_old, da_new, da_depths_old_v, da_depths_new_v, dt)

    # Outlier clamp only on Vz (Vx/Vy oscillate around zero; clamping them zeros out the real signal)
    vz_da = reject_outlier(buf_da_vz, vz_da)

    vx_da_ma = ma_update(ma_da_vx, vx_da);   vy_da_ma = ma_update(ma_da_vy, vy_da);   vz_da_ma = ma_update(ma_da_vz, vz_da)


    # Ground Truth


    R        = poses[frame_idx][:, :3]
    v_world  = (poses[frame_idx][:, 3] - poses[frame_idx - 1][:, 3]) / dt
    v_camera = R.T @ v_world
    gt_vx, gt_vy, gt_vz = v_camera[0], v_camera[1], v_camera[2]


    # Dead Reckoning: camera-frame velocity -> world-frame velocity (via R) -> position (x = x + v*dt)


    pos_st_ma += (R @ np.array([vx_st_ma, vy_st_ma, vz_st_ma])) * dt
    pos_da_ma += (R @ np.array([vx_da_ma, vy_da_ma, vz_da_ma])) * dt

    pos_gt = poses[frame_idx][:, 3]


    # Errors


    st_ma_errors.append(abs(vz_st_ma - gt_vz));   da_ma_errors.append(abs(vz_da_ma - gt_vz))

    st_ma_pos_errors.append(np.linalg.norm(pos_st_ma - pos_gt));   da_ma_pos_errors.append(np.linalg.norm(pos_da_ma - pos_gt))


    # Store History


    vx_st_ma_history.append(vx_st_ma);  vy_st_ma_history.append(vy_st_ma);  vz_st_ma_history.append(vz_st_ma)
    vx_da_ma_history.append(vx_da_ma);  vy_da_ma_history.append(vy_da_ma);  vz_da_ma_history.append(vz_da_ma)

    vx_gt_history.append(gt_vx);  vy_gt_history.append(gt_vy);  vz_gt_history.append(gt_vz)

    x_st_ma_hist.append(pos_st_ma[0]);   z_st_ma_hist.append(pos_st_ma[2])
    x_da_ma_hist.append(pos_da_ma[0]);   z_da_ma_hist.append(pos_da_ma[2])
    x_gt_hist.append(pos_gt[0]);         z_gt_hist.append(pos_gt[2])


    # Tracking Maintenance


    left_frame_old      = left_frame_new
    disparity_old        = disparity
    depth_anything_old   = depth_anything_new

    # SuperPoint: carry forward features
    feats_old = feats_new
    if len(sp_new) < MIN_FEATURES:
        feats_old = extract_superpoint(left_frame_old)


# Final Error Report in Terminal


print(f"  Stereo Depth    + SuperPoint/SuperGlue + Moving Average   Vz MAE: {np.mean(st_ma_errors):.4f} m/s")
print(f"  Depth Anything  + SuperPoint/SuperGlue + Moving Average   Vz MAE: {np.mean(da_ma_errors):.4f} m/s")

print(f"  Stereo Depth    + SuperPoint/SuperGlue + Moving Average   Position MAE: {np.mean(st_ma_pos_errors):.4f} m")
print(f"  Depth Anything  + SuperPoint/SuperGlue + Moving Average   Position MAE: {np.mean(da_ma_pos_errors):.4f} m")

cv2.destroyAllWindows()


# Save Velocity + Position Results to CSV


with open(VELOCITY_CSV_PATH, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "frame",
        "vx_st_ma", "vy_st_ma", "vz_st_ma",
        "vx_da_ma", "vy_da_ma", "vz_da_ma",
        "vx_gt", "vy_gt", "vz_gt",
    ])
    for i, frame_idx in enumerate(range(START_FRAME + 1, END_FRAME)):
        writer.writerow([
            frame_idx,
            vx_st_ma_history[i], vy_st_ma_history[i], vz_st_ma_history[i],
            vx_da_ma_history[i], vy_da_ma_history[i], vz_da_ma_history[i],
            vx_gt_history[i],    vy_gt_history[i],    vz_gt_history[i],
        ])

with open(POSITION_CSV_PATH, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "frame",
        "x_st_ma", "z_st_ma",
        "x_da_ma", "z_da_ma",
        "x_gt", "z_gt",
    ])
    for i, frame_idx in enumerate(range(START_FRAME + 1, END_FRAME)):
        writer.writerow([
            frame_idx,
            x_st_ma_hist[i], z_st_ma_hist[i],
            x_da_ma_hist[i], z_da_ma_hist[i],
            x_gt_hist[i],    z_gt_hist[i],
        ])

print(f"Saved velocity results to {VELOCITY_CSV_PATH}")
print(f"Saved position results to {POSITION_CSV_PATH}")


# Final Graph 1 — Velocity comparison: Vx / Vy / Vz, all three methods on each subplot


fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

axes[0].plot(vx_st_ma_history, color="royalblue",  label="Stereo Depth + SP/SG")
axes[0].plot(vx_da_ma_history, color="darkorange", label="Depth Anything + SP/SG")
axes[0].plot(vx_gt_history,    color="black",      label="Ground Truth")
axes[0].set_ylabel("Vx (m/s)"); axes[0].legend()

axes[1].plot(vy_st_ma_history, color="royalblue",  label="Stereo Depth + SP/SG")
axes[1].plot(vy_da_ma_history, color="darkorange", label="Depth Anything + SP/SG")
axes[1].plot(vy_gt_history,    color="black",      label="Ground Truth")
axes[1].set_ylabel("Vy (m/s)"); axes[1].legend()

axes[2].plot(vz_st_ma_history, color="royalblue",  label="Stereo Depth + SP/SG")
axes[2].plot(vz_da_ma_history, color="darkorange", label="Depth Anything + SP/SG")
axes[2].plot(vz_gt_history,    color="black",      label="Ground Truth")
axes[2].set_ylabel("Vz (m/s)"); axes[2].set_xlabel("Frame"); axes[2].legend()

fig.suptitle("Velocity Comparison: Stereo Depth vs Depth Anything vs Ground Truth")
plt.tight_layout()


# Final Graph 2 — Top-down (X-Z plane) trajectory comparison, all three methods on one plot


fig, ax = plt.subplots(figsize=(7, 7))
ax.plot(x_st_ma_hist, z_st_ma_hist, color="royalblue",  label="Stereo Depth + SP/SG")
ax.plot(x_da_ma_hist, z_da_ma_hist, color="darkorange", label="Depth Anything + SP/SG")
ax.plot(x_gt_hist,    z_gt_hist,    color="black",      label="Ground Truth")
ax.set_xlabel("X (m)"); ax.set_ylabel("Z (m)"); ax.set_title("Top Down Trajectory Comparison")
ax.axis("equal"); ax.legend()
plt.tight_layout()

plt.show()