# Importing Libraries


import numpy as np
import cv2
import os
import glob
import time
import torch
import open3d as o3d
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

MIN_FEATURES = 400
START_FRAME  = 0
END_FRAME    = 4530

# 3D Point Cloud Reconstruction Config (SIFT keypoints back-projected to 3D)
POINTCLOUD_FRAME_STRIDE     = 1     # use every Nth frame's SIFT points (raise to thin out the cloud / speed things up)
POINTCLOUD_MAX_PTS_PER_FRAME = 300  # cap points kept per frame (randomly subsampled if a frame exceeds this)
POINTCLOUD_MAX_DEPTH         = 50.0 # drop points beyond this depth (matches the clip already used in compute_depth)

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

def backproject_points(pts_2d, depths):
    # Back-projects 2D pixel points + per-point depth into 3D camera-frame coordinates
    # (pinhole model, using the same fx/cx/cy already used elsewhere in this script)
    pts_2d = np.asarray(pts_2d).reshape(-1, 2)
    depths = np.asarray(depths)
    X = (pts_2d[:, 0] - cx) * depths / fx
    Y = (pts_2d[:, 1] - cy) * depths / fx
    Z = depths
    return np.stack([X, Y, Z], axis=1)

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

# SuperPoint initial
feats_old = extract_superpoint(left_frame_old)

# SIFT initial
kps, _ = sift.detectAndCompute(left_frame_old, None)
pts_old_sift = np.array([kp.pt for kp in kps], dtype=np.float32).reshape(-1, 1, 2)


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


# History Lists


vx_sp_ma_history,   vy_sp_ma_history,   vz_sp_ma_history   = [], [], []
vx_sp_lp_history,   vy_sp_lp_history,   vz_sp_lp_history   = [], [], []

vx_sift_ma_history, vy_sift_ma_history, vz_sift_ma_history = [], [], []
vx_sift_lp_history, vy_sift_lp_history, vz_sift_lp_history = [], [], []

vx_gt_history, vy_gt_history, vz_gt_history = [], [], []


# Error Accumulators


sp_ma_errors,   sift_ma_errors   = [], []
sp_lp_errors,   sift_lp_errors   = [], []


# Dead-Reckoning Position Setup (world-frame position, one accumulator per method)
# Camera velocity -> world velocity is done via the GT rotation matrix R, then integrated: pos = pos + v_world * dt


pos_sp_ma   = poses[START_FRAME][:, 3].copy()
pos_sp_lp   = poses[START_FRAME][:, 3].copy()
pos_sift_ma = poses[START_FRAME][:, 3].copy()
pos_sift_lp = poses[START_FRAME][:, 3].copy()

# Top-down (X-Z plane) position history, per method + ground truth
x_sp_ma_hist,   z_sp_ma_hist   = [], []
x_sp_lp_hist,   z_sp_lp_hist   = [], []
x_sift_ma_hist, z_sift_ma_hist = [], []
x_sift_lp_hist, z_sift_lp_hist = [], []
x_gt_hist,      z_gt_hist      = [], []

# Position error accumulators (Euclidean distance to GT position), per method
sp_ma_pos_errors,   sift_ma_pos_errors   = [], []
sp_lp_pos_errors,   sift_lp_pos_errors   = [], []

# 3D point cloud accumulators (SIFT keypoints back-projected + placed in world frame)
pointcloud_world_gt  = []  # placed using ground-truth pose
pointcloud_world_est = []  # placed using GT rotation + SIFT+LP estimated position


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

    sp_depths_new = compute_depth(sp_new, disparity)
    sp_depths_old = compute_depth(sp_old, disparity_old)
    valid         = ~np.isnan(sp_depths_new) & ~np.isnan(sp_depths_old)
    sp_old, sp_new, sp_depths_old, sp_depths_new = sp_old[valid], sp_new[valid], sp_depths_old[valid], sp_depths_new[valid]

    vx_sp = vy_sp = vz_sp = 0.0
    if len(sp_depths_new) >= 6:  # 6 correspondences = 12 equations for the 6-DOF solve
        vx_sp, vy_sp, vz_sp = compute_velocity(sp_old, sp_new, sp_depths_old, sp_depths_new, dt)

    # Outlier clamp only on Vz (Vx/Vy oscillate around zero; clamping them zeros out the real signal)
    vz_sp = reject_outlier(buf_sp_vz, vz_sp)

    vx_sp_ma = ma_update(ma_sp_vx, vx_sp);   vy_sp_ma = ma_update(ma_sp_vy, vy_sp);   vz_sp_ma = ma_update(ma_sp_vz, vz_sp)
    vx_sp_lp = lp_update(lp_sp_vx, vx_sp);  vy_sp_lp = lp_update(lp_sp_vy, vy_sp);  vz_sp_lp = lp_update(lp_sp_vz, vz_sp)


    # SIFT + Optical Flow


    pts_new_sift, status, _ = cv2.calcOpticalFlowPyrLK(left_frame_old, left_frame_new, pts_old_sift, None)

    sift_old = pts_old_sift[status == 1]
    sift_new = pts_new_sift[status == 1]

    # Same minimum-point guard as the SuperPoint branch above.
    if len(sift_old) >= 8:
        F, mask = cv2.findFundamentalMat(sift_old, sift_new, cv2.FM_RANSAC)
        if mask is not None:
            sift_old = sift_old[mask.ravel() == 1]
            sift_new = sift_new[mask.ravel() == 1]
    else:
        sift_old = np.empty((0, 2), dtype=np.float32)
        sift_new = np.empty((0, 2), dtype=np.float32)

    sift_depths_new = compute_depth(sift_new, disparity)
    sift_depths_old = compute_depth(sift_old, disparity_old)
    valid           = ~np.isnan(sift_depths_new) & ~np.isnan(sift_depths_old)
    sift_old, sift_new, sift_depths_old, sift_depths_new = sift_old[valid], sift_new[valid], sift_depths_old[valid], sift_depths_new[valid]

    vx_sift = vy_sift = vz_sift = 0.0
    if len(sift_depths_new) >= 6:  # 6 correspondences = 12 equations for the 6-DOF solve
        vx_sift, vy_sift, vz_sift = compute_velocity(sift_old, sift_new, sift_depths_old, sift_depths_new, dt)

    # Outlier clamp only on Vz (Vx/Vy oscillate around zero; clamping them zeros out the real signal)
    vz_sift = reject_outlier(buf_sift_vz, vz_sift)

    vx_sift_ma = ma_update(ma_sift_vx, vx_sift);   vy_sift_ma = ma_update(ma_sift_vy, vy_sift);   vz_sift_ma = ma_update(ma_sift_vz, vz_sift)
    vx_sift_lp = lp_update(lp_sift_vx, vx_sift);  vy_sift_lp = lp_update(lp_sift_vy, vy_sift);  vz_sift_lp = lp_update(lp_sift_vz, vz_sift)


    # Ground Truth


    R        = poses[frame_idx][:, :3]
    v_world  = (poses[frame_idx][:, 3] - poses[frame_idx - 1][:, 3]) / dt
    v_camera = R.T @ v_world
    gt_vx, gt_vy, gt_vz = v_camera[0], v_camera[1], v_camera[2]


    # Dead Reckoning: camera-frame velocity -> world-frame velocity (via R) -> position (x = x + v*dt)


    pos_sp_ma   += (R @ np.array([vx_sp_ma,   vy_sp_ma,   vz_sp_ma]))   * dt
    pos_sp_lp   += (R @ np.array([vx_sp_lp,   vy_sp_lp,   vz_sp_lp]))   * dt
    pos_sift_ma += (R @ np.array([vx_sift_ma, vy_sift_ma, vz_sift_ma])) * dt
    pos_sift_lp += (R @ np.array([vx_sift_lp, vy_sift_lp, vz_sift_lp])) * dt

    pos_gt = poses[frame_idx][:, 3]


    # Point Cloud Accumulation (SIFT keypoints back-projected to 3D, placed in world frame)


    if len(sift_depths_new) > 0 and (frame_idx - START_FRAME) % POINTCLOUD_FRAME_STRIDE == 0:
        valid_depth = sift_depths_new < POINTCLOUD_MAX_DEPTH
        pc_pts_2d   = sift_new[valid_depth]
        pc_depths   = sift_depths_new[valid_depth]

        if len(pc_pts_2d) > POINTCLOUD_MAX_PTS_PER_FRAME:
            keep_idx  = np.random.choice(len(pc_pts_2d), POINTCLOUD_MAX_PTS_PER_FRAME, replace=False)
            pc_pts_2d = pc_pts_2d[keep_idx]
            pc_depths = pc_depths[keep_idx]

        if len(pc_pts_2d) > 0:
            cam_pts = backproject_points(pc_pts_2d, pc_depths)

            world_pts_gt  = (R @ cam_pts.T).T + pos_gt
            world_pts_est = (R @ cam_pts.T).T + pos_sift_lp

            pointcloud_world_gt.append(world_pts_gt)
            pointcloud_world_est.append(world_pts_est)


    # Errors


    sp_ma_errors.append(abs(vz_sp_ma - gt_vz));     sift_ma_errors.append(abs(vz_sift_ma - gt_vz))
    sp_lp_errors.append(abs(vz_sp_lp - gt_vz));     sift_lp_errors.append(abs(vz_sift_lp - gt_vz))

    sp_ma_pos_errors.append(np.linalg.norm(pos_sp_ma - pos_gt));     sift_ma_pos_errors.append(np.linalg.norm(pos_sift_ma - pos_gt))
    sp_lp_pos_errors.append(np.linalg.norm(pos_sp_lp - pos_gt));     sift_lp_pos_errors.append(np.linalg.norm(pos_sift_lp - pos_gt))


    # Store History


    vx_sp_ma_history.append(vx_sp_ma);  vy_sp_ma_history.append(vy_sp_ma);  vz_sp_ma_history.append(vz_sp_ma)
    vx_sp_lp_history.append(vx_sp_lp);  vy_sp_lp_history.append(vy_sp_lp);  vz_sp_lp_history.append(vz_sp_lp)

    vx_sift_ma_history.append(vx_sift_ma);  vy_sift_ma_history.append(vy_sift_ma);  vz_sift_ma_history.append(vz_sift_ma)
    vx_sift_lp_history.append(vx_sift_lp);  vy_sift_lp_history.append(vy_sift_lp);  vz_sift_lp_history.append(vz_sift_lp)

    vx_gt_history.append(gt_vx);  vy_gt_history.append(gt_vy);  vz_gt_history.append(gt_vz)

    x_sp_ma_hist.append(pos_sp_ma[0]);     z_sp_ma_hist.append(pos_sp_ma[2])
    x_sp_lp_hist.append(pos_sp_lp[0]);     z_sp_lp_hist.append(pos_sp_lp[2])
    x_sift_ma_hist.append(pos_sift_ma[0]); z_sift_ma_hist.append(pos_sift_ma[2])
    x_sift_lp_hist.append(pos_sift_lp[0]); z_sift_lp_hist.append(pos_sift_lp[2])
    x_gt_hist.append(pos_gt[0]);           z_gt_hist.append(pos_gt[2])


    # Tracking Maintenance


    left_frame_old = left_frame_new
    disparity_old  = disparity

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


print(f"  SuperPoint  + Moving Average   Vz MAE: {np.mean(sp_ma_errors):.4f} m/s")
print(f"  SuperPoint  + Low Pass Filter  Vz MAE: {np.mean(sp_lp_errors):.4f} m/s")
print(f"  SIFT        + Moving Average   Vz MAE: {np.mean(sift_ma_errors):.4f} m/s")
print(f"  SIFT        + Low Pass Filter  Vz MAE: {np.mean(sift_lp_errors):.4f} m/s")

print(f"  SuperPoint  + Moving Average   Position MAE: {np.mean(sp_ma_pos_errors):.4f} m")
print(f"  SuperPoint  + Low Pass Filter  Position MAE: {np.mean(sp_lp_pos_errors):.4f} m")
print(f"  SIFT        + Moving Average   Position MAE: {np.mean(sift_ma_pos_errors):.4f} m")
print(f"  SIFT        + Low Pass Filter  Position MAE: {np.mean(sift_lp_pos_errors):.4f} m")

cv2.destroyAllWindows()


# Final Graphs — 4 canvases, each with Vx / Vy / Vz vs Ground Truth


def plot_canvas(title, vx_est, vy_est, vz_est, color):
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    axes[0].plot(vx_est,       color=color,   label=title); axes[0].plot(vx_gt_history, color="black", label="Ground Truth"); axes[0].set_ylabel("Vx (m/s)"); axes[0].legend()
    axes[1].plot(vy_est,       color=color,   label=title); axes[1].plot(vy_gt_history, color="black", label="Ground Truth"); axes[1].set_ylabel("Vy (m/s)"); axes[1].legend()
    axes[2].plot(vz_est,       color=color,   label=title); axes[2].plot(vz_gt_history, color="black", label="Ground Truth"); axes[2].set_ylabel("Vz (m/s)"); axes[2].set_xlabel("Frame"); axes[2].legend()
    fig.suptitle(title)
    plt.tight_layout()

plot_canvas("SuperPoint + Moving Average",  vx_sp_ma_history,   vy_sp_ma_history,   vz_sp_ma_history,   color="royalblue")
plot_canvas("SuperPoint + Low Pass Filter", vx_sp_lp_history,   vy_sp_lp_history,   vz_sp_lp_history,   color="dodgerblue")
plot_canvas("SIFT + Moving Average",        vx_sift_ma_history, vy_sift_ma_history, vz_sift_ma_history, color="limegreen")
plot_canvas("SIFT + Low Pass Filter",       vx_sift_lp_history, vy_sift_lp_history, vz_sift_lp_history, color="mediumseagreen")


# Final Graphs — 4 top-down (X-Z plane) trajectory canvases: calculated vs ground truth


def plot_trajectory(title, x_est, z_est, color):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(x_est,     z_est,     color=color,   label=title)
    ax.plot(x_gt_hist, z_gt_hist, color="black", label="Ground Truth")
    ax.set_xlabel("X (m)"); ax.set_ylabel("Z (m)"); ax.set_title(title + " - Top Down")
    ax.axis("equal"); ax.legend()
    plt.tight_layout()

plot_trajectory("SuperPoint + Moving Average",  x_sp_ma_hist,   z_sp_ma_hist,   color="royalblue")
plot_trajectory("SuperPoint + Low Pass Filter", x_sp_lp_hist,   z_sp_lp_hist,   color="dodgerblue")
plot_trajectory("SIFT + Moving Average",        x_sift_ma_hist, z_sift_ma_hist, color="limegreen")
plot_trajectory("SIFT + Low Pass Filter",       x_sift_lp_hist, z_sift_lp_hist, color="mediumseagreen")

plt.show()


# 3D Point Cloud Reconstruction (SIFT keypoints, Open3D)
# Two separate clouds: one placed via ground-truth pose, one via GT rotation + SIFT+LP estimated position


pc_gt_all  = np.vstack(pointcloud_world_gt)  if pointcloud_world_gt  else np.empty((0, 3))
pc_est_all = np.vstack(pointcloud_world_est) if pointcloud_world_est else np.empty((0, 3))

pcd_gt = o3d.geometry.PointCloud()
pcd_gt.points = o3d.utility.Vector3dVector(pc_gt_all)
pcd_gt.paint_uniform_color([0.1, 0.1, 0.1])   # black-ish: ground-truth-pose reconstruction

pcd_est = o3d.geometry.PointCloud()
pcd_est.points = o3d.utility.Vector3dVector(pc_est_all)
pcd_est.paint_uniform_color([0.2, 0.8, 0.2])  # green: SIFT+LP estimated-pose reconstruction

print(f"GT-pose point cloud: {len(pc_gt_all)} points")
print(f"Estimated-pose point cloud: {len(pc_est_all)} points")

o3d.visualization.draw_geometries(
    [pcd_gt, pcd_est],
    window_name="SIFT 3D Point Cloud: Ground Truth (black) vs Estimated (green)"
)