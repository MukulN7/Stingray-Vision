# Importing Libraries


import numpy as np
import cv2
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
LEFT_VIDEO  = r"D:\Stingray Robotics\Sim Data 1\camA.mp4"
RIGHT_VIDEO = r"D:\Stingray Robotics\Sim Data 1\camB.mp4"
CALIB       = r"D:\Stingray Robotics\Odometry on Robo Sim\robotsimcalib.txt"
VELOCITY_CSV_PATH = r"D:\Stingray Robotics\velocity_output_sim9.csv"
POSITION_CSV_PATH = r"D:\Stingray Robotics\position_output_sim9.csv"

MIN_FEATURES = 400
MAX_FRAMES   = None  # set to an int to cap how many frames are processed; None runs the full video

cap_left  = cv2.VideoCapture(LEFT_VIDEO)
cap_right = cv2.VideoCapture(RIGHT_VIDEO)

fps = cap_left.get(cv2.CAP_PROP_FPS)
dt  = 1.0 / fps

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load Calibration


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

def compute_velocity(goodpts_old, goodpts_new, depths_old, depths_new, dt):
    # Evaluating the Jacobian at the midpoint of the frame interval (avg of old & new depth) 
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

    return cam_vel[0], cam_vel[1], cam_vel[2], cam_vel[3], cam_vel[4], cam_vel[5]


# Initial Feature Detection (first frame)


ret_l, left_frame_old  = cap_left.read()
ret_r, right_frame_old = cap_right.read()
left_frame_old  = cv2.cvtColor(left_frame_old,  cv2.COLOR_BGR2GRAY)
right_frame_old = cv2.cvtColor(right_frame_old, cv2.COLOR_BGR2GRAY)

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


# Rolling-window outlier clamp (median/MAD) 

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

# Moving Average buffers per axis, per method (linear + angular)
ma_sp_vx,   ma_sp_vy,   ma_sp_vz   = make_ma(), make_ma(), make_ma()
ma_sift_vx, ma_sift_vy, ma_sift_vz = make_ma(), make_ma(), make_ma()

ma_sp_wx,   ma_sp_wy,   ma_sp_wz   = make_ma(), make_ma(), make_ma()
ma_sift_wx, ma_sift_wy, ma_sift_wz = make_ma(), make_ma(), make_ma()

# Low Pass filters per axis, per method (linear + angular)
lp_sp_vx,   lp_sp_vy,   lp_sp_vz   = make_lp(), make_lp(), make_lp()
lp_sift_vx, lp_sift_vy, lp_sift_vz = make_lp(), make_lp(), make_lp()

lp_sp_wx,   lp_sp_wy,   lp_sp_wz   = make_lp(), make_lp(), make_lp()
lp_sift_wx, lp_sift_wy, lp_sift_wz = make_lp(), make_lp(), make_lp()


# History Lists


vx_sp_ma_history,   vy_sp_ma_history,   vz_sp_ma_history   = [], [], []
vx_sp_lp_history,   vy_sp_lp_history,   vz_sp_lp_history   = [], [], []

vx_sift_ma_history, vy_sift_ma_history, vz_sift_ma_history = [], [], []
vx_sift_lp_history, vy_sift_lp_history, vz_sift_lp_history = [], [], []


# Dead-Reckoning Position Setup (world-frame position, one accumulator per method)
# No ground-truth pose is available here, so orientation is instead tracked per method by integrating
# the Jacobian solve's angular velocity (R = R @ Rodrigues(omega*dt)), then: pos = pos + (R @ v_cam) * dt


pos_sp_ma   = np.zeros(3)
pos_sp_lp   = np.zeros(3)
pos_sift_ma = np.zeros(3)
pos_sift_lp = np.zeros(3)

R_sp_ma   = np.eye(3)
R_sp_lp   = np.eye(3)
R_sift_ma = np.eye(3)
R_sift_lp = np.eye(3)

# Top-down (X-Z plane) position history, per method
x_sp_ma_hist,   z_sp_ma_hist   = [], []
x_sp_lp_hist,   z_sp_lp_hist   = [], []
x_sift_ma_hist, z_sift_ma_hist = [], []
x_sift_lp_hist, z_sift_lp_hist = [], []


# Main Loop


total_frames = int(cap_left.get(cv2.CAP_PROP_FRAME_COUNT))
if MAX_FRAMES is not None:
    total_frames = min(total_frames, MAX_FRAMES)
pbar = tqdm(total=total_frames, desc="Processing frames")
frame_idx = 0
while True:

    frame_start_time = time.time()

    ret_l, left_frame_new  = cap_left.read()
    ret_r, right_frame_new = cap_right.read()
    if not ret_l or not ret_r:
        break
    if MAX_FRAMES is not None and frame_idx >= MAX_FRAMES:
        break

    left_frame_new = cv2.cvtColor(left_frame_new, cv2.COLOR_BGR2GRAY)
    left_img       = left_frame_new
    right_img      = cv2.cvtColor(right_frame_new, cv2.COLOR_BGR2GRAY)

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
        try:
            F, mask = cv2.findFundamentalMat(sp_old, sp_new, cv2.FM_RANSAC)
        except cv2.error:
            mask = None
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

    vx_sp = vy_sp = vz_sp = wx_sp = wy_sp = wz_sp = 0.0
    if len(sp_depths_new) >= 6:  # 6 correspondences = 12 equations for the 6-DOF solve
        vx_sp, vy_sp, vz_sp, wx_sp, wy_sp, wz_sp = compute_velocity(sp_old, sp_new, sp_depths_old, sp_depths_new, dt)

    # Outlier clamp only on Vz (Vx/Vy oscillate around zero; clamping them zeros out the real signal)
    vz_sp = reject_outlier(buf_sp_vz, vz_sp)

    vx_sp_ma = ma_update(ma_sp_vx, vx_sp);   vy_sp_ma = ma_update(ma_sp_vy, vy_sp);   vz_sp_ma = ma_update(ma_sp_vz, vz_sp)
    vx_sp_lp = lp_update(lp_sp_vx, vx_sp);  vy_sp_lp = lp_update(lp_sp_vy, vy_sp);  vz_sp_lp = lp_update(lp_sp_vz, vz_sp)

    wx_sp_ma = ma_update(ma_sp_wx, wx_sp);   wy_sp_ma = ma_update(ma_sp_wy, wy_sp);   wz_sp_ma = ma_update(ma_sp_wz, wz_sp)
    wx_sp_lp = lp_update(lp_sp_wx, wx_sp);  wy_sp_lp = lp_update(lp_sp_wy, wy_sp);  wz_sp_lp = lp_update(lp_sp_wz, wz_sp)


    # SIFT + Optical Flow


    if len(pts_old_sift) > 0:
        pts_new_sift, status, _ = cv2.calcOpticalFlowPyrLK(left_frame_old, left_frame_new, pts_old_sift, None)
        sift_old = pts_old_sift[status == 1]
        sift_new = pts_new_sift[status == 1]
    else:
        sift_old = np.empty((0, 2), dtype=np.float32)
        sift_new = np.empty((0, 2), dtype=np.float32)

    # Same minimum-point guard as the SuperPoint branch above.
    if len(sift_old) >= 8:
        try:
            F, mask = cv2.findFundamentalMat(sift_old, sift_new, cv2.FM_RANSAC)
        except cv2.error:
            mask = None
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

    vx_sift = vy_sift = vz_sift = wx_sift = wy_sift = wz_sift = 0.0
    if len(sift_depths_new) >= 6:  # 6 correspondences = 12 equations for the 6-DOF solve
        vx_sift, vy_sift, vz_sift, wx_sift, wy_sift, wz_sift = compute_velocity(sift_old, sift_new, sift_depths_old, sift_depths_new, dt)

    # Outlier clamp only on Vz (Vx/Vy oscillate around zero; clamping them zeros out the real signal)
    vz_sift = reject_outlier(buf_sift_vz, vz_sift)

    vx_sift_ma = ma_update(ma_sift_vx, vx_sift);   vy_sift_ma = ma_update(ma_sift_vy, vy_sift);   vz_sift_ma = ma_update(ma_sift_vz, vz_sift)
    vx_sift_lp = lp_update(lp_sift_vx, vx_sift);  vy_sift_lp = lp_update(lp_sift_vy, vy_sift);  vz_sift_lp = lp_update(lp_sift_vz, vz_sift)

    wx_sift_ma = ma_update(ma_sift_wx, wx_sift);   wy_sift_ma = ma_update(ma_sift_wy, wy_sift);   wz_sift_ma = ma_update(ma_sift_wz, wz_sift)
    wx_sift_lp = lp_update(lp_sift_wx, wx_sift);  wy_sift_lp = lp_update(lp_sift_wy, wy_sift);  wz_sift_lp = lp_update(lp_sift_wz, wz_sift)


    # Dead Reckoning: orientation update from angular velocity (R = R @ Rodrigues(omega*dt)),
    # then rotate camera-frame velocity into world frame before integrating position (x = x + v_world*dt)


    R_sp_ma   = R_sp_ma   @ cv2.Rodrigues(np.array([wx_sp_ma,   wy_sp_ma,   wz_sp_ma])   * dt)[0]
    R_sp_lp   = R_sp_lp   @ cv2.Rodrigues(np.array([wx_sp_lp,   wy_sp_lp,   wz_sp_lp])   * dt)[0]
    R_sift_ma = R_sift_ma @ cv2.Rodrigues(np.array([wx_sift_ma, wy_sift_ma, wz_sift_ma]) * dt)[0]
    R_sift_lp = R_sift_lp @ cv2.Rodrigues(np.array([wx_sift_lp, wy_sift_lp, wz_sift_lp]) * dt)[0]

    pos_sp_ma   += (R_sp_ma   @ np.array([vx_sp_ma,   vy_sp_ma,   vz_sp_ma]))   * dt
    pos_sp_lp   += (R_sp_lp   @ np.array([vx_sp_lp,   vy_sp_lp,   vz_sp_lp]))   * dt
    pos_sift_ma += (R_sift_ma @ np.array([vx_sift_ma, vy_sift_ma, vz_sift_ma])) * dt
    pos_sift_lp += (R_sift_lp @ np.array([vx_sift_lp, vy_sift_lp, vz_sift_lp])) * dt


    # Store History


    vx_sp_ma_history.append(vx_sp_ma);  vy_sp_ma_history.append(vy_sp_ma);  vz_sp_ma_history.append(vz_sp_ma)
    vx_sp_lp_history.append(vx_sp_lp);  vy_sp_lp_history.append(vy_sp_lp);  vz_sp_lp_history.append(vz_sp_lp)

    vx_sift_ma_history.append(vx_sift_ma);  vy_sift_ma_history.append(vy_sift_ma);  vz_sift_ma_history.append(vz_sift_ma)
    vx_sift_lp_history.append(vx_sift_lp);  vy_sift_lp_history.append(vy_sift_lp);  vz_sift_lp_history.append(vz_sift_lp)

    x_sp_ma_hist.append(pos_sp_ma[0]);     z_sp_ma_hist.append(pos_sp_ma[2])
    x_sp_lp_hist.append(pos_sp_lp[0]);     z_sp_lp_hist.append(pos_sp_lp[2])
    x_sift_ma_hist.append(pos_sift_ma[0]); z_sift_ma_hist.append(pos_sift_ma[2])
    x_sift_lp_hist.append(pos_sift_lp[0]); z_sift_lp_hist.append(pos_sift_lp[2])


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

    frame_idx += 1
    pbar.update(1)

pbar.close()
cap_left.release()
cap_right.release()
cv2.destroyAllWindows()


# Save Velocity + Position Results to CSV


with open(VELOCITY_CSV_PATH, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "frame",
        "vx_sp_ma", "vy_sp_ma", "vz_sp_ma",
        "vx_sp_lp", "vy_sp_lp", "vz_sp_lp",
        "vx_sift_ma", "vy_sift_ma", "vz_sift_ma",
        "vx_sift_lp", "vy_sift_lp", "vz_sift_lp",
    ])
    for i in range(len(vx_sp_ma_history)):
        writer.writerow([
            i + 1,
            vx_sp_ma_history[i],   vy_sp_ma_history[i],   vz_sp_ma_history[i],
            vx_sp_lp_history[i],   vy_sp_lp_history[i],   vz_sp_lp_history[i],
            vx_sift_ma_history[i], vy_sift_ma_history[i], vz_sift_ma_history[i],
            vx_sift_lp_history[i], vy_sift_lp_history[i], vz_sift_lp_history[i],
        ])

with open(POSITION_CSV_PATH, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "frame",
        "x_sp_ma", "z_sp_ma",
        "x_sp_lp", "z_sp_lp",
        "x_sift_ma", "z_sift_ma",
        "x_sift_lp", "z_sift_lp",
    ])
    for i in range(len(x_sp_ma_hist)):
        writer.writerow([
            i + 1,
            x_sp_ma_hist[i],   z_sp_ma_hist[i],
            x_sp_lp_hist[i],   z_sp_lp_hist[i],
            x_sift_ma_hist[i], z_sift_ma_hist[i],
            x_sift_lp_hist[i], z_sift_lp_hist[i],
        ])

print(f"Saved velocity results to {VELOCITY_CSV_PATH}")
print(f"Saved position results to {POSITION_CSV_PATH}")


# Final Graphs — 4 canvases, each with Vx / Vy / Vz


def plot_canvas(title, vx_est, vy_est, vz_est, color):
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    axes[0].plot(vx_est, color=color, label=title); axes[0].set_ylabel("Vx (m/s)"); axes[0].legend()
    axes[1].plot(vy_est, color=color, label=title); axes[1].set_ylabel("Vy (m/s)"); axes[1].legend()
    axes[2].plot(vz_est, color=color, label=title); axes[2].set_ylabel("Vz (m/s)"); axes[2].set_xlabel("Frame"); axes[2].legend()
    fig.suptitle(title)
    plt.tight_layout()

plot_canvas("SuperPoint + Moving Average",  vx_sp_ma_history,   vy_sp_ma_history,   vz_sp_ma_history,   color="royalblue")
plot_canvas("SuperPoint + Low Pass Filter", vx_sp_lp_history,   vy_sp_lp_history,   vz_sp_lp_history,   color="dodgerblue")
plot_canvas("SIFT + Moving Average",        vx_sift_ma_history, vy_sift_ma_history, vz_sift_ma_history, color="limegreen")
plot_canvas("SIFT + Low Pass Filter",       vx_sift_lp_history, vy_sift_lp_history, vz_sift_lp_history, color="mediumseagreen")


# Final Graphs — 4 top-down (X-Z plane) trajectory canvases


def plot_trajectory(title, x_est, z_est, color):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(x_est, z_est, color=color, label=title)
    ax.set_xlabel("X (m)"); ax.set_ylabel("Z (m)"); ax.set_title(title + " - Top Down")
    ax.axis("equal"); ax.legend()
    plt.tight_layout()

plot_trajectory("SuperPoint + Moving Average",  x_sp_ma_hist,   z_sp_ma_hist,   color="royalblue")
plot_trajectory("SuperPoint + Low Pass Filter", x_sp_lp_hist,   z_sp_lp_hist,   color="dodgerblue")
plot_trajectory("SIFT + Moving Average",        x_sift_ma_hist, z_sift_ma_hist, color="limegreen")
plot_trajectory("SIFT + Low Pass Filter",       x_sift_lp_hist, z_sift_lp_hist, color="mediumseagreen")

plt.show()