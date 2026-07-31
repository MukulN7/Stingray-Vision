# Importing Libraries


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
DATASET_PATH = r"E:\i20090804_084719_cv"
CALIB_PATH   = r"E:\ScottReef_20090804_084719.calib"
VELOCITY_CSV_PATH = r"D:\Stingray Robotics\velocity_output_scottreef.csv"
POSITION_CSV_PATH = r"D:\Stingray Robotics\position_output_scottreef.csv"

MIN_FEATURES = 400
START_FRAME  = 0

# Cap how many frames this run processes, counting from START_FRAME.
# Set to None to run the full dataset (all available frame pairs).
MAX_FRAMES = None

left_images  = sorted(glob.glob(os.path.join(DATASET_PATH, "*_LC16.*")))
right_images = sorted(glob.glob(os.path.join(DATASET_PATH, "*_RM16.*")))
n_pairs      = min(len(left_images), len(right_images))
left_images, right_images = left_images[:n_pairs], right_images[:n_pairs]

END_FRAME = len(left_images) if MAX_FRAMES is None else min(len(left_images), START_FRAME + MAX_FRAMES)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load Timestamps (parsed from filenames) and Calibration (raw, unrectified stereo)


def parse_timestamp(path):
    # Filenames look like PR_20090804_090219_414_LC16.ext -> HHMMSS_mmm
    stem = os.path.splitext(os.path.basename(path))[0]
    hhmmss, ms = stem.split("_")[2], stem.split("_")[3]
    h, m, s = int(hhmmss[0:2]), int(hhmmss[2:4]), int(hhmmss[4:6])
    return h * 3600 + m * 60 + s + int(ms) / 1000.0

timestamps = np.array([parse_timestamp(p) for p in left_images])

def parse_calib_cam(line):
    vals = list(map(float, line.split()))
    w, h = int(vals[0]), int(vals[1])
    K    = np.array(vals[2:11], dtype=np.float64).reshape(3, 3)
    dist = np.array(vals[11:15], dtype=np.float64)
    R    = np.array(vals[15:24], dtype=np.float64).reshape(3, 3)
    T    = np.array(vals[24:27], dtype=np.float64)
    return w, h, K, dist, R, T

with open(CALIB_PATH, "r") as f:
    calib_lines = [line for line in f.readlines() if line.strip() != ""]

_, _, K0, dist0, _, _       = parse_calib_cam(calib_lines[1])
img_w, img_h, K1, dist1, R1, T1 = parse_calib_cam(calib_lines[2])
image_size = (img_w, img_h)

# Raw calibration isn't rectified (right camera has a non-identity R), so stereo
# rectification is needed before disparity/feature matching will work correctly.
rect_R0, rect_R1, P0, P1, Q, roi0, roi1 = cv2.stereoRectify(
    K0, dist0, K1, dist1, image_size, R1, T1, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)

map0x, map0y = cv2.initUndistortRectifyMap(K0, dist0, rect_R0, P0, image_size, cv2.CV_32FC1)
map1x, map1y = cv2.initUndistortRectifyMap(K1, dist1, rect_R1, P1, image_size, cv2.CV_32FC1)

fx       = P0[0, 0]
cx       = P0[0, 2]
cy       = P0[1, 2]
baseline = abs(P1[0, 3] / fx)

def read_rectified(path, is_left):
    frame = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if is_left:
        return cv2.remap(frame, map0x, map0y, cv2.INTER_LINEAR)
    return cv2.remap(frame, map1x, map1y, cv2.INTER_LINEAR)


# Load SuperPoint + SuperGlue


# Denser keypoints + a looser match-filter threshold than KITTI's defaults: at 1 Hz
# the frame-to-frame baseline is much larger than KITTI's 10 Hz, so more candidate
# points and a more permissive match filter help LightGlue survive that motion.
extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)
matcher   = LightGlue(features="superpoint", filter_threshold=0.05).eval().to(device)

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

    return cam_vel[0], cam_vel[1], cam_vel[2]


# Initial Feature Detection (first frame)


left_frame_old  = read_rectified(left_images[START_FRAME],  is_left=True)
right_frame_old = read_rectified(right_images[START_FRAME], is_left=False)

# Disparity for the starting frame, so the first loop iteration has an "old" depth to average with
# Underwater seafloor imagery is low-texture/low-contrast compared to KITTI's road
# scenes, and this rig sits at a short, fairly constant altitude (~2-4 m). A larger
# blockSize averages over more of the weak texture, a lower uniquenessRatio stops
# good-but-not-super-distinctive matches from being thrown out, and preFilterCap
# helps normalize the lighting/caustics variation. numDisparities=128 (vs. the
# original 64) is sized from this calib file's actual fx (~1881 px, post-rectify)
# and baseline (~0.070 m): disparity = fx*baseline/Z, so 64 only reached down to
# Z ~ 2.06 m, while recorded altitude dips to ~2.0-2.1 m in places — 128 covers
# down to Z ~ 1.03 m with margin.
stereo_init   = cv2.StereoSGBM_create(minDisparity=0, numDisparities=128, blockSize=7,
                    P1=8*3*7**2, P2=32*3*7**2, disp12MaxDiff=1,
                    uniquenessRatio=5, speckleWindowSize=100, speckleRange=32,
                    preFilterCap=31, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
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

# Outlier clamp buffers per method (used for Vz only) — Vz is the down-facing camera's
# principal-ray axis here (altitude direction), so it should stay near zero and slowly
# varying; Vx/Vy carry the real, legitimately-oscillating survey motion (unlike KITTI's
# forward-facing rig, where Z was the direction of travel). That's still the axis
# expected to be steady, so the clamp assignment below is unchanged from KITTI.
buf_sp_vz   = make_buffer()
buf_sift_vz = make_buffer()

# Hold-last-valid-estimate state: when a frame doesn't have enough correspondences
# (e.g. during a fast turn), we reuse the last real measurement instead of feeding a
# fabricated 0.0 into the MA/LP filters, which would otherwise bias the smoothed
# estimate toward zero right when the true velocity is largest.
last_valid_sp   = {"vx": 0.0, "vy": 0.0, "vz": 0.0}
last_valid_sift = {"vx": 0.0, "vy": 0.0, "vz": 0.0}

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


# Dead-Reckoning Position Setup (one accumulator per method)
# No ground-truth/IMU orientation is available for this dataset, so camera-frame
# velocity is integrated directly (pos = pos + v_cam * dt), starting at the origin.


pos_sp_ma   = np.zeros(3)
pos_sp_lp   = np.zeros(3)
pos_sift_ma = np.zeros(3)
pos_sift_lp = np.zeros(3)

# Top-down (X-Z plane) position history, per method
x_sp_ma_hist,   z_sp_ma_hist   = [], []
x_sp_lp_hist,   z_sp_lp_hist   = [], []
x_sift_ma_hist, z_sift_ma_hist = [], []
x_sift_lp_hist, z_sift_lp_hist = [], []


# Main Loop


for frame_idx in tqdm(range(START_FRAME + 1, END_FRAME), desc="Processing frames"):

    frame_start_time = time.time()

    dt            = timestamps[frame_idx] - timestamps[frame_idx - 1]
    left_frame_new = read_rectified(left_images[frame_idx],  is_left=True)
    left_img       = left_frame_new
    right_img      = read_rectified(right_images[frame_idx], is_left=False)

    # Disparity (shared for both methods)
    # Same underwater-tuned params as the initial disparity computation above.
    stereo    = cv2.StereoSGBM_create(minDisparity=0, numDisparities=128, blockSize=7,
                    P1=8*3*7**2, P2=32*3*7**2, disp12MaxDiff=1,
                    uniquenessRatio=5, speckleWindowSize=100, speckleRange=32,
                    preFilterCap=31, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
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
        # Slightly looser threshold than OpenCV's default (3.0 -> not needed to change,
        # but made explicit + confidence raised) since larger inter-frame motion at 1 Hz
        # produces bigger, still-valid pixel offsets that a tight tolerance would reject.
        F, mask = cv2.findFundamentalMat(sp_old, sp_new, cv2.FM_RANSAC, ransacReprojThreshold=2.5, confidence=0.995)
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

    vx_sp, vy_sp, vz_sp = last_valid_sp["vx"], last_valid_sp["vy"], last_valid_sp["vz"]
    if len(sp_depths_new) >= 6:  # 6 correspondences = 12 equations for the 6-DOF solve
        vx_sp, vy_sp, vz_sp = compute_velocity(sp_old, sp_new, sp_depths_old, sp_depths_new, dt)
        last_valid_sp["vx"], last_valid_sp["vy"], last_valid_sp["vz"] = vx_sp, vy_sp, vz_sp

    # Outlier clamp only on Vz (Vx/Vy oscillate around zero; clamping them zeros out the real signal)
    vz_sp = reject_outlier(buf_sp_vz, vz_sp)

    vx_sp_ma = ma_update(ma_sp_vx, vx_sp);   vy_sp_ma = ma_update(ma_sp_vy, vy_sp);   vz_sp_ma = ma_update(ma_sp_vz, vz_sp)
    vx_sp_lp = lp_update(lp_sp_vx, vx_sp);  vy_sp_lp = lp_update(lp_sp_vy, vy_sp);  vz_sp_lp = lp_update(lp_sp_vz, vz_sp)


    # SIFT + Optical Flow


    # Wider window + more pyramid levels than KLT's defaults (21x21, 3 levels):
    # at 1 Hz the frame-to-frame displacement is much larger than KITTI's 10 Hz,
    # so KLT needs a bigger search range to converge instead of losing the point.
    lk_params = dict(
        winSize=(31, 31),
        maxLevel=4,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    pts_new_sift, status, _ = cv2.calcOpticalFlowPyrLK(left_frame_old, left_frame_new, pts_old_sift, None, **lk_params)

    sift_old = pts_old_sift[status == 1]
    sift_new = pts_new_sift[status == 1]

    # Same minimum-point guard as the SuperPoint branch above.
    if len(sift_old) >= 8:
        F, mask = cv2.findFundamentalMat(sift_old, sift_new, cv2.FM_RANSAC, ransacReprojThreshold=2.5, confidence=0.995)
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

    vx_sift, vy_sift, vz_sift = last_valid_sift["vx"], last_valid_sift["vy"], last_valid_sift["vz"]
    if len(sift_depths_new) >= 6:  # 6 correspondences = 12 equations for the 6-DOF solve
        vx_sift, vy_sift, vz_sift = compute_velocity(sift_old, sift_new, sift_depths_old, sift_depths_new, dt)
        last_valid_sift["vx"], last_valid_sift["vy"], last_valid_sift["vz"] = vx_sift, vy_sift, vz_sift

    # Outlier clamp only on Vz (Vx/Vy oscillate around zero; clamping them zeros out the real signal)
    vz_sift = reject_outlier(buf_sift_vz, vz_sift)

    vx_sift_ma = ma_update(ma_sift_vx, vx_sift);   vy_sift_ma = ma_update(ma_sift_vy, vy_sift);   vz_sift_ma = ma_update(ma_sift_vz, vz_sift)
    vx_sift_lp = lp_update(lp_sift_vx, vx_sift);  vy_sift_lp = lp_update(lp_sift_vy, vy_sift);  vz_sift_lp = lp_update(lp_sift_vz, vz_sift)


    # Dead Reckoning: integrate camera-frame velocity directly (x = x + v*dt)
    # No ground-truth/IMU orientation is available for this dataset.


    pos_sp_ma   += np.array([vx_sp_ma,   vy_sp_ma,   vz_sp_ma])   * dt
    pos_sp_lp   += np.array([vx_sp_lp,   vy_sp_lp,   vz_sp_lp])   * dt
    pos_sift_ma += np.array([vx_sift_ma, vy_sift_ma, vz_sift_ma]) * dt
    pos_sift_lp += np.array([vx_sift_lp, vy_sift_lp, vz_sift_lp]) * dt


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
    for i, frame_idx in enumerate(range(START_FRAME + 1, END_FRAME)):
        writer.writerow([
            frame_idx,
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
    for i, frame_idx in enumerate(range(START_FRAME + 1, END_FRAME)):
        writer.writerow([
            frame_idx,
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