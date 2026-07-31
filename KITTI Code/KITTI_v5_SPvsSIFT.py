# Importing Libraries


import numpy as np
import cv2
import os
import glob
import time
import torch
import matplotlib.pyplot as plt
from lightglue import LightGlue, SuperPoint
from lightglue.utils import rbd


# Configuration


# Paths
DATASET_PATH = r"D:\Stingray Robotics\archive\dataset\sequences\07"
LEFT_DIR  = os.path.join(DATASET_PATH, "image_0")
RIGHT_DIR = os.path.join(DATASET_PATH, "image_1")
CALIB     = os.path.join(DATASET_PATH, "calib.txt")
TIMES     = os.path.join(DATASET_PATH, "times.txt")
POSES     = r"D:\Stingray Robotics\archive\data_odometry_poses\dataset\poses\07.txt"

MIN_FEATURES = 350
START_FRAME  = 0
END_FRAME    = 400

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

    cam_vel = np.linalg.pinv(J) @ np.array(img_disp_vec)
    return cam_vel[0], cam_vel[1], cam_vel[2]


# Initial Feature Detection (first frame)


left_frame_old = cv2.imread(left_images[START_FRAME], cv2.IMREAD_GRAYSCALE)

# SuperPoint initial
feats_old = extract_superpoint(left_frame_old)

# SIFT initial
kps, _ = sift.detectAndCompute(left_frame_old, None)
pts_old_sift = np.array([kp.pt for kp in kps], dtype=np.float32).reshape(-1, 1, 2)


# Kalman Filter Setup (one for each)


def make_kf():
    return {"x": 5.0, "P": 1.0, "Q": 1.5, "R": 10}

def kf_update(kf, measurement):
    kf["P"] = kf["P"] + kf["Q"]
    K       = kf["P"] / (kf["P"] + kf["R"])
    kf["x"] = kf["x"] + K * (measurement - kf["x"])
    kf["P"] = (1 - K) * kf["P"]
    return kf["x"]

kf_sp   = make_kf()   # SuperPoint/SuperGlue Kalman
kf_sift = make_kf()   # SIFT Kalman


# Live Plot Setup — all three Vz on one graph


vz_sp_history   = []
vz_sift_history = []
vz_gt_history   = []

plt.ion()
fig, ax = plt.subplots()
line_sp,   = ax.plot([], [], color="blue",  label="Vz SuperPoint")
line_sift, = ax.plot([], [], color="green", label="Vz SIFT")
line_gt,   = ax.plot([], [], color="red",   label="Vz Ground Truth")
ax.set_xlabel("Frame")
ax.set_ylabel("Vz (m/s)")
ax.set_title("Vz Comparison: SIFT vs SuperPoint vs Ground Truth")
ax.legend()


# Error Accumulators


sp_errors   = []
sift_errors = []


# Main Loop


for frame_idx in range(START_FRAME + 1, END_FRAME):

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
    vz_sp_kf = kf_update(kf_sp, vz_sp)


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
    vz_sift_kf = kf_update(kf_sift, vz_sift)


    # Ground Truth 


    R        = poses[frame_idx][:, :3]
    v_world  = (poses[frame_idx][:, 3] - poses[frame_idx - 1][:, 3]) / dt
    v_camera = R.T @ v_world
    gt_vx, gt_vy, gt_vz = v_camera[0], v_camera[1], v_camera[2]


    # Error


    sp_errors.append(abs(vz_sp_kf - gt_vz))
    sift_errors.append(abs(vz_sift_kf - gt_vz))


    # Display


    fps           = 1.0 / (time.time() - frame_start_time)
    display_frame = cv2.cvtColor(left_frame_new, cv2.COLOR_GRAY2BGR)

    # SuperPoint velocities
    cv2.putText(display_frame, f"[SP]   Vx:{vx_sp:.2f} Vy:{vy_sp:.2f} Vz:{vz_sp_kf:.2f} m/s",   (20,  40), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 200,   0), 2)
    # SIFT velocities
    cv2.putText(display_frame, f"[SIFT] Vx:{vx_sift:.2f} Vy:{vy_sift:.2f} Vz:{vz_sift_kf:.2f} m/s", (20,  75), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,   255,   0), 2)
    # Ground truth velocities
    cv2.putText(display_frame, f"[GT]   Vx:{gt_vx:.2f} Vy:{gt_vy:.2f} Vz:{gt_vz:.2f} m/s",      (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,   200, 255), 2)
    cv2.putText(display_frame, f"FPS: {fps:.1f}", (20, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 180), 2)

    cv2.imshow("SIFT vs SuperPoint — Live Velocity", display_frame)


    # Live Plot Update


    vz_sp_history.append(vz_sp_kf)
    vz_sift_history.append(vz_sift_kf)
    vz_gt_history.append(gt_vz)

    line_sp.set_data(range(len(vz_sp_history)),   vz_sp_history)
    line_sift.set_data(range(len(vz_sift_history)), vz_sift_history)
    line_gt.set_data(range(len(vz_gt_history)),   vz_gt_history)
    ax.relim()
    ax.autoscale_view()
    plt.pause(0.001)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


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


print(f"  SuperPoint + SuperGlue  MAE: {np.mean(sp_errors):.4f} m/s")
print(f"  SIFT + Optical Flow     MAE: {np.mean(sift_errors):.4f} m/s")

if np.mean(sp_errors) < np.mean(sift_errors):
    print(" SuperPoint + SuperGlue is MORE accurate")
else:
    print(" SIFT + Optical Flow is MORE accurate")

cv2.destroyAllWindows()
