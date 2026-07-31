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

# SuperPoint parameters
MIN_FEATURES = 350          # Min features to keep tracking before re-detecting
MAX_FRAMES   = 400          # To limit playback, None = all

# Frame range
START_FRAME = 0
END_FRAME   = 800

# Image Paths
left_images  = sorted(glob.glob(os.path.join(LEFT_DIR,  "*.png")))
right_images = sorted(glob.glob(os.path.join(RIGHT_DIR, "*.png")))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load timestamps from times.txt


with open(TIMES, "r") as f:
    timestamps = np.array([float(line.strip()) for line in f.readlines()])


# Load ground truth poses from poses file


with open(POSES, "r") as f:
    poses = [np.array(list(map(float, line.strip().split()))).reshape(3, 4) for line in f.readlines()]


# Load camera intrinsics from calib.txt


with open(CALIB, "r") as f:
    lines = f.readlines()

# Left camera projection matrix (P0)
P0 = np.array(
    list(map(float, lines[0].split()[1:])),
    dtype=np.float32
).reshape(3, 4)

# Right camera projection matrix (P1)
P1 = np.array(
    list(map(float, lines[1].split()[1:])),
    dtype=np.float32
).reshape(3, 4)

# Camera parameters
fx = P0[0, 0]
cx = P0[0, 2]
cy = P0[1, 2]

# Baseline (meters)
baseline = abs(P1[0, 3] / fx)


# Load SuperPoint + SuperGlue Models  ← CHANGED (was: sift = cv2.SIFT_create())


extractor = SuperPoint(max_num_keypoints=1024).eval().to(device)
matcher   = LightGlue(features="superpoint").eval().to(device)


# Helper: extract SuperPoint keypoints from a grayscale numpy frame


def extract_superpoint(gray_frame):
    tensor = torch.from_numpy(gray_frame).float() / 255.0
    tensor = tensor.unsqueeze(0).unsqueeze(0).to(device)   # shape: [1, 1, H, W]
    feats  = extractor.extract(tensor)
    return feats


# Feature Detection


# Load first frame in grayscale
left_frame_old = cv2.imread(left_images[START_FRAME], cv2.IMREAD_GRAYSCALE)

# Detect SuperPoint features of first frame  ← CHANGED (was: sift.detectAndCompute)
feats_old = extract_superpoint(left_frame_old)
pts_old   = rbd(feats_old)["keypoints"].cpu().numpy().reshape(-1, 1, 2)


# Kalman Filter Setup (1D, for Vz)


kf_x  = 5.0    # Initial state estimate (rough starting Vz)
kf_P  = 1.0    # Initial estimate uncertainty
kf_Q  = 1.5    # Process noise  (how much Vz can physically change per frame)
kf_R  = 10     # Measurement noise (how noisy our Vz estimate is)


# Live Vz Comparision Plot Setup


vz_history    = []
gt_vz_history = []

plt.ion()
fig, ax = plt.subplots()
line_vz,    = ax.plot([], [], color="blue",  label="Estimated Vz")
line_gt_vz, = ax.plot([], [], color="red",   label="GT Vz")
ax.set_xlabel("Frame")
ax.set_ylabel("Vz (m/s)")
ax.set_title("Vz: Estimated vs Ground Truth")
ax.legend()


# Main Loop


for frame_idx in range(START_FRAME + 1, END_FRAME):

    frame_start_time = time.time()


    # Feature Matching with SuperPoint + SuperGlue  ← CHANGED (was: calcOpticalFlowPyrLK)


    left_frame_new = cv2.imread(left_images[frame_idx], cv2.IMREAD_GRAYSCALE)

    feats_new    = extract_superpoint(left_frame_new)
    matches01    = matcher({"image0": feats_old, "image1": feats_new})

    f0, f1, m01  = rbd(feats_old), rbd(feats_new), rbd(matches01)
    matched_idxs = m01["matches"]                                      # shape: [N, 2]

    goodpts_old  = f0["keypoints"][matched_idxs[:, 0]].cpu().numpy()
    goodpts_new  = f1["keypoints"][matched_idxs[:, 1]].cpu().numpy()


    # Track Filtering


    # Filtering based on RANSAC  ← unchanged (removed LK status filter, not needed)
    F, mask = cv2.findFundamentalMat(goodpts_old, goodpts_new, cv2.FM_RANSAC)
    goodpts_old = goodpts_old[mask.ravel() == 1]
    goodpts_new = goodpts_new[mask.ravel() == 1]


    # Depth Estimation


    # Loading the current image pair in grayscale
    left_img  = cv2.imread(left_images[frame_idx],  cv2.IMREAD_GRAYSCALE)
    right_img = cv2.imread(right_images[frame_idx], cv2.IMREAD_GRAYSCALE)

    # Compute disparity map of the newer image pair
    stereo    = cv2.StereoSGBM_create(minDisparity=0, numDisparities=64, blockSize=5,
                    P1=8*3*5**2, P2=32*3*5**2, disp12MaxDiff=1,
                    uniquenessRatio=10, speckleWindowSize=100, speckleRange=32)
    disparity = stereo.compute(left_img, right_img).astype(np.float32) / 16.0

    # Depth estimation for each tracked point
    depths = []
    for pt in goodpts_new:
        x, y = map(int, pt.ravel())
        if 0 <= x < disparity.shape[1] and 0 <= y < disparity.shape[0]:
            d = disparity[y, x]
            if d > 0:  # Valid disparity
                Z = (fx * baseline) / d
                depths.append(Z if Z < 50.0 else np.nan)  # Clip depth at 50m
            else:
                depths.append(np.nan)  # Invalid depth
        else:
            depths.append(np.nan)  # Out of bounds

    depths = np.array(depths)


    # Making sure to only keep points with valid depth estimates


    valid_mask  = ~np.isnan(depths)
    goodpts_old = goodpts_old[valid_mask]
    goodpts_new = goodpts_new[valid_mask]
    depths      = depths[valid_mask]


    # Coordinate Normalization


    # Normalizing pixel coordinates of tracked points to camera coordinates
    normalized_pts = []
    for pt in goodpts_new:
        x, y   = pt.ravel()
        x_norm = (x - cx) / fx
        y_norm = (y - cy) / fx
        normalized_pts.append((x_norm, y_norm))
    normalized_pts = np.array(normalized_pts)


    # Jacobian Construction


    # Constructing the Jacobian matrix for each feature point based on the normalized coordinates and depth
    jacobian_rows = []
    for (normalized_x, normalized_y), depth_Z in zip(normalized_pts, depths):

        feature_jacobian = np.array([
            [
                -1.0 / depth_Z,
                0,
                normalized_x / depth_Z,
                normalized_x * normalized_y,
                -(1 + normalized_x**2),
                normalized_y
            ],
            [
                0,
                -1.0 / depth_Z,
                normalized_y / depth_Z,
                1 + normalized_y**2,
                -normalized_x * normalized_y,
                -normalized_x
            ]
        ])

        jacobian_rows.append(feature_jacobian)

    # Stacking all feature Jacobians into a single jacobian matrix
    J = np.vstack(jacobian_rows)


    # Velocity Estimation


    img_disp_vec = []

    # Compute real dt from timestamps for this frame transition
    dt = timestamps[frame_idx] - timestamps[frame_idx - 1]

    for old_pt, new_pt in zip(goodpts_old, goodpts_new):

        old_x, old_y = old_pt.ravel()
        new_x, new_y = new_pt.ravel()

        dx = (new_x - old_x) / (dt * fx)
        dy = (new_y - old_y) / (dt * fx)

        img_disp_vec.extend([dx, dy])

    img_disp_vec = np.array(img_disp_vec)

    # Estimating camera velocity
    cam_vel = np.linalg.pinv(J) @ img_disp_vec

    vx = cam_vel[0]
    vy = cam_vel[1]
    vz = cam_vel[2]

    # Kalman Filter — Predict then Update
    kf_P  = kf_P + kf_Q
    kf_K  = kf_P / (kf_P + kf_R)
    kf_x  = kf_x + kf_K * (vz - kf_x)
    kf_P  = (1 - kf_K) * kf_P
    vz_kf = kf_x


    # Display


    fps          = 1.0 / (time.time() - frame_start_time)
    display_frame = cv2.cvtColor(left_frame_new, cv2.COLOR_GRAY2BGR)

    # Ground truth velocity transformed into the camera frame
    R        = poses[frame_idx][:, :3]
    v_world  = (poses[frame_idx][:, 3] - poses[frame_idx - 1][:, 3]) / dt
    v_camera = R.T @ v_world

    gt_vx = v_camera[0]
    gt_vy = v_camera[1]
    gt_vz = v_camera[2]

    cv2.putText(display_frame, f"Vx: {vx:.3f}  GT: {gt_vx:.3f} m/s", (20,  40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255,   0), 2)
    cv2.putText(display_frame, f"Vy: {vy:.3f}  GT: {gt_vy:.3f} m/s", (20,  80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255,   0), 2)
    cv2.putText(display_frame, f"Vz: {vz_kf:.3f}  GT: {gt_vz:.3f} m/s", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255,   0), 2)

    cv2.imshow("Pipeline 01", display_frame)

    # Depth Map Display
    depth_vis                        = np.clip(disparity, 0, 64)
    depth_vis                        = (depth_vis / 64 * 255).astype(np.uint8)
    depth_vis                        = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
    cv2.imshow("Depth Map", depth_vis)

    # Live Vz Plot Update
    vz_history.append(vz_kf)
    gt_vz_history.append(gt_vz)
    line_vz.set_data(range(len(vz_history)), vz_history)
    line_gt_vz.set_data(range(len(gt_vz_history)), gt_vz_history)
    ax.relim()
    ax.autoscale_view()
    plt.pause(0.001)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


    # Tracking Maintenance


    # Updating previous frame features to the current frame2  ← CHANGED
    left_frame_old = left_frame_new
    feats_old      = feats_new

    # Re-detect SuperPoint if too few tracked points remain  ← CHANGED (was: sift.detectAndCompute)
    if len(goodpts_new) < MIN_FEATURES:
        feats_old = extract_superpoint(left_frame_old)


cv2.destroyAllWindows()