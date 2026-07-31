# Importing Libraries

import numpy as np
import cv2
import time


# Configuration

LEFT_CAM_ID  = 1                  # left camera index  
RIGHT_CAM_ID = 2                  # right camera index
CALIB_FILE   = "stereo_calib.txt" # output of cc5.py
MIN_FEATURES = 350


# Load Calibration from stereo_calib.txt

def load_calib(path):
    with open(path, "r") as f:
        content = f.read()

    blocks = content.strip().split("\n\n")

    def parse_matrix(label):
        for i, block in enumerate(blocks):
            if block.strip().startswith(label):
                lines = block.strip().splitlines()[1:]  # skip the label line
                return np.array([list(map(float, l.split())) for l in lines])
        raise ValueError(f"Label '{label}' not found in calibration file.")

    K_left   = parse_matrix("K_left")
    K_right  = parse_matrix("K_right")
    R        = parse_matrix("R")
    T        = parse_matrix("T")

    return K_left, K_right, R, T

K_left, K_right, R, T = load_calib(CALIB_FILE)

fx       = K_left[0, 0]
fy       = K_left[1, 1]
cx       = K_left[0, 2]
cy       = K_left[1, 2]
baseline = abs(T[0, 0])           # metres


# Open Cameras

cap_left  = cv2.VideoCapture(LEFT_CAM_ID)
cap_right = cv2.VideoCapture(RIGHT_CAM_ID)

ret_l, frame_l = cap_left.read()
left_frame_old  = cv2.cvtColor(frame_l, cv2.COLOR_BGR2GRAY)


# Initial Feature Detection

sift      = cv2.SIFT_create()
keypoints, _ = sift.detectAndCompute(left_frame_old, None)
pts_old   = np.array([kp.pt for kp in keypoints], dtype=np.float32).reshape(-1, 1, 2)

stereo_bm = cv2.StereoBM_create(numDisparities=64, blockSize=15)
prev_time = time.time()


# Main Loop

while True:

    frame_start = time.time()

    # Real dt
    curr_time = time.time()
    dt        = curr_time - prev_time
    prev_time = curr_time
    if dt <= 0:
        dt = 1e-3

    # Grab frames
    ret_l, left_color  = cap_left.read()
    ret_r, right_color = cap_right.read()
    if not ret_l or not ret_r:
        break

    left_frame_new = cv2.cvtColor(left_color,  cv2.COLOR_BGR2GRAY)
    right_img      = cv2.cvtColor(right_color, cv2.COLOR_BGR2GRAY)


    # Feature Tracking

    pts_new, status, _ = cv2.calcOpticalFlowPyrLK(left_frame_old, left_frame_new, pts_old, None)

    goodpts_old = pts_old[status == 1]
    goodpts_new = pts_new[status == 1]


    # RANSAC Filtering

    if len(goodpts_old) < 8:
        left_frame_old = left_frame_new
        keypoints, _ = sift.detectAndCompute(left_frame_old, None)
        pts_old = np.array([kp.pt for kp in keypoints], dtype=np.float32).reshape(-1, 1, 2)
        continue

    F_mat, mask = cv2.findFundamentalMat(goodpts_old, goodpts_new, cv2.FM_RANSAC)
    if mask is None:
        left_frame_old = left_frame_new
        continue

    goodpts_old = goodpts_old[mask.ravel() == 1]
    goodpts_new = goodpts_new[mask.ravel() == 1]


    # Depth Estimation

    disparity = stereo_bm.compute(left_frame_new, right_img).astype(np.float32) / 16.0

    depths = []
    for pt in goodpts_new:
        x, y = map(int, pt.ravel())
        if 0 <= x < disparity.shape[1] and 0 <= y < disparity.shape[0]:
            d = disparity[y, x]
            depths.append((fx * baseline) / d if d > 0 else np.nan)
        else:
            depths.append(np.nan)

    depths = np.array(depths)

    valid       = ~np.isnan(depths)
    goodpts_old = goodpts_old[valid]
    goodpts_new = goodpts_new[valid]
    depths      = depths[valid]

    if len(depths) == 0:
        left_frame_old = left_frame_new
        continue


    # Coordinate Normalization  

    normalized_pts = [((pt[0] - cx) / fx, (pt[1] - cy) / fy) for pt in goodpts_new]
    normalized_pts = np.array(normalized_pts)


    # Jacobian Construction

    jacobian_rows = []
    for (nx, ny), Z in zip(normalized_pts, depths):
        jacobian_rows.append(np.array([
            [-1/Z,    0,  nx/Z,      nx*ny, -(1+nx**2),  ny],
            [   0, -1/Z,  ny/Z, 1+ny**2,      -nx*ny,  -nx]
        ]))
    J = np.vstack(jacobian_rows)


    # Image Displacement Vector 

    img_disp_vec = []
    for old_pt, new_pt in zip(goodpts_old, goodpts_new):
        img_disp_vec.extend([
            (new_pt[0] - old_pt[0]) / (dt * fx),
            (new_pt[1] - old_pt[1]) / (dt * fy)
        ])
    img_disp_vec = np.array(img_disp_vec)


    # Velocity Estimation

    cam_vel     = np.linalg.pinv(J) @ img_disp_vec
    vx, vy, vz  = cam_vel[0], cam_vel[1], cam_vel[2]


    # Display

    fps           = 1.0 / (time.time() - frame_start)
    display_frame = cv2.cvtColor(left_frame_new, cv2.COLOR_GRAY2BGR)

    cv2.putText(display_frame, f"Vx: {vx:.3f} m/s", (20,  40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.putText(display_frame, f"Vy: {vy:.3f} m/s", (20,  75), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.putText(display_frame, f"Vz: {vz:.3f} m/s", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.putText(display_frame, f"FPS: {fps:.1f}",   (20, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)

    cv2.imshow("Pipeline 02 — Live Velocity", display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


    # Tracking Maintenance

    left_frame_old = left_frame_new
    pts_old        = goodpts_new.reshape(-1, 1, 2)

    if len(pts_old) < MIN_FEATURES:
        keypoints, _ = sift.detectAndCompute(left_frame_old, None)
        pts_old = np.array([kp.pt for kp in keypoints], dtype=np.float32).reshape(-1, 1, 2)


cap_left.release()
cap_right.release()
cv2.destroyAllWindows()