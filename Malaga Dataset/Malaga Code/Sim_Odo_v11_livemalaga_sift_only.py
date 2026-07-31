# Importing Libraries

import numpy as np
import cv2
# import torch
# from lightglue import LightGlue, SuperPoint
# from lightglue.utils import rbd
from collections import deque


# Configuration

CALIB        = r"D:\Stingray Robotics\Visual Odometry Complete Pipeline\KITTI Odometry Full Pipeline\Sim Code\robotsimcalib.txt"
LEFT_VIDEO   = r"D:\Stingray Robotics\Visual Odometry Complete Pipeline\KITTI Odometry Full Pipeline\Sim Code\output3.mp4"
RIGHT_VIDEO  = r"D:\Stingray Robotics\Visual Odometry Complete Pipeline\KITTI Odometry Full Pipeline\Sim Code\output4.mp4"
MIN_FEATURES = 400

# --- Malaga ground-truth files -----------------------------------------
# ASSUMPTION: output3.mp4 / output4.mp4 were built by concatenating this
# exact dataset's _left.jpg / _right.jpg sequence in chronological order,
# one video frame per image, no drops. If that's not the case, frame N in
# the video will NOT correspond to the N-th image below, and the true
# velocity overlay will be wrong. See LEFT_IMAGE_TIMES construction below.
IMAGES_TXT   = r"D:\Malaga Urban Dataset\malaga-urban-dataset-plain-text\malaga-urban-dataset_IMAGES.txt"
GPS_TXT      = r"D:\Malaga Urban Dataset\malaga-urban-dataset-plain-text\malaga-urban-dataset_GPS.txt"

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load Calibration

with open(CALIB, "r") as f:
    lines = f.readlines()

P0 = np.array(list(map(float, lines[0].split()[1:])), dtype=np.float32).reshape(3, 4)
P1 = np.array(list(map(float, lines[1].split()[1:])), dtype=np.float32).reshape(3, 4)

fx       = P0[0, 0]
cx       = P0[0, 2]
cy       = P0[1, 2]
baseline = abs(P1[0, 3] / fx)


# Load SuperPoint + LightGlue

# extractor = SuperPoint(max_num_keypoints=1024).eval().to(device)
# matcher   = LightGlue(features="superpoint").eval().to(device)

# def extract_superpoint(gray_frame):
#     tensor = torch.from_numpy(gray_frame).float() / 255.0
#     tensor = tensor.unsqueeze(0).unsqueeze(0).to(device)
#     return extractor.extract(tensor)
# 
# 
# Load SIFT

sift = cv2.SIFT_create()


# ---------------------------------------------------------------------
# Ground truth loading: derive velocity from GPS Local X/Y/Z positions
# ---------------------------------------------------------------------

def load_left_frame_timestamps(images_txt_path):
    """Return sorted list of UNIX timestamps, one per LEFT frame, in the
    same order the frames appear in the file (== assumed video order)."""
    times = []
    with open(images_txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or "_left.jpg" not in line:
                continue
            # img_CAMERA1_1261228439.115674_left.jpg
            ts = float(line.split("_")[2])
            times.append(ts)
    return np.array(times, dtype=np.float64)


def load_gps_true_velocity(gps_txt_path):
    """GPS speed/dir/Local-V* columns are essentially unpopulated in this
    dataset (near-all-zero), so true velocity is derived by differentiating
    Local X/Y/Z position over GPS time (~1 Hz). Returns time array +
    per-axis velocity (dataset local frame, Z up) + speed magnitude."""
    t, x, y, z = [], [], [], []
    with open(gps_txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("%"):
                continue
            cols = line.split()
            t.append(float(cols[0]))
            x.append(float(cols[8]))   # Local X
            y.append(float(cols[9]))   # Local Y
            z.append(float(cols[10]))  # Local Z

    t = np.array(t, dtype=np.float64)
    x = np.array(x, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    z = np.array(z, dtype=np.float64)

    order = np.argsort(t)
    t, x, y, z = t[order], x[order], y[order], z[order]

    vx = np.gradient(x, t)
    vy = np.gradient(y, t)
    vz = np.gradient(z, t)
    speed = np.sqrt(vx**2 + vy**2 + vz**2)
    return t, vx, vy, vz, speed


def true_velocity_at(frame_time, gps_t, vx, vy, vz, speed):
    """Linear-interpolate GPS-derived velocity at an arbitrary timestamp.
    Values outside the GPS time range are clamped to the nearest end."""
    tvx = np.interp(frame_time, gps_t, vx)
    tvy = np.interp(frame_time, gps_t, vy)
    tvz = np.interp(frame_time, gps_t, vz)
    tsp = np.interp(frame_time, gps_t, speed)
    return tvx, tvy, tvz, tsp


left_frame_times = load_left_frame_timestamps(IMAGES_TXT)
gps_t, gps_vx, gps_vy, gps_vz, gps_speed = load_gps_true_velocity(GPS_TXT)


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

# Fallback dt from container FPS, used only if a frame's true timestamp
# can't be resolved (e.g. video is longer than the timestamp list).
fallback_dt = 1.0 / cap_left.get(cv2.CAP_PROP_FPS)

ret_l, left_raw = cap_left.read()
left_frame_old  = cv2.cvtColor(left_raw, cv2.COLOR_BGR2GRAY)

# feats_old = extract_superpoint(left_frame_old)

kps, _ = sift.detectAndCompute(left_frame_old, None)
pts_old_sift = np.array([kp.pt for kp in kps], dtype=np.float32).reshape(-1, 1, 2)

frame_idx = 0  # index into left_frame_times; frame 0 already consumed above
prev_frame_time = left_frame_times[0] if len(left_frame_times) > 0 else 0.0


# Rolling-window outlier clamp (median/MAD), applied only to Vz
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

# buf_sp_vz   = make_buffer()
buf_sift_vz = make_buffer()

# ma_sp_vx,   ma_sp_vy,   ma_sp_vz   = make_ma(), make_ma(), make_ma()
ma_sift_vx, ma_sift_vy, ma_sift_vz = make_ma(), make_ma(), make_ma()

# lp_sp_vx,   lp_sp_vy,   lp_sp_vz   = make_lp(), make_lp(), make_lp()
lp_sift_vx, lp_sift_vy, lp_sift_vz = make_lp(), make_lp(), make_lp()


# ---------------------------------------------------------------------
# Overlay drawing helper — true vs calculated, side by side
# ---------------------------------------------------------------------

def draw_overlay(img_bgr, true_vals, sift_vals):
    """true_vals  = (vx, vy, vz, speed)
       sp_vals    = (vx_ma, vy_ma, vz_ma, vx_lp, vy_lp, vz_lp)
       sift_vals  = (vx_ma, vy_ma, vz_ma, vx_lp, vy_lp, vz_lp)"""
    h, w = img_bgr.shape[:2]
    panel_w = 330
    overlay = img_bgr.copy()
    cv2.rectangle(overlay, (0, 0), (panel_w, 190), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, img_bgr, 0.45, 0, img_bgr)

    font = cv2.FONT_HERSHEY_SIMPLEX
    y = 22
    lh = 20

    def put(text, color=(255, 255, 255), bold=False):
        nonlocal y
        cv2.putText(img_bgr, text, (10, y), font, 0.5, color, 2 if bold else 1, cv2.LINE_AA)
        y += lh

    put("TRUE (GPS)", (0, 255, 255), bold=True)
    put(f"  Vx={true_vals[0]:+.3f} Vy={true_vals[1]:+.3f} Vz={true_vals[2]:+.3f}")
    put(f"  speed={true_vals[3]:.3f} m/s")

    ## removed SuperPoint
    # removed
    # removed
    # removed

    put("SIFT (MA)", (0, 165, 255), bold=True)
    put(f"  Vx={sift_vals[0]:+.3f} Vy={sift_vals[1]:+.3f} Vz={sift_vals[2]:+.3f}")
    put("SIFT (LP)", (0, 165, 255), bold=True)
    put(f"  Vx={sift_vals[3]:+.3f} Vy={sift_vals[4]:+.3f} Vz={sift_vals[5]:+.3f}")

    return img_bgr


# Main Loop

stereo = cv2.StereoSGBM_create(minDisparity=0, numDisparities=64, blockSize=5,
             P1=8*3*5**2, P2=32*3*5**2, disp12MaxDiff=1,
             uniquenessRatio=10, speckleWindowSize=100, speckleRange=32)

cv2.namedWindow("Visual Odometry - True vs Calculated", cv2.WINDOW_NORMAL)

while True:
    ret_l, left_raw  = cap_left.read()
    ret_r, right_raw = cap_right.read()
    if not ret_l or not ret_r:
        print("End of video")
        break

    frame_idx += 1

    # Resolve this frame's real timestamp + true dt from the image list
    if frame_idx < len(left_frame_times):
        frame_time = left_frame_times[frame_idx]
        dt = frame_time - prev_frame_time
        if dt <= 0:
            dt = fallback_dt
    else:
        frame_time = prev_frame_time + fallback_dt
        dt = fallback_dt
    prev_frame_time = frame_time

    left_frame_new = cv2.cvtColor(left_raw,  cv2.COLOR_BGR2GRAY)
    left_img       = left_frame_new
    right_img      = cv2.cvtColor(right_raw, cv2.COLOR_BGR2GRAY)

    disparity = stereo.compute(left_img, right_img).astype(np.float32) / 16.0

    # # SuperPoint + LightGlue
    # feats_new    = extract_superpoint(left_frame_new)
    # matches01    = matcher({"image0": feats_old, "image1": feats_new})
    # f0, f1, m01  = rbd(feats_old), rbd(feats_new), rbd(matches01)
    # matched_idxs = m01["matches"]

    # sp_old = np.ascontiguousarray(f0["keypoints"][matched_idxs[:, 0]].cpu().numpy(), dtype=np.float32)
    # sp_new = np.ascontiguousarray(f1["keypoints"][matched_idxs[:, 1]].cpu().numpy(), dtype=np.float32)

    # if len(sp_old) >= 8:
    #     try:
    #         F, mask = cv2.findFundamentalMat(sp_old, sp_new, cv2.FM_RANSAC)
    #         if mask is not None:
    #             sp_old = sp_old[mask.ravel() == 1]
    #             sp_new = sp_new[mask.ravel() == 1]
    #     except cv2.error:
    #         pass

    # sp_depths  = compute_depth(sp_new, disparity)
    # valid      = ~np.isnan(sp_depths)
    # sp_old, sp_new, sp_depths = sp_old[valid], sp_new[valid], sp_depths[valid]

    # vx_sp = vy_sp = vz_sp = 0.0
    # if len(sp_depths) > 0:
    #     vx_sp, vy_sp, vz_sp = compute_velocity(sp_old, sp_new, sp_depths, dt)

    # vz_sp = reject_outlier(buf_sp_vz, vz_sp)

    # vx_sp_ma = ma_update(ma_sp_vx, vx_sp);   vy_sp_ma = ma_update(ma_sp_vy, vy_sp);   vz_sp_ma = ma_update(ma_sp_vz, vz_sp)
    # vx_sp_lp = lp_update(lp_sp_vx, vx_sp);  vy_sp_lp = lp_update(lp_sp_vy, vy_sp);  vz_sp_lp = lp_update(lp_sp_vz, vz_sp)

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

    vx_sift_ma = ma_update(ma_sift_vx, vx_sift);   vy_sift_ma = ma_update(ma_sift_vy, vy_sift);   vz_sift_ma = ma_update(ma_sift_vz, vz_sift)
    vx_sift_lp = lp_update(lp_sift_vx, vx_sift);  vy_sift_lp = lp_update(lp_sift_vy, vy_sift);  vz_sift_lp = lp_update(lp_sift_vz, vz_sift)

    # True velocity from GPS at this frame's timestamp
    tvx, tvy, tvz, tspeed = true_velocity_at(frame_time, gps_t, gps_vx, gps_vy, gps_vz, gps_speed)

    # Draw overlay + show live video
    display_frame = left_raw.copy() if left_raw.ndim == 3 else cv2.cvtColor(left_frame_new, cv2.COLOR_GRAY2BGR)
    display_frame = draw_overlay(
        display_frame,
        (tvx, tvy, tvz, tspeed),
        (vx_sift_ma, vy_sift_ma, vz_sift_ma, vx_sift_lp, vy_sift_lp, vz_sift_lp),
    )
    cv2.imshow("Visual Odometry - True vs Calculated", display_frame)

    if cv2.waitKey(1) & 0xFF in (ord('q'), 27):  # 'q' or ESC to quit
        break

    # Tracking Maintenance
    left_frame_old = left_frame_new

    # feats_old = feats_new
    # if len(sp_new) < MIN_FEATURES:
        # feats_old = extract_superpoint(left_frame_old)

    pts_old_sift = sift_new.reshape(-1, 1, 2)
    if len(pts_old_sift) < MIN_FEATURES:
        kps, _ = sift.detectAndCompute(left_frame_old, None)
        pts_old_sift = np.array([kp.pt for kp in kps], dtype=np.float32).reshape(-1, 1, 2)

cap_left.release()
cap_right.release()
cv2.destroyAllWindows()
print("Done.")