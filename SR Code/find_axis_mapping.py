"""
Brute-force search for the camera-to-body axis mapping.

Since this rig is nadir-facing, the camera axes (X-right, Y-down,
Z-forward/optical) don't map onto vehicle body axes (X-forward, Y-right,
Z-down) in the "obvious" forward-facing-camera way. Rather than guess,
this script tries every permutation and sign combination of the camera
velocity components, rotates each candidate into the nav frame using the
same rotation_matrix() as rotate_velocity_to_navframe.py, and scores it
against ground-truth North/East/Down velocity. Whichever mapping gives
the lowest error is (very likely) the correct one for this mount.

Run this once, read off the winning mapping, then hardcode it into
rotate_velocity_to_navframe.py's v_body assignment.
"""

import itertools
import numpy as np
import pandas as pd

POSE_EST_PATH   = r"D:\Stingray Robotics\ScottReef GT\ScottReef_20090804_084719_SLAM\stereo_pose_est.data"
VELOCITY_CSV_IN = r"D:\Stingray Robotics\SR Output\SR Output 2\velocity_output002_scottreef.csv"

# Use one variant to search with -- sp_ma has the least filter lag of the
# saved variants, which matters less here since we're comparing whole-
# dataset error, not individual transitions.
VARIANT = "sp_ma"

ROLL_OFFSET_DEG  = 0.0
PITCH_OFFSET_DEG = 0.0
YAW_OFFSET_DEG   = 180.0


def load_pose_est(path):
    ts, roll, pitch, yaw, N, E, D = [], [], [], [], [], [], []
    with open(path) as f:
        for line in f:
            if line.startswith("%") or line.startswith("ORIGIN") or not line.strip():
                continue
            parts = line.split()
            ts.append(float(parts[1]))
            N.append(float(parts[4]))
            E.append(float(parts[5]))
            D.append(float(parts[6]))
            roll.append(float(parts[8]))   # swapped: was parts[7]
            pitch.append(float(parts[7]))  # swapped: was parts[8]
            yaw.append(float(parts[9]))
    return (np.array(ts), np.array(roll), np.array(pitch), np.array(yaw),
            np.array(N), np.array(E), np.array(D))


def rotation_matrix(phi, theta, psi):
    cphi, sphi = np.cos(phi), np.sin(phi)
    cth, sth = np.cos(theta), np.sin(theta)
    cpsi, spsi = np.cos(psi), np.sin(psi)
    return np.array([
        [cpsi * cth, -spsi * cphi + cpsi * sth * sphi,  spsi * sphi + cpsi * cphi * sth],
        [spsi * cth,  cpsi * cphi + sphi * sth * spsi, -cpsi * sphi + sth * spsi * cphi],
        [-sth,        cth * sphi,                        cth * cphi],
    ])


ts, roll, pitch, yaw, gt_N, gt_E, gt_D = load_pose_est(POSE_EST_PATH)
vel_df = pd.read_csv(VELOCITY_CSV_IN)
n = min(len(vel_df), len(ts) - 1)

v_cam_all = vel_df.loc[:n - 1, [f"vx_{VARIANT}", f"vy_{VARIANT}", f"vz_{VARIANT}"]].to_numpy(dtype=float)

# Ground-truth nav-frame velocity (North/East/Down), same length convention
# as rotate_velocity_to_navframe.py (row i spans pose sample i -> i+1).
dt = np.diff(ts[:n + 1])
gt_vel = np.column_stack([
    np.diff(gt_N[:n + 1]),
    np.diff(gt_E[:n + 1]),
    np.diff(gt_D[:n + 1]),
]) / dt[:, None]

YAW_OFFSET_CANDIDATES_DEG = [0.0, 90.0, 180.0, 270.0]

best = None
for yaw_off_deg in YAW_OFFSET_CANDIDATES_DEG:
    roll_mid  = (roll[:n] + roll[1:n + 1]) / 2 + np.radians(ROLL_OFFSET_DEG)
    pitch_mid = (pitch[:n] + pitch[1:n + 1]) / 2 + np.radians(PITCH_OFFSET_DEG)
    yaw_mid   = (yaw[:n] + yaw[1:n + 1]) / 2 + np.radians(yaw_off_deg)

    for perm in itertools.permutations(range(3)):
        for signs in itertools.product([1, -1], repeat=3):
            errs = np.empty(n)
            for i in range(n):
                R = rotation_matrix(roll_mid[i], pitch_mid[i], yaw_mid[i])
                v_body = np.array([v_cam_all[i, perm[0]] * signs[0],
                                    v_cam_all[i, perm[1]] * signs[1],
                                    v_cam_all[i, perm[2]] * signs[2]])
                v_nav = R @ v_body
                errs[i] = np.linalg.norm(v_nav - gt_vel[i])
            mae = errs.mean()
            if best is None or mae < best[0]:
                best = (mae, perm, signs, yaw_off_deg)

mae, perm, signs, yaw_off_deg = best
axis_names = ["cam_vx", "cam_vy", "cam_vz"]
print(f"Best mapping found: MAE = {mae:.4f} m/s, YAW_OFFSET_DEG = {yaw_off_deg}")
print(f"v_body[0] (mapped to rotation_matrix's 'X') = {signs[0]:+d} * {axis_names[perm[0]]}")
print(f"v_body[1] (mapped to rotation_matrix's 'Y') = {signs[1]:+d} * {axis_names[perm[1]]}")
print(f"v_body[2] (mapped to rotation_matrix's 'Z') = {signs[2]:+d} * {axis_names[perm[2]]}")
print()
print("Hardcode this into rotate_velocity_to_navframe.py as:")
print(f"YAW_OFFSET_DEG = {yaw_off_deg}")
print(f"v_body = np.array([{signs[0]:+d} * v_cam[{perm[0]}], "
      f"{signs[1]:+d} * v_cam[{perm[1]}], {signs[2]:+d} * v_cam[{perm[2]}]])")
