"""
Rotates already-computed camera-frame velocities into the navigation frame
using orientation (roll/pitch/yaw) from stereo_pose_est.data, then
re-integrates position from the rotated velocities.

Assumption: frame N in the velocity/position CSVs corresponds, in order,
to pose row N in stereo_pose_est.data (both are chronological). If your
VO pipeline skipped/subsampled frames from the full dataset, this
alignment will drift -- check a few frames' timestamps against your
source images to confirm before trusting the output.
"""

import numpy as np
import pandas as pd

POSE_EST_PATH    = r"D:\Stingray Robotics\ScottReef GT\ScottReef_20090804_084719_SLAM\stereo_pose_est.data"
VELOCITY_CSV_IN  = r"D:\Stingray Robotics\SR Output\SR Output 2\velocity_output002_scottreef.csv"
VELOCITY_CSV_OUT = r"D:\Stingray Robotics\SR Output\SR Output 2\velocity_output0012_scottreef_rotated.csv"
POSITION_CSV_OUT = r"D:\Stingray Robotics\SR Output\SR Output 2\position_output0012_scottreef_rotated.csv"

VARIANTS = ["sp_ma", "sp_lp", "sift_ma", "sift_lp"]

# Angle offsets (degrees) applied to the pose-file orientation before building
# the rotation matrix. Tune these if the rotated result comes out flipped or
# rotated relative to ground truth -- e.g. a heading/sign convention mismatch
# commonly shows up as needing +-180 on yaw, or +-90 on roll/pitch.
ROLL_OFFSET_DEG  = 0.0
PITCH_OFFSET_DEG = 0.0
YAW_OFFSET_DEG   = -90.0


def load_pose_est(path):
    """Parse timestamp + roll/pitch/yaw (radians) from stereo_pose_est.data."""
    ts, roll, pitch, yaw = [], [], [], []
    with open(path) as f:
        for line in f:
            if line.startswith("%") or line.startswith("ORIGIN") or not line.strip():
                continue
            parts = line.split()
            ts.append(float(parts[1]))
            roll.append(float(parts[7]))
            pitch.append(float(parts[8]))
            yaw.append(float(parts[9]))
    return np.array(ts), np.array(roll), np.array(pitch), np.array(yaw)


def rotation_matrix(phi, theta, psi):
    """Body -> nav frame rotation (roll phi, pitch theta, yaw psi), radians."""
    cphi, sphi = np.cos(phi), np.sin(phi)
    cth, sth = np.cos(theta), np.sin(theta)
    cpsi, spsi = np.cos(psi), np.sin(psi)
    return np.array([
        [cpsi * cth, -spsi * cphi + cpsi * sth * sphi,  spsi * sphi + cpsi * cphi * sth],
        [spsi * cth,  cpsi * cphi + sphi * sth * spsi, -cpsi * sphi + sth * spsi * cphi],
        [-sth,        cth * sphi,                        cth * cphi],
    ])


ts, roll, pitch, yaw = load_pose_est(POSE_EST_PATH)
vel_df = pd.read_csv(VELOCITY_CSV_IN)
n = len(vel_df)

# Each velocity row spans pose sample i -> i+1: use midpoint orientation
# and the real dt between those pose timestamps.
roll_mid  = (roll[:n] + roll[1:n + 1]) / 2 + np.radians(ROLL_OFFSET_DEG)
pitch_mid = (pitch[:n] + pitch[1:n + 1]) / 2 + np.radians(PITCH_OFFSET_DEG)
yaw_mid   = (yaw[:n] + yaw[1:n + 1]) / 2 + np.radians(YAW_OFFSET_DEG)
dt = np.diff(ts[:n + 1])

vel_rows, pos_rows = [], []
pos = {v: np.zeros(3) for v in VARIANTS}

for i in range(n):
    R = rotation_matrix(roll_mid[i], pitch_mid[i], yaw_mid[i])
    vel_row = {"frame": vel_df.loc[i, "frame"]}
    pos_row = {"frame": vel_df.loc[i, "frame"]}

    for v in VARIANTS:
        v_cam = vel_df.loc[i, [f"vx_{v}", f"vy_{v}", f"vz_{v}"]].to_numpy(dtype=float)
        v_nav = R @ v_cam
        vel_row[f"vx_{v}"], vel_row[f"vy_{v}"], vel_row[f"vz_{v}"] = v_nav

        pos[v] = pos[v] + v_nav * dt[i]
        pos_row[f"x_{v}"], pos_row[f"y_{v}"], pos_row[f"z_{v}"] = pos[v]

    vel_rows.append(vel_row)
    pos_rows.append(pos_row)

pd.DataFrame(vel_rows).to_csv(VELOCITY_CSV_OUT, index=False)
pd.DataFrame(pos_rows).to_csv(POSITION_CSV_OUT, index=False)

print(f"Saved rotated velocity to {VELOCITY_CSV_OUT}")
print(f"Saved rotated position to {POSITION_CSV_OUT}")