"""
Exhaustive search for the correct rotation setup.

Ground truth velocity is kept exactly as-is in the nav frame (no rotation
applied to it -- it's already North/East/Down from stereo_pose_est.data).

Everything else is searched, because none of it can be safely assumed:
  - WHICH raw pose column (parts[7], parts[8], parts[9]) is actually roll,
    which is pitch, which is yaw -- all 6 assignments are tried.
  - The SIGN of each angle (mirrored convention) -- all 8 sign combos.
  - An additive OFFSET on each angle (0/90/180/270 deg) -- all 64 combos.
  - WHICH computed velocity axis (vx/vy/vz) feeds which slot of the body
    frame the rotation matrix expects -- all 6 axis permutations.
  - The SIGN of each velocity axis -- all 8 sign combos.

Every combination (angle role x angle sign x angle offset x axis perm x
axis sign) is scored by rotating the calculated velocity and comparing it
to ground truth. The lowest-error combination is printed, then used to
produce the final rotated velocity/position CSVs for all variants.

Fully vectorized over frames, so the search itself only takes seconds
even though it's checking on the order of ~150,000 combinations.
"""

import itertools
import numpy as np
import pandas as pd

POSE_EST_PATH    = r"D:\Stingray Robotics\ScottReef GT\ScottReef_20090804_084719_SLAM\stereo_pose_est.data"
VELOCITY_CSV_IN  = r"D:\Stingray Robotics\SR Output\SR Output 2\velocity_output002_scottreef.csv"
VELOCITY_CSV_OUT = r"D:\Stingray Robotics\SR Output\SR Output 2\velocity_output0022_scottreef_rotated.csv"
POSITION_CSV_OUT = r"D:\Stingray Robotics\SR Output\SR Output 2\position_output0022_scottreef_rotated.csv"

VARIANTS       = ["sp_ma", "sp_lp", "sift_ma", "sift_lp"]
SEARCH_VARIANT = "sp_ma"   # used to score candidates (least filter lag of the saved variants)
OFFSET_CANDIDATES_DEG = [0.0, 90.0, 180.0, 270.0]


def load_pose_est(path):
    """Parse timestamp, North/East/Down, and the three raw orientation
    columns. The orientation columns are deliberately left UNLABELED --
    which one is roll/pitch/yaw is determined by the search below, not
    assumed here."""
    ts, N, E, D, a7, a8, a9 = [], [], [], [], [], [], []
    with open(path) as f:
        for line in f:
            if line.startswith("%") or line.startswith("ORIGIN") or not line.strip():
                continue
            parts = line.split()
            ts.append(float(parts[1]))
            N.append(float(parts[4]))
            E.append(float(parts[5]))
            D.append(float(parts[6]))
            a7.append(float(parts[7]))
            a8.append(float(parts[8]))
            a9.append(float(parts[9]))
    return (np.array(ts), np.array(N), np.array(E), np.array(D),
            np.array(a7), np.array(a8), np.array(a9))


def rotation_matrix_batch(phi, theta, psi):
    """Vectorized body -> nav DCM. phi/theta/psi are arrays of length n;
    returns R with shape (n, 3, 3)."""
    cphi, sphi = np.cos(phi), np.sin(phi)
    cth, sth   = np.cos(theta), np.sin(theta)
    cpsi, spsi = np.cos(psi), np.sin(psi)
    n = len(phi)
    R = np.empty((n, 3, 3))
    R[:, 0, 0] = cpsi * cth
    R[:, 0, 1] = -spsi * cphi + cpsi * sth * sphi
    R[:, 0, 2] =  spsi * sphi + cpsi * cphi * sth
    R[:, 1, 0] = spsi * cth
    R[:, 1, 1] =  cpsi * cphi + sphi * sth * spsi
    R[:, 1, 2] = -cpsi * sphi + sth * spsi * cphi
    R[:, 2, 0] = -sth
    R[:, 2, 1] = cth * sphi
    R[:, 2, 2] = cth * cphi
    return R


ts, gt_N, gt_E, gt_D, a7, a8, a9 = load_pose_est(POSE_EST_PATH)
raw_angles = [a7, a8, a9]  # unlabeled on purpose

vel_df = pd.read_csv(VELOCITY_CSV_IN)
n = min(len(vel_df), len(ts) - 1)

dt = np.diff(ts[:n + 1])
gt_vel = np.column_stack([
    np.diff(gt_N[:n + 1]),
    np.diff(gt_E[:n + 1]),
    np.diff(gt_D[:n + 1]),
]) / dt[:, None]   # ground truth stays in the nav frame, untouched

mid_angle = [(a[:n] + a[1:n + 1]) / 2.0 for a in raw_angles]

v_cam_search = vel_df.loc[:n - 1, [f"vx_{SEARCH_VARIANT}", f"vy_{SEARCH_VARIANT}",
                                    f"vz_{SEARCH_VARIANT}"]].to_numpy(dtype=float)

print("Searching angle-role x angle-sign x angle-offset x axis-perm x axis-sign combinations...")

best = None
count = 0
for angle_perm in itertools.permutations(range(3)):          # raw col -> phi/theta/psi
    for angle_signs in itertools.product([1, -1], repeat=3):
        for offsets_deg in itertools.product(OFFSET_CANDIDATES_DEG, repeat=3):
            phi   = angle_signs[0] * mid_angle[angle_perm[0]] + np.radians(offsets_deg[0])
            theta = angle_signs[1] * mid_angle[angle_perm[1]] + np.radians(offsets_deg[1])
            psi   = angle_signs[2] * mid_angle[angle_perm[2]] + np.radians(offsets_deg[2])
            R = rotation_matrix_batch(phi, theta, psi)

            for axis_perm in itertools.permutations(range(3)):    # cam axis -> body X/Y/Z
                for axis_signs in itertools.product([1, -1], repeat=3):
                    v_body = np.column_stack([
                        axis_signs[0] * v_cam_search[:, axis_perm[0]],
                        axis_signs[1] * v_cam_search[:, axis_perm[1]],
                        axis_signs[2] * v_cam_search[:, axis_perm[2]],
                    ])
                    v_nav = np.einsum("nij,nj->ni", R, v_body)
                    mae = np.mean(np.abs(v_nav - gt_vel))
                    count += 1
                    if best is None or mae < best[0]:
                        best = (mae, angle_perm, angle_signs, offsets_deg, axis_perm, axis_signs)

mae, angle_perm, angle_signs, offsets_deg, axis_perm, axis_signs = best
angle_names = ["raw_col7", "raw_col8", "raw_col9"]
axis_names  = ["cam_vx", "cam_vy", "cam_vz"]

print(f"\nSearched {count} combinations.")
print(f"BEST: MAE = {mae:.4f} m/s\n")
print("Angle roles (rotation_matrix(phi, theta, psi)):")
print(f"  phi   = {angle_signs[0]:+d} * {angle_names[angle_perm[0]]} + {offsets_deg[0]:.0f} deg")
print(f"  theta = {angle_signs[1]:+d} * {angle_names[angle_perm[1]]} + {offsets_deg[1]:.0f} deg")
print(f"  psi   = {angle_signs[2]:+d} * {angle_names[angle_perm[2]]} + {offsets_deg[2]:.0f} deg")
print("\nCamera velocity -> body axis mapping:")
print(f"  v_body[0] (X) = {axis_signs[0]:+d} * {axis_names[axis_perm[0]]}")
print(f"  v_body[1] (Y) = {axis_signs[1]:+d} * {axis_names[axis_perm[1]]}")
print(f"  v_body[2] (Z) = {axis_signs[2]:+d} * {axis_names[axis_perm[2]]}")

# --- Apply the winning combination to all variants and save results ---
phi_best   = angle_signs[0] * mid_angle[angle_perm[0]] + np.radians(offsets_deg[0])
theta_best = angle_signs[1] * mid_angle[angle_perm[1]] + np.radians(offsets_deg[1])
psi_best   = angle_signs[2] * mid_angle[angle_perm[2]] + np.radians(offsets_deg[2])
R_best = rotation_matrix_batch(phi_best, theta_best, psi_best)

vel_out = {"frame": vel_df.loc[:n - 1, "frame"].values}
pos_out = {"frame": vel_df.loc[:n - 1, "frame"].values}

for v in VARIANTS:
    v_cam_v = vel_df.loc[:n - 1, [f"vx_{v}", f"vy_{v}", f"vz_{v}"]].to_numpy(dtype=float)
    v_body_v = np.column_stack([
        axis_signs[0] * v_cam_v[:, axis_perm[0]],
        axis_signs[1] * v_cam_v[:, axis_perm[1]],
        axis_signs[2] * v_cam_v[:, axis_perm[2]],
    ])
    v_nav_v = np.einsum("nij,nj->ni", R_best, v_body_v)
    pos_v = np.cumsum(v_nav_v * dt[:, None], axis=0)

    vel_out[f"vx_{v}"], vel_out[f"vy_{v}"], vel_out[f"vz_{v}"] = v_nav_v[:, 0], v_nav_v[:, 1], v_nav_v[:, 2]
    pos_out[f"x_{v}"], pos_out[f"y_{v}"], pos_out[f"z_{v}"] = pos_v[:, 0], pos_v[:, 1], pos_v[:, 2]

pd.DataFrame(vel_out).to_csv(VELOCITY_CSV_OUT, index=False)
pd.DataFrame(pos_out).to_csv(POSITION_CSV_OUT, index=False)

print(f"\nSaved rotated velocity to {VELOCITY_CSV_OUT}")
print(f"Saved rotated position to {POSITION_CSV_OUT}")
