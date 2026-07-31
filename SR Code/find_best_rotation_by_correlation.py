"""
Same exhaustive search as find_and_apply_best_rotation.py, but scored by
CORRELATION instead of MAE.

Why: MAE conflates polarity error with amplitude/scale error. Since the VO
velocity is running at roughly half the amplitude of ground truth (likely
from the MA/LP filter not settling within each survey leg) and is time-
delayed, the MAE-optimal combination is not guaranteed to be the correctly-
SIGNED one -- a wrong-sign combo could coincidentally produce a smaller
absolute deviation given the amplitude mismatch.

Correlation is invariant to amplitude and constant offset -- it only asks
"do these two signals rise and fall together". If flipping Vx/Vy really is
the fix, this will show it clearly and unambiguously.

This does NOT fix the amplitude/lag problem -- it only answers the sign/
orientation question cleanly. Amplitude and lag are separate problems (see
diagnose_scale_and_lag.py) that this script deliberately ignores by scoring
on correlation, not error magnitude.
"""

import itertools
import numpy as np
import pandas as pd

POSE_EST_PATH   = r"D:\Stingray Robotics\ScottReef GT\ScottReef_20090804_084719_SLAM\stereo_pose_est.data"
VELOCITY_CSV_IN = r"D:\Stingray Robotics\SR Output\SR Output 2\velocity_output002_scottreef.csv"
VELOCITY_CSV_OUT = r"D:\Stingray Robotics\SR Output\SR Output 2\velocity_output0024_scottreef_rotated.csv"
POSITION_CSV_OUT = r"D:\Stingray Robotics\SR Output\SR Output 2\position_output0024_scottreef_rotated.csv"

VARIANTS       = ["sp_ma", "sp_lp", "sift_ma", "sift_lp"]
SEARCH_VARIANT = "sp_ma"
OFFSET_CANDIDATES_DEG = [0.0, 90.0, 180.0, 270.0]


def load_pose_est(path):
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


def per_axis_correlation(v_nav, gt_vel):
    """Mean Pearson correlation across the 3 axes. +1 = perfectly in
    phase, -1 = perfectly inverted, 0 = unrelated -- independent of
    amplitude or constant offset."""
    corrs = []
    for k in range(3):
        a = v_nav[:, k] - v_nav[:, k].mean()
        b = gt_vel[:, k] - gt_vel[:, k].mean()
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        corrs.append(np.dot(a, b) / denom if denom > 1e-12 else 0.0)
    return np.mean(corrs)


ts, gt_N, gt_E, gt_D, a7, a8, a9 = load_pose_est(POSE_EST_PATH)
raw_angles = [a7, a8, a9]

vel_df = pd.read_csv(VELOCITY_CSV_IN)
n = min(len(vel_df), len(ts) - 1)

dt = np.diff(ts[:n + 1])
gt_vel = np.column_stack([
    np.diff(gt_N[:n + 1]),
    np.diff(gt_E[:n + 1]),
    np.diff(gt_D[:n + 1]),
]) / dt[:, None]

mid_angle = [(a[:n] + a[1:n + 1]) / 2.0 for a in raw_angles]

v_cam_search = vel_df.loc[:n - 1, [f"vx_{SEARCH_VARIANT}", f"vy_{SEARCH_VARIANT}",
                                    f"vz_{SEARCH_VARIANT}"]].to_numpy(dtype=float)

print("Searching by CORRELATION (not MAE) -- isolating sign/orientation from amplitude/lag...")

best = None   # (correlation, ...) -- we MAXIMIZE this time
count = 0
for angle_perm in itertools.permutations(range(3)):
    for angle_signs in itertools.product([1, -1], repeat=3):
        for offsets_deg in itertools.product(OFFSET_CANDIDATES_DEG, repeat=3):
            phi   = angle_signs[0] * mid_angle[angle_perm[0]] + np.radians(offsets_deg[0])
            theta = angle_signs[1] * mid_angle[angle_perm[1]] + np.radians(offsets_deg[1])
            psi   = angle_signs[2] * mid_angle[angle_perm[2]] + np.radians(offsets_deg[2])
            R = rotation_matrix_batch(phi, theta, psi)

            for axis_perm in itertools.permutations(range(3)):
                for axis_signs in itertools.product([1, -1], repeat=3):
                    v_body = np.column_stack([
                        axis_signs[0] * v_cam_search[:, axis_perm[0]],
                        axis_signs[1] * v_cam_search[:, axis_perm[1]],
                        axis_signs[2] * v_cam_search[:, axis_perm[2]],
                    ])
                    v_nav = np.einsum("nij,nj->ni", R, v_body)
                    corr = per_axis_correlation(v_nav, gt_vel)
                    count += 1
                    if best is None or corr > best[0]:
                        best = (corr, angle_perm, angle_signs, offsets_deg, axis_perm, axis_signs)

corr, angle_perm, angle_signs, offsets_deg, axis_perm, axis_signs = best
angle_names = ["raw_col7", "raw_col8", "raw_col9"]
axis_names  = ["cam_vx", "cam_vy", "cam_vz"]

print(f"\nSearched {count} combinations.")
print(f"BEST mean correlation = {corr:.4f}  (1.0 = perfect, this is what MAE was blind to)\n")
print("Angle roles (rotation_matrix(phi, theta, psi)):")
print(f"  phi   = {angle_signs[0]:+d} * {angle_names[angle_perm[0]]} + {offsets_deg[0]:.0f} deg")
print(f"  theta = {angle_signs[1]:+d} * {angle_names[angle_perm[1]]} + {offsets_deg[1]:.0f} deg")
print(f"  psi   = {angle_signs[2]:+d} * {angle_names[angle_perm[2]]} + {offsets_deg[2]:.0f} deg")
print("\nCamera velocity -> body axis mapping:")
print(f"  v_body[0] (X) = {axis_signs[0]:+d} * {axis_names[axis_perm[0]]}")
print(f"  v_body[1] (Y) = {axis_signs[1]:+d} * {axis_names[axis_perm[1]]}")
print(f"  v_body[2] (Z) = {axis_signs[2]:+d} * {axis_names[axis_perm[2]]}")

# Show per-axis correlation for the winner
phi_b   = angle_signs[0] * mid_angle[angle_perm[0]] + np.radians(offsets_deg[0])
theta_b = angle_signs[1] * mid_angle[angle_perm[1]] + np.radians(offsets_deg[1])
psi_b   = angle_signs[2] * mid_angle[angle_perm[2]] + np.radians(offsets_deg[2])
R_b = rotation_matrix_batch(phi_b, theta_b, psi_b)
v_body_b = np.column_stack([
    axis_signs[0] * v_cam_search[:, axis_perm[0]],
    axis_signs[1] * v_cam_search[:, axis_perm[1]],
    axis_signs[2] * v_cam_search[:, axis_perm[2]],
])
v_nav_b = np.einsum("nij,nj->ni", R_b, v_body_b)
for k, label in enumerate(["vx", "vy", "vz"]):
    a = v_nav_b[:, k] - v_nav_b[:, k].mean()
    b = gt_vel[:, k] - gt_vel[:, k].mean()
    r = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    print(f"  {label}: correlation = {r:+.3f}")

# ---------------------------------------------------------------------------
# Apply the SAME winning rotation/axis-mapping (found on SEARCH_VARIANT) to
# ALL FOUR variants, then save velocity and position outputs in exactly the
# same column layout/order as the original input files.
# ---------------------------------------------------------------------------
vel_out_cols = {"frame": vel_df["frame"].values[:n]}
pos_out_cols = {"frame": vel_df["frame"].values[:n]}

for variant in VARIANTS:
    v_cam_v = vel_df.loc[:n - 1, [f"vx_{variant}", f"vy_{variant}", f"vz_{variant}"]].to_numpy(dtype=float)
    v_body_v = np.column_stack([
        axis_signs[0] * v_cam_v[:, axis_perm[0]],
        axis_signs[1] * v_cam_v[:, axis_perm[1]],
        axis_signs[2] * v_cam_v[:, axis_perm[2]],
    ])
    v_nav_v = np.einsum("nij,nj->ni", R_b, v_body_v)

    vel_out_cols[f"vx_{variant}"] = v_nav_v[:, 0]
    vel_out_cols[f"vy_{variant}"] = v_nav_v[:, 1]
    vel_out_cols[f"vz_{variant}"] = v_nav_v[:, 2]

    # Position = cumulative integration of velocity over dt (same dt used
    # for the GT comparison above). Position output only has x/z, matching
    # the original pipeline's format (y/vertical is not carried into position).
    pos_out_cols[f"x_{variant}"] = np.cumsum(v_nav_v[:, 0] * dt)
    pos_out_cols[f"z_{variant}"] = np.cumsum(v_nav_v[:, 2] * dt)

vel_out_df = pd.DataFrame(vel_out_cols)
pos_out_df = pd.DataFrame(pos_out_cols)

vel_out_df.to_csv(VELOCITY_CSV_OUT, index=False)
pos_out_df.to_csv(POSITION_CSV_OUT, index=False)
print(f"\nSaved rotated velocity ({len(vel_out_df)} rows) to:\n  {VELOCITY_CSV_OUT}")
print(f"Saved rotated position ({len(pos_out_df)} rows) to:\n  {POSITION_CSV_OUT}")