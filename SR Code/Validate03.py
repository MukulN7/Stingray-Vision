import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

POSE_FILE = r"D:\Stingray Robotics\ScottReef GT\ScottReef_20090804_084719_SLAM\stereo_pose_est.data"
VEL_CSV   = r"D:\Stingray Robotics\SR Output\SR Output 2\velocity_output0020_scottreef_rotated.csv"
POS_CSV   = r"D:\Stingray Robotics\SR Output\SR Output 2\position_output0020_scottreef_rotated.csv"
methods = ["sp_ma", "sp_lp", "sift_ma", "sift_lp"]
labels  = {"sp_ma": "SuperPoint+MA", "sp_lp": "SuperPoint+LP",
           "sift_ma": "SIFT+MA",     "sift_lp": "SIFT+LP"}

# --- Parse ground truth pose file ---
rows = []
with open(POSE_FILE) as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("%") or line.startswith("ORIGIN"):
            continue
        rows.append(line.split())

ts   = np.array([float(r[1]) for r in rows])
gt_N = np.array([float(r[4]) for r in rows])   # North
gt_E = np.array([float(r[5]) for r in rows])   # East
gt_D = np.array([float(r[6]) for r in rows])   # Depth

# --- Ground-truth velocity: nav-frame finite difference (North/East/Down) ---
# VEL_CSV/POS_CSV are already rotated into the nav frame by
# rotate_velocity_to_navframe.py, so GT stays in the nav frame too --
# no rotation into camera frame here, or the two sides won't match.
dt    = np.diff(ts)
gt_vx = np.diff(gt_N) / dt   # North
gt_vy = np.diff(gt_E) / dt   # East
gt_vz = np.diff(gt_D) / dt   # Down

# top-down ground-truth reference (nav frame, North/East)
gt_x, gt_z = gt_N, gt_E

# --- Load estimates ---
vel = pd.read_csv(VEL_CSV)
pos = pd.read_csv(POS_CSV)

n      = min(len(vel), len(gt_vx))
frames = vel["frame"].values[:n]
gt_vx, gt_vy, gt_vz = gt_vx[:n], gt_vy[:n], gt_vz[:n]

# --- Mean absolute error ---
print(f"{'Method':15s} {'MAE_vx':>8s} {'MAE_vy':>8s} {'MAE_vz':>8s}")
for m in methods:
    mae_x = np.mean(np.abs(vel[f"vx_{m}"].values[:n] - gt_vx))
    mae_y = np.mean(np.abs(vel[f"vy_{m}"].values[:n] - gt_vy))
    mae_z = np.mean(np.abs(vel[f"vz_{m}"].values[:n] - gt_vz))
    print(f"{labels[m]:15s} {mae_x:8.4f} {mae_y:8.4f} {mae_z:8.4f}")

# --- Velocity plots (one figure per method) ---
for m in methods:
    fig, axs = plt.subplots(3, 1, figsize=(14, 7), sharex=True)
    fig.suptitle(labels[m])
    for ax, axis, gt in zip(axs, ["vx", "vy", "vz"], [gt_vx, gt_vy, gt_vz]):
        ax.plot(frames, vel[f"{axis}_{m}"].values[:n], color="tab:blue", label=labels[m])
        ax.plot(frames, gt, color="black", label="Ground Truth")
        ax.set_ylabel(f"V{axis[1]} (m/s)")
        ax.legend()
    axs[-1].set_xlabel("Frame")
    plt.tight_layout()

# --- Top-down trajectory plot ---
plt.figure(figsize=(8, 8))
plt.plot(gt_x - gt_x[0], gt_z - gt_z[0], color="black", label="Ground Truth")
for m in methods:
    x_m, z_m = pos[f"x_{m}"], pos[f"z_{m}"]
    plt.plot(x_m - x_m.iloc[0], z_m - z_m.iloc[0], label=labels[m])
plt.xlabel("X (m)")
plt.ylabel("Z (m)")
plt.title("Top-Down Trajectory")
plt.legend()
plt.axis("equal")

plt.show()