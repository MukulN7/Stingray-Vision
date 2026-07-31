"""
validate_velocity_simple.py

Frame-by-frame validation of velocity_output002_scottreef.csv against the
DVL ground-truth velocity (vx, vy, vz) recorded directly in the .auv RDI
lines. Raw per-axis comparison only (no speed magnitude) - estimated
vx/vy/vz is in the camera's own frame, ground truth vx/vy/vz is in the
vehicle body frame, so the axes aren't guaranteed to point the same way,
but this is the raw axis-by-axis comparison as requested.

Frame -> timestamp recovery (frame N -> row N of stereo_pose_est.data) and
nearest-time matching against the DVL log work exactly as before - see the
earlier script/explanation for why that's needed.
"""

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------
# Config
# ---------------------------------------------------------------
AUV_FILES = [
    "/mnt/user-data/uploads/20090804_0847_RAW.auv",
    "/mnt/user-data/uploads/20090804_0900_RAW.auv",
    "/mnt/user-data/uploads/20090804_1000_RAW.auv",
    "/mnt/user-data/uploads/20090804_1100_RAW.auv",
    "/mnt/user-data/uploads/20090804_1200_RAW.auv",
]
VELOCITY_CSV_PATH = "/mnt/user-data/uploads/velocity_output002_scottreef.csv"
STEREO_POSE_PATH = "/mnt/user-data/uploads/stereo_pose_est.data"
OUTPUT_CSV_PATH = "/mnt/user-data/outputs/frame_by_frame_validation.csv"
OUTPUT_PLOT_DIR = "/mnt/user-data/outputs"

MATCH_TOLERANCE_S = 0.5  # max gap allowed between a frame and its matched DVL sample

# DVL occasionally drops bottom-lock for a single ping and reports a wild spike
# (e.g. frame 835: vx=-3.3, vy=-2.84 m/s, versus a normal range of about +-0.5).
# Any ground-truth sample with |v| beyond this on any axis is treated as a bad
# reading and dropped before matching/plotting.
GT_OUTLIER_LIMIT_MPS = 1.0

METHODS = ["sp_ma", "sp_lp", "sift_ma", "sift_lp"]
METHOD_LABELS = {
    "sp_ma": "SuperPoint+MA",
    "sp_lp": "SuperPoint+LP",
    "sift_ma": "SIFT+MA",
    "sift_lp": "SIFT+LP",
}

# ---------------------------------------------------------------
# Parse .auv ground truth (RDI / DVL lines -> vehicle body-frame velocity)
# ---------------------------------------------------------------
RDI_LINE = re.compile(r"^RDI:\s+([\d.]+)\s+(.*)$")
KV = re.compile(r"([A-Za-z_]+\d*):(-?\d+\.?\d*)")

def parse_auv_rdi(paths):
    rows = []
    for path in paths:
        with open(path, "r", errors="ignore") as f:
            for line in f:
                m = RDI_LINE.match(line.strip())
                if not m:
                    continue
                kv = dict(KV.findall(m.group(2)))
                try:
                    rows.append({
                        "timestamp": float(m.group(1)),
                        "vx_gt": float(kv["vx"]),
                        "vy_gt": float(kv["vy"]),
                        "vz_gt": float(kv["vz"]),
                    })
                except KeyError:
                    continue
    df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)

    # Drop single-ping DVL bottom-lock-loss spikes (see GT_OUTLIER_LIMIT_MPS).
    bad = (df.vx_gt.abs() > GT_OUTLIER_LIMIT_MPS) | \
          (df.vy_gt.abs() > GT_OUTLIER_LIMIT_MPS) | \
          (df.vz_gt.abs() > GT_OUTLIER_LIMIT_MPS)
    if bad.any():
        print(f"Dropping {bad.sum()} DVL outlier sample(s) beyond "
              f"+-{GT_OUTLIER_LIMIT_MPS} m/s: "
              f"{df.loc[bad, ['timestamp','vx_gt','vy_gt','vz_gt']].to_dict('records')}")
        df = df.loc[~bad].reset_index(drop=True)
    return df

# ---------------------------------------------------------------
# Parse stereo_pose_est.data (only used to recover a timestamp per frame)
# ---------------------------------------------------------------
def parse_stereo_pose_timestamps(path):
    rows = []
    with open(path, "r", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("%") or line.startswith("ORIGIN"):
                continue
            p = line.split()
            rows.append({"timestamp": float(p[1])})
    return pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)

# ---------------------------------------------------------------
# Load everything
# ---------------------------------------------------------------
gt_df = parse_auv_rdi(AUV_FILES)
pose_df = parse_stereo_pose_timestamps(STEREO_POSE_PATH)
vel_df = pd.read_csv(VELOCITY_CSV_PATH)

idx = (vel_df["frame"].astype(int) - 1).clip(0, len(pose_df) - 1)
vel_df["timestamp"] = pose_df.iloc[idx]["timestamp"].values

lo, hi = vel_df.timestamp.min(), vel_df.timestamp.max()
ok = gt_df.timestamp.min() <= lo and hi <= gt_df.timestamp.max()
print(f"Matched frame timestamps span {lo:.1f} -> {hi:.1f} (epoch), "
      f"{'inside' if ok else 'OUTSIDE'} the .auv log span "
      f"({gt_df.timestamp.min():.1f} -> {gt_df.timestamp.max():.1f}) - "
      f"{'looks plausible.' if ok else 'CHECK ALIGNMENT.'}")

# ---------------------------------------------------------------
# Frame-by-frame nearest-time match against ground truth
# ---------------------------------------------------------------
merged = pd.merge_asof(
    vel_df.sort_values("timestamp"),
    gt_df.sort_values("timestamp"),
    on="timestamp",
    direction="nearest",
    tolerance=MATCH_TOLERANCE_S,
).dropna(subset=["vx_gt"]).reset_index(drop=True)

print(f"Matched {len(merged)}/{len(vel_df)} frames to a DVL sample within {MATCH_TOLERANCE_S}s.\n")

# ---------------------------------------------------------------
# Per-axis error (no speed magnitude)
# ---------------------------------------------------------------
out_cols = ["frame", "timestamp", "vx_gt", "vy_gt", "vz_gt"]
print(f"{'method':16s}{'axis':6s}{'MAE':>10s}{'RMSE':>10s}{'corr':>8s}")
for m in METHODS:
    for axis in ["x", "y", "z"]:
        est, gt = merged[f"v{axis}_{m}"], merged[f"v{axis}_gt"]
        err = est - gt
        mae, rmse = err.abs().mean(), np.sqrt((err**2).mean())
        corr = est.corr(gt)
        print(f"{m:16s}{axis:>6s}{mae:10.4f}{rmse:10.4f}{corr:8.3f}")
    out_cols += [f"vx_{m}", f"vy_{m}", f"vz_{m}"]

merged[out_cols].to_csv(OUTPUT_CSV_PATH, index=False)
print(f"\nFull frame-by-frame table saved to {OUTPUT_CSV_PATH}")

# ---------------------------------------------------------------
# Plots - one figure per method, 3 rows (Vx/Vy/Vz) x 2 cols (GT | method)
# ---------------------------------------------------------------
frame_num = merged["frame"]
for m in METHODS:
    label = METHOD_LABELS[m]
    fig, axs = plt.subplots(3, 2, figsize=(15, 8), sharex=True)
    fig.suptitle(label, fontsize=13)

    axes_letters = ["x", "y", "z"]
    axes_titles = ["Vx (m/s)", "Vy (m/s)", "Vz (m/s)"]

    for row, (letter, ylabel) in enumerate(zip(axes_letters, axes_titles)):
        axs[row, 0].plot(frame_num, merged[f"v{letter}_gt"], color="black", label="Ground Truth")
        axs[row, 0].set_ylabel(ylabel)
        axs[row, 0].legend(loc="upper right")

        axs[row, 1].plot(frame_num, merged[f"v{letter}_{m}"], color="tab:blue", label=label)
        axs[row, 1].legend(loc="upper right")

    axs[0, 0].set_title("Ground Truth")
    axs[0, 1].set_title(label)
    axs[2, 0].set_xlabel("Frame")
    axs[2, 1].set_xlabel("Frame")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = f"{OUTPUT_PLOT_DIR}/velocity_comparison_{m}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")
