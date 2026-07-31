"""
validate_velocity_simple.py

Frame-by-frame validation of velocity_output002_scottreef.csv against the
DVL ground-truth velocity (vx, vy, vz) recorded directly in the .auv RDI
lines - no differentiation of position, no pipeline re-run needed.

A few things worth knowing before reading the numbers:

1) The .auv file does NOT hand you one ground-truth sample per camera
   frame. The DVL (RDI:) logs at its own ~3-4 Hz rate, completely
   independent of the camera's ~1 Hz frame rate. So "frame by frame"
   here means: for each estimated frame, find the closest-in-time RDI
   sample and compare against that.

2) The CSV has no timestamp for each frame, so step 1 needs a clock.
   stereo_pose_est.data supplies exactly that: it lists one pose per
   processed stereo pair with a real Unix timestamp, in the same time
   order the frames were produced. This script matches frame N to row N
   of that file to recover a timestamp, then uses it to find the
   nearest DVL sample. This is a positional approximation (see the
   printed sanity check) - it is not as good as a true per-frame
   timestamp, but it needs no pipeline re-run.

3) The calibration file you supplied (ScottReef_20090804_084719.calib)
   only contains the STEREO rig geometry - intrinsics for both cameras
   and the left/right extrinsic baseline (~7cm). It does not contain
   the camera-to-vehicle mounting rotation, so it can't be used to
   rotate the estimated velocity into the vehicle body frame. That
   information simply isn't in this file.

   Net effect: estimated vx/vy/vz (camera frame) and ground-truth
   vx/vy/vz (vehicle body frame) are still not the same physical axes.
   Both are reported below for reference, but SPEED (sqrt(vx^2+vy^2+
   vz^2)) is the number to trust, since it's the same regardless of
   which frame either vector is expressed in.
"""

import re
import numpy as np
import pandas as pd

# ---------------------------------------------------------------
# Config
# ---------------------------------------------------------------
AUV_FILES = [
    "D:\Stingray Robotics\ScottReef\SR AUV GT\20090804_0847.RAW.auv",
    "D:\Stingray Robotics\ScottReef\SR AUV GT\20090804_0900_RAW.auv",
    "D:\Stingray Robotics\ScottReef\SR AUV GT\20090804_1000_RAW.auv",
    "D:\Stingray Robotics\ScottReef\SR AUV GT\20090804_1100_RAW.auv",
    "D:\Stingray Robotics\ScottReef\SR AUV GT\20090804_1200_RAW.auv",
]
VELOCITY_CSV_PATH = "D:\Stingray Robotics\ScottReef\SR Output\SR Output 2\velocity_output002_scottreef.csv"
STEREO_POSE_PATH = "D:\Stingray Robotics\ScottReef\ScottReef GT\stereo_pose_est.data"
OUTPUT_CSV_PATH = "D:\Stingray Robotics\ScottReef\SR Output\SR Output 2\velocity_output0025_scottreef.csv"

MATCH_TOLERANCE_S = 0.5   # max gap allowed between a frame and its matched DVL sample
METHODS = ["sp_ma", "sp_lp", "sift_ma", "sift_lp"]

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
    return pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)

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

# Recover a timestamp per CSV frame: frame N -> row N of stereo_pose_est.data
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

merged["speed_gt"] = np.sqrt(merged.vx_gt**2 + merged.vy_gt**2 + merged.vz_gt**2)

out_cols = ["frame", "timestamp", "vx_gt", "vy_gt", "vz_gt", "speed_gt"]
print(f"{'method':28s}{'MAE speed':>12s}{'RMSE speed':>12s}{'corr':>8s}")
for m in METHODS:
    merged[f"speed_{m}"] = np.sqrt(merged[f"vx_{m}"]**2 + merged[f"vy_{m}"]**2 + merged[f"vz_{m}"]**2)
    err = merged[f"speed_{m}"] - merged["speed_gt"]
    mae, rmse = err.abs().mean(), np.sqrt((err**2).mean())
    corr = merged[f"speed_{m}"].corr(merged["speed_gt"])
    print(f"{m:28s}{mae:12.4f}{rmse:12.4f}{corr:8.3f}")
    out_cols += [f"vx_{m}", f"vy_{m}", f"vz_{m}", f"speed_{m}"]

merged[out_cols].to_csv(OUTPUT_CSV_PATH, index=False)
print(f"\nFull frame-by-frame table saved to {OUTPUT_CSV_PATH}")
