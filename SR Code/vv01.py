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
   the camera-to-vehicle mounting rotation, so there's no exact way to
   rotate the estimated (camera-frame) velocity into the vehicle body
   frame.

   WORLD-FRAME ROTATION (new in this version)
   --------------------------------------------
   Every matched RDI sample carries not just vx/vy/vz but also the
   vehicle's attitude at that instant: h (heading/yaw), p (pitch),
   r (roll), all in degrees. rotation_matrix(phi, theta, psi) builds
   the body -> nav (world) rotation for a given roll/pitch/yaw.

   For the GROUND TRUTH this is exact: the DVL vx/vy/vz are reported
   in the vehicle body frame, and h/p/r describe that same body's
   attitude, so R_body->nav @ [vx,vy,vz]_gt gives the true world-frame
   velocity.

   For the ESTIMATE this is an *approximation*: vx/vy/vz from the
   pipeline are in the camera frame, not the vehicle body frame, and
   (per point 3 above) the camera-to-body mounting rotation is
   unknown. In the absence of that mounting rotation, this script
   applies the same vehicle attitude (h/p/r) directly to the estimated
   vector, i.e. it treats the camera frame as if it were coincident
   with the body frame. This is the best available common frame given
   the data on hand - if a camera-to-vehicle mounting rotation is ever
   supplied, replace R below (for the estimate only) with
   R_body->nav @ R_cam->body.

   Net effect: vx_nav/vy_nav/vz_nav for gt and for each method are now
   expressed in the same fixed world axes and ARE directly comparable
   component-by-component, not just via SPEED as before. Speed is
   still reported too, since it's frame-invariant and unaffected by
   the camera-mounting caveat above.
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
# Body -> Nav (world) frame rotation, roll/pitch/yaw in radians
# ---------------------------------------------------------------
def rotation_matrix(phi, theta, psi):
    """Body -> nav frame rotation (roll phi, pitch theta, yaw psi), radians."""
    cphi, sphi = np.cos(phi), np.sin(phi)
    cth, sth = np.cos(theta), np.sin(theta)
    cpsi, spsi = np.cos(psi), np.sin(psi)
    return np.array([
        [cpsi * cth, -spsi * cphi + cpsi * sth * sphi,  spsi * sphi + cpsi * cphi * sth],
        [spsi * cth,  cpsi * cphi + sphi * sth * spsi, -cpsi * sphi + sth * spsi * cphi],
        [-sth,        cth * sphi,                       cth * cphi],
    ])

# ---------------------------------------------------------------
# Parse .auv ground truth (RDI / DVL lines -> vehicle body-frame
# velocity + attitude, needed to rotate into nav frame)
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
                        "heading_deg": float(kv["h"]),
                        "pitch_deg": float(kv["p"]),
                        "roll_deg": float(kv["r"]),
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

# ---------------------------------------------------------------
# Rotate ground truth AND every estimate into the common nav/world
# frame, using the attitude (h/p/r) of the matched RDI sample.
# See the module docstring for the caveat on the estimate side.
# ---------------------------------------------------------------
def rotate_columns(df, vx_col, vy_col, vz_col, out_prefix):
    """Apply R_body->nav (from this row's h/p/r) to (vx,vy,vz) columns."""
    vx_nav = np.empty(len(df))
    vy_nav = np.empty(len(df))
    vz_nav = np.empty(len(df))
    phi = np.radians(df["roll_deg"].values)
    theta = np.radians(df["pitch_deg"].values)
    psi = np.radians(df["heading_deg"].values)
    v_body = df[[vx_col, vy_col, vz_col]].values
    for i in range(len(df)):
        R = rotation_matrix(phi[i], theta[i], psi[i])
        v_nav = R @ v_body[i]
        vx_nav[i], vy_nav[i], vz_nav[i] = v_nav
    df[f"{out_prefix}_vx_nav"] = vx_nav
    df[f"{out_prefix}_vy_nav"] = vy_nav
    df[f"{out_prefix}_vz_nav"] = vz_nav

rotate_columns(merged, "vx_gt", "vy_gt", "vz_gt", "gt")
for m in METHODS:
    rotate_columns(merged, f"vx_{m}", f"vy_{m}", f"vz_{m}", m)

merged["speed_gt"] = np.sqrt(merged.vx_gt**2 + merged.vy_gt**2 + merged.vz_gt**2)

out_cols = ["frame", "timestamp", "heading_deg", "pitch_deg", "roll_deg",
            "vx_gt", "vy_gt", "vz_gt", "gt_vx_nav", "gt_vy_nav", "gt_vz_nav", "speed_gt"]

print(f"{'method':28s}{'MAE speed':>12s}{'RMSE speed':>12s}{'corr(speed)':>13s}"
      f"{'MAE vx_nav':>12s}{'MAE vy_nav':>12s}{'MAE vz_nav':>12s}")
for m in METHODS:
    merged[f"speed_{m}"] = np.sqrt(merged[f"vx_{m}"]**2 + merged[f"vy_{m}"]**2 + merged[f"vz_{m}"]**2)
    err = merged[f"speed_{m}"] - merged["speed_gt"]
    mae, rmse = err.abs().mean(), np.sqrt((err**2).mean())
    corr = merged[f"speed_{m}"].corr(merged["speed_gt"])

    mae_vx = (merged[f"{m}_vx_nav"] - merged["gt_vx_nav"]).abs().mean()
    mae_vy = (merged[f"{m}_vy_nav"] - merged["gt_vy_nav"]).abs().mean()
    mae_vz = (merged[f"{m}_vz_nav"] - merged["gt_vz_nav"]).abs().mean()

    print(f"{m:28s}{mae:12.4f}{rmse:12.4f}{corr:13.3f}{mae_vx:12.4f}{mae_vy:12.4f}{mae_vz:12.4f}")

    out_cols += [f"vx_{m}", f"vy_{m}", f"vz_{m}",
                 f"{m}_vx_nav", f"{m}_vy_nav", f"{m}_vz_nav", f"speed_{m}"]

merged[out_cols].to_csv(OUTPUT_CSV_PATH, index=False)
print(f"\nFull frame-by-frame table (with world-frame vx/vy/vz) saved to {OUTPUT_CSV_PATH}")