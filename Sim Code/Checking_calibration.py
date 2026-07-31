"""
vo_diagnostics.py

Standalone sanity-check script for a stereo visual odometry pipeline.
Does NOT depend on torch / SuperPoint / LightGlue - only cv2 + numpy,
so it can run even in environments without the full VO stack installed.

Checks performed:
  1. Calibration resolution vs actual video resolution
  2. Raw calibration values (fx, cx, cy, baseline) sanity printout
  3. Reported FPS vs actual frame count / implied duration
  4. Duplicate-frame detection (both left and right videos)
  5. Manual depth check on a handful of frames (prints median disparity/depth
     so you can sanity check against known scene scale)
  6. Manual velocity hand-check: computes SIFT+optical-flow velocity for a
     few consecutive frame pairs using the *exact* same math as the main
     pipeline, but prints every intermediate quantity (dt, disparity, depth,
     pixel displacement) so a bug can be isolated by inspection.

Edit the CONFIG block below to point at your calib file and videos, then run:
    python vo_diagnostics.py
"""

import numpy as np
import cv2


# ============================================================
# CONFIG - edit these paths to match your setup
# ============================================================

CALIB       = r"D:\Stingray Robotics\Visual Odometry Complete Pipeline\KITTI Odometry Full Pipeline\Sim Code\robotsimcalib.txt"
LEFT_VIDEO  = r"D:\Stingray Robotics\Visual Odometry Complete Pipeline\KITTI Odometry Full Pipeline\Sim Code\output3.mp4"
RIGHT_VIDEO = r"D:\Stingray Robotics\Visual Odometry Complete Pipeline\KITTI Odometry Full Pipeline\Sim Code\output4.mp4"

# If you know the true physical baseline you configured in the simulator
# (in meters), put it here for a direct comparison. Set to None to skip.
EXPECTED_BASELINE_M = None

# If you know the true simulated run duration (seconds), put it here to
# cross-check against implied duration from FPS * frame_count. Set to None
# to skip.
EXPECTED_DURATION_S = None

# How many frame pairs to use for duplicate-frame scan and manual hand-check
N_FRAMES_TO_SCAN   = 200
N_FRAMES_HANDCHECK = 5


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ============================================================
# 1 & 2. Load calibration and print raw values
# ============================================================

section("1-2. CALIBRATION FILE CONTENTS")

with open(CALIB, "r") as f:
    lines = f.readlines()

P0 = np.array(list(map(float, lines[0].split()[1:])), dtype=np.float32).reshape(3, 4)
P1 = np.array(list(map(float, lines[1].split()[1:])), dtype=np.float32).reshape(3, 4)

fx       = P0[0, 0]
fy       = P0[1, 1]
cx       = P0[0, 2]
cy       = P0[1, 2]
baseline = abs(P1[0, 3] / fx)

print(f"P0:\n{P0}")
print(f"P1:\n{P1}")
print(f"fx = {fx:.4f}")
print(f"fy = {fy:.4f}")
print(f"cx = {cx:.4f}")
print(f"cy = {cy:.4f}")
print(f"baseline (from P1[0,3]/fx) = {baseline:.6f} m")

if EXPECTED_BASELINE_M is not None:
    ratio = baseline / EXPECTED_BASELINE_M
    print(f"\nExpected baseline: {EXPECTED_BASELINE_M} m")
    print(f"Ratio (calib/expected): {ratio:.3f}")
    if abs(ratio - 1.0) > 0.05:
        print("  !! MISMATCH: baseline does not match expected value.")
        print("     This directly scales computed depth (Z = fx*baseline/disparity)")
        print("     and therefore scales computed velocity by roughly the same factor.")
else:
    print("\n(Set EXPECTED_BASELINE_M in the config block to cross-check this "
          "against your simulator's actual rig setup.)")


# ============================================================
# 3. Video resolution vs calibration principal point
# ============================================================

section("3. VIDEO RESOLUTION vs CALIBRATION")

cap_left  = cv2.VideoCapture(LEFT_VIDEO)
cap_right = cv2.VideoCapture(RIGHT_VIDEO)

if not cap_left.isOpened() or not cap_right.isOpened():
    raise RuntimeError("Could not open one or both video files. Check paths.")

w_l = int(cap_left.get(cv2.CAP_PROP_FRAME_WIDTH))
h_l = int(cap_left.get(cv2.CAP_PROP_FRAME_HEIGHT))
w_r = int(cap_right.get(cv2.CAP_PROP_FRAME_WIDTH))
h_r = int(cap_right.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Left video resolution:  {w_l} x {h_l}")
print(f"Right video resolution: {w_r} x {h_r}")
if (w_l, h_l) != (w_r, h_r):
    print("  !! Left and right video resolutions do not match!")

print(f"\ncx, cy from calib: {cx:.2f}, {cy:.2f}")
print(f"Expected (w/2, h/2) from left video: {w_l/2:.2f}, {h_l/2:.2f}")

cx_diff_pct = abs(cx - w_l / 2) / (w_l / 2) * 100
cy_diff_pct = abs(cy - h_l / 2) / (h_l / 2) * 100
print(f"cx difference: {cx_diff_pct:.1f}% of half-width")
print(f"cy difference: {cy_diff_pct:.1f}% of half-height")

if cx_diff_pct > 15 or cy_diff_pct > 15:
    print("  !! Large mismatch between calib principal point and video center.")
    print("     This strongly suggests the calibration file describes a DIFFERENT")
    print("     resolution than these videos (fx/cx/cy are resolution-dependent).")
else:
    print("  OK: principal point roughly matches video center.")


# ============================================================
# 4. FPS / dt / duration cross-check
# ============================================================

section("4. FPS / dt / DURATION CHECK")

fps_l = cap_left.get(cv2.CAP_PROP_FPS)
fps_r = cap_right.get(cv2.CAP_PROP_FPS)
count_l = int(cap_left.get(cv2.CAP_PROP_FRAME_COUNT))
count_r = int(cap_right.get(cv2.CAP_PROP_FRAME_COUNT))

dt = 1.0 / fps_l

print(f"Left  video: FPS={fps_l:.4f}, frame_count={count_l}, "
      f"implied duration={count_l / fps_l:.3f} s")
print(f"Right video: FPS={fps_r:.4f}, frame_count={count_r}, "
      f"implied duration={count_r / fps_r:.3f} s")
print(f"dt used in pipeline (1/fps_left) = {dt * 1000:.3f} ms")

if abs(fps_l - fps_r) > 1e-3:
    print("  !! Left and right videos report different FPS - they should match "
          "for synchronized stereo capture.")

if count_l != count_r:
    print(f"  !! Left and right frame counts differ ({count_l} vs {count_r}). "
          "Streams may be desynchronized.")

if EXPECTED_DURATION_S is not None:
    implied = count_l / fps_l
    ratio = implied / EXPECTED_DURATION_S
    print(f"\nExpected simulated duration: {EXPECTED_DURATION_S} s")
    print(f"Implied duration from video: {implied:.3f} s")
    print(f"Ratio (implied/expected): {ratio:.3f}")
    if abs(ratio - 1.0) > 0.05:
        print("  !! MISMATCH: the video's reported FPS does not match the actual")
        print("     physical duration of the simulated run. This means dt is wrong")
        print("     by roughly this same factor, which directly and inversely")
        print("     scales every computed velocity (v = pixel_disp / (dt * fx)).")
        print(f"     A dt that is {ratio:.2f}x too LARGE would make computed")
        print(f"     velocity ~{1/ratio:.2f}x too SMALL - matching an underestimate.")
else:
    print("\n(Set EXPECTED_DURATION_S in the config block if you know how long "
          "the simulated run actually was, to directly check for a dt scaling bug.)")


# ============================================================
# 5. Duplicate-frame scan
# ============================================================

section(f"5. DUPLICATE-FRAME SCAN (first {N_FRAMES_TO_SCAN} frames)")

def scan_duplicates(path, n):
    cap = cv2.VideoCapture(path)
    prev = None
    dup_indices = []
    for i in range(n):
        ret, frame = cap.read()
        if not ret:
            break
        if prev is not None and np.array_equal(frame, prev):
            dup_indices.append(i)
        prev = frame
    cap.release()
    return dup_indices

dup_left  = scan_duplicates(LEFT_VIDEO, N_FRAMES_TO_SCAN)
dup_right = scan_duplicates(RIGHT_VIDEO, N_FRAMES_TO_SCAN)

print(f"Left video:  {len(dup_left)} exact duplicate frame(s) found "
      f"{'at indices ' + str(dup_left) if dup_left else ''}")
print(f"Right video: {len(dup_right)} exact duplicate frame(s) found "
      f"{'at indices ' + str(dup_right) if dup_right else ''}")

if dup_left or dup_right:
    print("  !! Duplicate frames detected. If the simulator rendered new content")
    print("     less often than the video's frame rate (e.g. content updates at")
    print("     20fps but exported at 60fps), real inter-frame motion is smaller")
    print("     than dt implies, causing velocity UNDERESTIMATION.")
else:
    print("  OK: no exact duplicate frames in the scanned range.")


# ============================================================
# 6. Manual depth + velocity hand-check on real frames
# ============================================================

section(f"6. MANUAL DEPTH/VELOCITY HAND-CHECK ({N_FRAMES_HANDCHECK} frame pairs)")

cap_left  = cv2.VideoCapture(LEFT_VIDEO)
cap_right = cv2.VideoCapture(RIGHT_VIDEO)

stereo = cv2.StereoSGBM_create(
    minDisparity=0, numDisparities=64, blockSize=5,
    P1=8 * 3 * 5 ** 2, P2=32 * 3 * 5 ** 2, disp12MaxDiff=1,
    uniquenessRatio=10, speckleWindowSize=100, speckleRange=32,
)

sift = cv2.SIFT_create()

ret_l, left_raw = cap_left.read()
ret_r, right_raw = cap_right.read()
if not (ret_l and ret_r):
    raise RuntimeError("Could not read first frame pair.")

left_old  = cv2.cvtColor(left_raw, cv2.COLOR_BGR2GRAY)
right_old = cv2.cvtColor(right_raw, cv2.COLOR_BGR2GRAY)

kps, _ = sift.detectAndCompute(left_old, None)
pts_old = np.array([kp.pt for kp in kps], dtype=np.float32).reshape(-1, 1, 2)
print(f"Initial SIFT keypoints detected: {len(pts_old)}")

for frame_idx in range(N_FRAMES_HANDCHECK):
    ret_l, left_raw  = cap_left.read()
    ret_r, right_raw = cap_right.read()
    if not (ret_l and ret_r):
        print(f"End of video at frame {frame_idx}")
        break

    left_new  = cv2.cvtColor(left_raw,  cv2.COLOR_BGR2GRAY)
    right_new = cv2.cvtColor(right_raw, cv2.COLOR_BGR2GRAY)

    disparity = stereo.compute(left_new, right_new).astype(np.float32) / 16.0
    valid_disp = disparity[disparity > 0]

    print(f"\n--- Frame pair {frame_idx} ---")
    if valid_disp.size > 0:
        med_disp = np.median(valid_disp)
        med_depth = (fx * baseline) / med_disp
        print(f"Median valid disparity: {med_disp:.3f} px")
        print(f"Implied median depth:   {med_depth:.3f} m  "
              f"(sanity check this against your known scene scale)")
    else:
        print("No valid disparity pixels found in this frame.")

    pts_new, status, _ = cv2.calcOpticalFlowPyrLK(left_old, left_new, pts_old, None)
    old_pts = pts_old[status == 1]
    new_pts = pts_new[status == 1]
    print(f"Tracked points: {len(new_pts)} / {len(pts_old)}")

    if len(old_pts) > 0:
        disp_px = new_pts.reshape(-1, 2) - old_pts.reshape(-1, 2)
        med_disp_px = np.median(np.linalg.norm(disp_px, axis=1))
        print(f"Median pixel displacement per frame: {med_disp_px:.4f} px")
        print(f"  -> implied angular rate proxy (disp_px / (dt*fx)): "
              f"{med_disp_px / (dt * fx):.6f} (rad/s-ish, unitless per-axis units)")

    left_old = left_new
    pts_old = new_pts.reshape(-1, 1, 2)
    if len(pts_old) < 400:
        kps, _ = sift.detectAndCompute(left_old, None)
        pts_old = np.array([kp.pt for kp in kps], dtype=np.float32).reshape(-1, 1, 2)

cap_left.release()
cap_right.release()

section("DONE")
print("Review the flagged (!!) lines above first - they point at the most")
print("likely source of a constant scale/bias error in the pipeline.")