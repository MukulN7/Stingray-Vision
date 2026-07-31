# Importing Libraries


import numpy as np
import cv2
import re


# Configuration


IN_CALIB  = r"D:\Stingray Robotics\Depth Map on Live Stereo Camera Feed\stereo_calibC.txt"
OUT_CALIB = r"D:\Stingray Robotics\Homegrown Data\calib  C.txt"


# Parse Homegrown Calib File


with open(IN_CALIB, "r") as f:
    text = f.read()

w, h = map(int, re.search(r"Image size \(w x h\):\s*(\d+)\s*x\s*(\d+)", text).groups())

def read_block(label, rows):
    block = re.search(label + r"\s*\n((?:.+\n?){" + str(rows) + "})", text).group(1)
    return np.array([list(map(float, line.split())) for line in block.strip().splitlines()])

K_left  = read_block("K_left",  3)
K_right = read_block("K_right", 3)
D_left  = read_block("D_left",  1).reshape(-1)
D_right = read_block("D_right", 1).reshape(-1)
R       = read_block("R", 3)
T       = read_block("T", 3).reshape(3, 1)


# Stereo Rectify -> Rectified Projection Matrices


_, _, P0, P1, _, _, _ = cv2.stereoRectify(
    K_left, D_left, K_right, D_right, (w, h), R, T,
    flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
)


# Save in KITTI Format


with open(OUT_CALIB, "w") as f:
    for name, P in [("P0", P0), ("P1", P1)]:
        vals = " ".join(f"{v:.12e}" for v in P.flatten())
        f.write(f"{name}: {vals}\n")

print(f"Saved KITTI-format calib to {OUT_CALIB}")