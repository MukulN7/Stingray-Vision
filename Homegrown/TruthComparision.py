import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---- load true GPS data ----
lat = pd.read_csv(r"D:\Stingray Robotics\Homegrown Data\Final Dataset\latitude.csv")
lon = pd.read_csv(r"D:\Stingray Robotics\Homegrown Data\Final Dataset\longitude.csv")

t_gt = lat['elapsed time'].values
lat_vals = lat['value'].values
lon_vals = lon['value'].values

# convert lat/lon to local flat-earth x,z (meters) relative to first gps point
R = 6371000
lat0, lon0 = lat_vals[0], lon_vals[0]

east = (lon_vals - lon0) * R * np.cos(np.radians(lat0)) * np.pi / 180
north = (lat_vals - lat0) * R * np.pi / 180

# rotate ground truth so initial motion direction points along +z (forward),
# same idea as aligning world frame to vehicle frame in the KITTI/Malaga scripts
heading = np.arctan2(east[-1] - east[0], north[-1] - north[0])
cos_h, sin_h = np.cos(heading), np.sin(heading)

x_gt = east * cos_h - north * sin_h
z_gt = east * sin_h + north * cos_h

# ---- load estimated data ----
est = pd.read_csv(r"D:\Stingray Robotics\position_outputD.csv")
frames = est['frame'].values

# figure out fps from total gps duration vs total frame count
duration = t_gt[-1] - t_gt[0]
FPS = frames[-1] / duration
t_est = frames / FPS

methods = ['sp_ma', 'sp_lp', 'sift_ma', 'sift_lp']

# interpolate ground truth onto estimated timestamps and compute error
print("Method Comparison (MAE against GPS ground truth)")
print("-" * 55)

x_gt_i = np.interp(t_est, t_gt, x_gt)
z_gt_i = np.interp(t_est, t_gt, z_gt)

for m in methods:
    x_est = est[f'x_{m}'].values
    z_est = est[f'z_{m}'].values
    mae_x = np.mean(np.abs(x_est - x_gt_i))
    mae_z = np.mean(np.abs(z_est - z_gt_i))
    mae_total = np.mean(np.sqrt((x_est - x_gt_i) ** 2 + (z_est - z_gt_i) ** 2))
    print(f"{m:10s} | MAE_x: {mae_x:6.2f} m | MAE_z: {mae_z:6.2f} m | MAE_total: {mae_total:6.2f} m")

# ---- top-down trajectory plot ----
plt.figure(figsize=(8, 8))
plt.plot(x_gt, z_gt, 'k-', linewidth=1, label='Ground Truth (GPS)', alpha=0.5)

colors = ['red', 'blue', 'green', 'orange']
for m, c in zip(methods, colors):
    plt.plot(est[f'x_{m}'], est[f'z_{m}'], color=c, linewidth=1, label=m)

plt.xlabel('X - lateral (m)')
plt.ylabel('Z - forward (m)')
plt.title('Top-Down Trajectory Comparison')
plt.legend()
plt.axis('equal')
plt.grid(True)
plt.tight_layout()
plt.savefig('trajectory_comparison.png', dpi=150)
plt.show()