
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Config - change these if your file names / column names differ

LAT_FILE = r"D:\Stingray Robotics\Homegrown Data\Final Dataset\latitude.csv"
LON_FILE = r"D:\Stingray Robotics\Homegrown Data\Final Dataset\longitude.csv"
ALT_FILE = r"D:\Stingray Robotics\Homegrown Data\Final Dataset\altitude.csv"
VEL_FILE = r"D:\Stingray Robotics\velocity_outputE.csv"
POS_FILE = r"D:\Stingray Robotics\position_outputE.csv"

# Each entry is one VO variant to check. Must match the "_<method>" suffix
# used in velocity_outputE.csv / position_outputE.csv column names.
METHODS = ["sp_ma", "sp_lp", "sift_ma", "sift_lp"]

EARTH_RADIUS = 6_371_000  # meters



# Step 1: Load GPS logs

lat_df = pd.read_csv(LAT_FILE)
lon_df = pd.read_csv(LON_FILE)
alt_df = pd.read_csv(ALT_FILE)

gps_time = lat_df["elapsed time"].values
gps_lat = lat_df["value"].values
gps_lon = lon_df["value"].values
gps_alt = alt_df["value"].values


# Step 2: Convert GPS lat/lon to local X (East) / Z (North) meters
# Uses a simple flat-earth (equirectangular) approximation, referenced to
# the first GPS point. Good enough for short trajectories like this.

lat0, lon0 = gps_lat[0], gps_lon[0]

d_lat = np.radians(gps_lat - lat0)
d_lon = np.radians(gps_lon - lon0)

gps_x = d_lon * EARTH_RADIUS * np.cos(np.radians(lat0))  # East  -> X
gps_z = d_lat * EARTH_RADIUS                              # North -> Z

# GPS velocity, via simple finite differences between consecutive fixes
dt = np.diff(gps_time)
gps_vx = np.diff(gps_x) / dt
gps_vz = np.diff(gps_z) / dt
gps_vy = np.diff(gps_alt) / dt  # vertical speed, from altitude
gps_vel_time = gps_time[1:]     # velocity[i] belongs to the interval ending here



# Step 3: Load VO output

vel_df = pd.read_csv(VEL_FILE)
pos_df = pd.read_csv(POS_FILE)

n_frames = len(pos_df)

# We don't have a per-frame timestamp, so assume the VO run spans the same
# total duration as the GPS log, sampled at a constant frame rate.
frame_time = np.linspace(0, gps_time[-1], n_frames)



# Step 4: Interpolate GPS ground truth onto the VO frame timeline

gt_x = np.interp(frame_time, gps_time, gps_x)
gt_z = np.interp(frame_time, gps_time, gps_z)
gt_vx = np.interp(frame_time, gps_vel_time, gps_vx)
gt_vz = np.interp(frame_time, gps_vel_time, gps_vz)
gt_vy = np.interp(frame_time, gps_vel_time, gps_vy)



# Step 5 & 6: MAE + plots, for every method

def mae(a, b):
    return np.mean(np.abs(a - b))


frames = np.arange(n_frames)

for m in METHODS:
    vx = vel_df[f"vx_{m}"].values
    vy = vel_df[f"vy_{m}"].values
    vz = vel_df[f"vz_{m}"].values
    x = pos_df[f"x_{m}"].values
    z = pos_df[f"z_{m}"].values

    print(f"\n--- {m} ---")
    print(f"MAE Vx : {mae(vx, gt_vx):.4f} m/s")
    print(f"MAE Vy : {mae(vy, gt_vy):.4f} m/s")
    print(f"MAE Vz : {mae(vz, gt_vz):.4f} m/s")
    print(f"MAE X  : {mae(x, gt_x):.4f} m")
    print(f"MAE Z  : {mae(z, gt_z):.4f} m")

    # Trajectory plot (top-down, X vs Z) 
    plt.figure(figsize=(7, 7))
    plt.plot(x, z, color="green", label=m)
    plt.plot(gt_x, gt_z, color="black", label="GPS Ground Truth")
    plt.xlabel("X (m)")
    plt.ylabel("Z (m)")
    plt.title(f"{m} - Top Down")
    plt.legend()
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(f"trajectory_{m}.png", dpi=150)
    plt.close()

    # Velocity plot (Vx, Vy, Vz vs frame) 
    fig, axs = plt.subplots(3, 1, figsize=(14, 7), sharex=True)

    axs[0].plot(frames, vx, color="blue", label=m)
    axs[0].plot(frames, gt_vx, color="black", label="GPS Ground Truth")
    axs[0].set_ylabel("Vx (m/s)")
    axs[0].legend()

    axs[1].plot(frames, vy, color="blue", label=m)
    axs[1].plot(frames, gt_vy, color="black", label="GPS Ground Truth")
    axs[1].set_ylabel("Vy (m/s)")
    axs[1].legend()

    axs[2].plot(frames, vz, color="blue", label=m)
    axs[2].plot(frames, gt_vz, color="black", label="GPS Ground Truth")
    axs[2].set_ylabel("Vz (m/s)")
    axs[2].set_xlabel("Frame")
    axs[2].legend()

    fig.suptitle(m)
    plt.tight_layout()
    plt.savefig(f"velocity_{m}.png", dpi=150)
    plt.close()

print("\nDone. Plots saved as trajectory_<method>.png and velocity_<method>.png")