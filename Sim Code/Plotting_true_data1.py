import pandas as pd
import matplotlib.pyplot as plt

# Load Excel files
xx = pd.read_csv(r"D:\Stingray Robotics\Visual Odometry Complete Pipeline\KITTI Odometry Full Pipeline\Sim Code\plot_data_pos_x.csv")
yy = pd.read_csv(r"D:\Stingray Robotics\Visual Odometry Complete Pipeline\KITTI Odometry Full Pipeline\Sim Code\plot_data_pos_y.csv")
zz = pd.read_csv(r"D:\Stingray Robotics\Visual Odometry Complete Pipeline\KITTI Odometry Full Pipeline\Sim Code\plot_data_pos_z.csv")

# Compute velocity = delta_position / delta_time
def velocity(df):
    dt = df["elapsed time"].diff()
    dp = df["value"].diff()
    return df["elapsed time"], dp / dt

tx, vx = velocity(xx)
ty, vy = velocity(yy)
tz, vz = velocity(zz)

# Plot
fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
fig.suptitle("Velocity from Position Data")

axes[0].plot(tx, vx, color="green")
axes[0].set_ylabel("Vx (m/s)")
axes[0].grid(True)

axes[1].plot(ty, vy, color="green")
axes[1].set_ylabel("Vy (m/s)")
axes[1].grid(True)

axes[2].plot(tz, vz, color="green")
axes[2].set_ylabel("Vz (m/s)")
axes[2].set_xlabel("Time (s)")
axes[2].grid(True)

plt.tight_layout()
plt.savefig("velocity_plot.png", dpi=150)
plt.show()