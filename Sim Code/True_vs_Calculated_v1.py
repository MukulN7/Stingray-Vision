import pandas as pd
import matplotlib.pyplot as plt

# Load true position data and compute true velocity
def velocity(df):
    dt = df["elapsed time"].diff()
    dp = df["value"].diff()
    return df["elapsed time"], dp / dt

xx = pd.read_csv(r"D:\Stingray Robotics\Visual Odometry Complete Pipeline\KITTI Odometry Full Pipeline\Sim Code\plot_data_xx_2.csv")
yy = pd.read_csv(r"D:\Stingray Robotics\Visual Odometry Complete Pipeline\KITTI Odometry Full Pipeline\Sim Code\plot_data_yy_2.csv")
zz = pd.read_csv(r"D:\Stingray Robotics\Visual Odometry Complete Pipeline\KITTI Odometry Full Pipeline\Sim Code\plot_data_zz_2.csv")

t_true_x, v_true_x = velocity(xx)
t_true_y, v_true_y = velocity(yy)
t_true_z, v_true_z = velocity(zz)

# Load calculated velocity data
df = pd.read_csv(r"D:\Stingray Robotics\Visual Odometry Complete Pipeline\KITTI Odometry Full Pipeline\Sim Code\velocities_v9.txt")
time = df["timestamp"] - df["timestamp"].iloc[0]

# Define the 4 combinations: (title, vx column, vy column, vz column)
combinations = [
    ("SP + Moving Average",   "vx_sp_ma",   "vy_sp_ma",   "vz_sp_ma"),
    ("SP + Low Pass Filter",  "vx_sp_lp",   "vy_sp_lp",   "vz_sp_lp"),
    ("SIFT + Moving Average", "vx_sift_ma", "vy_sift_ma", "vz_sift_ma"),
    ("SIFT + Low Pass Filter","vx_sift_lp", "vy_sift_lp", "vz_sift_lp"),
]

for title, col_vx, col_vy, col_vz in combinations:

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8))
    fig.suptitle(title, fontsize=14)

    ax1.plot(time, df[col_vx], color="blue", label="Calculated")
    ax1.plot(t_true_x, v_true_x, color="red", label="True")
    ax1.set_ylabel("Vx (m/s)")
    ax1.grid(True)
    ax1.legend()

    ax2.plot(time, df[col_vy], color="blue", label="Calculated")
    ax2.plot(t_true_y, v_true_y, color="red", label="True")
    ax2.set_ylabel("Vy (m/s)")
    ax2.grid(True)
    ax2.legend()

    ax3.plot(time, df[col_vz], color="blue", label="Calculated")
    ax3.plot(t_true_z, v_true_z, color="red", label="True")
    ax3.set_ylabel("Vz (m/s)")
    ax3.set_xlabel("Time (s)")
    ax3.grid(True)
    ax3.legend()

    plt.tight_layout()

    # Save each graph as a separate file
    filename = title.replace(" ", "_").replace("+", "").replace("__", "_") + "_vs_true.png"
    plt.savefig(f"D:\\Stingray Robotics\\Visual Odometry Complete Pipeline\\KITTI Odometry Full Pipeline\\Sim Code\\SimOutput\\{filename}", dpi=150)
    print(f"Saved: {filename}")

plt.show()