import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv(r"D:\Stingray Robotics\Visual Odometry Complete Pipeline\KITTI Odometry Full Pipeline\Sim Code\SimOutput\velocities_v7.txt")

# Time axis starting from 0
time = df["timestamp"] - df["timestamp"].iloc[0]

# Define the 4 combinations: (title, vx column, vy column, vz column)
combinations = [
    ("SP + Moving Average",  "vx_sp_ma",   "vy_sp_ma",   "vz_sp_ma"),
    ("SP + Low Pass Filter", "vx_sp_lp",   "vy_sp_lp",   "vz_sp_lp"),
    ("SIFT + Moving Average","vx_sift_ma",  "vy_sift_ma",  "vz_sift_ma"),
    ("SIFT + Low Pass Filter","vx_sift_lp", "vy_sift_lp",  "vz_sift_lp"),
]

for title, col_vx, col_vy, col_vz in combinations:

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8))
    fig.suptitle(title, fontsize=14)

    ax1.plot(time, df[col_vx], color="green")
    ax1.set_ylabel("Vx (m/s)")
    ax1.grid(True)

    ax2.plot(time, df[col_vy], color="green")
    ax2.set_ylabel("Vy (m/s)")
    ax2.grid(True)

    ax3.plot(time, df[col_vz], color="green")
    ax3.set_ylabel("Vz (m/s)")
    ax3.set_xlabel("Time (s)")
    ax3.grid(True)

    plt.tight_layout()

    # Save each graph as a separate file
    filename = title.replace(" ", "_").replace("+", "").replace("__", "_") + ".png"
    plt.savefig(f"D:\\Stingray Robotics\\Visual Odometry Complete Pipeline\\KITTI Odometry Full Pipeline\\Sim Code\\SimOutput\\{filename}", dpi=150)
    plt.close()
    print(f"Saved: {filename}")
