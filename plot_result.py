import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

poses_file = 'poses_1'
timestamps_file = 'timestamps_1'

poses_data = None
timestamps_data = None

try:
    poses_data = pd.read_csv(poses_file, header=None)
    timestamps_data = pd.read_csv(timestamps_file, header=None)
except Exception as e:
    print(f"Error reading files: {e}")
    exit()

positions = poses_data.iloc[:, :3].values
timestamps = timestamps_data[0].values

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')
line, = ax.plot([], [], [], '-o', label='Trajectory', color='blue', markersize=3)

# Setting the labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Video Odometry')

# Setting the view limits
range_padding = 2.0
x_min, x_max = positions[:, 0].min() - range_padding, positions[:, 0].max() + range_padding
y_min, y_max = positions[:, 1].min() - range_padding, positions[:, 1].max() + range_padding
z_min, z_max = positions[:, 2].min() - range_padding, positions[:, 2].max() + range_padding
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.set_zlim(z_min, z_max)

# Animation function
def update(num, positions, line):
    line.set_data(positions[:num, 0:2].T)  # X and Y
    line.set_3d_properties(positions[:num, 2])  # Z
    return line,

ani = FuncAnimation(fig, update, frames=len(positions), fargs=(positions, line),
                    interval=100, blit=False)

plt.show()
