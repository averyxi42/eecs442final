import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

pcd = np.load('point cloud.npy')
ax = plt.subplot(projection='3d')
ax.scatter(pcd[0],pcd[1],pcd[2])
ax.set_aspect('equal')
plt.show()