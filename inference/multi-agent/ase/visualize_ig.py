import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

points = np.array([
    [0, 0, 0],  # point 0
    [1, 0, 0],  # point 1
    [1, 1, 0],  # point 2
    [0, 1, 1],  # point 3
])

# 每个连接是一对索引 (start_idx, end_idx)
connectivity = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制点
x, y, z = points[:, 0], points[:, 1], points[:, 2]
ax.scatter(x, y, z, c='r', s=50)

# 绘制连接线
for i, j in connectivity:
    xs, ys, zs = [points[i, 0], points[j, 0]], [points[i, 1], points[j, 1]], [points[i, 2], points[j, 2]]
    ax.plot(xs, ys, zs, 'b')

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.show()
