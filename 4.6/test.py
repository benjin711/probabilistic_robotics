import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Generate some random data
x = np.random.normal(0, 1, 1000)
y = np.random.normal(0, 1, 1000)

# Create a 2D histogram
hist, xedges, yedges = np.histogram2d(x, y, bins=(50, 50))

# Construct arrays for the anchor positions of the bars.
xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25, indexing="ij")
xpos = xpos.ravel()
ypos = ypos.ravel()
zpos = 0

# Construct arrays with the dimensions for the bars.
dx = dy = 0.5 * np.ones_like(zpos)
dz = hist.ravel()

# Create a colormap
cmap = plt.cm.get_cmap('jet')  # 'jet' is the name of the colormap
max_height = np.max(dz)   # get the maximum bar height
min_height = np.min(dz)
# scale each dz value to range between 0 and 1
rgba = [cmap((k-min_height)/max_height) for k in dz] 

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=rgba, zsort='average')

plt.show()