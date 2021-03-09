import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import time

from lloyd import Field
from scipy.spatial import Voronoi, voronoi_plot_2d

np.random.seed(42)

new_positions = np.random.rand(100, 2)
field = Field(new_positions)

# print(points)
# plt = voronoi_plot_2d(field.voronoi, show_vertices=False, line_colors='orange', line_width=2, line_alpha=0.6, point_size=2)
# plt.show()

# fig = voronoi_plot_2d(Voronoi(points))
# plt.show()

# for i in range(6): 
#     field.relax()
#     new_positions = field.get_points()

#     # print(new_positions)
#     print(new_positions[:, 0])
#     # plt = voronoi_plot_2d(field.voronoi, show_vertices=False, line_colors='orange', line_width=2, line_alpha=0.6, point_size=2)
#     # plt.show()

#     fig = voronoi_plot_2d(Voronoi(points))
#     plt.show()

fig, ax = plt.subplots()
mat1, = ax.plot(new_positions[:, 0], new_positions[:, 1], c='g', linestyle='None', marker='o')

def init():
    mat1.set_data([], [])
    return mat1,

def update(i):
    field.relax()
    new_positions = field.get_points()

    mat1.set_xdata(new_positions[:, 0])
    mat1.set_ydata(new_positions[:, 1])
    return mat1,

myAnimation = animation.FuncAnimation(fig, update, init_func=init, interval=50, blit=True, repeat=True)
plt.draw()

plt.show()
