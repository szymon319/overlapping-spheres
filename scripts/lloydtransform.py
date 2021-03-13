import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
# import time
# import umap, os

from lloyd import Field
from scipy.spatial import Voronoi, voronoi_plot_2d

np.random.seed(42)

new_positions = np.random.rand(100, 2)
print(new_positions)

field = Field(new_positions)

# print(points)
# plt = voronoi_plot_2d(field.voronoi, show_vertices=False, line_colors='orange', line_width=2, line_alpha=0.6, point_size=2)
# plt.show()

# fig = voronoi_plot_2d(Voronoi(points))
# plt.show()

# for i in range(6): 
#     field.relax()
#     new_positions = field.get_points()

#     print(new_positions)
#     print(new_positions[:, 0])
#     # plt = voronoi_plot_2d(field.voronoi, show_vertices=False, line_colors='orange', line_width=2, line_alpha=0.6, point_size=2)
#     # plt.show()

#     fig = voronoi_plot_2d(Voronoi(points))
#     plt.show()

fig, ax = plt.subplots()
mat1, = ax.plot(new_positions[:, 0], new_positions[:, 1], c='g', linestyle='None', marker='o')

# def plot(vor, name, e=0.3):
#     plot = voronoi_plot_2d(vor, show_vertices=False, line_colors='y', line_alpha=0.5, point_size=5)
#     plot.set_figheight(14)
#     plot.set_figwidth(20)
#     plt.axis([field.bb_points[0]-e, field.bb_points[1]+e, field.bb_points[2]-e, field.bb_points[3]+e])
#     if not os.path.exists('plots'): os.makedirs('plots')
#     if len(str(name)) < 2: name = '0' + str(name)
#     plot.savefig( 'plots/' + str(name) + '.png' )

def init():
    mat1.set_data([], [])
    return mat1,


def update(i):
    field.relax()
    new_positions = field.get_points()

    mat1.set_xdata(new_positions[:, 0])
    mat1.set_ydata(new_positions[:, 1])
    return mat1,

myAnimation = animation.FuncAnimation(fig, update, init_func=init, interval=500, blit=True, repeat=True)
plt.draw()

plt.show()

# X = np.random.rand(1000, 4)
# X = umap.UMAP().fit_transform(X)

# field = Field(X)
# for i in range(20):
#     print(' * running iteration', i)
#     plot(field.voronoi, i)
#     field.relax()
