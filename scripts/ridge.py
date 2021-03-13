from overlappingspheres.functionality import randomly_scatter
from scipy import stats
from scipy.spatial import Voronoi, voronoi_plot_2d
from shapely.geometry import LineString, Point, Polygon

import math
import matplotlib.pyplot as plt
import numpy as np

unitsquare = Polygon([(0, 0), (30, 0), (30, 30), (0, 30)])
mrandom = np.array(randomly_scatter(50, unitsquare))

# print(mrandom[3, 0])
# print(np.isclose(mrandom[3, 0], 26.76538703))
mrandom[np.logical_and(np.isclose(26.76538703, mrandom[:, 0]), np.isclose(2.60816498, mrandom[:, 1])), 2] += 12
print(mrandom)

mpoints = np.delete(mrandom, np.s_[2:3], axis=1)
# mpoints = np.array([[0, 0], [0, 1], [0, 2],
#                    [1, 0], [1, 1], [1, 2],
#                    [2, 0], [2, 1], [2, 2]])
vor = Voronoi(mpoints)

print(vor.ridge_vertices)

# fig = plt.figure()
# plt.plot(vor.vertices[:, 0], vor.vertices[:, 1], 'ko', ms=8)

# for vpair in vor.ridge_vertices:
#     if vpair[0] >= 0 and vpair[1] >= 0:
#         v0 = vor.vertices[vpair[0]]
#         print(v0)
#         v1 = vor.vertices[vpair[1]]
#         print(v1)
#         plt.plot([v0[0], v1[0]], [v0[1], v1[1]], 'k', linewidth=2)

# plt.show()


def slope(x1, y1, x2, y2):
    if x1 == x2:
        return "vertical"
    elif y1 == y2:
        return "horizontal"
    else:
        m = (y2 - y1) / (x2 - x1)
        return m


def sq_distance(x1, x2):
    return sum(map(lambda x: (x[0] - x[1])**2, zip(x1, x2)))


def get_min_point(point, points):
    dists = list(map(lambda x: sq_distance(x, point), points))
    return points[dists.index(min(dists))]


mmetric = 0

for vpair in vor.ridge_vertices:
    if vpair[0] >= 0 and vpair[1] >= 0:
        v0 = vor.vertices[vpair[0]]
        v1 = vor.vertices[vpair[1]]
        # print(type(v0))
        print(v0, v1)

        len_vpair = math.hypot(v1[0] - v0[0], v1[1] - v0[1])
        center_vpair = np.array([0.5 * (v1[0] + v0[0]), 0.5 * (v1[1] + v0[1])])
        print(center_vpair)

        slope_vpair = slope(v0[0], v0[1], center_vpair[0], center_vpair[1])
        # print(type(slope_vpair))
        print(slope_vpair)

        if slope_vpair == "vertical":
            dy = 0
            dx = 1e-5
        elif slope_vpair == "horizontal":
            dy = 1e-5
            dx = 0
        else:
            dy = math.sqrt((1e-5)**2 / (float(slope_vpair)**2 + 1))
            dx = - float(slope_vpair) * dy

        center_vpair1 = []
        center_vpair1.append(center_vpair[0] + dx)
        center_vpair1.append(center_vpair[1] + dy)

        center_vpair2 = []
        center_vpair2.append(center_vpair[0] - dx)
        center_vpair2.append(center_vpair[1] - dy)

        # print(center_vpair2)
        # print(mpoints.tolist())

        l_input = [[center_vpair1, center_vpair2],
                    mpoints.tolist()]

        output = list(map(list, zip(l_input[0], map(lambda pt: get_min_point(pt, l_input[1]), l_input[0]))))
        print(output)
        print(output[0][1])

        if mrandom[np.logical_and(output[0][1][0] == mrandom[:, 0], output[0][1][1] == mrandom[:, 1]), 2] == \
    mrandom[np.logical_and(output[1][1][0] == mrandom[:, 0], output[1][1][1] == mrandom[:, 1]), 2]:
            mmetric += len_vpair

print(mmetric)

fig = voronoi_plot_2d(vor)
plt.show()
