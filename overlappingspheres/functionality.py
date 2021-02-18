from scipy.spatial import KDTree, Voronoi, voronoi_plot_2d
from shapely.geometry import LineString, Point, Polygon

import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import random
import sys


def randompoint_on(poly, celltype: int):
    """
    A function that takes a name and returns a greeting.

    Parameters
    ----------
    name : str, optional
        The name to greet (default is "")

    Returns
    -------
    str
        The greeting
    """
    min_x, min_y, max_x, max_y = poly.bounds

    x = random.uniform(min_x, max_x)
    x_line = LineString([(x, min_y), (x, max_y)])
    x_line_intercept_min, x_line_intercept_max = x_line.intersection(poly).xy[1].tolist()
    y = random.uniform(x_line_intercept_min, x_line_intercept_max)
    # celltype = random.randint(0, 1)

    # return Point([x, y])
    return np.array([x, y, celltype])


def randomly_scatter(n, poly):
    """
    A function that takes a name and returns a greeting.

    Parameters
    ----------
    name : str, optional
        The name to greet (default is "")

    Returns
    -------
    str
        The greeting
    """
    points = [randompoint_on(poly, 0) for i in range(int(n / 2))] + [randompoint_on(poly, 1) for i in range(int(n / 2), n)]
    return points


# unitsquare = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
# print(randomly_scatter(100, unitsquare))

def forces_total(pt, pts, equation="inverse"):
    """
    A function that takes a name and returns a greeting.

    Parameters
    ----------
    name : str, optional
        The name to greet (default is "")

    Returns
    -------
    str
        The greeting
    """
    sum = [0, 0]
    for pointpt in pts:
        if pointpt[0] == pt[0] and pointpt[1] == pt[1]:
            continue
        # deltaX = point.x - pt.x
        # deltaY = point.y - pt.y
        deltaX = pointpt[0] - pt[0]
        deltaY = pointpt[1] - pt[1]

        angleInDegrees = math.atan2(deltaY, deltaX) * 180 / math.pi
        # print(angleInDegrees)

        # distance = pt.distance(point)
        distance = math.sqrt(((pointpt[0] - pt[0]) ** 2) + ((pointpt[1] - pt[1]) ** 2))
        epsilon = 0.05
        threshold2 = 1 / 2
        threshold5 = 1 / 5

        if equation == "inverse":
            force = 1 / distance
        elif equation == "inverse square":
            force = + ((1 / (distance + threshold5)) ** 2) - (1 / (distance + threshold5))
        # elif equation == "Overlapping spheres":
        #     if distance > threshold2:
        #         force = 0
        #     elif distance > threshold20:
        #         force = - ((1 / distance) ** (1 / 2))
        #     else:
        #         force = 1 / distance
        elif equation == "Lennard-Jones":
            if distance > threshold2:
                force = 0
            else:
                force = ((epsilon * 1 / distance) ** 12) - ((epsilon * 1 / distance) ** 6)
        else:
            raise ValueError

        print(pointpt)
        if pointpt[2] == pt[2]:
            forceX = - force * math.cos(math.radians(angleInDegrees))
            forceY = - force * math.sin(math.radians(angleInDegrees))
        else:
            forceX = - force * math.cos(math.radians(angleInDegrees)) * 10
            forceY = - force * math.sin(math.radians(angleInDegrees)) * 10

        sum[0] += forceX
        sum[1] += forceY

    return sum


def advance(board, timestamp):
    """
    A function taking some arguments and returning the minimum number among the arguments.

    Parameters
    ----------
    args : int, float
        The numbers from which to return the minimum

    Returns
    -------
    int, float
        The minimum
    """
    # newstate = set()
    newstate = []

    for pointpt in board:
        forces_shifted = forces_total(pointpt, board, "inverse square")
        # print(forces_shifted)
        newstate.append([pointpt[0] + timestamp * forces_shifted[0], pointpt[1] + timestamp * forces_shifted[1], pointpt[2]])
        # print(newstate)

    return np.array(newstate)


def shift(noofpoints):
    """
    A function taking some arguments and returning the minimum number among the arguments.

    Parameters
    ----------
    args : int, float
        The numbers from which to return the minimum

    Returns
    -------
    int, float
        The minimum
    """
    unitsquare = Polygon([(0, 0), (5, 0), (5, 5), (0, 5)])
    pointsg = randomly_scatter(noofpoints, unitsquare)
    # pointsm = randomly_scatter(noofpoints, unitsquare)

    # xsg = [pointpt[0] for pointpt in pointsg]
    # xsm = [pointpt.x for pointpt in pointsm]
    # ysg = [pointpt[1] for pointpt in pointsg]
    # ysm = [pointpt.y for pointpt in pointsm]

    # tuplesg = list(zip(xsg, ysg))
    # tuplesm = list(zip(xsm, ysm))

    # shiftedg = set(tuplesg)

    return np.array(pointsg)


def build_voronoi(points):
    """
    Build a Voronoi map from self.points. For background on self.voronoi attrs, see:
    https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.spatial.Voronoi.html
    """
    x = points[:, 0]
    y = points[:, 0]
    bounding_box = [min(x), max(x), min(y), max(y)]

    eps = sys.float_info.epsilon
    voronoi = Voronoi(points)
    filtered_regions = [] # list of regions with vertices inside Voronoi map

    for region in voronoi.regions:
        inside_map = True    # is this region inside the Voronoi map?
        for index in region: # index = the idx of a vertex in the current region

            # check if index is inside Voronoi map (indices == -1 are outside map)
            if index == -1:
                inside_map = False
                break

            # check if the current coordinate is in the Voronoi map's bounding box
            else:
                coords = voronoi.vertices[index]
                if not (bounding_box[0] - eps <= coords[0] and
                        bounding_box[1] + eps >= coords[0] and
                        bounding_box[2] - eps <= coords[1] and
                        bounding_box[3] + eps >= coords[1]):
                    inside_map = False
                    break

        # store hte region if it has vertices and is inside Voronoi map
        if region != [] and inside_map:
            filtered_regions.append(region)


def find_centroid(vertices):
    """
    Find the centroid of a Voroni region described by `vertices`, and return a
    np array with the x and y coords of that centroid.
    The equation for the method used here to find the centroid of a 2D polygon
    is given here: https://en.wikipedia.org/wiki/Centroid#Of_a_polygon
    @params: np.array `vertices` a numpy array with shape n,2
    @returns np.array a numpy array that defines the x, y coords
      of the centroid described by `vertices`
    """
    area = 0
    centroid_x = 0
    centroid_y = 0

    for i in range(len(vertices) - 1):
        step = (vertices[i, 0] * vertices[i+1, 1]) - (vertices[i+1, 0] * vertices[i, 1])
        area += step
        centroid_x += (vertices[i, 0] + vertices[i+1, 0]) * step
        centroid_y += (vertices[i, 1] + vertices[i+1, 1]) * step
    area /= 2
    centroid_x = (1.0/(6.0*area)) * centroid_x
    centroid_y = (1.0/(6.0*area)) * centroid_y
    return np.array([[centroid_x, centroid_y]])


# shiftedg = set(shift(100))

# fig, ax = plt.subplots()

# xg, yg = zip(*shiftedg)
# mat, = ax.plot(xg, yg, color='green', marker='o')

# ax.axis([-5, 5, -5, 5])

# myAnimation = animation.FuncAnimation(fig, animate, interval=50, blit=False, repeat=True)
# plt.draw()
# plt.show()

points = np.array([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2],
                   [2, 0], [2, 1], [2, 2]])

vor = Voronoi(points)
fig = voronoi_plot_2d(vor)
plt.show()
