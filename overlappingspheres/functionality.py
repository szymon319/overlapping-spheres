from scipy import spatial
from scipy.spatial.distance import cdist
from shapely.geometry import LineString, Point, Polygon

import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import random


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

def forces_total(pt, ptspts, old, equation="inverse"):
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

    if old == "news":
        pts = cutoff(pt, ptspts, 0.05)
    elif old == "old":
        pts = ptspts
    else:
        raise ValueError

    # for pointpt in pts:
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


def advance(board, timestamp, old):
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
        forces_shifted = forces_total(pointpt, board, old, "inverse square")
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
    unitsquare = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    # unitsquare = Polygon([(0, 0), (5, 0), (5, 5), (0, 5)])
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


def cutoff(ptt, ptts, cellsize):
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
    b = ptts[:, :-1]

    tree = spatial.KDTree(b)
    nearby_points = []
    for results in tree.query_ball_point(([ptt[0], ptt[1]]), cellsize):
        # print(results)
        my_list = list(b[results])
        my_listappend = my_list + [ptts[:, -1][results]]
        # print(my_listappend)
        nearby_points.append(my_listappend)
        # print(nearby_points)

    return np.array(nearby_points)


def fractional(board):
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
    # length = []
    xsg = []
    xsm = []
    ysg = []
    ysm = []

    for pointpt in board:
        if pointpt[2] == 0:
            xsg.append(pointpt[0])
            ysg.append(pointpt[1])
        else:
            xsm.append(pointpt[0])
            ysm.append(pointpt[1])

    return cdist(xsg, ysg, 'euclidean') + cdist(xsm, ysm, 'euclidean')


# shiftedg = set(shift(100))

# fig, ax = plt.subplots()

# xg, yg = zip(*shiftedg)
# mat, = ax.plot(xg, yg, color='green', marker='o')

# ax.axis([-5, 5, -5, 5])

# myAnimation = animation.FuncAnimation(fig, animate, interval=50, blit=False, repeat=True)
# plt.draw()
# plt.show()

# pt = [0, 0, 0]
# pts = np.array([[1, 1, 1], [0.02, 0, 1]])
# print(pts[:, :-1])
# y = cutoff(pt, pts, 0.05)
# print(y)

test = np.array([[0, 0, 1], [5, 0, 1]])
print(fractional(test))
