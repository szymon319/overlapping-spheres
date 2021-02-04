from shapely.geometry import LineString, Point, Polygon

import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import random


def randompoint_on(poly):
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

    return Point([x, y])


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
    points = [randompoint_on(poly) for i in range(n)]
    return points


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
        if pointpt == pt:
            continue
        # deltaX = point.x - pt.x
        # deltaY = point.y - pt.y
        deltaX = pointpt[0] - pt[0]
        deltaY = pointpt[1] - pt[1]

        angleInDegrees = math.atan2(deltaY, deltaX) * 180 / math.pi
        # print(angleInDegrees)

        # distance = pt.distance(point)
        distance = math.sqrt(((pointpt[0] - pt[0]) ** 2) + ((pointpt[1] - pt[1]) ** 2))

        if equation == "inverse":
            force = 1 / distance
        elif equation == "inverse square":
            force = ((1 / distance) ** 2) - (1 / distance)
        else:
            raise ValueError

        forceX = - force * math.cos(math.radians(angleInDegrees))
        forceY = - force * math.sin(math.radians(angleInDegrees))

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
    newstate = set()

    for pointpt in board:
        forces_shifted = forces_total(pointpt, board)
        # print(forces_shifted)
        newstate.add((pointpt[0] + timestamp * forces_shifted[0], pointpt[1] + timestamp * forces_shifted[1]))
        # print(newstate)

    return newstate


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
    pointsg = randomly_scatter(noofpoints, unitsquare)
    # pointsm = randomly_scatter(noofpoints, unitsquare)

    xsg = [pointpt.x for pointpt in pointsg]
    # xsm = [pointpt.x for pointpt in pointsm]
    ysg = [pointpt.y for pointpt in pointsg]
    # ysm = [pointpt.y for pointpt in pointsm]

    tuplesg = list(zip(xsg, ysg))
    # tuplesm = list(zip(xsm, ysm))

    # shiftedg = set(tuplesg)

    return tuplesg

# shiftedg = set(shift(100))

# fig, ax = plt.subplots()

# xg, yg = zip(*shiftedg)
# mat, = ax.plot(xg, yg, color='green', marker='o')

# ax.axis([-5, 5, -5, 5])

# myAnimation = animation.FuncAnimation(fig, animate, interval=50, blit=False, repeat=True)
# plt.draw()
# plt.show()
