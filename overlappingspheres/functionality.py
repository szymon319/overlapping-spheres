from shapely.geometry import LineString, Point, Polygon

import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
# import numpy as np
import random

timestamp = 0.0001
unitsquare = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
print(type(unitsquare))


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


def forces_total(pt, pts):
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

        # force = 1 / distance
        force = ((1 / distance) ** 2) - (1 / distance)

        forceX = - force * math.cos(math.radians(angleInDegrees))
        forceY = - force * math.sin(math.radians(angleInDegrees))

        sum[0] += forceX
        sum[1] += forceY

    return sum


def advance(board):
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


points = randomly_scatter(100, unitsquare)
x, y = unitsquare.exterior.xy

xs = [pointpt.x for pointpt in points]
ys = [pointpt.y for pointpt in points]
tuples = list(zip(xs, ys))

shifted = set(tuples)

fig, ax = plt.subplots()
x, y = zip(*shifted)
mat, = ax.plot(x, y, 'o')


def animate(i):
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
    global shifted
    shifted = advance(shifted)
    # print(shifted)
    x, y = zip(*shifted)
    mat.set_data(x, y)
    return mat,


ax.axis([-15, 15, -15, 15])
# plt.plot(x, y, "r")

myAnimation = animation.FuncAnimation(fig, animate, interval=50, blit=False, repeat=True)
plt.draw()
plt.show()
