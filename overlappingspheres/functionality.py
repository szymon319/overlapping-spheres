from scipy import spatial
from scipy.spatial.distance import cdist
from scipy.spatial import Voronoi, voronoi_plot_2d
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
        # pts = cutoff(pt, ptspts, 0.05)
        pts = cutoff(pt, ptspts, 5)
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
        # moo = 
        threshold2 = 1 / 2
        threshold5 = 1 / 5

        if equation == "inverse":
            force = 1 / distance
        elif equation == "inverse square":
            # force = + ((1 / (distance + threshold5)) ** 2) - (1 / (distance + threshold5))
            force_base = + ((1 / (distance + 2 * threshold5)) ** 2) - (1 / (distance + threshold5))
            if distance < 1.707:
                force = 1 / distance
            else:
                force = force_base

            if distance > 2:
                force2 = force_base * 1 / 10
            elif distance < 1.707:
                force2 = 10 / distance
            else:
                force2 = force_base * 10
        elif equation == "paper":
            if distance < 2:
                force_base = - 2 * math.log(1 + (distance - 2) / 2)
            else:
                force_base = - (distance - 2) * math.exp(- 5 * (distance - 2) / 2)
            force = 5 * force_base
            force2 = 50 * force_base
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

        mu = 0
        sigma = threshold5
        # print(pointpt)
        if pointpt[2] == pt[2]:
            forceX = - force * math.cos(math.radians(angleInDegrees)) + np.random.normal(mu, sigma)
            forceY = - force * math.sin(math.radians(angleInDegrees)) + np.random.normal(mu, sigma)
        else:
            forceX = - force2 * math.cos(math.radians(angleInDegrees)) + np.random.normal(mu, sigma)
            forceY = - force2 * math.sin(math.radians(angleInDegrees)) + np.random.normal(mu, sigma)

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
        forces_shifted = forces_total(pointpt, board, old, "paper")
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
    unitsquare = Polygon([(0, 0), (30, 0), (30, 30), (0, 30)])
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
    coordsg = np.array(list(zip(xsg, ysg))).reshape(len(xsg), -1)
    coordsm = np.array(list(zip(xsm, ysm))).reshape(len(xsm), -1)
    # print(coordsg)
    # print(coordsm)

    return (np.sum(cdist(coordsg, coordsg, 'euclidean')) + np.sum(cdist(coordsm, coordsm, 'euclidean')))


def voronoipts(ptts):
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
    # Calculate Voronoi Regions
    xsg = []
    xsm = []
    ysg = []
    ysm = []

    for pointpt in ptts:
        if pointpt[2] == 0:
            xsg.append(pointpt[0])
            ysg.append(pointpt[1])
        else:
            xsm.append(pointpt[0])
            ysm.append(pointpt[1])

    coordsg = np.array(list(zip(xsg, ysg))).reshape(len(xsg), -1)
    # coordsm = np.array(list(zip(xsm, ysm))).reshape(len(xsm), -1)

    # poly_shapes, pts, poly_to_pt_assignments = voronoi_regions_from_coords(coords, boundary_shape)
    vor = Voronoi(coordsg)

    return vor


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

# test = np.array([[0, 0, 1], [5, 0, 1], [5, 0, 0], [2, 2, 1]])
# print(fractional(test))

fig = voronoi_plot_2d(voronoipts(shift(100)))
plt.show()
