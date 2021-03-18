from lloyd import Field
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

np.random.seed(42)
random.seed(42)


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

def forces_total(pt, ptspts, old, counter, equation="inverse"):
    """
    Calculates the total force on pt as a result of all other points ptspts
    Parameters
    ----------
    pt the current cell
    ptspts the other cells
    old ??? no idea
    counter ??? no idea
    equation which force function to use

    Returns
    -------
    the total force acing on the cell pt
    """
    if old == "news":
        pts = cutoff(pt, ptspts, 4)
    elif old == "old":
        pts = ptspts
    else:
        raise ValueError

    total_force = np.zeros(2)

    for pointpt in pts:

        this_pos = np.array([pt[0], pt[1]])
        other_pos = np.array([pointpt[0], pointpt[1]])

        this_cell_type = pt[2]
        other_cell_type = pointpt[2]

        if np.allclose(this_pos, other_pos):
            continue

        this_to_other = other_pos - this_pos
        distance = np.linalg.norm(this_to_other)
        deviation = distance - 2.0
        unit_this_to_other = this_to_other / distance

        if equation == "inverse":
            force = 1 / distance
            force2 = 1 / distance
        # elif equation == "inverse square":
        #     # force = + ((1 / (distance + threshold5)) ** 2) - (1 / (distance + threshold5))
        #     force_base = + ((1 / (distance + 2 * threshold5)) ** 2) - (1 / (distance + threshold5))
        #     if distance < 1.707:
        #         force = 1 / distance
        #     else:
        #         force = force_base

        #     if distance > 2:
        #         force2 = force_base * 1 / 10
        #     elif distance < 1.707:
        #         force2 = 10 / distance
        #     else:
        #         force2 = force_base * 10
        elif equation == "paper":

            # Calculate the spring constant, which is reduced in the case of heterogeneous
            # interactions between cells further apart than their rest length
            spring_const = 25
            if distance > 2 and this_cell_type != other_cell_type:
                spring_const *= 0.1
            decay_const = 5.0

            # Calculate the force based on the equation on Page 7 of Osborne et al
            if distance < 2:
                force = spring_const * 2.0 * np.log1p(0.5 * deviation) * unit_this_to_other
            else:
                force = spring_const * deviation * np.exp(-0.5 * decay_const * deviation) * unit_this_to_other

            total_force += force
            # print(f"this: {this_pos}, other: {other_pos}, dist: {distance}, force: {force}")

        # elif equation == "Overlapping spheres":
        #     if distance > threshold2:
        #         force = 0
        #     elif distance > threshold20:
        #         force = - ((1 / distance) ** (1 / 2))
        #     else:
        #         force = 1 / distance
        # elif equation == "Lennard-Jones":
        #     if distance > threshold2:
        #         force = 0
        #     else:
        #         force = ((epsilon * 1 / distance) ** 12) - ((epsilon * 1 / distance) ** 6)
        else:
            raise ValueError

    # Add on the random force
    normal_mean = 0
    normal_std = 1
    normal_strength = 5.0
    total_force += normal_strength * np.random.normal(normal_mean, normal_std, 2)

    return total_force


def advance(board, timestamp, old, my_counter):
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
        forces_shifted = forces_total(pointpt, board, old, my_counter, "paper")
        # print(forces_shifted)
        x_check = pointpt[0] + timestamp * forces_shifted[0]
        y_check = pointpt[1] + timestamp * forces_shifted[1]
        if x_check > 30:
            x_coord = 30
        elif x_check < 0:
            x_coord = 0
        else:
            x_coord = x_check
        if y_check > 30:
            y_coord = 30
        elif y_check < 0:
            y_coord = 0
        else:
            y_coord = y_check
        newstate.append([x_coord, y_coord, pointpt[2]])
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
    pointsg = np.array(randomly_scatter(noofpoints, unitsquare))
    # print(pointsg)
    # pointsm = randomly_scatter(noofpoints, unitsquare)

    # xsg = []
    # xsm = []
    # ysg = []
    # ysm = []

    # for pointpt in pointsg:
    #     if pointpt[2] == 0:
    #         xsg.append(pointpt[0])
    #         ysg.append(pointpt[1])
    #     else:
    #         xsm.append(pointpt[0])
    #         ysm.append(pointpt[1])

    # coordsg = np.array(list(zip(xsg, ysg))).reshape(len(xsg), -1)

    coordsg = np.delete(pointsg, np.s_[2:3], axis=1)
    # print(coordsg)
    # field = Field(coordsg)
    # for i in range(int(3e4)):
    #     field.relax()
    #     new_positions = field.get_points()

    # outputg = np.c_[new_positions, pointsg[:, 2]]
    outputg = np.c_[coordsg, pointsg[:, 2]]

    # xsg = [pointpt[0] for pointpt in pointsg]
    # xsm = [pointpt.x for pointpt in pointsm]
    # ysg = [pointpt[1] for pointpt in pointsg]
    # ysm = [pointpt.y for pointpt in pointsm]

    # tuplesg = list(zip(xsg, ysg))
    # tuplesm = list(zip(xsm, ysm))

    # shiftedg = set(tuplesg)

    return outputg


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


def metricpoints(board):
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
    mpoints = np.delete(board, np.s_[2:3], axis=1)
    vor = Voronoi(mpoints)

    mmetric = 0
    for vpair in vor.ridge_vertices:
        if vpair[0] >= 0 and vpair[1] >= 0:
            v0 = vor.vertices[vpair[0]]
            v1 = vor.vertices[vpair[1]]

            len_vpair = math.sqrt(sq_distance(v0, v1))
            center_vpair = np.array([0.5 * (v1[0] + v0[0]), 0.5 * (v1[1] + v0[1])])
            slope_vpair = slope(v0[0], v0[1], center_vpair[0], center_vpair[1])

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

            l_input = [
                [center_vpair1, center_vpair2],
                mpoints.tolist()
            ]

            output = list(map(list, zip(l_input[0], map(lambda pt: get_min_point(pt, l_input[1]), l_input[0]))))
            if board[np.logical_and(output[0][1][0] == board[:, 0], output[0][1][1] == board[:, 1]), 2] == board[np.logical_and(output[1][1][0] == board[:, 0], output[1][1][1] == board[:, 1]), 2]:
                mmetric += len_vpair
    return mmetric


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

# fig = voronoi_plot_2d(voronoipts(shift(20)))
# plt.show()

# print()
