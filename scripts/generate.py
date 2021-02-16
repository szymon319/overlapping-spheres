from shapely.geometry import LineString, Point, Polygon

import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import random

from overlappingspheres.functionality import randompoint_on
from overlappingspheres.functionality import randomly_scatter
from overlappingspheres.functionality import forces_total
from overlappingspheres.functionality import advance
from overlappingspheres.functionality import shift
# from overlappingspheres.functionality import animate
from overlappingspheres.functionality import cutoff
from overlappingspheres.functionality import fractional

pt = np.array([0.01, 0, 1])
# print(pt[0])
pts = np.array([[0, 0, 1], [5, 0, 1], [0.02, 0, 1]])
# print(pts)
# print(cutoff(pt, pts, 0.01))

# shiftedg = set(shift(100))
# shiftedm = set(shift(100))


def get_colour(t):
    if t == 0:
        return 'g'
    else:
        return 'm'


# main = 100
main = 1000
main2 = int(main / 2)

fig, ax = plt.subplots()
test = shift(main)
# test = np.array([[0, 0, 1], [5, 0, 1]])
# test = np.array([[2.4, 0, 1], [2.6, 0, 1]])

# print(fractional(test))
# print((shift(100)))
# print(type(shift(100)))
# xg, yg = zip(*shiftedg)
# xm, ym = zip(*shiftedm)

C = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]])
test_colors = test[:, 2]
# print(test_colors.shape)
# exit(0)

# test_colors.apply(lambda x: )
# print(test_colors)
# exit(0)

# mat, = ax.plot(test[:, 0], test[:, 1], c = C / 255.0, marker='o')
# mat, = ax.plot(test[:, 0], test[:, 1], color=[get_colour(i) for i in test[:, 2]], marker='o')
# mat, = ax.scatter(test[:, 0], test[:, 1], c=np.apply_along_axis(lambda x: [0,1.0,0], 1, test), marker='o')
mat1, = ax.plot(test[:main2, 0], test[:main2, 1], c='g', linestyle='None', marker='o')
mat2, = ax.plot(test[main2:, 0], test[main2:, 1], c='m', linestyle='None', marker='o')


def init():
    mat1.set_data([], [])
    mat2.set_data([], [])
    return mat1,


xsfractional = []


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
    # global shiftedg
    global test
    # print(test)

    # tuplesg = list(zip(test[:, 0], test[:, 1]))
    # shiftedg = set(tuplesg)
    # if not shiftedg:
    #     shiftedg = shift(100)

    # test = advance(test, 0.15)
    # test = advance(test, 0.015)
    test = advance(test, 0.0015, "news")
    # test = advance(test, 0.0015, "old")

    # shiftedm = advance(shiftedm, 0.0001)
    # print(shifted)

    # xsg = [pointpt[0] for pointpt in test]
    # ysg = [pointpt[1] for pointpt in test]
    # xm, ym = zip(*shiftedm)

    # fig, ax = plt.subplots()
    # mat, = ax.plot(xg, yg, color='green', marker='o')
    # print(type(xg))
    # print((xg))
    xsfractional.append(fractional(test))
    print(fractional(test))
    # mat.set_data(xsg, ysg)
    mat1.set_data(test[:main2, 0], test[:main2, 1])
    mat2.set_data(test[main2:, 0], test[main2:, 1])
    # mat.set_color('blue')
    return mat1, mat2

    # newpoints = (xs[i], ys[i], "g",
    #              xs[i], ys[i], "m")

    # newpoints = (xg, yg, "g",
    #              xm, ym, "m")

    # animlist = plt.plot(*newpoints, linestyle="None", marker="o")
    # return animlist


ax.axis([-1, 5, -1, 5])
# plt.plot(x, y, "r")

myAnimation = animation.FuncAnimation(fig, animate, init_func=init, interval=50, blit=True, repeat=True)
plt.draw()

# plt.plot(xsfractional)
plt.show()
