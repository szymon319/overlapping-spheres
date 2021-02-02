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

shiftedg = set(shift(100))

fig, ax = plt.subplots()

xg, yg = zip(*shiftedg)
mat, = ax.plot(xg, yg, color='green', marker='o')


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
    global shiftedg
    # global shiftedm
    # if not shiftedg:
    #     shiftedg = shift(100)
    shiftedg = advance(shiftedg, 0.0001)
    # shiftedm = advance(shiftedm)
    # print(shifted)

    xg, yg = zip(*shiftedg)
    # xm, ym = zip(*shiftedm)

    # fig, ax = plt.subplots()
    # mat, = ax.plot(xg, yg, color='green', marker='o')

    mat.set_data(xg, yg)
    return mat,

    # newpoints = (xs[i], ys[i], "g",
    #              xs[i], ys[i], "m")

    # newpoints = (xg, yg, "g",
    #              xm, ym, "m")

    # animlist = plt.plot(*newpoints, linestyle="None", marker="o")
    # return animlist


ax.axis([-5, 5, -5, 5])
# plt.plot(x, y, "r")

myAnimation = animation.FuncAnimation(fig, animate, interval=50, blit=False, repeat=True)
plt.draw()
plt.show()
