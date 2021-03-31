import ast
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def from_np_array(array_string):
    array_string = ','.join(array_string.replace('[ ', '[').split())
    return np.array(ast.literal_eval(array_string))


df1 = pd.read_csv("df1.csv", converters={"0 points": from_np_array, "1 points": from_np_array})
coord0 = np.array(df1["0 points"][0])
coord1 = np.array(df1["1 points"][0])
# print(coord0)
# print((coord0.shape))

vmin = -10
vmax = 50

fig, ax = plt.subplots()
s = ((ax.get_window_extent().width / (vmax - vmin + 1.) * 72. / fig.dpi))

mat1, = ax.plot(coord0[:, 0], coord0[:, 1], c='g', linestyle='None', marker='o', markersize=2 * s)
mat2, = ax.plot(coord1[:, 0], coord1[:, 1], c='m', linestyle='None', marker='o', markersize=2 * s)

ax.axis([-10, 50, -10, 50])
plt.draw()
plt.show()
