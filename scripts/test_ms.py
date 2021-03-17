import matplotlib.pyplot as plt

vmin = -10
vmax = 10

fig, ax = plt.subplots()
s = ((2 * ax.get_window_extent().width / (vmax - vmin + 1.) * 72. / fig.dpi) ** 2)

x = [0, 2, 4, 6, 8, 10]
y = [0] * len(x)
ax.axis([-10, 10, -10, 10])
ax.scatter(x, y, s=s)
plt.show()
