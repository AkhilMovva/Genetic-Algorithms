import numpy as np
import matplotlib.pyplot as plt

import ver_1
import ver_2
import ver_3

x = np.arange(0, ver_1.gen_size).tolist()

y1 = ver_1.outer_ymax_mean #mean of maximum values
y2 = ver_2.outer_ymax_mean
y3 = ver_3.outer_ymax_mean

plt.figure()
plt.plot(x, y1, x, y2, x ,y3)
plt.legend(['v1', 'v2', 'v3'])

y1 = ver_1.outer_ymax_std #standard deviation of maximum values
y2 = ver_2.outer_ymax_std
y3 = ver_3.outer_ymax_std

plt.figure()
plt.plot(x, y1, x, y2, x ,y3)
plt.legend(['v1', 'v2', 'v3'])

y1 = ver_1.outer_yavg_mean #mean of average values
y2 = ver_2.outer_yavg_mean
y3 = ver_3.outer_yavg_mean

plt.figure()
plt.plot(x, y1, x, y2, x ,y3)
plt.legend(['v1', 'v2', 'v3'])

y1 = ver_1.outer_yavg_std #standard deviation of average values
y2 = ver_2.outer_yavg_std
y3 = ver_3.outer_yavg_std

plt.figure()
plt.plot(x, y1, x, y2, x ,y3)
plt.legend(['v1', 'v2', 'v3'])
