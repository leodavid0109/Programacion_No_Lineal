# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 16:10:21 2023

@author: 000095840
"""


from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt

x = np.outer(np.linspace(-3, 3, 30), np.ones(30))
y = x.copy().T # transpose
z = x**4 +x**2 *(1-2*y) + 2*y**2 - 2*x*y + 4.5*x - 4*y + 4
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(x, y, z,cmap='viridis', edgecolor='none')

x1 = np.outer(np.linspace(-0.1, 0.1, 30), np.ones(30))
y1 = x1.copy().T # transpose
z1 = (x1-y1**2) * (x1 - 4*y1** 2)
fig1 = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(x1, y1, z1,cmap='viridis', edgecolor='none')
plt.show()
