# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 05:21:16 2022

@author: linhai
"""

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
folder = "D2"
#image = plt.imread(folder +'/X.png')
#mask = plt.imread(folder +'/Y.png')

image = Image.open(folder +'/Xn25.png').convert("L")
mask = Image.open(folder +'/Ypn25.png').convert("L")
image = np.asarray(image)
#mask = np.asarray(image)

fig, ax = plt.subplots()
ax.imshow(image, cmap='gray')
ax.imshow(mask, cmap='twilight', alpha=0.5)
fig.show()
fig.savefig(folder +'/noise25.png')