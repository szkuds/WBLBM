from wblbm.masks import needle_tank_mask
import numpy as np

Ny, Nx = 100, 200
theta = np.pi / 3
Rd = 30

mask, Rx, Ry = needle_tank_mask(Ny, Nx, theta, Rd)

print(Rx)