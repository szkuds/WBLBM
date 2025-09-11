import numpy as np
import matplotlib.pyplot as plt

# Parameters
Ny, Nx = 401, 401
theta = 0  # radians
Rd = 60

# Generate mask
from wblbm.masks import vertical_needle_tank_mask  # Replace with actual import if needed
mask, Rx, Ry = vertical_needle_tank_mask(Ny, Nx, theta, Rd)

# Plot the mask
plt.figure(figsize=(8, 8))
plt.imshow(mask, origin="lower", cmap="gray")
plt.title("Needle Tank Mask")
plt.xlabel("x")
plt.ylabel("y")
plt.colorbar(label="Mask Value (1=Fluid, 0=Solid)")
plt.scatter(Rx, Ry, color="red", s=50, label="Droplet Center")
plt.legend()
plt.tight_layout()
plt.show()