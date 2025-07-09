import numpy as np
from wblbm.grid.grid import Grid
from wblbm.lattice.lattice import Lattice


class Initialise:
    """
    Base class for initialisation of the simulation.
    """

    def __init__(self, grid_shape, lattice_type):
        self.grid = Grid(grid_shape)
        self.lattice = Lattice(lattice_type)
        self.nx, self.ny = grid_shape
        self.q = self.lattice.q  # Number of discrete velocities

    def initialise_population(self, initial_value=1.0):
        """
        Initialise the population distribution f with shape (nx, ny, q, 1).
        """
        f = np.full((self.nx, self.ny, self.q, 1), initial_value, dtype=np.float64)
        return f

    def set_boundary_conditions(self):
        """
        Set the boundary conditions for the simulation.
        This method should be overridden by subclasses.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")
