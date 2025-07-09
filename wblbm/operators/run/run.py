import jax.numpy as jnp
import numpy as np
from typing import Optional, Callable
from wblbm.grid import Grid
from wblbm.lattice import Lattice
from wblbm.operators.initialise.init import Initialise
from wblbm.operators.update.update import Update
from wblbm.operators.update.update_multiphase import UpdateMultiphase
from wblbm.operators.macroscopic.macroscopic import Macroscopic


class Run:
    """
    Main class to run lattice Boltzmann simulations.
    """

    def __init__(
            self,
            grid_shape: tuple,
            lattice_type: str = "D2Q9",
            tau: float = 1.0,
            nt: int = 1000,
            multiphase: bool = False,
            kappa: float = 0.1,
            beta: float = 1.0,
            rho_l: float = 1.0,
            rho_v: float = 0.1,
            save_interval: int = 100,
            output_callback: Optional[Callable] = None
    ):
        """
        Initialize the Run class.

        Args:
            grid_shape (tuple): Shape of the simulation grid (nx, ny)
            lattice_type (str): Type of lattice (default: "D2Q9")
            tau (float): Relaxation time parameter
            nt (int): Number of time steps
            multiphase (bool): Whether to run multiphase simulation
            kappa (float): Interface width parameter (for multiphase)
            beta (float): Surface tension parameter (for multiphase)
            rho_l (float): Liquid density (for multiphase)
            rho_v (float): Vapor density (for multiphase)
            save_interval (int): Interval for saving/outputting results
            output_callback (Callable): Optional callback function for custom output handling
        """
        self.grid_shape = grid_shape
        self.lattice_type = lattice_type
        self.tau = tau
        self.nt = nt
        self.multiphase = multiphase
        self.save_interval = save_interval
        self.output_callback = output_callback

        # Initialize components
        self.grid = Grid(grid_shape)
        self.lattice = Lattice(lattice_type)
        self.initialise = Initialise(grid_shape, lattice_type)

        # Initialize update operator
        if multiphase:
            self.update = UpdateMultiphase(
                self.grid, self.lattice, tau, kappa, beta, rho_l, rho_v
            )
        else:
            self.update = Update(self.grid, self.lattice, tau)

        # Initialize macroscopic calculator for output
        self.macroscopic = Macroscopic(self.grid, self.lattice)

        # Storage for results
        self.results = {
            'rho': [],
            'u': [],
            'iterations': []
        }

    def init(self, initial_density: float = 1.0,
             initial_velocity: Optional[jnp.ndarray] = None,
             custom_init: Optional[Callable] = None) -> jnp.ndarray:
        """
        Initialize the population distribution.

        Args:
            initial_density (float): Initial density value
            initial_velocity (jnp.ndarray): Initial velocity field
            custom_init (Callable): Custom initialization function

        Returns:
            jnp.ndarray: Initial population distribution
        """
        if custom_init is not None:
            return custom_init()

        # Initialize population
        f = self.initialise.initialise_population(initial_density)
        f = jnp.array(f)

        # If initial velocity is provided, adjust the population
        if initial_velocity is not None:
            from wblbm.operators.equilibrium.equilibirum import Equilibrium
            equilibrium = Equilibrium(self.grid, self.lattice)

            # Create density field
            rho = jnp.full((self.grid.nx, self.grid.ny, 1, 1), initial_density)

            # Reshape velocity to match expected format
            if initial_velocity.shape != (self.grid.nx, self.grid.ny, 1, 2):
                u = jnp.broadcast_to(
                    initial_velocity.reshape(1, 1, 1, -1),
                    (self.grid.nx, self.grid.ny, 1, 2)
                )
            else:
                u = initial_velocity

            # Calculate equilibrium distribution
            feq = equilibrium(rho, u)
            f = feq.reshape(self.grid.nx, self.grid.ny, self.lattice.q, 1)

        return f

    def run(self,
            initial_density: float = 1.0,
            initial_velocity: Optional[jnp.ndarray] = None,
            custom_init: Optional[Callable] = None,
            verbose: bool = True) -> dict:
        """
        Main function to run the LBM simulation.

        Args:
            initial_density (float): Initial density value
            initial_velocity (jnp.ndarray): Initial velocity field
            custom_init (Callable): Custom initialization function
            verbose (bool): Whether to print progress

        Returns:
            dict: Dictionary containing simulation results
        """
        # Initialize
        f_prev = self.init(initial_density, initial_velocity, custom_init)

        if verbose:
            print(f"Starting LBM simulation with {self.nt} time steps...")
            print(f"Grid shape: {self.grid_shape}")
            print(f"Lattice type: {self.lattice_type}")
            print(f"Multiphase: {self.multiphase}")

        # Main simulation loop
        for it in range(self.nt):
            # Update step
            f_next = self.update(f_prev)
            f_prev = f_next

            # Save results at specified intervals
            if it % self.save_interval == 0 or it == self.nt - 1:
                rho, u = self.macroscopic(f_prev)
                self.results['rho'].append(np.array(rho))
                self.results['u'].append(np.array(u))
                self.results['iterations'].append(it)

                # Custom output callback
                if self.output_callback:
                    self.output_callback(it, rho, u, f_prev)

                if verbose:
                    print(f"Step {it}/{self.nt}: avg_rho={np.mean(rho):.4f}, "
                          f"max_u={np.max(np.sqrt(u[..., 0] ** 2 + u[..., 1] ** 2)):.6f}")

        if verbose:
            print("Simulation completed!")

        return self.results

    def get_final_state(self) -> tuple:
        """
        Get the final density and velocity fields.

        Returns:
            tuple: (rho, u) final density and velocity fields
        """
        if not self.results['rho']:
            raise ValueError("No simulation results available. Run simulation first.")

        return self.results['rho'][-1], self.results['u'][-1]

    def save_results(self, filename: str):
        """
        Save simulation results to a file.

        Args:
            filename (str): Output filename
        """
        np.savez(filename, **self.results)
        print(f"Results saved to {filename}")
