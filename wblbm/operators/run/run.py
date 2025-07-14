import jax.numpy as jnp
import numpy as np
from wblbm.grid import Grid
from wblbm.lattice import Lattice
from wblbm.operators.initialise.init import Initialise
from wblbm.operators.update.update import Update
from wblbm.operators.update.update_multiphase import UpdateMultiphase
from wblbm.operators.macroscopic.macroscopic_multiphase import MacroscopicMultiphase
from wblbm.utils.io import SimulationIO
from wblbm.utils.profiler import JAXProfiler
from wblbm.operators.boundary_condition.boundary_condition import BoundaryCondition


class Run:
    """
    Main class to configure and run lattice Boltzmann simulations.
    """

    def __init__(
        self,
        grid_shape: tuple,
        lattice_type: str = "D2Q9",
        tau: float = 1.0,
        nt: int = 1000,
        multiphase: bool = False,
        kappa: float = 0.1,
        rho_l: float = 1.0,
        rho_v: float = 0.1,
        interface_width: int = 4,
        save_interval: int = 100,
        results_dir: str = "results",
        bc_config: dict = None,
        force_enabled: bool = False,
        force_obj=None,  # New parameter for force object
    ):
        self.grid_shape = grid_shape
        self.nt = nt
        self.multiphase = multiphase
        self.save_interval = save_interval

        # Store multiphase params
        self.rho_l = rho_l
        self.rho_v = rho_v
        self.interface_width = interface_width if multiphase else None
        self.force_enabled = force_enabled
        self.force_obj = force_obj

        # Initialize core components
        self.grid = Grid(grid_shape)
        self.lattice = Lattice(lattice_type)
        self.initialiser = Initialise(self.grid, self.lattice)

        # Boundary condition handler
        self.boundary_condition = None
        if bc_config is not None:
            self.boundary_condition = BoundaryCondition(
                self.grid, self.lattice, bc_config
            )

        # Select the appropriate update and macroscopic operators
        if multiphase:
            from wblbm.operators.macroscopic.macroscopic_multiphase import (
                MacroscopicMultiphase,
            )

            self.update = UpdateMultiphase(
                self.grid,
                self.lattice,
                tau,
                kappa,
                interface_width,
                rho_l,
                rho_v,
                bc_config=bc_config,
                force_enabled=force_enabled,
            )
            self.macroscopic_multiphase = MacroscopicMultiphase(
                self.grid,
                self.lattice,
                kappa,
                interface_width,
                rho_l,
                rho_v,
                force_enabled=force_enabled,  # Add this
            )
        else:
            from wblbm.operators.macroscopic.macroscopic import Macroscopic

            self.update = Update(
                self.grid,
                self.lattice,
                tau,
                bc_config=bc_config,
                force_enabled=force_enabled,
            )
            self.macroscopic = Macroscopic(
                self.grid, self.lattice, force_enabled=force_enabled
            )

        # Prepare config dictionary for the IO handler
        self.config = {
            "grid_shape": grid_shape,
            "lattice_type": lattice_type,
            "tau": tau,
            "nt": nt,
            "multiphase": multiphase,
            "save_interval": save_interval,
            "kappa": kappa if multiphase else None,
            "beta": self.update.macroscopic.beta if multiphase else None,
            "rho_l": rho_l if multiphase else None,
            "rho_v": rho_v if multiphase else None,
            "interface_width": self.interface_width,
            "bc_config": bc_config,
            "force_enabled": force_enabled,
            "force_obj": str(type(force_obj)) if force_obj is not None else None,
        }
        self.io_handler = SimulationIO(base_dir=results_dir, config=self.config)

    def run(self, init_type: str = "standard", verbose: bool = True):
        """
        Main function to run the LBM simulation.

        Args:
            init_type (str): Type of initialisation ('standard' or 'multiphase_bubble').
            verbose (bool): Whether to print progress updates to the console.
        """
        # Initialize the population distribution based on the simulation type
        if self.multiphase and init_type == "multiphase_droplet":
            f_prev = self.initialiser.initialise_multiphase_droplet(
                self.rho_l, self.rho_v, self.interface_width
            )
        elif self.multiphase and init_type == "multiphase_bubble":
            f_prev = self.initialiser.initialise_multiphase_bubble(
                self.rho_l, self.rho_v, self.interface_width
            )
        else:
            f_prev = self.initialiser.initialise_standard()

        if verbose:
            print(f"Starting LBM simulation with {self.nt} time steps...")
            print(f"Config -> Grid: {self.grid_shape}, Multiphase: {self.multiphase}")

        # Main simulation loop
        for it in range(self.nt):
            force = None
            if self.force_enabled and self.force_obj is not None:
                # Compute force using the force object (optionally pass rho if needed)
                if self.multiphase:
                    rho, _, _ = self.macroscopic_multiphase(f_prev)
                else:
                    rho, _ = self.macroscopic(f_prev)
                force = self.force_obj.compute_force(rho)
            elif self.force_enabled:
                force = jnp.ones((self.grid.nx, self.grid.ny, 1, 2)) * jnp.array(
                    [0.01, 0.0]
                )
            f_next = (
                self.update(f_prev, force=force)
                if self.force_enabled
                else self.update(f_prev)
            )
            # Apply boundary conditions if present
            if self.boundary_condition is not None:
                f_next = self.boundary_condition(f_next, f_next)
            f_prev = f_next

            # Save data at the specified interval
            if it % self.save_interval == 0 or it == self.nt - 1:
                if self.multiphase:
                    rho, u, force = self.macroscopic_multiphase(f_prev)
                    data_to_save = {
                        "rho": np.array(rho),
                        "u": np.array(u),
                        "force": np.array(force),
                    }
                else:
                    rho, u = self.macroscopic(f_prev)
                    data_to_save = {"rho": np.array(rho), "u": np.array(u)}

                self.io_handler.save_data_step(it, data_to_save)

                if verbose:
                    avg_rho = np.mean(rho)
                    max_u = np.max(np.sqrt(u[..., 0] ** 2 + u[..., 1] ** 2))
                    print(
                        f"Step {it}/{self.nt}: avg_rho={avg_rho:.4f}, max_u={max_u:.6f}"
                    )

        if verbose:
            print("Simulation completed!")
            print(f"Results saved in: {self.io_handler.run_dir}")

    def run_with_profiling(
        self,
        init_type: str = "standard",
        verbose: bool = True,
        profile_steps: int = 100,
    ):
        """
        Run simulation with JAX profiling enabled for a subset of steps.

        Args:
            profile_steps (int): Number of steps to profile (should be small, e.g., 100-1000)
        """
        # Initialize as normal
        if self.multiphase and init_type == "multiphase_bubble":
            f_prev = self.initialiser.initialise_multiphase_bubble(
                self.rho_l, self.rho_v, self.interface_width
            )
        else:
            f_prev = self.initialiser.initialise_standard()

        if verbose:
            print(
                f"Starting LBM simulation with profiling for {profile_steps} steps..."
            )

        # Warm up JAX compilation first (shorter warmup)
        for it in range(5):
            force = None
            if self.force_enabled and self.force_obj is not None:
                if self.multiphase:
                    rho, _, _ = self.macroscopic_multiphase(f_prev)
                else:
                    rho, _ = self.macroscopic(f_prev)
                force = self.force_obj.compute_force(rho)
            elif self.force_enabled:
                force = jnp.ones((self.grid.nx, self.grid.ny, 1, 2)) * jnp.array(
                    [0.01, 0.0]
                )
            f_next = (
                self.update(f_prev, force=force)
                if self.force_enabled
                else self.update(f_prev)
            )
            if self.boundary_condition is not None:
                f_next = self.boundary_condition(f_next, f_next)
            f_prev = f_next

        # Profile the actual operations
        with JAXProfiler("./profiler_output"):
            for it in range(profile_steps):
                force = None
                if self.force_enabled and self.force_obj is not None:
                    if self.multiphase:
                        rho, _, _ = self.macroscopic_multiphase(f_prev)
                    else:
                        rho, _ = self.macroscopic(f_prev)
                    force = self.force_obj.compute_force(rho)
                elif self.force_enabled:
                    force = jnp.ones((self.grid.nx, self.grid.ny, 1, 2)) * jnp.array(
                        [0.01, 0.0]
                    )
                f_next = (
                    self.update(f_prev, force=force)
                    if self.force_enabled
                    else self.update(f_prev)
                )
                if self.boundary_condition is not None:
                    f_next = self.boundary_condition(f_next, f_next)
                f_prev = f_next
                # Ensure computation completes before continuing
                if hasattr(f_next, "block_until_ready"):
                    f_next.block_until_ready()

                if verbose and it % 10 == 0:
                    print(f"Profiling step {it}/{profile_steps}")

        print("Profiling completed! Check ./profiler_output directory")
