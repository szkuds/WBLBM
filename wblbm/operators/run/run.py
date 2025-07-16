import jax.numpy as jnp
import numpy as np
from wblbm.grid import Grid
from wblbm.lattice import Lattice
from wblbm.operators.initialise.init import Initialise
from wblbm.operators.update.update import Update
from wblbm.operators.update.update_multiphase import UpdateMultiphase
from wblbm.operators.macroscopic.macroscopic_multiphase import MacroscopicMultiphase
from wblbm.operators.macroscopic.macroscopic_multiphase_wetting import MacroscopicWetting
from wblbm.operators.update.update_mutliphase_wetting import UpdateMultiphaseWetting
from wblbm.utils.io import SimulationIO
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
            force_obj=None,
            wetting_enabled: bool = False,
            hysteresis_params: dict = None,
            phi_value: float = 1.0,      # <-- Added
            d_rho_value: float = 0.0     # <-- Added
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
        self.wetting_enabled = wetting_enabled
        self.hysteresis_params = hysteresis_params
        self.phi_value = phi_value
        self.d_rho_value = d_rho_value

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
        if multiphase and not wetting_enabled:
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
                force_enabled=force_enabled,
            )
        elif wetting_enabled and multiphase:
            self.update = UpdateMultiphaseWetting(
                self.grid,
                self.lattice,
                tau,
                kappa,
                interface_width,
                rho_l,
                rho_v,
                bc_config=bc_config,
                force_enabled=force_enabled,
                wetting_enabled=wetting_enabled,
            )
            self.macroscopic_multiphase_wetting = MacroscopicWetting(
                self.grid,
                self.lattice,
                kappa,
                interface_width,
                rho_l,
                rho_v,
                force_enabled=force_enabled,
                wetting_enabled=wetting_enabled,
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

        # Initialize Wetting and Hysteresis if enabled
        self.wetting = None
        self.hysteresis = None
        self.contact_angle = None
        self.contact_line_location = None
        if wetting_enabled and multiphase:
            from wblbm import GradientWetting
            from wblbm.operators.wetting.contact_angle import ContactAngle
            from wblbm.operators.wetting.contact_line_location import ContactLineLocation
            self.wetting = GradientWetting(self.lattice, rho_l, rho_v)
            rho_mean = (rho_l + rho_v) / 2
            self.contact_angle = ContactAngle(rho_mean)
            self.contact_line_location = ContactLineLocation(rho_mean)

        if hysteresis_params is not None and wetting_enabled:
            from wblbm.operators.wetting.hysteresis import Hysteresis
            self.hysteresis = Hysteresis(**hysteresis_params)

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
            "wetting_enabled": wetting_enabled,
            "hysteresis_params": hysteresis_params,
            "phi_value": phi_value,      # <-- Added
            "d_rho_value": d_rho_value   # <-- Added
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
        elif self.multiphase and init_type == "wetting_chemical_step":
            f_prev = self.initialiser.initialise_wetting_chemical_step(
                self.rho_l, self.rho_v, self.interface_width
            )
        elif self.multiphase and init_type == "wetting":
            f_prev = self.initialiser.initialise_wetting(
                self.rho_l, self.rho_v, self.interface_width
            )
        else:
            f_prev = self.initialiser.initialise_standard()

        # Initialize wetting parameters if wetting is enabled
        phi_left = None
        phi_right = None
        d_rho_left = None
        d_rho_right = None
        if self.wetting_enabled:
            nx = self.grid.nx
            phi_left = jnp.ones(nx)
            phi_left = phi_left.at[:].set(self.phi_value)
            d_rho_left = jnp.zeros(nx)
            d_rho_left = d_rho_left.at[:].set(self.d_rho_value)
            phi_right = jnp.ones(nx)
            phi_right = phi_right.at[:].set(self.phi_value)
            d_rho_right = jnp.zeros(nx)
            d_rho_right = d_rho_right.at[:].set(self.d_rho_value)

        # Initialize hysteresis state variables if hysteresis is enabled
        left_step_passed = False
        right_step_passed = False
        cah_window_left_philic = False
        cah_window_right_philic = False
        cah_window_left_phobic = False
        cah_window_right_phobic = False
        pinned_count_left = 0
        pinned_count_right = 0

        if verbose:
            print(f"Starting LBM simulation with {self.nt} time steps...")
            print(f"Config -> Grid: {self.grid_shape}, Multiphase: {self.multiphase}, Wetting: {self.wetting_enabled}")

        # Main simulation loop
        for it in range(self.nt):
            force = None
            if self.force_enabled and self.force_obj is not None:
                if self.wetting_enabled and self.multiphase:
                    rho, _, _ = self.macroscopic_multiphase_wetting(f_prev)
                elif self.multiphase:
                    rho, _, _ = self.macroscopic_multiphase(f_prev)
                else:
                    rho, _ = self.macroscopic(f_prev)
                mask: bool = rho > 0.95 * self.rho_v + 0.05 * self.rho_l
                force = (self.force_obj.compute_force(rho)) * rho * mask
            elif self.force_enabled:
                force = jnp.ones((self.grid.nx, self.grid.ny, 1, 2)) * jnp.array([0.0, 0.01])

            # Apply hysteresis adjustments if enabled (compute wetting parameters dynamically)
            if self.hysteresis is not None and self.wetting_enabled:
                contact_angle_func = self.contact_angle.compute
                cll_func = self.contact_line_location.compute
                get_rho_func = lambda f: self.macroscopic_multiphase_wetting(f)[0]
                (phi_left, phi_right, d_rho_left, d_rho_right,
                 left_step_passed, right_step_passed,
                 cah_window_left_philic, cah_window_right_philic,
                 cah_window_left_phobic, cah_window_right_phobic,
                 pinned_count_left, pinned_count_right,
                 p_m_left, p_m_right) = self.hysteresis.apply(
                    f_prev, it, left_step_passed, right_step_passed,
                    cah_window_left_philic, cah_window_right_philic,
                    cah_window_left_phobic, cah_window_right_phobic,
                    pinned_count_left, pinned_count_right,
                    self.update, contact_angle_func, cll_func, get_rho_func
                )
            if self.wetting_enabled:
                f_next = self.update(
                    f_prev, force=force, phi_left=phi_left, phi_right=phi_right,
                    d_rho_left=d_rho_left, d_rho_right=d_rho_right
                )
            else:
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
                    rho, u, force_tot = self.macroscopic_multiphase(f_prev)
                    data_to_save = {
                        "rho": np.array(rho),
                        "u": np.array(u),
                        "force": np.array(force),
                    }
                else:
                    rho, u = self.macroscopic(f_prev)
                    data_to_save = {"rho": np.array(rho), "u": np.array(u), "force_tot": np.array(force_tot)}

                self.io_handler.save_data_step(it, data_to_save)

                if verbose:
                    avg_rho = np.mean(rho)
                    max_u = np.max(np.sqrt(u[..., 0] ** 2 + u[..., 1] ** 2))
                    print(f"Step {it}/{self.nt}: avg_rho={avg_rho:.4f}, max_u={max_u:.6f}")

        if verbose:
            print("Simulation completed!")
            print(f"Results saved in: {self.io_handler.run_dir}")