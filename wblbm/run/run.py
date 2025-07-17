import numpy as np


class SimulationFactory:
    @staticmethod
    def create_simulation(simulation_type, **kwargs):
        if simulation_type == "singlephase":
            from wblbm.simulations.singlephase import SinglePhaseSimulation

            return SinglePhaseSimulation(**kwargs)
        elif simulation_type == "multiphase":
            from wblbm.simulations.multiphase import MultiphaseSimulation

            return MultiphaseSimulation(**kwargs)
        elif simulation_type == "wetting":
            from wblbm.simulations.wetting import WettingSimulation

            return WettingSimulation(**kwargs)
        else:
            raise ValueError(f"Unknown simulation type: {simulation_type}")


class Run:
    """
    Main class to configure and run lattice Boltzmann simulations using the simulation factory.
    """

    def __init__(
        self,
        simulation_type="singlephase",
        save_interval=100,
        results_dir="results",
        **kwargs,
    ):
        self.simulation = SimulationFactory.create_simulation(simulation_type, **kwargs)
        self.save_interval = save_interval
        self.results_dir = results_dir
        self.config = self._build_config(
            simulation_type=simulation_type,
            save_interval=save_interval,
            results_dir=results_dir,
            **kwargs,
        )
        from wblbm.utils.io import SimulationIO

        self.io_handler = SimulationIO(base_dir=results_dir, config=self.config)

    def _build_config(self, **kwargs):
        # Simple config builder for demonstration; extend as needed
        return dict(**kwargs)

    def _save_data(self, it, fprev):
        # Save data using the simulation's macroscopic operator
        if hasattr(self.simulation, "macroscopic"):
            macroscopic = self.simulation.macroscopic
            try:
                result = macroscopic(fprev)
                if isinstance(result, tuple) and len(result) == 3:
                    rho, u, force = result
                    data_to_save = {
                        "rho": np.array(rho),
                        "u": np.array(u),
                        "force": np.array(force),
                    }
                else:
                    rho, u = result
                    data_to_save = {
                        "rho": np.array(rho),
                        "u": np.array(u),
                    }
            except Exception:
                data_to_save = {"f": np.array(fprev)}
        else:
            data_to_save = {"f": np.array(fprev)}
        self.io_handler.save_data_step(it, data_to_save)

    def run(self, init_type="standard", verbose=True):
        fprev = self.simulation.initialize_fields(init_type)
        nt = getattr(self.simulation, "nt", 1000)
        if verbose:
            print(f"Starting LBM simulation with {nt} time steps...")
            print(
                f"Config -> Grid: {self.simulation.grid_shape}, Multiphase: {self.simulation.multiphase}, Wetting: {self.simulation.wetting_enabled}"
            )
        for it in range(nt):
            fprev = self.simulation.run_timestep(fprev, it)
            if it % self.save_interval == 0 or it == nt - 1:
                self._save_data(it, fprev)
                if verbose and hasattr(self.simulation, "macroscopic"):
                    result = self.simulation.macroscopic(fprev)
                    if isinstance(result, tuple) and len(result) >= 2:
                        rho, u = result[:2]
                        avg_rho = np.mean(rho)
                        max_u = np.max(np.sqrt(u[..., 0] ** 2 + u[..., 1] ** 2))
                        print(
                            f"Step {it}/{nt}: avg_rho={avg_rho:.4f}, max_u={max_u:.6f}"
                        )
        if verbose:
            print("Simulation completed!")
            print(f"Results saved in: {self.io_handler.run_dir}")
