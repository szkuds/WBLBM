import os
import json
import numpy as np
from datetime import datetime
from typing import Dict

class SimulationIO:
    """
    Handles all I/O operations for the simulation, including logging and saving results.
    """
    def __init__(self, base_dir: str = "results", config: Dict = None):
        """
        Initializes the IO handler.

        Args:
            base_dir (str): The base directory to store simulation results.
            config (Dict, optional): A dictionary containing the simulation configuration to save.
        """
        self.base_dir = base_dir
        self.run_dir = self._create_timestamped_directory()
        self.data_dir = os.path.join(self.run_dir, "data")
        os.makedirs(self.data_dir, exist_ok=True)

        if config:
            self.save_config(config)

    def _create_timestamped_directory(self) -> str:
        """Creates a unique, timestamped directory for a single simulation run."""
        timestamp = datetime.now().strftime("%Y-%m-%d/%H-%M-%S")
        run_dir = os.path.join(self.base_dir, timestamp)
        os.makedirs(run_dir, exist_ok=True)
        print(f"Created results directory: {run_dir}")
        return run_dir

    def save_config(self, config: Dict):
        """Saves the simulation configuration to a JSON file."""
        config_path = os.path.join(self.run_dir, "config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        print(f"Configuration saved to {config_path}")

    def save_data_step(self, iteration: int, data: Dict[str, np.ndarray]):
        """Saves the data for a single timestep to a compressed .npz file."""
        filename = os.path.join(self.data_dir, f"timestep_{iteration}.npz")
        np.savez(filename, **data)
