import json
import jax.numpy as jnp
from typing import Any, Dict
from pathlib import Path
import numpy as np
from datetime import datetime
import os


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        # Handle JAX arrays
        if isinstance(obj, jnp.ndarray):
            return obj.tolist()
        # Handle custom force objects
        if hasattr(obj, '__class__') and hasattr(obj, '__dict__'):
            result = {
                '__class__': obj.__class__.__name__,
                '__module__': obj.__class__.__module__
            }
            for key, value in obj.__dict__.items():
                try:
                    json.dumps(value, cls=CustomJSONEncoder)
                    result[key] = value
                except (TypeError, ValueError):
                    result[key] = str(value)
            return result
        # Handle other numpy arrays if present
        if hasattr(obj, 'tolist'):
            return obj.tolist()
        return super().default(obj)


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
        """Saves the simulation configuration to a JSON file using CustomJSONEncoder."""
        config_path = os.path.join(self.run_dir, "config.json")

        # Rename boundary condition details if present (avoids duplication)
        if "bc_config" in config:
            config["boundary_conditions"] = config.pop("bc_config")

        with open(config_path, "w") as f:
            json.dump(config, f, indent=4, cls=CustomJSONEncoder)
        print(f"Configuration saved to {config_path}")

    def save_data_step(self, iteration: int, data: Dict[str, np.ndarray]):
        """Saves the data for a single timestep to a compressed .npz file."""
        filename = os.path.join(self.data_dir, f"timestep_{iteration}.npz")
        np.savez(filename, **data)
