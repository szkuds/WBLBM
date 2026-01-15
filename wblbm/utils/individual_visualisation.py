#!/usr/bin/env python3
"""
Standalone visualization utility for LBM simulation data.
Usage: python visualisation.py /path/to/data/directory
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import json


def visualise_from_dir(data_dir, output_dir=None):
    """
    Visualizes simulation results from .npz files in the given directory.

    Args:
        data_dir: Path to directory containing .npz files
        output_dir: Optional output directory for plots (defaults to data_dir/plots)
    """
    if not os.path.isdir(data_dir):
        print(f"Error: Directory '{data_dir}' does not exist.")
        sys.exit(1)

    # Find .npz files
    files = [f for f in os.listdir(data_dir) if f.endswith(".npz")]
    if not files:
        print(f"No .npz files found in '{data_dir}'")
        sys.exit(1)

    # Setup output directory
    if output_dir is None:
        output_dir = os.path.join(data_dir, "../plots")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving plots to: {output_dir}")

    # Sort files by timestep
    files.sort(key=lambda f: int(f.split("_")[-1].split(".")[0]))

    # Try to load config if available
    config_path = os.path.join(os.path.dirname(data_dir), "config.json")
    config = {}
    if os.path.isfile(config_path):
        with open(config_path) as f:
            config = json.load(f)

    density_ratio = config.get("rho_l", 1) / config.get("rho_v", 1)
    use_log_scale = density_ratio > 100

    for filename in files:
        file_path = os.path.join(data_dir, filename)
        timestep = int(filename.split("_")[-1].split(".")[0])

        data = np.load(file_path)
        rho = data["rho"]
        u = data["u"]
        force = data.get("force", None)

        n_plots = 2 if force is None else 3
        fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))

        # Plot density
        if use_log_scale:
            density_data = rho[:, :, 0, 0].T
            density_data = np.maximum(density_data, np.min(density_data[density_data > 0]) * 1e-10)
            im1 = axes[0].imshow(density_data, origin="lower", cmap="viridis",
                                 norm=plt.matplotlib.colors.LogNorm())
            axes[0].set_title("Density (Log Scale)")
        else:
            im1 = axes[0].imshow(rho[:, :, 0, 0].T, origin="lower", cmap="viridis")
            axes[0].set_title("Density")
        plt.colorbar(im1, ax=axes[0], label="Density")

        # Plot velocity with quiver overlay
        vel_mag = np.sqrt(u[:, :, 0, 0] ** 2 + u[:, :, 0, 1] ** 2)
        im2 = axes[1].imshow(vel_mag.T, origin="lower", cmap="plasma")
        axes[1].set_title("Velocity")
        plt.colorbar(im2, ax=axes[1], label="Velocity Magnitude")

        nx, ny = u.shape[0], u.shape[1]
        x, y = np.arange(nx), np.arange(ny)
        X, Y = np.meshgrid(x, y)
        skip = max(1, min(nx, ny) // 10)

        axes[1].quiver(X[::skip, ::skip], Y[::skip, ::skip],
                       u[:, :, 0, 0].T[::skip, ::skip],
                       u[:, :, 0, 1].T[::skip, ::skip],
                       color="white", scale=None, scale_units="xy", angles="xy")

        # Plot force if available
        if force is not None:
            force_mag = np.sqrt(force[:, :, 0, 0] ** 2 + force[:, :, 0, 1] ** 2)
            im3 = axes[2].imshow(force_mag.T, origin="lower", cmap="cividis")
            axes[2].set_title("Force")
            plt.colorbar(im3, ax=axes[2], label="Force Magnitude")

        for ax in axes:
            ax.set_xlabel("x")
            ax.set_ylabel("y")

        plt.suptitle(f"LBM Simulation - Timestep {timestep}")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        plot_filename = os.path.join(output_dir, f"timestep_{timestep}.png")
        plt.savefig(plot_filename)
        plt.close(fig)
        print(f"Saved: {plot_filename}")

    print(f"Finished generating {len(files)} plots.")


def main():
    parser = argparse.ArgumentParser(description="Visualize LBM simulation .npz files")
    parser.add_argument("data_dir", nargs="?", help="Path to directory containing .npz files")
    parser.add_argument("-o", "--output", help="Output directory for plots", default=None)

    args = parser.parse_args()

    # If no data_dir provided via command line, prompt for it
    if args.data_dir is None:
        data_dir = input("Enter path to directory containing .npz files: ").strip()
        output_dir = input("Enter output directory (press Enter to use default): ").strip() or None
    else:
        data_dir = args.data_dir
        output_dir = args.output

    visualise_from_dir(data_dir, output_dir)


if __name__ == "__main__":
    main()
