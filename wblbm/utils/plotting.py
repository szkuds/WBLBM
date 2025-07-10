import os
import numpy as np
import matplotlib.pyplot as plt


def visualize_all_steps(sim_instance, title="LBM Simulation Results"):
    """
    Visualizes simulation results by loading and plotting every saved timestep.

    Args:
        sim_instance: The completed simulation instance from the Run class.
        title (str): The base title for the plots.
    """
    try:
        # Get the directory where data files are stored
        data_dir = sim_instance.io_handler.data_dir

        # Create a new directory within the run to store the plots
        plot_dir = os.path.join(sim_instance.io_handler.run_dir, "plots")
        os.makedirs(plot_dir, exist_ok=True)
        print(f"Saving plots to: {plot_dir}")

        # Get all data files and sort them by timestep
        files = [f for f in os.listdir(data_dir) if f.endswith('.npz')]
        if not files:
            print("No data files found to visualize.")
            return

        # Sort files based on the iteration number in the filename
        files.sort(key=lambda f: int(f.split('_')[-1].split('.')[0]))

        # Loop through each timestep file and generate a plot
        for filename in files:
            file_path = os.path.join(data_dir, filename)
            timestep = int(filename.split('_')[-1].split('.')[0])

            data = np.load(file_path)
            final_rho = data['rho']
            final_u = data['u']

            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            # Plot density
            im1 = axes[0].imshow(final_rho[:, :, 0, 0].T, origin='lower', cmap='viridis')
            axes[0].set_title(f'Density (t={timestep})')
            plt.colorbar(im1, ax=axes[0], label="Density")

            # Plot velocity magnitude
            vel_mag = np.sqrt(final_u[:, :, 0, 0] ** 2 + final_u[:, :, 0, 1] ** 2)
            im2 = axes[1].imshow(vel_mag.T, origin='lower', cmap='plasma')
            axes[1].set_title('Velocity Magnitude')
            plt.colorbar(im2, ax=axes[1], label="Velocity")

            for ax in axes:
                ax.set_xlabel('x')
                ax.set_ylabel('y')

            plt.suptitle(f"{title} - Timestep {timestep}")
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])

            # Save the figure to the plots directory
            plot_filename = os.path.join(plot_dir, f"timestep_{timestep}.png")
            plt.savefig(plot_filename)
            plt.close(fig)  # Close the figure to free up memory

        print(f"Finished generating plots for all {len(files)} timesteps.")

    except ImportError:
        print("Matplotlib not found. Please install it to visualize results.")
    except Exception as e:
        print(f"An error occurred during visualization: {e}")

