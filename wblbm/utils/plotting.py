import os
import numpy as np
import matplotlib.pyplot as plt


def visualise(sim_instance, title="LBM Simulation Results"):
    """
    Visualizes simulation results by loading and plotting every saved timestep.
    This version includes a vector plot overlay on the velocity magnitude heatmap.

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
            final_force = data.get('force', None)

            fig, axes = plt.subplots(1, 2 if final_force is None else 3, figsize=(12 if final_force is None else 18, 5))

            # Plot density
            im1 = axes[0].imshow(final_rho[:, :, 0, 0].T, origin='lower', cmap='viridis')
            axes[0].set_title(f'Density (t={timestep})')
            plt.colorbar(im1, ax=axes[0], label="Density")

            # --- Velocity Plotting with Vector Overlay ---

            # 1. Plot velocity magnitude heatmap
            vel_mag = np.sqrt(final_u[:, :, 0, 0] ** 2 + final_u[:, :, 0, 1] ** 2)
            im2 = axes[1].imshow(vel_mag.T, origin='lower', cmap='plasma')
            axes[1].set_title('Velocity Magnitude & Vectors')
            plt.colorbar(im2, ax=axes[1], label="Velocity Magnitude")

            # 2. Overlay velocity vector plot (quiver)
            nx, ny = final_u.shape[0], final_u.shape[1]
            x = np.arange(0, nx)
            y = np.arange(0, ny)
            X, Y = np.meshgrid(x, y)

            # Extract velocity components
            U = final_u[:, :, 0, 0]
            V = final_u[:, :, 0, 1]

            # Downsample the vectors to avoid a cluttered plot
            # Plot one vector every `skip` grid points
            skip = 10

            # Plotting quiver requires transposing U and V to match the meshgrid and imshow orientation
            axes[1].quiver(X[::skip, ::skip], Y[::skip, ::skip], U.T[::skip, ::skip], V.T[::skip, ::skip],
                           color='white', scale=None, scale_units='xy', angles='xy')

            # --- End of Velocity Plotting Section ---

            if final_force is not None:
                # Plot force magnitude and vectors
                force_mag = np.sqrt(final_force[:, :, 0, 0] ** 2 + final_force[:, :, 0, 1] ** 2)
                im3 = axes[2].imshow(force_mag.T, origin='lower', cmap='cividis')
                axes[2].set_title('Force Magnitude & Vectors')
                plt.colorbar(im3, ax=axes[2], label="Force Magnitude")
                U_force = final_force[:, :, 0, 0]
                V_force = final_force[:, :, 0, 1]
                axes[2].quiver(X[::skip, ::skip], Y[::skip, ::skip], U_force.T[::skip, ::skip], V_force.T[::skip, ::skip],
                               color='white', scale=None, scale_units='xy', angles='xy')

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
