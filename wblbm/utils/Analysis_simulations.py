import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def plots(global_csv_path: str):
    """
    Plot droplet height and base diameter vs average X position (X_avg/R0)
    in separate plots for different U0 values. Saves figures to the same folder.
    """
    if not os.path.exists('/Users/isoldeholweg/TUD_LBM/results/Global_results_a/droplet_deformation_vs_U0.csv'):
        print(f"File not found: {global_csv_path}")
        return

    # Load global CSV
    df = pd.read_csv('/Users/isoldeholweg/TUD_LBM/results/Global_results_a/droplet_deformation_vs_U0.csv')

    # Ensure output folder exists
    output_dir = os.path.dirname(global_csv_path)
    os.makedirs(output_dir, exist_ok=True)

    # Unique U0 values
    U0_values = sorted(df['U0'].unique())

    # --- Electrowetting number range ---
    if "eta" in df.columns:
        eta_min = df["eta"].min()
        eta_max = df["eta"].max()
        eta_unique = np.sort(df["eta"].unique())

        print("\nElectrowetting number (eta) summary:")
        print(f"  Min eta = {eta_min:.4e}")
        print(f"  Max eta = {eta_max:.4e}")
        print(f"  Unique eta values = {eta_unique}")
    else:
        print("Column 'eta' not found in global CSV.")

    # Marker size and transparency
    marker_size = 20  # smaller points
    alpha_value = 0.7  # slightly transparent for overlap

    # --- Plot 1: Height ---
    plt.figure(figsize=(10, 6))
    for U0 in U0_values:
        df_U0 = df[df['U0'] == U0]
        plt.scatter(df_U0['X_avg_R0'], df_U0['Height'],
                    label=f'U0={U0}', marker='+', s=marker_size, alpha=alpha_value)

    plt.xlabel(r'$X_\mathrm{avg}/R_0$', fontsize=14)
    plt.ylabel('Droplet Height', fontsize=14)
    plt.title('Droplet Height vs Average X Position', fontsize=16)
    #plt.ylim(17.9,18.1) --> literally every point is on 18
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    height_fig_path = os.path.join(output_dir, 'droplet_height_vs_Xavg.png')
    plt.savefig(height_fig_path, dpi=150)
    plt.close()
    print(f"Height plot saved to {height_fig_path}")

    # --- Plot 2: Base Diameter ---
    plt.figure(figsize=(10, 6))
    for U0 in U0_values:
        df_U0 = df[df['U0'] == U0]
        plt.scatter(df_U0['X_avg_R0'], df_U0['Base_diameter'],
                    label=f'U0={U0}', marker='o', s=marker_size, alpha=alpha_value)

    plt.xlabel(r'$X_\mathrm{avg}/R_0$', fontsize=14)
    plt.ylabel('Droplet Base Diameter', fontsize=14)
    plt.title('Droplet Base Diameter vs Average X Position', fontsize=16)
   # plt.ylim(28,30)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    base_fig_path = os.path.join(output_dir, 'droplet_base_diameter_vs_Xavg.png')
    plt.savefig(base_fig_path, dpi=150)
    plt.close()
    print(f"Base diameter plot saved to {base_fig_path}")

    # --- Plot 3: Centre of Mass Velocity and Acceleration ---
    plt.figure(figsize=(10, 6))
    for U0 in U0_values:
        df_U0 = df[df['U0'] == U0]
        plt.scatter(df_U0['X_avg_R0'], df_U0['v_cm'],
                    label=f'Velocity at U0={U0}', marker='^', s=marker_size, alpha=alpha_value)
        #plt.scatter(df_U0['X_avg_R0'], df_U0['a_cm'],
                    #label= f'Acceleration at U0={U0}', marker='x', s=marker_size, alpha=alpha_value)
    plt.xlabel(r'$X_\mathrm{avg}/R_0$', fontsize=14)
    plt.ylabel('Centre of Mass Velocity', fontsize=14)
    plt.title('Centre of Mass Velocity vs Average X Position', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'centre_of_mass_velocity_vs_Xavg.png'), dpi=150)
    plt.close()

    # --- Plot 4: Centre of Mass Capillary Number ---
    plt.figure(figsize=(10, 6))
    for U0 in U0_values:
        df_U0 = df[df['U0'] == U0]
        plt.scatter(df_U0['X_avg_R0'], df_U0['Ca'],
                    label=f'U0={U0}', marker='s', s=marker_size, alpha=alpha_value)
    plt.xlabel(r'$X_\mathrm{avg}/R_0$', fontsize=14)
    plt.ylabel('Capillary Number', fontsize=14)
    plt.title('Ca vs Average X Position', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Ca_vs_Xavg.png'), dpi=150)
    plt.close()

    # --- Plot 5: Contact angles ---
    plt.figure(figsize=(10, 6))
    for U0 in U0_values:
        df_U0 = df[df['U0'] == U0]
        # Left contact angle
        plt.scatter(df_U0['X_avg_R0'], df_U0['CA_left'],
                    label=f'U0={U0} left', marker='o', s=18, alpha=0.7)

        # Right contact angle
        plt.scatter(df_U0['X_avg_R0'], df_U0['CA_right'],
                    label=f'U0={U0} right', marker='x', s=18, alpha=0.7)

    plt.xlabel(r'$X_\mathrm{avg}/R_0$', fontsize=14)
    plt.ylabel('Contact angle (degrees)', fontsize=14)
    plt.title('Contact angles vs Average X Position', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=9, ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'CA_left_right_vs_Xavg.png'), dpi=200)
    plt.close()

    # --- Plot 6: Mean Contact angles ---
    plt.figure(figsize=(10, 6))
    for U0 in U0_values:
        df_U0 = df[df['U0'] == U0]
        # Left contact angle
        plt.scatter(df_U0['X_avg_R0'], df_U0['CA_left_mean'],
                    label=f'U0={U0} left', marker='o', s=18, alpha=0.7)

        # Right contact angle
        plt.scatter(df_U0['X_avg_R0'], df_U0['CA_right_mean'],
                    label=f'U0={U0} right', marker='x', s=18, alpha=0.7)

    plt.xlabel(r'$X_\mathrm{avg}/R_0$', fontsize=14)
    plt.ylabel('Average Contact angle (degrees)', fontsize=14)
    plt.title('Average Contact angles vs Average X Position', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=9, ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'CA_left_right_mean_vs_Xavg.png'), dpi=200)
    plt.close()

    # --- Plot 7: side by side histogram of mean contact angles ---
    df_unique = df[["U0", "CA_left_mean", "CA_right_mean"]].drop_duplicates()
    df_unique = df_unique.sort_values("U0")
    x = np.arange(len(df_unique))
    width = 0.6  # width of bars
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))  # side-by-side subplots

    # --- Left contact angles ---
    axes[0].bar(x, df_unique["CA_left_mean"], width=width)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(df_unique["U0"])
    axes[0].set_xlabel(r"$U_0$", fontsize=14)
    axes[0].set_ylabel("Mean Left Contact Angle (°)", fontsize=14)
    axes[0].set_title("Left Contact Angle", fontsize=16)
    axes[0].set_ylim(98.6, 98.8)  # zoomed in
    axes[0].grid(axis="y", alpha=0.3)

    # --- Right contact angles ---
    axes[1].bar(x, df_unique["CA_right_mean"], width=width)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(df_unique["U0"])
    axes[1].set_xlabel(r"$U_0$", fontsize=14)
    axes[1].set_ylabel("Mean Right Contact Angle (°)", fontsize=14)
    axes[1].set_title("Right Contact Angle", fontsize=16)
    axes[1].set_ylim(99.3, 99.5)  # different y-axis zoom
    axes[1].grid(axis="y", alpha=0.3)

    plt.suptitle("Mean Contact Angles vs $U_0$", fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # leave space for suptitle

    save_path = os.path.join(output_dir, "mean_contact_angles_subplots_vs_U0.png")
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"All plots saved to {output_dir}")

if __name__ == "__main__":
    # Update this path to your global results CSV
    global_csv_path = "/Users/isoldeholweg/TUD_LBM/results/Global_results_a/droplet_deformation_vs_U0.csv"
    plots(global_csv_path)
