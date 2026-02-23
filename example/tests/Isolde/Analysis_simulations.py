import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------------
# Plotting settings
# ------------------------
plt.rcParams.update({
    "font.size": 14,
    "axes.linewidth": 1.5,
    "xtick.major.width": 1.2,
    "ytick.major.width": 1.2,
    "xtick.minor.width": 1.0,
    "ytick.minor.width": 1.0,
})

MARKER_SIZE = 5
MARKER_EDGE_WIDTH = 1.5
MARKERS = ["o", "s", "^", "d", "v", "x", "+"]
COLOUR = ['r', 'g', 'b','c', 'm', 'y', 'k']


def plots(global_csv_path: str):

    if not os.path.exists(global_csv_path):
        print(f"File not found: {global_csv_path}")
        return

    df = pd.read_csv(global_csv_path)

    output_dir = os.path.dirname(global_csv_path)
    os.makedirs(output_dir, exist_ok=True)

    U0_values = sorted(df["U0"].unique())

    # ------------------------
    # Helper plotting function
    # ------------------------
    def open_marker_plot(x, y, marker, label, colour):
        plt.plot(
            x, y,
            linestyle="None",
            marker=marker,
            markersize=MARKER_SIZE,
            markerfacecolor="none",
            markeredgecolor=colour,
            markeredgewidth=MARKER_EDGE_WIDTH,
            label=label
        )

    # ------------------------
    # Plot 1: Droplet height
    # ------------------------
    plt.figure(figsize=(8, 6))
    for i, U0 in enumerate(U0_values):
        d = df[df["U0"] == U0]
        open_marker_plot(
            d["X_avg_R0"], d["Height"],
            MARKERS[i % len(MARKERS)],
            fr"$U_0={U0}$",
            COLOUR[i % len(COLOUR)],
        )

    plt.xlabel(r"$X_{\mathrm{avg}}/R_0$")
    plt.ylabel("Droplet height (lattice units)")
    plt.title("Droplet height vs average position")
    plt.legend(frameon=False, loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "droplet_height_vs_Xavg.png"), dpi=300)
    plt.close()

    # ------------------------
    # Plot 2: Base diameter
    # ------------------------
    plt.figure(figsize=(8, 6))
    for i, U0 in enumerate(U0_values):
        d = df[df["U0"] == U0]
        open_marker_plot(
            d["X_avg_R0"], d["Base_diameter"],
            MARKERS[i % len(MARKERS)],
            fr"$U_0={U0}$",
            COLOUR[i % len(COLOUR)],
        )

    plt.xlabel(r"$X_{\mathrm{avg}}/R_0$")
    plt.ylabel("Droplet base diameter (lattice units)")
    plt.title("Droplet base diameter vs average position")
    plt.legend(frameon=False, loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "droplet_base_diameter_vs_Xavg.png"), dpi=300)
    plt.close()

    # ------------------------
    # Plot 3: Velocity
    # ------------------------
    plt.figure(figsize=(8, 6))
    for i, U0 in enumerate(U0_values):
        d = df[df["U0"] == U0]
        open_marker_plot(
            d["X_avg_R0"], d["v_cm"],
            MARKERS[i % len(MARKERS)],
            fr"$U_0={U0}$",
            COLOUR[i % len(COLOUR)],
        )

    plt.xlabel(r"$X_{\mathrm{avg}}/R_0$")
    plt.ylabel("Centre-of-mass velocity (lattice units)")
    plt.title("Droplet velocity vs average position")
    plt.legend(frameon=False, loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "centre_of_mass_velocity_vs_Xavg.png"), dpi=300)
    plt.close()

    # ------------------------
    # Plot 4: Capillary number
    # ------------------------
    plt.figure(figsize=(8, 6))
    for i, U0 in enumerate(U0_values):
        d = df[df["U0"] == U0]
        open_marker_plot(
            d["X_avg_R0"], d["Ca"],
            MARKERS[i % len(MARKERS)],
            fr"$U_0={U0}$",
            COLOUR[i % len(COLOUR)],
        )

    plt.xlabel(r"$X_{\mathrm{avg}}/R_0$")
    plt.ylabel("Capillary number")
    plt.title("Capillary number vs average position")
    plt.legend(frameon=False, loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "Ca_vs_Xavg.png"), dpi=300)
    plt.close()

    # ------------------------
    # Plot 5: Contact angles
    # ------------------------
    plt.figure(figsize=(8, 6))
    for i, U0 in enumerate(U0_values):
        d = df[df["U0"] == U0]

        open_marker_plot(
            d["X_avg_R0"], d["CA_left"],
            "o", fr"$U_0={U0}$ left",
            COLOUR[i % len(COLOUR)],
        )
        open_marker_plot(
            d["X_avg_R0"], d["CA_right"],
            "*", fr"$U_0={U0}$ right",
            COLOUR[i % len(COLOUR)],
        )

    plt.xlabel(r"$X_{\mathrm{avg}}/R_0$")
    plt.ylabel("Contact angle (°)")
    plt.title("Contact angles vs average position")
    plt.legend(frameon=False, fontsize=12, loc='best', ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "CA_left_right_vs_Xavg.png"), dpi=300)
    plt.close()

    # ------------------------
    # Plot 6: Mean contact angles
    # ------------------------
    plt.figure(figsize=(8, 6))
    for i, U0 in enumerate(U0_values):
        d = df[df["U0"] == U0]

        open_marker_plot(
            d["X_avg_R0"], d["CA_left_mean"],
            "o", fr"$U_0={U0}$ left",
            COLOUR[i % len(COLOUR)],
        )
        open_marker_plot(
            d["X_avg_R0"], d["CA_right_mean"],
            "x", fr"$U_0={U0}$ right",
            COLOUR[i % len(COLOUR)],
        )

    plt.xlabel(r"$X_{\mathrm{avg}}/R_0$")
    plt.ylabel("Mean contact angle (°)")
    plt.title("Mean contact angles vs average position")
    plt.legend(frameon=False, fontsize=12, loc='best', ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "CA_left_right_mean_vs_Xavg.png"), dpi=300)
    plt.close()


if __name__ == "__main__":
    global_csv_path = (
        "/Users/isoldeholweg/TUD_LBM/results/"
        "Global_results_10f/droplet_deformation_vs_U0.csv"
    )
    plots(global_csv_path)
