import os
from typing import Dict

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import TABLEAU_COLORS

FIGURE_DPI = 200
FIGURE_FORMAT = "png"
USE_TEX = False  # set True if a full LaTeX installation is available

plt.rcParams.update({
    "text.usetex": USE_TEX,
    "font.size": 12,
    "axes.labelsize": 16,
    "axes.titlesize": 14,
    "legend.fontsize": 10,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
})

COLORS = list(TABLEAU_COLORS.values())
MARKERS = ["o", "s", "D", "^", "v", "<", ">", "p", "*", "h", "H", "+", "x"]

REGIME_NAMES = {0: "Pre-step", 1: "Step interaction", 2: "Post-step"}

# Param display info (used in Figs 3-5)
_PARAM_INFO = {
    "H": {"label": r"$H$", "col": "H"},
    "inclination_angle": {"label": r"Inclination $\alpha$ (°)", "col": "inclination_angle"},
    "mid_ca": {"label": r"$(\theta_A+\theta_R)/2$ (°)", "col": "mid_ca"},
}

# Outcome labels for phase diagrams
_OUTCOME_LABELS = {
    0: "Pinned",
    1: "Sliding (no crossing)",
    2: "Step-crossing (small def.)",
    3: "Step-crossing (strong def.)",
}
_OUTCOME_MARKERS = {0: "x", 1: "^", 2: "o", 3: "s"}
_OUTCOME_COLORS = {0: COLORS[0], 1: COLORS[1], 2: COLORS[2], 3: COLORS[3]}


def _save(fig, path):
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def classify_outcome(sim_df: pd.DataFrame) -> int:
    """
    Heuristic classification of the simulation outcome:
        0 = pinned  (droplet barely moved)
        1 = sliding (moved but did not cross step)
        2 = step-crossing, small deformation
        3 = step-crossing, strong deformation
    """
    #x = sim_df["total_displacement_R0"].values
    regimes = sim_df["regime"].values
    #height_n = sim_df["height_norm"].values
    #bd_n = sim_df["base_diameter_norm"].values

    #total_displacement = abs(x[-1] - x[0]) if len(x) > 1 else 0.0

    reached_step = np.any(regimes >= 1)
    crossed_step = np.any(regimes >= 2)

    #if total_displacement < 0.3:
    #    return 0  # pinned

    if not reached_step or not crossed_step:
        return 1  # sliding, did not fully cross

    # crossed step → check deformation
    step_mask = regimes == 1
    return 2  # small deformation


def figure5_phase_diagrams(df_all: pd.DataFrame, output_dir: str):
    """
    Phase diagrams in three parameter planes:
        (a) H  vs  (θ_A+θ_R)/2
        (b) H  vs  inclination α
        (c) α  vs  (θ_A+θ_R)/2
    Each point is one simulation, coloured/shaped by its outcome.
    """
    outcomes: Dict[str, Dict] = {}
    for sn, g in df_all.groupby("sim_name"):
        outcomes[str(sn)] = dict(
            outcome=classify_outcome(g),
            H=g["H"].iloc[0],
            inclination_angle=g["inclination_angle"].iloc[0],
            mid_ca=g["mid_ca"].iloc[0],
        )
    odf = pd.DataFrame(outcomes).T
    if odf.empty:
        print("  [skip] No data for phase diagrams")
        return

    panels = [
        ("H", "mid_ca",
         r"$H$", r"$(\theta_A+\theta_R)/2$ (°)", "(a)"),
        ("H", "inclination_angle",
         r"$H$", r"Inclination $\alpha$ (°)", "(b)"),
        ("inclination_angle", "mid_ca",
         r"Inclination $\alpha$ (°)", r"$(\theta_A+\theta_R)/2$ (°)", "(c)"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    for idx, (xp, yp, xl, yl, lbl) in enumerate(panels):
        ax = axes[idx]
        for oc in sorted(odf["outcome"].unique()):
            sub = odf[odf["outcome"] == oc]
            ax.scatter(
                sub[xp].astype(float), sub[yp].astype(float),
                marker=_OUTCOME_MARKERS.get(oc, "o"),
                c=_OUTCOME_COLORS.get(oc, "gray"),
                s=70,
                label=_OUTCOME_LABELS.get(oc, f"Regime {oc}"),
                edgecolors="k", linewidths=0.5,
            )
        ax.set_xlabel(xl, fontsize=14)
        ax.set_ylabel(yl, fontsize=14)
        ax.set_title(lbl, fontsize=13)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Figure 5 – Phase diagrams", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(fig, os.path.join(output_dir, f"fig5_phase_diagrams.{FIGURE_FORMAT}"))


def main():
    df = pd.read_csv(
        "/Users/sbszkudlarek/TUD_LBM_data/26_03_26-Phase_space_figure_update/regime_averages.csv")
    output_dir = "/Users/sbszkudlarek/TUD_LBM_data/26_03_26-Phase_space_figure_update"
    figure5_phase_diagrams(df, output_dir)


if __name__ == "__main__":
    main()
