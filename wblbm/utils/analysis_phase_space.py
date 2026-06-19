"""
analysis_phase_space.py
=======================
Thin convenience wrapper that runs only the phase-diagram analysis
(Figure 5) from analysis_paper_figures.py.

Usage
-----
    python analysis_phase_space.py
"""

from wblbm.utils.analysis_paper_figures import (
    discover_simulations,
    process_simulation,
    figure5_phase_diagrams,
    classify_outcome,
    _regime_averages,
)

import os
import pandas as pd
from typing import List


def main():
    parent_dir = input(
        "Enter parent directory containing simulation results (absolute path): "
    ).strip()

    if not os.path.isdir(parent_dir):
        print(f"Directory not found: {parent_dir}")
        return

    output_dir = os.path.join(parent_dir, "Paper_figures")
    os.makedirs(output_dir, exist_ok=True)

    sim_dirs = discover_simulations(parent_dir)
    print(f"Found {len(sim_dirs)} simulation(s)")

    frames: List[pd.DataFrame] = []
    for sd in sim_dirs:
        print(f"Processing {sd} …")
        try:
            df = process_simulation(sd)
        except Exception as exc:
            print(f"  [error] {exc}")
            df = None
        if df is not None:
            frames.append(df)

    if not frames:
        print("No data was produced.  Check directory structure.")
        return

    df_all = pd.concat(frames, ignore_index=True)

    # Save phase-space specific CSV
    summary_rows = []
    for sn, g in df_all.groupby("sim_name"):
        summary_rows.append(dict(
            sim_name=sn,
            inclination_angle=g["inclination_angle"].iloc[0],
            theta_a_pre=g["theta_a_pre"].iloc[0],
            theta_r_pre=g["theta_r_pre"].iloc[0],
            theta_a_post=g["theta_a_post"].iloc[0],
            theta_r_post=g["theta_r_post"].iloc[0],
            H=g["H"].iloc[0],
            mid_ca=g["mid_ca"].iloc[0],
            Ca_cm_mean=g["Ca_cm"].mean(),
            We_cm_mean=g["We_cm"].mean(),
            total_displacement_R0=abs(g["x_over_R"].iloc[-1] - g["x_over_R"].iloc[0]),
            outcome=classify_outcome(g),
        ))
    summary = pd.DataFrame(summary_rows)
    csv_path = os.path.join(output_dir, "phase_space_summary.csv")
    summary.to_csv(csv_path, index=False)
    print(f"Saved phase-space summary → {csv_path}")

    # Generate phase diagrams
    figure5_phase_diagrams(df_all, output_dir)
    print(f"\nPhase diagram saved to {output_dir}")


if __name__ == "__main__":
    main()

