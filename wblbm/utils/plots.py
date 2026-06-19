"""
plot_droplet_data.py
--------------------
Reads a simulation CSV and plots contact angles (ca_left, ca_right) and
capillary numbers (Ca_cl_left, Ca_cl_right) against the normalised droplet
position x/R on a dual-y-axis figure.

Usage
-----
    python plot_droplet_data.py
    # → prompted for the path to the CSV file or its parent directory
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd


# ── 1. Locate the CSV ────────────────────────────────────────────────────────

def resolve_csv(raw: str) -> Path:
    """Accept either a directory (picks the first *.csv inside) or a file."""
    p = Path(raw).expanduser().resolve()
    if p.is_dir():
        csvs = sorted(p.glob("*.csv"))
        if not csvs:
            sys.exit(f"No CSV files found in directory: {p}")
        if len(csvs) > 1:
            print("Multiple CSV files found:")
            for i, f in enumerate(csvs):
                print(f"  [{i}] {f.name}")
            idx = int(input("Enter the number of the file to use: ").strip())
            return csvs[idx]
        return csvs[0]
    if p.suffix.lower() == ".csv" and p.is_file():
        return p
    sys.exit(f"Path is not a CSV file or a directory: {p}")


raw_path = input("Enter the path to the CSV file or its parent directory:\n> ").strip()
csv_path = resolve_csv(raw_path)
print(f"\nReading: {csv_path}")


# ── 2. Load the required columns ─────────────────────────────────────────────

COLS = ["x_over_R", "ca_left", "ca_right", "Ca_cl_left", "Ca_cl_right"]

try:
    df = pd.read_csv(csv_path, usecols=COLS)
except ValueError as e:
    sys.exit(f"Column error: {e}\nMake sure the CSV contains: {COLS}")

df = df.sort_values("x_over_R").reset_index(drop=True)
print(f"Loaded {len(df):,} rows.\n")


# ── 3. Build the dual-axis plot ───────────────────────────────────────────────

fig, ax1 = plt.subplots(figsize=(9, 5))
ax2 = ax1.twinx()

# Colour palette — two hues, solid for left / dashed for right
CA_LEFT_COLOR  = "#1f77b4"   # blue
CA_RIGHT_COLOR = "#1f77b4"
CL_LEFT_COLOR  = "#d62728"   # red
CL_RIGHT_COLOR = "#d62728"

x = df["x_over_R"]

# Contact angles on the left y-axis
lns1 = ax1.scatter(x, df["ca_left"],  color=CA_LEFT_COLOR,  marker="o", s=18,
                   label=r"$\theta_\mathrm{trailing}$")
lns2 = ax1.scatter(x, df["ca_right"], color=CA_RIGHT_COLOR, marker="s", s=18,
                   label=r"$\theta_\mathrm{leading}$")

# Capillary numbers on the right y-axis
lns3 = ax2.scatter(x, df["Ca_cl_left"],  color=CL_LEFT_COLOR,  marker="^", s=18,
                   label=r"$Ca_\mathrm{trailing}$")
lns4 = ax2.scatter(x, df["Ca_cl_right"], color=CL_RIGHT_COLOR, marker="D", s=18,
                   label=r"$Ca_\mathrm{leading}$")

# ── Labels & formatting ───────────────────────────────────────────────────────

ax1.set_xlabel(r"$x \, / \, R$", fontsize=13)
ax1.set_ylabel(r"Contact angle $\theta$ (°)", fontsize=13, color=CA_LEFT_COLOR)
ax2.set_ylabel(r"Capillary number $Ca$",       fontsize=13, color=CL_LEFT_COLOR)

ax1.tick_params(axis="y", labelcolor=CA_LEFT_COLOR)
ax2.tick_params(axis="y", labelcolor=CL_RIGHT_COLOR)

ax1.yaxis.set_minor_locator(ticker.AutoMinorLocator())
ax2.yaxis.set_minor_locator(ticker.AutoMinorLocator())
ax1.xaxis.set_minor_locator(ticker.AutoMinorLocator())
ax1.grid(which="major", ls="--", alpha=0.4)
ax1.grid(which="minor", ls=":",  alpha=0.2)

# Combined legend
handles = [lns1, lns2, lns3, lns4]
ax1.legend(handles, [h.get_label() for h in handles],
           loc="best", framealpha=0.9, fontsize=11)

plt.title(r"Contact angles and capillary numbers vs $x/R$", fontsize=13, pad=10)
fig.tight_layout()

# ── 4. Save & show ───────────────────────────────────────────────────────────

out_path = csv_path.parent / (csv_path.stem + "_ca_Ca_plot.png")
fig.savefig(out_path, dpi=150)
print(f"Figure saved to: {out_path}")
plt.show()