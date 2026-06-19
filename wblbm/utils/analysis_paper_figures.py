"""
analysis_paper_figures.py
=========================
Generates the five paper figures from LBM droplet-on-incline simulations
that include a chemical step.

Directory layout expected (results produced by the pipeline)
------------------------------------------------------------
<parent_dir>/
  <timestamp_uniqueid>/
    <run_chemical_step_info>/      (any name containing "chem_step")
      config.json
      data/
        timestep_0.npz
        timestep_1000.npz
        ...

config.json stores all Run(...) kwargs with bc_config renamed to
boundary_conditions.  Chemical-step angles live inside
  boundary_conditions -> chemical_step -> ca_advancing_pre_step, etc.

Figures
-------
1. Inclined setup schematic (no data needed)
2. High-density combo figure (Ca, contact angles, deformation)
3. Local Ca and We vs x/R (3 subfigures, varying one parameter at a time)
4. Regime-averaged Ca and We vs parameters
5. Phase diagrams in parameter planes

All data is also exported to CSV.

Usage
-----
    python analysis_paper_figures.py

You will be prompted for the parent directory that contains all
simulation result folders.
"""

import os
import re
import json
import math
from typing import Dict, List, Tuple, Optional

import jax.numpy as jnp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TABLEAU_COLORS
from matplotlib.patches import Circle

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
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


# ═══════════════════════════════════════════════════════════════════════════
# I.  LOW-LEVEL HELPERS  (reused from Analysis_baseline.py for consistency)
# ═══════════════════════════════════════════════════════════════════════════

def load_config(sim_dir: str) -> Dict:
    """Load configuration from config.json."""
    config_path = os.path.join(sim_dir, "config.json")
    with open(config_path, "r") as f:
        return json.load(f)


def clean_name(s: str) -> str:
    """Make a directory name human-readable for legend labels."""
    parts = s.split("_", 1)
    if parts[0].replace("-", "").isdigit() and len(parts) == 2:
        title_part = parts[1]
    else:
        title_part = s
    return title_part.replace("_", " ").capitalize()


def extract_iteration_number(filename: str) -> Optional[int]:
    """Extract iteration number from an npz filename like timestep_1000.npz."""
    match = re.search(r"timestep_(\d+)", filename)
    return int(match.group(1)) if match else None


def extract_npz_data(npz_file: str) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Extract rho, u_x, u_y from a saved .npz file."""
    data = np.load(npz_file, allow_pickle=True)
    rho = jnp.array(data["rho"])
    u = jnp.array(data["u"])
    return rho, u[..., 0], u[..., 1]


# ═══════════════════════════════════════════════════════════════════════════
# II.  PHYSICAL / GEOMETRIC QUANTITIES
# ═══════════════════════════════════════════════════════════════════════════

def surface_tension(kappa: float, w: float, rho_l: float, rho_g: float) -> float:
    """σ = (2/3) κ |ρ_l − ρ_g|² / w"""
    return (2.0 / 3.0) * kappa * abs(rho_l - rho_g) ** 2 / w


def hysteresis_parameter(theta_r_deg: float, theta_a_deg: float) -> float:
    """H = cos(θ_R) − cos(θ_A).  Positive when θ_A > θ_R."""
    return math.cos(math.radians(theta_r_deg)) - math.cos(math.radians(theta_a_deg))


def midpoint_angle(theta_a_deg: float, theta_r_deg: float) -> float:
    """(θ_A + θ_R) / 2  in degrees."""
    return (theta_a_deg + theta_r_deg) / 2.0


def calculate_contact_line_location(
    rho: jnp.ndarray, rho_mean: float
) -> Tuple[float, float]:
    """Return left and right contact-line x-positions via linear interpolation."""
    arr = rho[:, 1, 0, 0]
    mask = jnp.array(arr < rho_mean, dtype=int)
    diff = jnp.diff(mask)

    idx_left = int(jnp.where(diff == -1)[0][0])
    idx_right = int(jnp.where(diff == 1)[0][0]) + 1

    x_left = idx_left + (rho_mean - arr[idx_left]) / (
        arr[idx_left + 1] - arr[idx_left]
    )
    x_right = idx_right - (rho_mean - arr[idx_right]) / (
        arr[idx_right - 1] - arr[idx_right]
    )
    return float(x_left), float(x_right)


def calculate_contact_angle(
    rho: jnp.ndarray, rho_mean: float
) -> Tuple[float, float]:
    """Return left (trailing) and right (leading) contact angles in degrees."""
    a0 = rho[:, 1, 0, 0]
    a1 = rho[:, 2, 0, 0]

    def _masks(arr):
        m = jnp.array(arr < rho_mean, dtype=int)
        return jnp.diff(m)

    d0 = _masks(a0)
    d1 = _masks(a1)

    il0 = int(jnp.where(d0 == -1)[0][0])
    il1 = int(jnp.where(d1 == -1)[0][0])
    ir0 = int(jnp.where(d0 == 1)[0][0]) + 1
    ir1 = int(jnp.where(d1 == 1)[0][0]) + 1

    def _interp_left(arr, idx):
        return idx + (rho_mean - arr[idx]) / (arr[idx + 1] - arr[idx])

    def _interp_right(arr, idx):
        return idx - (rho_mean - arr[idx]) / (arr[idx - 1] - arr[idx])

    xl0 = _interp_left(a0, il0)
    xl1 = _interp_left(a1, il1)
    xr0 = _interp_right(a0, ir0)
    xr1 = _interp_right(a1, ir1)

    ca_left = float(jnp.rad2deg(math.pi / 2 + jnp.arctan(xl0 - xl1)))
    ca_right = float(jnp.rad2deg(math.pi / 2 + jnp.arctan(xr1 - xr0)))
    return ca_left, ca_right


def calculate_center_of_mass(
    rho: jnp.ndarray, rho_mean: float
) -> Tuple[float, float]:
    """Centre-of-mass coordinates weighted by liquid density."""
    rho_2d = rho[:, :, 0, 0]
    mask = rho_2d > rho_mean
    xi, yi = jnp.indices(rho_2d.shape)
    total = jnp.sum(mask * rho_2d)
    return float(jnp.sum(xi * mask * rho_2d) / total), float(
        jnp.sum(yi * mask * rho_2d) / total
    )


def calculate_avg_x_location(rho, rho_mean, half_nx):
    """Average x-position of the liquid phase, centred on the grid."""
    rho2d = rho[:, :, 0, 0]
    mask = rho2d > rho_mean
    x_indices = jnp.arange(rho2d.shape[0]) - half_nx
    masked_x = jnp.where(mask, jnp.expand_dims(x_indices, 1), 0)
    return float(jnp.sum(masked_x) / jnp.sum(mask))


def calculate_droplet_height(rho, rho_mean, cl_left, cl_right, wall_y=0):
    """Height at the midpoint of the base diameter."""
    rho2d = rho[:, :, 0, 0]
    mask = rho2d > rho_mean
    if not jnp.any(mask):
        return 0.0
    mid_x = int(0.5 * (cl_left + cl_right))
    col = mask[mid_x, :]
    if not jnp.any(col):
        return 0.0
    return float(jnp.max(jnp.where(col)[0]) - wall_y)


def calculate_droplet_base_diameter(cl_left, cl_right):
    return cl_right - cl_left


def calculate_velocities(cl_left, cl_right, cm_x, plot_every):
    """Finite-difference velocities from position arrays (jnp arrays)."""
    def _fd(x):
        x_prev = jnp.roll(x, shift=1).at[0].set(x[0])
        return (x - x_prev) / plot_every
    return _fd(cl_left), _fd(cl_right), _fd(cm_x)


def calculate_accelerations(v_left, v_right, v_cm, plot_every):
    """Finite-difference accelerations from velocity arrays."""
    def _fd(v):
        v_prev = jnp.roll(v, shift=1).at[0].set(v[0])
        return (v - v_prev) / plot_every
    return _fd(v_left), _fd(v_right), _fd(v_cm)


# ═══════════════════════════════════════════════════════════════════════════
# III.  CONFIG PARAMETER EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════

def _get_chem_step_params(config: Dict) -> Dict:
    """
    Extract the chemical-step contact angles from the config.

    Returns dict with keys:
        ca_adv_pre, ca_rec_pre, ca_adv_post, ca_rec_post
    Falls back to the hysteresis params if chemical_step block is absent.
    """
    bc = config.get("boundary_conditions", config.get("bc_config", {}))
    cs = bc.get("chemical_step", {})
    hp = bc.get("hysteresis_params", {})

    ca_adv_pre = cs.get("ca_advancing_pre_step", hp.get("ca_advancing", 90.0))
    ca_rec_pre = cs.get("ca_receding_pre_step", hp.get("ca_receding", 80.0))
    ca_adv_post = cs.get("ca_advancing_post_step", 70.0)
    ca_rec_post = cs.get("ca_receding_post_step", 60.0)

    return dict(
        ca_adv_pre=ca_adv_pre,
        ca_rec_pre=ca_rec_pre,
        ca_adv_post=ca_adv_post,
        ca_rec_post=ca_rec_post,
    )


def _get_step_x(config: Dict) -> Optional[float]:
    """
    Return the x-coordinate of the chemical step in grid units.

    The config stores chemical_step_location as a fraction of NX and
    chemical_step_edge tells which boundary owns the step.
    """
    bc = config.get("boundary_conditions", config.get("bc_config", {}))
    cs = bc.get("chemical_step", {})
    frac = cs.get("chemical_step_location", None)
    if frac is None:
        return None
    grid = config.get("grid_shape", [201, 101])
    nx = grid[0]
    return float(frac * nx)


# ═══════════════════════════════════════════════════════════════════════════
# IV.  REGIME DETECTION
# ═══════════════════════════════════════════════════════════════════════════

def classify_regimes(
    x_avg: np.ndarray, step_x: Optional[float], R0: float, margin_R: float = 1.5
) -> np.ndarray:
    """
    Assign an integer regime label to every time step:
        0 = pre-step,  1 = step interaction,  2 = post-step.
    If no step location is known, every point is labelled 0.
    """
    labels = np.zeros(len(x_avg), dtype=int)
    if step_x is None:
        return labels
    lower = step_x - margin_R * R0
    upper = step_x + margin_R * R0
    labels[(x_avg >= lower) & (x_avg <= upper)] = 1
    labels[x_avg > upper] = 2
    return labels


def classify_outcome(sim_df: pd.DataFrame) -> int:
    """
    Heuristic classification of the simulation outcome:
        0 = pinned  (droplet barely moved)
        1 = sliding (moved but did not cross step)
        2 = step-crossing, small deformation
        3 = step-crossing, strong deformation
    """
    x = sim_df["x_over_R"].values
    regimes = sim_df["regime"].values
    height_n = sim_df["height_norm"].values
    bd_n = sim_df["base_diameter_norm"].values

    total_displacement = abs(x[-1] - x[0]) if len(x) > 1 else 0.0

    reached_step = np.any(regimes >= 1)
    crossed_step = np.any(regimes >= 2)

    if total_displacement < 0.3:
        return 0  # pinned

    if not reached_step or not crossed_step:
        return 1  # sliding, did not fully cross

    # crossed step → check deformation
    step_mask = regimes == 1
    if np.any(step_mask):
        deformation = np.std(height_n[step_mask]) + np.std(bd_n[step_mask])
        if deformation > 0.15:
            return 3  # strong deformation
    return 2  # small deformation


# ═══════════════════════════════════════════════════════════════════════════
# V.  PROCESS A SINGLE SIMULATION RUN
# ═══════════════════════════════════════════════════════════════════════════

def process_simulation(sim_dir: str) -> Optional[pd.DataFrame]:
    """
    Walk through every timestep in a simulation directory and return a
    DataFrame with all per-timestep quantities needed for the paper.
    """
    config = load_config(sim_dir)
    data_dir = os.path.join(sim_dir, "data")
    if not os.path.isdir(data_dir):
        print(f"  [skip] no data/ in {sim_dir}")
        return None

    npz_files = sorted(
        [f for f in os.listdir(data_dir)
         if f.endswith(".npz") and not f.startswith("._")],
        key=lambda x: extract_iteration_number(x) or 0,
    )
    if not npz_files:
        print(f"  [skip] no .npz files in {data_dir}")
        return None

    # ── constants from config ──
    RHO_L = config.get("rho_l", 1.0)
    RHO_V = config.get("rho_v", 0.001)
    RHO_MEAN = (RHO_L + RHO_V) / 2.0
    grid = config.get("grid_shape", [201, 101])
    HALF_NX = grid[0] // 2
    PLOT_EVERY = config.get("save_interval", config.get("plot_every", 1))
    TAU = config.get("tau", 0.99)
    KAPPA = config.get("kappa", 0.04)
    IW = config.get("interface_width", 5)
    SIGMA_LG = config.get("sigma_lg", None)
    sigma = surface_tension(KAPPA, IW, RHO_L, RHO_V)
    if SIGMA_LG is not None:
        sigma = SIGMA_LG          # prefer explicit value if present
    if sigma == 0:
        sigma = 1e-10             # guard against division by zero
    nu = (TAU - 0.5) / 3.0       # kinematic viscosity (LBM units)
    mu = nu * RHO_L               # dynamic viscosity
    R0 = config.get("R_ZERO", 27)
    inc_angle = config.get("inclination_angle", 0.0)

    cs = _get_chem_step_params(config)
    theta_a_pre = cs["ca_adv_pre"]
    theta_r_pre = cs["ca_rec_pre"]
    theta_a_post = cs["ca_adv_post"]
    theta_r_post = cs["ca_rec_post"]

    H_pre = hysteresis_parameter(theta_r_pre, theta_a_pre)
    mid_ca_pre = midpoint_angle(theta_a_pre, theta_r_pre)
    H_post = hysteresis_parameter(theta_r_post, theta_a_post)
    mid_ca_post = midpoint_angle(theta_a_post, theta_r_post)

    step_x = _get_step_x(config)

    # ── per-timestep accumulation ──
    iterations = []
    avg_ux_list = []
    avg_uy_list = []
    avg_x_list = []
    cl_left_list = []
    cl_right_list = []
    ca_left_list = []
    ca_right_list = []
    cm_x_list = []
    cm_y_list = []
    height_list = []
    base_diam_list = []

    for npz_file in npz_files:
        it = extract_iteration_number(npz_file)
        if it is None:
            continue
        npz_path = os.path.join(data_dir, npz_file)
        rho, ux, uy = extract_npz_data(npz_path)

        rho_2d = rho[:, :, 0, 0]
        mask = rho_2d > RHO_MEAN
        n_liq = float(jnp.sum(mask))
        if n_liq == 0:
            continue

        avg_ux = float(jnp.sum(jnp.where(mask, ux[:, :, 0], 0)) / n_liq)
        avg_uy = float(jnp.sum(jnp.where(mask, uy[:, :, 0], 0)) / n_liq)

        try:
            cl_l, cl_r = calculate_contact_line_location(rho, RHO_MEAN)
            ca_l, ca_r = calculate_contact_angle(rho, RHO_MEAN)
        except (IndexError, ValueError):
            continue

        cm_x, cm_y = calculate_center_of_mass(rho, RHO_MEAN)
        avg_x = calculate_avg_x_location(rho, RHO_MEAN, HALF_NX)
        height = calculate_droplet_height(rho, RHO_MEAN, cl_l, cl_r)
        base_d = calculate_droplet_base_diameter(cl_l, cl_r)

        iterations.append(it)
        avg_ux_list.append(avg_ux)
        avg_uy_list.append(avg_uy)
        avg_x_list.append(avg_x)
        cl_left_list.append(cl_l)
        cl_right_list.append(cl_r)
        ca_left_list.append(ca_l)
        ca_right_list.append(ca_r)
        cm_x_list.append(cm_x)
        cm_y_list.append(cm_y)
        height_list.append(height)
        base_diam_list.append(base_d)

    if not iterations:
        return None

    # ── convert to arrays ──
    iterations = np.array(iterations)
    avg_ux_arr = jnp.array(avg_ux_list)
    cl_left_arr = jnp.array(cl_left_list)
    cl_right_arr = jnp.array(cl_right_list)
    cm_x_arr = jnp.array(cm_x_list)
    height_arr = jnp.array(height_list)
    base_diam_arr = jnp.array(base_diam_list)

    # ── velocities & accelerations ──
    v_left, v_right, v_cm = calculate_velocities(
        cl_left_arr, cl_right_arr, cm_x_arr, PLOT_EVERY
    )
    a_left, a_right, a_cm = calculate_accelerations(
        v_left, v_right, v_cm, PLOT_EVERY
    )

    # ── Capillary numbers  Ca = μ v / σ ──
    Ca_drop = np.array(avg_ux_arr * ((TAU - 0.5) / 3)) / sigma
    Ca_CL_L = np.array(v_left * ((TAU - 0.5) / 3)) / sigma
    Ca_CL_R = np.array(v_right * ((TAU - 0.5) / 3)) / sigma
    Ca_cm = np.array(v_cm * ((TAU - 0.5) / 3)) / sigma

    # ── Weber numbers  We = ρ v² L / σ ──
    We_CL_L = np.array(v_left ** 2 * R0 * RHO_L) / sigma
    We_CL_R = np.array(v_right ** 2 * R0 * RHO_L) / sigma
    We_drop = np.array(avg_ux_arr ** 2 * R0 * RHO_L) / sigma
    We_cm = np.array(v_cm ** 2 * R0 * RHO_L) / sigma

    # ── normalised quantities ──
    it_max = iterations.max() if len(iterations) > 0 else 1
    iterations_norm = iterations / it_max
    avg_x_norm = np.array(avg_x_list) / R0
    height_norm = np.array(height_arr) / R0
    base_diam_norm = np.array(base_diam_arr) / R0

    # Use raw cm_x for regime classification (not normalised relative to grid centre)
    regime_labels = classify_regimes(np.array(cm_x_list), step_x, R0)

    # ── build DataFrame ──
    df = pd.DataFrame({
        "iteration": iterations,
        "iteration_norm": iterations_norm,
        "avg_ux": avg_ux_list,
        "avg_uy": avg_uy_list,
        "avg_x": avg_x_list,
        "x_over_R": avg_x_norm,
        "cl_left": cl_left_list,
        "cl_right": cl_right_list,
        "v_cl_left": np.array(v_left),
        "v_cl_right": np.array(v_right),
        "v_cm": np.array(v_cm),
        "a_cl_left": np.array(a_left),
        "a_cl_right": np.array(a_right),
        "a_cm": np.array(a_cm),
        "ca_left": ca_left_list,
        "ca_right": ca_right_list,
        "cm_x": cm_x_list,
        "cm_y": cm_y_list,
        "height": height_list,
        "base_diameter": base_diam_list,
        "height_norm": np.array(height_norm),
        "base_diameter_norm": np.array(base_diam_norm),
        "Ca_drop": Ca_drop,
        "Ca_cl_left": Ca_CL_L,
        "Ca_cl_right": Ca_CL_R,
        "Ca_cm": Ca_cm,
        "We_drop": We_drop,
        "We_cl_left": We_CL_L,
        "We_cl_right": We_CL_R,
        "We_cm": We_cm,
        "regime": regime_labels,
    })

    # ── simulation-level parameters (constant per run) ──
    df["sim_dir"] = sim_dir
    df["sim_name"] = os.path.basename(sim_dir)
    df["inclination_angle"] = inc_angle
    df["theta_a_pre"] = theta_a_pre
    df["theta_r_pre"] = theta_r_pre
    df["theta_a_post"] = theta_a_post
    df["theta_r_post"] = theta_r_post
    df["H"] = H_pre
    df["mid_ca"] = mid_ca_pre
    df["H_post"] = H_post
    df["mid_ca_post"] = mid_ca_post
    df["R0"] = R0
    df["sigma"] = sigma
    df["rho_l"] = RHO_L
    df["rho_v"] = RHO_V
    df["mu"] = mu
    df["tau"] = TAU

    return df


# ═══════════════════════════════════════════════════════════════════════════
# VI.  DISCOVER SIMULATIONS
# ═══════════════════════════════════════════════════════════════════════════

def discover_simulations(parent_dir: str) -> List[str]:
    """
    Walk *parent_dir* and return every directory that contains both
    config.json and data/.

    Directories whose name contains "init" (e.g. run_wetting_init) are
    skipped because they only initialise the system and do not contain
    meaningful simulation results.

    This handles the pipeline layout:
        <parent>/<date>/<pipeline_ts>/<stage_name>/config.json + data/
    """
    sim_dirs: List[str] = []
    for root, dirs, files in os.walk(parent_dir):
        if "config.json" in files and "data" in dirs:
            dirname = os.path.basename(root)
            if "init" in dirname.lower():
                continue
            sim_dirs.append(root)
    return sorted(sim_dirs)


# ═══════════════════════════════════════════════════════════════════════════
# VII.  FIGURE HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _save(fig, path):
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def _dual_axis_plot(ax_ca, x, ca, we, color, marker, label, ms=3):
    """
    Plot Ca on *ax_ca* (left y-axis, solid) and We on a twinned right
    y-axis (dashed).  The twin axis is created once and cached on the
    axes object.
    """
    ax_ca.plot(
        x, ca,
        marker=marker, ms=ms, color=color, label=label,
        linestyle="-", linewidth=0.8, alpha=0.85,
    )
    # Create or retrieve the twin axes for We
    if not hasattr(ax_ca, "_we_twin"):
        ax_we = ax_ca.twinx()
        ax_ca._we_twin = ax_we
        ax_we.set_ylabel("We", fontsize=14)
    else:
        ax_we = ax_ca._we_twin

    ax_we.plot(
        x, we,
        marker=marker, ms=ms, color=color, label=label + " (We)",
        linestyle="--", linewidth=0.8, alpha=0.55,
    )
    return ax_we


# ═══════════════════════════════════════════════════════════════════════════
# VIII.  FIGURE 1 – Inclined setup schematic
# ═══════════════════════════════════════════════════════════════════════════

def figure1_setup_schematic(output_dir: str):
    """
    Produce a schematic of the inclined surface with a chemical step,
    showing a droplet on each side with their contact-angle labels.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    # ── inclined surface ──
    angle_deg = 20
    angle_rad = math.radians(angle_deg)
    L = 10.0
    x0, y0 = 0.0, L * math.sin(angle_rad)
    x1, y1 = L * math.cos(angle_rad), 0.0
    ax.plot([x0, x1], [y0, y1], "k-", linewidth=2.5)

    # Horizontal reference line (dashed)
    ax.plot([x1 - 1.5, x1 + 0.5], [y1, y1], "k--", linewidth=0.8, alpha=0.5)
    # Arc for inclination angle
    arc_r = 1.5
    arc_angles = np.linspace(0, angle_rad, 30)
    ax.plot(
        x1 + arc_r * np.cos(np.pi - arc_angles),
        y1 + arc_r * np.sin(arc_angles) * 0,   # flat arc along horizontal
        color="gray", linewidth=0.8,
    )
    # Angle annotation
    ax.annotate(
        r"$\alpha$",
        xy=(x1 - 1.0, 0.25),
        fontsize=14, ha="center", color="k",
    )

    # ── chemical step (vertical line) ──
    frac = 0.50
    sx = x0 + frac * (x1 - x0)
    sy = y0 + frac * (y1 - y0)
    perp_len = 0.7
    dx_perp = -math.sin(angle_rad) * perp_len
    dy_perp = math.cos(angle_rad) * perp_len
    ax.plot(
        [sx - dx_perp * 0.3, sx + dx_perp],
        [sy - dy_perp * 0.3, sy + dy_perp],
        color="red", linewidth=2, linestyle="--", label="Chemical step",
    )
    ax.annotate(
        "Chemical\nstep",
        xy=(sx + dx_perp * 1.1, sy + dy_perp * 1.1),
        fontsize=10, color="red", ha="center",
    )

    # ── droplet on hydrophobic side (uphill / left of step) ──
    r1 = 0.50
    drop1_frac = 0.25
    dx1 = x0 + drop1_frac * (x1 - x0)
    dy1 = y0 + drop1_frac * (y1 - y0) + r1
    c1 = Circle(
        (dx1, dy1), r1,
        facecolor=COLORS[0], edgecolor="k", linewidth=0.8,
        alpha=0.55, label="Droplet (hydrophobic side)",
    )
    ax.add_patch(c1)
    ax.annotate(
        r"$\theta_{A}, \theta_{R}$",
        xy=(dx1, dy1 + r1 + 0.2), fontsize=11, ha="center", color=COLORS[0],
        fontweight="bold",
    )

    # ── droplet on hydrophilic side (downhill / right of step) ──
    r2 = 0.60
    drop2_frac = 0.75
    dx2 = x0 + drop2_frac * (x1 - x0)
    dy2 = y0 + drop2_frac * (y1 - y0) + r2 * 0.75
    c2 = Circle(
        (dx2, dy2), r2,
        facecolor=COLORS[1], edgecolor="k", linewidth=0.8,
        alpha=0.55, label="Droplet (hydrophilic side)",
    )
    ax.add_patch(c2)
    ax.annotate(
        r"$\theta'_{A}, \theta'_{R}$",
        xy=(dx2, dy2 + r2 + 0.2), fontsize=11, ha="center", color=COLORS[1],
        fontweight="bold",
    )

    # ── gravity arrow ──
    ax.annotate(
        "", xy=(x1 + 0.3, -0.6), xytext=(x1 + 0.3, 0.6),
        arrowprops=dict(arrowstyle="->", lw=1.5, color="gray"),
    )
    ax.text(x1 + 0.55, 0.0, r"$g$", fontsize=13, color="gray", ha="left")

    # ── region labels ──
    ax.text(
        dx1, dy1 - r1 - 0.5, "Hydrophobic", fontsize=9, ha="center",
        color=COLORS[0], fontstyle="italic",
    )
    ax.text(
        dx2, dy2 - r2 - 0.5, "Hydrophilic", fontsize=9, ha="center",
        color=COLORS[1], fontstyle="italic",
    )

    ax.set_xlim(-0.8, L * math.cos(angle_rad) + 1.2)
    ax.set_ylim(-1.2, L * math.sin(angle_rad) + 1.8)
    ax.set_aspect("equal")
    ax.legend(fontsize=9, loc="upper right")
    ax.set_title("Figure 1 – Inclined setup with chemical step", fontsize=14)
    ax.axis("off")

    _save(fig, os.path.join(output_dir, f"fig1_setup_schematic.{FIGURE_FORMAT}"))


# ═══════════════════════════════════════════════════════════════════════════
# IX.  FIGURE 2 – High-density combo figure
# ═══════════════════════════════════════════════════════════════════════════

def figure2_high_density_combo(df_all: pd.DataFrame, output_dir: str):
    """
    Select one high-density simulation and produce a 2×2 combo figure:
        (a) Ca vs normalised iteration
        (b) Contact angles (trailing & leading) vs normalised iteration
        (c) Ca vs x/R
        (d) Height & base diameter (normalised) vs x/R

    If a low-density run exists, overlay it in lighter colour for
    comparison.
    """
    rho_ls = df_all.groupby("sim_name")["rho_l"].first()
    median_rho = rho_ls.median()
    high_names = rho_ls[rho_ls >= median_rho].index.tolist()
    low_names = rho_ls[rho_ls < median_rho].index.tolist()

    if not high_names:
        print("  [skip] No high-density simulation found for Figure 2")
        return

    hi_name = high_names[0]
    df_hi = df_all[df_all["sim_name"] == hi_name].copy()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Load low-density comparison data if available
    df_lo = None
    if low_names:
        df_lo = df_all[df_all["sim_name"] == low_names[0]].copy()

    # ── (a) Ca vs normalised iteration ──
    ax = axes[0, 0]
    ax.scatter(
        df_hi["iteration_norm"], df_hi["Ca_cm"],
        s=30, color=COLORS[0], alpha=0.7, edgecolors='none',
        label=f"High dens. ({clean_name(hi_name)})",
    )
    if df_lo is not None:
        ax.scatter(
            df_lo["iteration_norm"], df_lo["Ca_cm"],
            s=20, color=COLORS[1], alpha=0.45, marker='s', edgecolors='none',
            label=f"Low dens. ({clean_name(low_names[0])})",
        )
    ax.set_xlabel(r"it / it$_{\mathrm{max}}$")
    ax.set_ylabel("Ca (centre of mass)")
    ax.legend(fontsize=8)
    ax.set_title("(a) Ca vs normalised iteration")
    ax.grid(True, alpha=0.3)

    # ── (b) contact angles vs normalised iteration ──
    ax = axes[0, 1]
    ax.scatter(
        df_hi["iteration_norm"], df_hi["ca_left"],
        s=30, color=COLORS[2], alpha=0.7, edgecolors='none',
        label="Trailing (left)",
    )
    ax.scatter(
        df_hi["iteration_norm"], df_hi["ca_right"],
        s=30, color=COLORS[3], alpha=0.7, marker='o', edgecolors='none',
        label="Leading (right)",
    )
    ax.set_xlabel(r"it / it$_{\mathrm{max}}$")
    ax.set_ylabel(r"Contact angle (°)")
    ax.legend(fontsize=8)
    ax.set_title("(b) Contact angles vs normalised iteration")
    ax.grid(True, alpha=0.3)

    # ── (c) Ca vs x/R ──
    ax = axes[1, 0]
    ax.scatter(
        df_hi["x_over_R"], df_hi["Ca_cm"],
        s=30, color=COLORS[0], alpha=0.7, edgecolors='none',
    )
    if df_lo is not None:
        ax.scatter(
            df_lo["x_over_R"], df_lo["Ca_cm"],
            s=20, color=COLORS[1], alpha=0.45, marker='s', edgecolors='none',
            label="Low density",
        )
        ax.legend(fontsize=8)
    ax.set_xlabel(r"$x / R_0$")
    ax.set_ylabel("Ca (centre of mass)")
    ax.set_title("(c) Ca vs normalised position")
    ax.grid(True, alpha=0.3)

    # ── (d) height & base diameter normalised vs x/R ──
    ax = axes[1, 1]
    ax.scatter(
        df_hi["x_over_R"], df_hi["height_norm"],
        s=30, color=COLORS[4], alpha=0.7, edgecolors='none',
        label="Height / R0",
    )
    ax.scatter(
        df_hi["x_over_R"], df_hi["base_diameter_norm"],
        s=30, color=COLORS[5], alpha=0.7, marker='s', edgecolors='none',
        label="Base diam. / R0",
    )
    ax.set_xlabel(r"$x / R_0$")
    ax.set_ylabel("Normalised length")
    ax.legend(fontsize=8)
    ax.set_title("(d) Droplet deformation vs position")
    ax.grid(True, alpha=0.3)

    fig.suptitle("Figure 2 – High-density case", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save(fig, os.path.join(output_dir, f"fig2_high_density_combo.{FIGURE_FORMAT}"))


# ═══════════════════════════════════════════════════════════════════════════
# X.  FIGURE 3 – Local Ca and We vs x/R (three subfigures)
# ═══════════════════════════════════════════════════════════════════════════

def _group_by_param(
    df_all: pd.DataFrame,
    vary: str,
    hold: List[str],
    tol_frac: float = 0.05,
) -> Dict[float, pd.DataFrame]:
    """
    Group simulations where *hold* parameters are approximately constant
    and *vary* changes.

    Returns {value_of_vary: sub-DataFrame}.
    """
    sim_params = (
        df_all.groupby("sim_name")[[vary] + hold]
        .first()
        .reset_index()
    )
    if sim_params.empty:
        return {}

    # Round the held parameters and find the most-populated combination
    rounded = {}
    for h in hold:
        rounded[h] = sim_params[h].round(2)

    if hold:
        combo_key = [rounded[h] for h in hold]
        sim_params["_combo"] = list(zip(*combo_key)) if len(hold) > 1 else combo_key[0]
        combo_counts = sim_params["_combo"].value_counts()
        best_combo = combo_counts.idxmax()
        # Filter to simulations matching that combination within tolerance
        mask = pd.Series(True, index=sim_params.index)
        if len(hold) > 1:
            for i, h in enumerate(hold):
                ref = best_combo[i]
                mask &= (sim_params[h] - ref).abs() <= tol_frac * max(1, abs(ref))
        else:
            ref = best_combo
            mask &= (sim_params[hold[0]] - ref).abs() <= tol_frac * max(1, abs(ref))
        sim_params = sim_params[mask]

    groups = {}
    for _, row in sim_params.iterrows():
        sn = row["sim_name"]
        val = round(float(row[vary]), 4)
        groups[val] = df_all[df_all["sim_name"] == sn]
    return dict(sorted(groups.items()))


def figure3_local_Ca_We(df_all: pd.DataFrame, output_dir: str):
    """
    Three subfigures, each with dual y-axes (Ca left, We right) vs x/R.
        (a) vary H
        (b) vary inclination angle
        (c) vary (θ_A + θ_R)/2
    """
    vary_keys = ["H", "inclination_angle", "mid_ca"]
    subfig_labels = ["(a)", "(b)", "(c)"]

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    for idx, vary in enumerate(vary_keys):
        hold = [k for k in vary_keys if k != vary]
        groups = _group_by_param(df_all, vary, hold)

        ax_ca = axes[idx]

        for gi, (val, gdf) in enumerate(groups.items()):
            c = COLORS[gi % len(COLORS)]
            m = MARKERS[gi % len(MARKERS)]
            lbl = f"{_PARAM_INFO[vary]['label']}={val:.3g}"
            _dual_axis_plot(
                ax_ca,
                gdf["x_over_R"].values,
                gdf["Ca_cm"].values,
                gdf["We_cm"].values,
                color=c, marker=m, label=lbl, ms=3,
            )

        ax_ca.set_xlabel(r"$x / R_0$")
        ax_ca.set_ylabel("Ca")
        ax_ca.set_title(f"{subfig_labels[idx]} vary {_PARAM_INFO[vary]['label']}")
        ax_ca.grid(True, alpha=0.3)

        # Build a combined legend from both axes
        lines_ca, labels_ca = ax_ca.get_legend_handles_labels()
        if hasattr(ax_ca, "_we_twin"):
            lines_we, labels_we = ax_ca._we_twin.get_legend_handles_labels()
        else:
            lines_we, labels_we = [], []
        ax_ca.legend(
            lines_ca + lines_we, labels_ca + labels_we,
            fontsize=7, loc="upper left",
        )

    fig.suptitle(
        "Figure 3 – Local Ca and We vs position", fontsize=15,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(fig, os.path.join(output_dir, f"fig3_local_Ca_We.{FIGURE_FORMAT}"))


# ═══════════════════════════════════════════════════════════════════════════
# XI.  FIGURE 4 – Regime-averaged Ca and We vs parameters
# ═══════════════════════════════════════════════════════════════════════════

def _regime_averages(df_all: pd.DataFrame) -> pd.DataFrame:
    """Compute per-simulation, per-regime averages of Ca and We."""
    rows: List[Dict] = []
    for (sn, reg), g in df_all.groupby(["sim_name", "regime"]):
        rows.append(dict(
            sim_name=sn,
            regime=int(reg),
            regime_name=REGIME_NAMES.get(int(reg), "?"),
            Ca_cm_mean=g["Ca_cm"].mean(),
            Ca_cm_std=g["Ca_cm"].std(),
            We_cm_mean=g["We_cm"].mean(),
            We_cm_std=g["We_cm"].std(),
            H=g["H"].iloc[0],
            inclination_angle=g["inclination_angle"].iloc[0],
            mid_ca=g["mid_ca"].iloc[0],
            theta_a_pre=g["theta_a_pre"].iloc[0],
            theta_r_pre=g["theta_r_pre"].iloc[0],
            theta_a_post=g["theta_a_post"].iloc[0],
            theta_r_post=g["theta_r_post"].iloc[0],
            rho_l=g["rho_l"].iloc[0],
        ))
    return pd.DataFrame(rows)


def figure4_regime_averaged(df_all: pd.DataFrame, output_dir: str):
    """
    Three subfigures: regime-averaged Ca (left axis) and We (right axis)
    plotted against each varied parameter.
    """
    vary_keys = ["H", "inclination_angle", "mid_ca"]
    subfig_labels = ["(a)", "(b)", "(c)"]
    regime_base_colors = {0: 0, 1: 3, 2: 6}
    regime_markers = {0: "o", 1: "s", 2: "D"}

    ra = _regime_averages(df_all)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    for idx, vary in enumerate(vary_keys):
        ax = axes[idx]
        ax_we = ax.twinx()

        for reg in sorted(ra["regime"].unique()):
            sub = ra[ra["regime"] == reg].sort_values(vary)
            c = COLORS[regime_base_colors.get(reg, 0) % len(COLORS)]
            m = regime_markers.get(reg, "o")
            lbl = REGIME_NAMES.get(reg, "?")

            ax.errorbar(
                sub[vary], sub["Ca_cm_mean"], yerr=sub["Ca_cm_std"],
                marker=m, ms=6, color=c, linestyle="-", capsize=3,
                label=f"Ca – {lbl}",
            )
            ax_we.errorbar(
                sub[vary], sub["We_cm_mean"], yerr=sub["We_cm_std"],
                marker=m, ms=6, color=c, linestyle="--", alpha=0.55, capsize=3,
                label=f"We – {lbl}",
            )

        ax.set_xlabel(_PARAM_INFO[vary]["label"])
        ax.set_ylabel("Ca (mean)")
        ax_we.set_ylabel("We (mean)")
        ax.set_title(f"{subfig_labels[idx]} vary {_PARAM_INFO[vary]['label']}")
        ax.grid(True, alpha=0.3)

        # Combined legend
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax_we.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="best")

    fig.suptitle(
        "Figure 4 – Regime-averaged Ca and We vs parameters", fontsize=15,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(fig, os.path.join(output_dir, f"fig4_regime_averaged.{FIGURE_FORMAT}"))


# ═══════════════════════════════════════════════════════════════════════════
# XII.  FIGURE 5 – Phase diagrams
# ═══════════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════════
# XIII.  MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

def main():
    """
    Main entry point.  Discovers simulations, processes every run,
    saves combined CSV files, and generates all five paper figures.
    """
    parent_dir = input(
        "Enter parent directory containing simulation results (absolute path): "
    ).strip()

    if not os.path.isdir(parent_dir):
        print(f"Directory not found: {parent_dir}")
        return

    output_dir = os.path.join(parent_dir, "Paper_figures")
    os.makedirs(output_dir, exist_ok=True)

    # ── discover & process ──
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

    # ── save master CSV (all timesteps, all simulations) ──
    csv_path = os.path.join(output_dir, "all_simulations_data.csv")
    df_all.to_csv(csv_path, index=False)
    print(f"\nSaved combined CSV → {csv_path}  ({len(df_all)} rows)")

    # ── save regime-averaged CSV ──
    ra = _regime_averages(df_all)
    ra_path = os.path.join(output_dir, "regime_averages.csv")
    ra.to_csv(ra_path, index=False)
    print(f"Saved regime averages → {ra_path}")

    # ── save per-simulation summary CSV ──
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
            rho_l=g["rho_l"].iloc[0],
            rho_v=g["rho_v"].iloc[0],
            sigma=g["sigma"].iloc[0],
            R0=g["R0"].iloc[0],
            Ca_cm_mean=g["Ca_cm"].mean(),
            We_cm_mean=g["We_cm"].mean(),
            total_displacement_R0=abs(g["x_over_R"].iloc[-1] - g["x_over_R"].iloc[0]),
            outcome=classify_outcome(g),
            n_timesteps=len(g),
        ))
    summary = pd.DataFrame(summary_rows)
    summary_path = os.path.join(output_dir, "simulation_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"Saved simulation summary → {summary_path}")

    # ── generate figures ──
    print("\nGenerating figures …")
    figure1_setup_schematic(output_dir)
    figure2_high_density_combo(df_all, output_dir)
    figure3_local_Ca_We(df_all, output_dir)
    figure4_regime_averaged(df_all, output_dir)
    figure5_phase_diagrams(df_all, output_dir)

    print(f"\nAll outputs saved to {output_dir}")


if __name__ == "__main__":
    main()

