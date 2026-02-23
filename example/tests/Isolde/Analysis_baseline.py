import os
import json
import jax.numpy as jnp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from typing import Dict, Tuple, Optional
import math
from matplotlib.colors import TABLEAU_COLORS


def load_config(sim_dir: str) -> Dict:
    """Load configuration from config.json"""
    config_path = os.path.join(sim_dir, 'config.json')
    with open(config_path, 'r') as f:
        return json.load(f)


def clean_name(s: str) -> str:
    parts = s.split("_", 1)
    if parts[0].replace("-", "").isdigit() and len(parts) == 2:
        # looks like a timestamp prefix, drop it
        title_part = parts[1]
    else:
        # no timestamp prefix, keep whole string
        title_part = s
    readable = title_part.replace("_", " ")
    return readable.capitalize()


def extract_npz_data(npz_file: str) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Extract rho, u_x, u_y from npz file"""
    data = jnp.load(npz_file)
    rho = data['rho']
    u = data['u']
    u_x = u[..., 0]
    u_y = u[..., 1]
    return rho, u_x, u_y


def calculate_contact_angle(rho: jnp.ndarray, RHO_MEAN: float) -> Tuple[float, float]:
    """Calculate contact angles left and right"""
    array_i_j0 = rho[:, 1, 0, 0]
    array_i_jpos1 = rho[:, 2, 0, 0]

    mask_i_j0 = array_i_j0 < RHO_MEAN
    mask_i_jpos1 = array_i_jpos1 < RHO_MEAN

    mask_int_i_j0 = jnp.array(mask_i_j0, dtype=int)
    mask_int_i_jpos1 = jnp.array(mask_i_jpos1, dtype=int)

    diff_mask_i_j0 = jnp.diff(mask_int_i_j0)
    diff_mask_i_jpos1 = jnp.diff(mask_int_i_jpos1)

    transition_index_left_i_j0 = jnp.where(diff_mask_i_j0 == -1)[0]
    transition_index_left_i_jpos1 = jnp.where(diff_mask_i_jpos1 == -1)[0]
    transition_index_right_i_j0 = jnp.where(diff_mask_i_j0 == 1)[0] + 1
    transition_index_right_i_jpos1 = jnp.where(diff_mask_i_jpos1 == 1)[0] + 1

    index_left_i_j0 = int(transition_index_left_i_j0[0])
    index_left_i_jpos1 = int(transition_index_left_i_jpos1[0])
    index_right_i_j0 = int(transition_index_right_i_j0[0])
    index_right_i_jpos1 = int(transition_index_right_i_jpos1[0])

    x_val_left_j0 = index_left_i_j0 + (RHO_MEAN - array_i_j0[index_left_i_j0]) / (
            array_i_j0[index_left_i_j0 + 1] - array_i_j0[index_left_i_j0])
    x_val_left_jpos1 = index_left_i_jpos1 + (RHO_MEAN - array_i_jpos1[index_left_i_jpos1]) / (
            array_i_jpos1[index_left_i_jpos1 + 1] - array_i_jpos1[index_left_i_jpos1])
    x_val_right_j0 = index_right_i_j0 - (RHO_MEAN - array_i_j0[index_right_i_j0]) / (
            array_i_j0[index_right_i_j0 - 1] - array_i_j0[index_right_i_j0])
    x_val_right_jpos1 = index_right_i_jpos1 - (RHO_MEAN - array_i_jpos1[index_right_i_jpos1]) / (
            array_i_jpos1[index_right_i_jpos1 - 1] - array_i_jpos1[index_right_i_jpos1])

    contact_angle_left = jnp.rad2deg(math.pi / 2 + jnp.arctan(x_val_left_j0 - x_val_left_jpos1))
    contact_angle_right = jnp.rad2deg(math.pi / 2 + jnp.arctan(x_val_right_jpos1 - x_val_right_j0))

    return contact_angle_left, contact_angle_right


def calculate_contact_line_location(rho: jnp.ndarray, RHO_MEAN: float) -> Tuple[float, float]:
    """Calculate contact line locations left and right"""
    array_i_j0 = rho[:, 1, 0, 0]

    mask_i_j0 = array_i_j0 < RHO_MEAN
    mask_int_i_j0 = jnp.array(mask_i_j0, dtype=int)
    diff_mask_i_j0 = jnp.diff(mask_int_i_j0)

    transition_index_left_i_j0 = jnp.where(diff_mask_i_j0 == -1)[0]
    transition_index_right_i_j0 = jnp.where(diff_mask_i_j0 == 1)[0] + 1

    index_left_i_j0 = int(transition_index_left_i_j0[0])
    index_right_i_j0 = int(transition_index_right_i_j0[0])

    x_val_left_j0 = index_left_i_j0 + (RHO_MEAN - array_i_j0[index_left_i_j0]) / (
            array_i_j0[index_left_i_j0 + 1] - array_i_j0[index_left_i_j0])
    x_val_right_j0 = index_right_i_j0 - (RHO_MEAN - array_i_j0[index_right_i_j0]) / (
            array_i_j0[index_right_i_j0 - 1] - array_i_j0[index_right_i_j0])

    return x_val_left_j0, x_val_right_j0


def calculate_center_of_mass(rho: jnp.ndarray, rho_mean: float) -> Tuple[float, float]:
    """Calculate center of mass of droplet"""
    rho_2d = rho[:, :, 0, 0]
    mask = rho_2d > rho_mean
    x_indices, y_indices = jnp.indices(rho_2d.shape)

    total_mass = jnp.sum(mask * rho_2d)
    cm_x = jnp.sum(x_indices * mask * rho_2d) / total_mass
    cm_y = jnp.sum(y_indices * mask * rho_2d) / total_mass

    return cm_x, cm_y


def calculate_droplet_base_diameter(cl_left, cl_right):
    """Calculate droplet base diameter"""
    return cl_right - cl_left


def calculate_droplet_height(rho: jnp.ndarray, rho_mean: float) -> float:
    """
    Calculate droplet height using interpolation, similar to contact line calculation.
    Returns height as a float.
    """
    # Extract 2D slice along the vertical axis (y-axis)
    rho2d = rho[:, :, 0, 0]

    # Find the x locations where rho crosses rho_mean (mask)
    mask = rho2d > rho_mean

    # If no liquid is detected, return 0
    if not jnp.any(mask):
        return 0.0

    # Find top and bottom indices of the liquid column for each x
    y_indices = jnp.arange(rho2d.shape[1])
    heights = []

    for x in range(rho2d.shape[0]):
        column = rho2d[x, :]
        liquid_mask = column > rho_mean
        if jnp.any(liquid_mask):
            # bottom index
            index_bottom = jnp.where(liquid_mask)[0][0]
            # top index
            index_top = jnp.where(liquid_mask)[0][-1]

            # Linear interpolation at bottom
            if index_bottom > 0:
                y_bottom = index_bottom - (rho_mean - column[index_bottom]) / (
                    column[index_bottom - 1] - column[index_bottom]
                )
            else:
                y_bottom = index_bottom

            # Linear interpolation at top
            if index_top < rho2d.shape[1] - 1:
                y_top = index_top + (rho_mean - column[index_top]) / (
                    column[index_top + 1] - column[index_top]
                )
            else:
                y_top = index_top

            heights.append(y_top - y_bottom)

    # Take the maximum height among all columns
    if heights:
        return float(jnp.max(jnp.array(heights)))
    else:
        return 0.0


def calculate_avg_x_location(rho, rho_mean, half_nx):
    rho2d = rho[:, :, 0, 0]
    mask = rho2d > rho_mean
    x_indices = jnp.arange(rho2d.shape[0]) - half_nx
    x_indices_2d = jnp.expand_dims(x_indices, axis=1)
    masked_x = jnp.where(mask, x_indices_2d, 0)
    avg_x_location = jnp.sum(masked_x) / jnp.sum(mask)
    return avg_x_location


def calculate_velocities(contact_line_left: jnp.ndarray, contact_line_right: jnp.ndarray,
                         centre_of_mass_x: jnp.ndarray, plot_every: int) -> Tuple[
    jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Calculate velocities from position time series"""
    x_t_left = contact_line_left
    x_t_right = contact_line_right
    x_t_cm = centre_of_mass_x

    x_tneg1_left_ = jnp.roll(x_t_left, shift=1)
    x_tneg1_left = x_tneg1_left_.at[0].set(x_t_left[0])

    x_tneg1_right_ = jnp.roll(x_t_right, shift=1)
    x_tneg1_right = x_tneg1_right_.at[0].set(x_t_right[0])

    x_tneg1_cm_ = jnp.roll(x_t_cm, shift=1)
    x_tneg1_cm = x_tneg1_cm_.at[0].set(x_t_cm[0])

    v_left = (x_t_left - x_tneg1_left) / plot_every
    v_right = (x_t_right - x_tneg1_right) / plot_every
    v_cm = (x_t_cm - x_tneg1_cm) / plot_every

    return v_left, v_right, v_cm

def calculate_accelerations(v_left: jnp.ndarray, v_right: jnp.ndarray, v_cm: jnp.ndarray, plot_every: int) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Calculate accelerations from velocity time series"""

    v_tneg1_left_ = jnp.roll(v_left, shift=1)
    v_tneg1_left = v_tneg1_left_.at[0].set(v_left[0])

    v_tneg1_right_ = jnp.roll(v_right, shift=1)
    v_tneg1_right = v_tneg1_right_.at[0].set(v_right[0])

    v_tneg1_cm_ = jnp.roll(v_cm, shift=1)
    v_tneg1_cm = v_tneg1_cm_.at[0].set(v_cm[0])

    a_left = (v_left - v_tneg1_left) / plot_every
    a_right = (v_right - v_tneg1_right) / plot_every
    a_cm = (v_cm - v_tneg1_cm) / plot_every

    return a_left, a_right, a_cm

def electrowetting_number(U0, eps0, eps_r, gamma):
    return (eps0 * eps_r * U0**2) / (2 * gamma * 1) #d= 1 in our simulation

def surface_tension(kappa,w,rho_l, rho_g):
    return 2/3*kappa*np.abs(rho_l-rho_g)**2

def extract_iteration_number(filename: str) -> Optional[int]:
    """Extract iteration number from npz filename"""
    match = re.search(r'timestep_(\d+)', filename)
    if match:
        return int(match.group(1))
    return None

def process_single_simulation(sim_dir: str, output_dir: str) -> str:
    """Process a single simulation using data_plots logic"""
    config = load_config(sim_dir)

    # Create analysis output directory
    analysis_dir = os.path.join(output_dir, 'Analysis')
    os.makedirs(analysis_dir, exist_ok=True)

    # Extract data directory
    data_dir = os.path.join(sim_dir, 'data')
    if not os.path.exists(data_dir):
        return None

    # Get all npz files
    npz_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.npz')],
                       key=lambda x: extract_iteration_number(x) or 0)

    if not npz_files:
        return None

    # Extract constants
    RHO_L = config.get('rho_l', 1.0)
    RHO_G = config.get('rho_v', 0.001)
    RHO_MEAN = (config.get('rho_l', 1.0) + config.get('rho_v', 0.001)) / 2
    NX = config.get('grid_shape', [256, 256])
    HALF_NX = NX[0] // 2
    PLOT_EVERY = config.get('plot_every', 1)
    EPS0 = 8.854e-12
    EPS_R = 1
    KAPPA = config.get('kappa', 0.4)
    W = config.get('interface_width', 5)

    gamma = surface_tension(KAPPA, W, RHO_L, RHO_G)

    # Process all timesteps
    iterations = []
    avg_u_x_list = []
    avg_u_y_list = []
    avg_x_location_list = []
    contact_line_left_list = []
    contact_line_right_list = []
    contact_angle_left_list = []
    contact_angle_right_list = []
    cm_x_list = []
    cm_y_list = []
    height_list = []
    base_diameter_list = []

    for npz_file in npz_files:
        npz_path = os.path.join(data_dir, npz_file)
        rho, u_x, u_y = extract_npz_data(npz_path)

        iteration = extract_iteration_number(npz_file)
        if iteration is None:
            continue

        # Calculate metrics
        rho_2d = rho[:, :, 0, 0]
        mask = rho_2d > RHO_MEAN
        masked_u_x = jnp.where(mask, u_x[:, :, 0], 0)
        masked_u_y = jnp.where(mask, u_y[:, :, 0], 0)

        avg_u_x = jnp.sum(masked_u_x) / jnp.sum(mask) if jnp.sum(mask) > 0 else 0
        avg_u_y = jnp.sum(masked_u_y) / jnp.sum(mask) if jnp.sum(mask) > 0 else 0
        avg_x_loc = calculate_avg_x_location(rho, RHO_MEAN, HALF_NX)
        cl_left, cl_right = calculate_contact_line_location(rho, RHO_MEAN)
        ca_left, ca_right = calculate_contact_angle(rho, RHO_MEAN)
        cm_x, cm_y = calculate_center_of_mass(rho, RHO_MEAN)
        height = calculate_droplet_height(rho, RHO_MEAN)
        base_diameter = calculate_droplet_base_diameter(cl_left, cl_right)

        iterations.append(iteration)
        avg_u_x_list.append(avg_u_x)
        avg_u_y_list.append(avg_u_y)
        avg_x_location_list.append(avg_x_loc)
        contact_line_left_list.append(cl_left)
        contact_line_right_list.append(cl_right)
        contact_angle_left_list.append(ca_left)
        contact_angle_right_list.append(ca_right)
        cm_x_list.append(cm_x)
        cm_y_list.append(cm_y)
        height_list.append(height)
        base_diameter_list.append(base_diameter)

    # Calculate derived quantities
    if_max_iteration = max(iterations) if iterations else 1
    iterations_norm = jnp.array(iterations) / if_max_iteration
    avg_x_loc_norm = jnp.array(avg_x_location_list) / config.get('R_ZERO', 27)

    v_left, v_right, v_cm = calculate_velocities(jnp.array(contact_line_left_list),
                                                 jnp.array(contact_line_right_list),
                                                 jnp.array(cm_x_list), PLOT_EVERY)
    a_left, a_right, a_cm = calculate_accelerations(v_left, v_right, v_cm, PLOT_EVERY)

    TAU = config.get('tau', 1.0)
    SIGMA_LG = config.get('sigma_lg', 0.1)

    Ca_drop = (jnp.array(avg_u_x_list) * ((TAU - 0.5) / 3)) / SIGMA_LG
    Ca_CL_L = (v_left * ((TAU - 0.5) / 3)) / SIGMA_LG
    Ca_CL_R = (v_right * ((TAU - 0.5) / 3)) / SIGMA_LG
    Ca_drop_cm = (v_cm * ((TAU - 0.5) / 3)) / SIGMA_LG

    R0 = config.get('R_ZERO', 27)
    height_norm = jnp.array(height_list) / R0
    base_diameter_norm = jnp.array(base_diameter_list) / R0

    INCLINATION_ANGLE = config.get('inclination_angle', 0)
    Ca_norm_inc = Ca_drop / jnp.sin(jnp.deg2rad(INCLINATION_ANGLE)) if INCLINATION_ANGLE > 0 else Ca_drop

    # Save to CSV
    csv_output = os.path.join(analysis_dir, 'processed_data.csv')
    output_data = pd.DataFrame({
        'Iteration': iterations,
        'Normalised iterations': iterations_norm,
        'Average X Location': avg_x_location_list,
        'Average X location normalised': avg_x_loc_norm,
        'Average X Velocity': avg_u_x_list,
        'Average Y Velocity': avg_u_y_list,
        'Contact line left': contact_line_left_list,
        'Contact line right': contact_line_right_list,
        'Contact line velocity left': v_left,
        'Contact line velocity right': v_right,
        'Centre of mass velocity': v_cm,
        'Contact line acceleration left': a_left,
        'Contact line acceleration right': a_right,
        'Centre of mass acceleration': a_cm,
        'Contact angle left': contact_angle_left_list,
        'Contact angle right': contact_angle_right_list,
        'Center of Mass X': cm_x_list,
        'Center of Mass Y': cm_y_list,
        'Ca': Ca_drop,
        'Ca left contact line': Ca_CL_L,
        'Ca right contact line': Ca_CL_R,
        'Ca centre of mass': Ca_drop_cm,
        'Ca normalised': Ca_norm_inc,
        'Height': height_list,
        'Base diameter': base_diameter_list,
        'Height normalised': height_norm,
        'Base diameter normalised': base_diameter_norm,

    })

    output_data.to_csv(csv_output, index=False)
    print(f"Processed {sim_dir}")
    print(f"  Saved to {csv_output}")

    # ------- save information of this simulation to global results file to compare multiple simulations later -----
    # Skip chemical step runs
    simulation_name = os.path.basename(sim_dir)
    if "chem" in simulation_name.lower():
        print(f"Skipping chemical step run: {simulation_name}")
        return None

    global_dir = os.path.join("/Users/isoldeholweg/TUD_LBM/results", "Global_results_2c")
    os.makedirs(global_dir, exist_ok=True)
    global_path = os.path.join(global_dir, "droplet_deformation_vs_U0.csv")

    U0 = config.get('U_0')

    eta = electrowetting_number(U0,EPS0,EPS_R,gamma)
    print('Electrowetting number:', eta)
    # Load existing global summary if it exists
    if os.path.exists(global_path):
        df_global = pd.read_csv(global_path)
        # Ensure required columns exist
        for col in ["simulation", "U0", 'eta', "Iteration", "X_avg_R0", "Height", "Base_diameter", "v_cm", "a_cm", "Ca", "CA_left", "CA_right", "CA_left_mean", "CA_right_mean", 'eta']:
            if col not in df_global.columns:
                df_global[col] = np.nan
    else:
        df_global = pd.DataFrame(columns=[
            "simulation", "U0", "Iteration", "X_avg_R0", 'eta',
            "Height", "Base_diameter", "v_cm", "a_cm", "Ca", "CA_left", "CA_right", "CA_left_mean", "CA_right_mean", 'eta',
        ])

    # Normalise positions and dimensions
    R0 = config.get('R_ZERO', 27)
    X_avg_norm = jnp.array(avg_x_location_list) / R0

    v_cm_list = v_cm
    a_cm_list = a_cm
    Ca_list = Ca_drop

    CA_left_mean = np.mean(contact_angle_left_list)
    CA_right_mean = np.mean(contact_angle_right_list)

    # Add rows for each iteration§
    for i, iteration in enumerate(iterations):
        # Skip duplicates
        if ((df_global["simulation"] == simulation_name) &
            (df_global["U0"] == U0) &
            (df_global["Iteration"] == iteration)).any():
            continue

        df_global.loc[len(df_global)] = {
            "simulation": simulation_name,
            "U0": U0,
            "Iteration": iteration,
            "X_avg_R0": float(X_avg_norm[i]),
            "Height": float(height_list[i]),
            "Base_diameter": float(base_diameter_list[i]),
            "v_cm": float(v_cm_list[i]),
            "a_cm": float(a_cm_list[i]),
            "Ca": float(Ca_list[i]),
            "CA_left": contact_angle_left_list[i],
            "CA_right": contact_angle_right_list[i],
            "CA_left_mean": float(CA_left_mean),
            "CA_right_mean": float(CA_right_mean),
            'eta': eta,
        }

    # Save updated global CSV
    df_global.to_csv(global_path, index=False)
    print(f"Updated global summary at {global_path}")
    # ---------------------------------------------------------------------------

    return csv_output


def compare_simulations(parent_dir: str, comparison_output_dir: str):
    """Compare multiple simulation results.
       Makes:
       (1) normal plots with all simulations
       (2) separate plots ONLY for baseline chemical-step runs (U0 = 0)
    """

    os.makedirs(comparison_output_dir, exist_ok=True)

    # -------------------------------------------------------
    # Find all simulation directories
    # -------------------------------------------------------
    simulation_dirs = []
    for root, dirs, files in os.walk(parent_dir):
        if 'Analysis' in dirs:
            simulation_dirs.append(root)

    if not simulation_dirs:
        print("No processed simulations found for comparison")
        return

    # -------------------------------------------------------
    # Load all processed data
    # -------------------------------------------------------
    simulation_data = {}
    for sim_dir in simulation_dirs:
        sim_name = os.path.basename(sim_dir)
        csv_path = os.path.join(sim_dir, 'Analysis', 'processed_data.csv')
        if os.path.exists(csv_path):
            config = load_config(sim_dir)
            data = pd.read_csv(csv_path)
            simulation_data[sim_name] = {'data': data, 'config': config}

    if not simulation_data:
        print("No simulation data found for comparison")
        return

    # -------------------------------------------------------
    # Split into electric and baseline chemical runs
    # -------------------------------------------------------
    electric_sims = {}
    baseline_sims = {}

    for sim_name, sim_info in simulation_data.items():

        name_lower = sim_name.lower()

        # Chemical step runs (baseline, no electric forcing)
        if "chem" in name_lower:
            baseline_sims[sim_name] = sim_info

        # Everything else = electric-field runs
        else:
            electric_sims[sim_name] = sim_info

    print(f"Found {len(electric_sims)} electric-field simulations")
    print(f"Found {len(baseline_sims)} baseline chemical-step simulations")

    # -------------------------------------------------------
    # Plot configuration
    # -------------------------------------------------------
    plot_configs = [
        {
            "x_column": "Normalised iterations",
            "y_column": "Ca",
            "title": "Ca vs Iteration",
            "xlabel": r"$\mathrm{it}/\mathrm{it}_{\mathrm{max}}$",
            "ylabel": r"$\mathrm{Ca}$",
            "x_tick_spacing": 0.1,
            "filename": "01_comparison_Ca_vs_It.png",
        },
        {
            "x_column": "Average X location normalised",
            "y_column": "Ca",
            "title": "Ca vs normalised average X location",
            "xlabel": r"$\frac{X_{\mathrm{avg}}}{R_0}$",
            "ylabel": r"$\mathrm{Ca}$",
            "x_tick_spacing": 1,
            "filename": "02_comparison_Ca_vs_X_avg.png",
        },
        {
            "x_column": "Normalised iterations",
            "y_columns": ["Ca left contact line", "Ca right contact line"],
            "markers": [".", "o"],
            "labels": ["Trailing", "Leading"],
            "title": "Ca contact line vs Iteration",
            "xlabel": r"$\mathrm{it}/\mathrm{it}_{\mathrm{max}}$",
            "ylabel": r"$\mathrm{Ca}$",
            "x_tick_spacing": 0.1,
            "filename": "03_comparison_contact_line_motion.png",
        },
        {
            "x_column": "Average X location normalised",
            "y_columns": ["Contact line velocity left", "Contact line velocity right"],
            "markers": [".", "o"],
            "labels": ["Trailing", "Leading"],
            "title": "Contact line velocity vs normalised average X location",
            "xlabel": r"$\frac{X_{\mathrm{avg}}}{R_0}$",
            "ylabel": "Contact Line Velocity",
            "x_tick_spacing": 1,
            "filename": "04_comparison_contact_line_motion_vs_X_avg.png",
        },
        {
            "x_column": "Average X location normalised",
            "y_columns": ["Height"],
            "markers": ["o"],
            "labels": ["Droplet height"],
            "title": "Droplet height vs Normalized X Location",
            "xlabel": r"$\frac{X_{\mathrm{avg}}}{R_0}$",
            "ylabel": r"$H$",
            "x_tick_spacing": 1,
            "filename": "09_comparison_droplet_height_vs_x_location.png",
        },
        {
            "x_column": "Average X location normalised",
            "y_columns": ["Base diameter"],
            "markers": ["o"],
            "labels": ["Droplet base diameter"],
            "title": "Droplet base diameter vs Normalized X Location",
            "xlabel": r"$\frac{X_{\mathrm{avg}}}{R_0}$",
            "ylabel": r"$D$",
            "x_tick_spacing": 1,
            "filename": "10_comparison_droplet_base_diameter_vs_x_location.png",
        },
        {
            "x_column": "Normalised iterations",
            "y_columns": ["Contact angle left", "Contact angle right"],
            "markers": [".", "o"],
            "labels": ["Trailing", "Leading"],
            "title": "Contact Angle vs Iterations",
            "xlabel": r"$\mathrm{it}/\mathrm{it}_{\mathrm{max}}$",
            "ylabel": "Contact Angle (degrees)",
            "x_tick_spacing": 0.1,
            "filename": "07_comparison_contact_angles_vs_iterations.png",
        },
        {
            "x_column": "Average X location normalised",
            "y_columns": ["Contact angle left", "Contact angle right"],
            "markers": [".", "o"],
            "labels": ["Trailing", "Leading"],
            "title": "Contact Angle vs Normalized X Location",
            "xlabel": r"$\frac{X_{\mathrm{avg}}}{R_0}$",
            "ylabel": r"$\theta$",
            "x_tick_spacing": 1,
            "filename": "08_comparison_contact_angles_vs_x_location.png",
        },
    ]

    colors = list(TABLEAU_COLORS.values())
    markers = ["o", "s", "D", "^", "v", "<", ">", "p", "*", "h", "H", "+", "x"]

    plt.rcParams['text.usetex'] = True

    # -------------------------------------------------------
    # Helper function for plotting
    # -------------------------------------------------------
    MARKER_SIZE = 6
    MARKER_EDGE_WIDTH = 1.5
    MARKERS = ["o", "s", "^", "d", "v", "x", "+", "*"]

    def make_plots(sim_dict, out_dir, tag):
        """Make comparison plots with improved style"""
        if not sim_dict:
            print(f"No simulations to plot for {tag}")
            return

        sorted_sims = sorted(
            sim_dict.items(),
            key=lambda x: x[1]['config'].get('advancing_ca', 0)
        )

        colors = list(TABLEAU_COLORS.values())

        for config in plot_configs:
            fig, ax = plt.subplots(figsize=(10, 6))  # natural figure size

            for i, (sim_name, sim_info) in enumerate(sorted_sims):
                color = colors[i % len(colors)]
                marker = MARKERS[i % len(MARKERS)]
                label = clean_name(sim_name)
                data = sim_info['data']

                if 'y_columns' in config:
                    for j, y_column in enumerate(config['y_columns']):
                        if y_column in data.columns:
                            ax.plot(
                                data[config['x_column']], data[y_column],
                                linestyle="None",
                                marker=marker,
                                markersize=MARKER_SIZE,
                                markerfacecolor="none",
                                markeredgewidth=MARKER_EDGE_WIDTH,
                                color=color,
                                alpha=1.0 if j == 0 else 0.8,
                                label=f"{label} ({config['labels'][j]})"
                            )
                else:
                    if config['y_column'] in data.columns:
                        ax.plot(
                            data[config['x_column']], data[config['y_column']],
                            linestyle="None",
                            marker=marker,
                            markersize=MARKER_SIZE,
                            markerfacecolor="none",
                            markeredgewidth=MARKER_EDGE_WIDTH,
                            color=color,
                            label=label
                        )

            ax.set_xlabel(config['xlabel'], fontsize=22)
            ax.set_ylabel(config['ylabel'], fontsize=22)
            ax.set_title(config.get('title', ''), fontsize=18)
            ax.tick_params(axis='both', labelsize=22)

            #Legend inside plot, not overlapping data
            ax.legend(frameon=False, fontsize=20, loc='best', ncol=config.get('ncol', 1))

            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"{tag}_{config['filename']}"), dpi=300)
            plt.close()
            print(f"Saved {tag}_{config['filename']}")

    # -------------------------------------------------------
    # (1) NORMAL plots (all simulations)
    # -------------------------------------------------------
    all_dir = os.path.join(comparison_output_dir, "all_simulations")
    os.makedirs(all_dir, exist_ok=True)
    make_plots(simulation_data, all_dir, "ALL")

    # -------------------------------------------------------
    # (2) BASELINE plots (chemical step only, U0 = 0)
    # -------------------------------------------------------
    baseline_dir = os.path.join(comparison_output_dir, "baseline_chemical")
    os.makedirs(baseline_dir, exist_ok=True)
    make_plots(baseline_sims, baseline_dir, "BASELINE")

    print("\nComparison analysis finished.")


def main():
    """Main analysis pipeline"""
    parent_dir = input("Enter parent directory (absolute path): ").strip()

    if not os.path.isdir(parent_dir):
        print(f"Directory not found: {parent_dir}")
        return

    # Find all simulation directories (excluding those with 'init' in name)
    sim_dirs = [d for d in os.listdir(parent_dir)
                if os.path.isdir(os.path.join(parent_dir, d))
                and 'init' not in d.lower()
                and 'comparison_analysis' not in d.lower()]

    if not sim_dirs:
        print("No simulations found (or all contain 'init' in name)")
        return

    print(f"Found {len(sim_dirs)} simulations to process")
    # Process each simulation
    for sim_name in sim_dirs:
        sim_path = os.path.join(parent_dir, sim_name)
        output_dir = sim_path
        process_single_simulation(sim_path, output_dir)

    # Compare simulations
    print("\nGenerating comparison analysis...")
    comparison_output_dir = os.path.join(parent_dir, 'Comparison_analysis')
    compare_simulations(parent_dir, comparison_output_dir)
    print(f"Comparison plots saved to {comparison_output_dir}")


if __name__ == '__main__':
    main()
