#!/usr/bin/env python3
"""
Animation script for NWO presentation.

Generates composite images showing:
- Top: Density field visualization
- Bottom: Dual-axis plot of contact angles and capillary numbers

Then combines images into a video.
"""

import os
import re
import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec

try:
    import jax.numpy as jnp
except ImportError:
    jnp = np

# Handle moviepy import for both v1.x and v2.x
ImageSequenceClip = None
_moviepy_error = None
try:
    # Try moviepy 2.x import first
    from moviepy import ImageSequenceClip
except ImportError as e1:
    try:
        # Try moviepy 1.x import
        from moviepy.editor import ImageSequenceClip
    except ImportError as e2:
        _moviepy_error = f"moviepy 2.x error: {e1}, moviepy 1.x error: {e2}"

# If still None, try alternative imports for moviepy 2.x
if ImageSequenceClip is None:
    try:
        from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
    except ImportError:
        pass

if ImageSequenceClip is None:
    try:
        from moviepy.video.VideoClip import ImageSequenceClip
    except ImportError:
        pass

if ImageSequenceClip is None and _moviepy_error:
    print(f"Warning: Could not import ImageSequenceClip from moviepy.")
    print(f"Details: {_moviepy_error}")
    print("Video creation will be skipped. Install/reinstall with: pip install moviepy")


# Configure matplotlib
plt.rcParams['text.usetex'] = False  # Set to True if LaTeX is available
plt.rcParams['font.size'] = 12


def load_config(sim_dir: str) -> Dict:
    """Load configuration from config.json"""
    config_path = os.path.join(sim_dir, 'config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)

    # Try constants.txt fallback
    constants_path = os.path.join(sim_dir, 'constants.txt')
    if os.path.exists(constants_path):
        return load_constants_txt(constants_path)

    raise FileNotFoundError(f"No config.json or constants.txt found in {sim_dir}")


def load_constants_txt(file_path: str) -> Dict:
    """Load constants from a constants.txt file"""
    config = {}
    with open(file_path, 'r') as f:
        constants_code = f.read()

    # Create a safe namespace
    local_namespace = {}
    global_namespace = {"jnp": jnp, "np": np, "int": int, "float": float}

    try:
        exec(constants_code, global_namespace, local_namespace)
        # Convert uppercase variables to config dict with lowercase keys
        for k, v in local_namespace.items():
            if k.isupper():
                config[k.lower()] = v
    except Exception as e:
        print(f"Warning: Could not parse constants.txt: {e}")

    return config


def extract_iteration_number(filename: str) -> Optional[int]:
    """Extract iteration number from npz filename"""
    # Try different patterns
    patterns = [
        r'timestep_(\d+)',
        r'it(\d+)',
        r'iteration_(\d+)',
        r'(\d+)\.npz$'
    ]
    for pattern in patterns:
        match = re.search(pattern, filename)
        if match:
            return int(match.group(1))
    return None


def extract_npz_data(npz_file: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract rho, u_x, u_y from npz file"""
    data = np.load(npz_file)
    rho = data['rho']
    u = data['u']
    u_x = u[..., 0]
    u_y = u[..., 1]
    return rho, u_x, u_y


def calculate_contact_angle(rho: np.ndarray, rho_mean: float) -> Tuple[float, float]:
    """Calculate contact angles left and right"""
    import math

    # Handle different array shapes
    if rho.ndim == 4:
        array_i_j0 = rho[:, 1, 0, 0]
        array_i_jpos1 = rho[:, 2, 0, 0]
    elif rho.ndim == 2:
        array_i_j0 = rho[:, 1]
        array_i_jpos1 = rho[:, 2]
    else:
        raise ValueError(f"Unexpected rho shape: {rho.shape}")

    mask_i_j0 = array_i_j0 < rho_mean
    mask_i_jpos1 = array_i_jpos1 < rho_mean

    mask_int_i_j0 = np.array(mask_i_j0, dtype=int)
    mask_int_i_jpos1 = np.array(mask_i_jpos1, dtype=int)

    diff_mask_i_j0 = np.diff(mask_int_i_j0)
    diff_mask_i_jpos1 = np.diff(mask_int_i_jpos1)

    transition_index_left_i_j0 = np.where(diff_mask_i_j0 == -1)[0]
    transition_index_left_i_jpos1 = np.where(diff_mask_i_jpos1 == -1)[0]
    transition_index_right_i_j0 = np.where(diff_mask_i_j0 == 1)[0] + 1
    transition_index_right_i_jpos1 = np.where(diff_mask_i_jpos1 == 1)[0] + 1

    if len(transition_index_left_i_j0) == 0 or len(transition_index_right_i_j0) == 0:
        return np.nan, np.nan

    index_left_i_j0 = int(transition_index_left_i_j0[0])
    index_left_i_jpos1 = int(transition_index_left_i_jpos1[0])
    index_right_i_j0 = int(transition_index_right_i_j0[0])
    index_right_i_jpos1 = int(transition_index_right_i_jpos1[0])

    x_val_left_j0 = index_left_i_j0 + (rho_mean - array_i_j0[index_left_i_j0]) / (
            array_i_j0[index_left_i_j0 + 1] - array_i_j0[index_left_i_j0] + 1e-10)
    x_val_left_jpos1 = index_left_i_jpos1 + (rho_mean - array_i_jpos1[index_left_i_jpos1]) / (
            array_i_jpos1[index_left_i_jpos1 + 1] - array_i_jpos1[index_left_i_jpos1] + 1e-10)
    x_val_right_j0 = index_right_i_j0 - (rho_mean - array_i_j0[index_right_i_j0]) / (
            array_i_j0[index_right_i_j0 - 1] - array_i_j0[index_right_i_j0] + 1e-10)
    x_val_right_jpos1 = index_right_i_jpos1 - (rho_mean - array_i_jpos1[index_right_i_jpos1]) / (
            array_i_jpos1[index_right_i_jpos1 - 1] - array_i_jpos1[index_right_i_jpos1] + 1e-10)

    contact_angle_left = np.rad2deg(math.pi / 2 + np.arctan(x_val_left_j0 - x_val_left_jpos1))
    contact_angle_right = np.rad2deg(math.pi / 2 + np.arctan(x_val_right_jpos1 - x_val_right_j0))

    return float(contact_angle_left), float(contact_angle_right)


def calculate_contact_line_location(rho: np.ndarray, rho_mean: float) -> Tuple[float, float]:
    """Calculate contact line locations left and right"""
    # Handle different array shapes
    if rho.ndim == 4:
        array_i_j0 = rho[:, 1, 0, 0]
    elif rho.ndim == 2:
        array_i_j0 = rho[:, 1]
    else:
        raise ValueError(f"Unexpected rho shape: {rho.shape}")

    mask_i_j0 = array_i_j0 < rho_mean
    mask_int_i_j0 = np.array(mask_i_j0, dtype=int)
    diff_mask_i_j0 = np.diff(mask_int_i_j0)

    transition_index_left_i_j0 = np.where(diff_mask_i_j0 == -1)[0]
    transition_index_right_i_j0 = np.where(diff_mask_i_j0 == 1)[0] + 1

    if len(transition_index_left_i_j0) == 0 or len(transition_index_right_i_j0) == 0:
        return np.nan, np.nan

    index_left_i_j0 = int(transition_index_left_i_j0[0])
    index_right_i_j0 = int(transition_index_right_i_j0[0])

    x_val_left_j0 = index_left_i_j0 + (rho_mean - array_i_j0[index_left_i_j0]) / (
            array_i_j0[index_left_i_j0 + 1] - array_i_j0[index_left_i_j0] + 1e-10)
    x_val_right_j0 = index_right_i_j0 - (rho_mean - array_i_j0[index_right_i_j0]) / (
            array_i_j0[index_right_i_j0 - 1] - array_i_j0[index_right_i_j0] + 1e-10)

    return float(x_val_left_j0), float(x_val_right_j0)


def calculate_center_of_mass(rho: np.ndarray, rho_mean: float) -> Tuple[float, float]:
    """Calculate center of mass of droplet"""
    if rho.ndim == 4:
        rho_2d = rho[:, :, 0, 0]
    elif rho.ndim == 2:
        rho_2d = rho
    else:
        raise ValueError(f"Unexpected rho shape: {rho.shape}")

    mask = rho_2d > rho_mean
    x_indices, y_indices = np.indices(rho_2d.shape)

    total_mass = np.sum(mask * rho_2d)
    if total_mass == 0:
        return np.nan, np.nan

    cm_x = np.sum(x_indices * mask * rho_2d) / total_mass
    cm_y = np.sum(y_indices * mask * rho_2d) / total_mass

    return float(cm_x), float(cm_y)


def calculate_avg_x_location(rho: np.ndarray, rho_mean: float, half_nx: int) -> float:
    """Calculate average x location of the droplet"""
    if rho.ndim == 4:
        rho_2d = rho[:, :, 0, 0]
    elif rho.ndim == 2:
        rho_2d = rho
    else:
        raise ValueError(f"Unexpected rho shape: {rho.shape}")

    mask = rho_2d > rho_mean
    x_indices = np.arange(rho_2d.shape[0]) - half_nx
    x_indices_2d = np.expand_dims(x_indices, axis=1)
    masked_x = np.where(mask, x_indices_2d, 0)

    total_mask = np.sum(mask)
    if total_mask == 0:
        return np.nan

    avg_x_location = np.sum(masked_x) / total_mask
    return float(avg_x_location)


def calculate_velocities(contact_line_left: np.ndarray, contact_line_right: np.ndarray,
                         centre_of_mass_x: np.ndarray, plot_every: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate velocities from position time series"""
    x_t_left = np.array(contact_line_left)
    x_t_right = np.array(contact_line_right)
    x_t_cm = np.array(centre_of_mass_x)

    x_tneg1_left = np.roll(x_t_left, shift=1)
    x_tneg1_left[0] = x_t_left[0]

    x_tneg1_right = np.roll(x_t_right, shift=1)
    x_tneg1_right[0] = x_t_right[0]

    x_tneg1_cm = np.roll(x_t_cm, shift=1)
    x_tneg1_cm[0] = x_t_cm[0]

    v_left = (x_t_left - x_tneg1_left) / plot_every
    v_right = (x_t_right - x_tneg1_right) / plot_every
    v_cm = (x_t_cm - x_tneg1_cm) / plot_every

    return v_left, v_right, v_cm


def process_simulation_data(sim_dir: str) -> Tuple[pd.DataFrame, Dict]:
    """
    Process simulation data and return DataFrame with all metrics.

    Checks for existing processed_data.csv first, otherwise generates from npz files.
    """
    # Check for existing processed data
    analysis_dir = os.path.join(sim_dir, 'Analysis')
    csv_path = os.path.join(analysis_dir, 'processed_data.csv')

    config = load_config(sim_dir)

    if os.path.exists(csv_path):
        print(f"Loading existing analysis data from {csv_path}")
        df = pd.read_csv(csv_path)
        return df, config

    print("Processed data not found. Generating from npz files...")

    # Find data directory
    data_dir = os.path.join(sim_dir, 'data')
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    # Get all npz files
    npz_files = sorted(
        [f for f in os.listdir(data_dir) if f.endswith('.npz')],
        key=lambda x: extract_iteration_number(x) or 0
    )

    if not npz_files:
        raise FileNotFoundError(f"No .npz files found in {data_dir}")

    # Extract constants
    rho_l = config.get('rho_l', config.get('RHO_L', 1.0))
    rho_v = config.get('rho_v', config.get('RHO_V', 0.001))
    rho_mean = (rho_l + rho_v) / 2

    grid_shape = config.get('grid_shape', config.get('NX', [256, 256]))
    if isinstance(grid_shape, (list, tuple)):
        nx = grid_shape[0]
    else:
        nx = grid_shape
    half_nx = nx // 2

    plot_every = config.get('plot_every', config.get('PLOT_EVERY', 1))
    tau = config.get('tau', config.get('TAU', 1.0))
    sigma_lg = config.get('sigma_lg', config.get('SIGMA_LG', 0.1))
    r_zero = config.get('r_zero', config.get('R_ZERO', 27))

    # Process all timesteps
    iterations = []
    avg_u_x_list = []
    avg_x_location_list = []
    contact_line_left_list = []
    contact_line_right_list = []
    contact_angle_left_list = []
    contact_angle_right_list = []
    cm_x_list = []

    print(f"Processing {len(npz_files)} timesteps...")
    for i, npz_file in enumerate(npz_files):
        npz_path = os.path.join(data_dir, npz_file)
        rho, u_x, u_y = extract_npz_data(npz_path)

        iteration = extract_iteration_number(npz_file)
        if iteration is None:
            continue

        # Calculate metrics
        if rho.ndim == 4:
            rho_2d = rho[:, :, 0, 0]
            u_x_2d = u_x[:, :, 0]
        else:
            rho_2d = rho
            u_x_2d = u_x

        mask = rho_2d > rho_mean
        masked_u_x = np.where(mask, u_x_2d, 0)

        total_mask = np.sum(mask)
        avg_u_x = np.sum(masked_u_x) / total_mask if total_mask > 0 else 0
        avg_x_loc = calculate_avg_x_location(rho, rho_mean, half_nx)
        cl_left, cl_right = calculate_contact_line_location(rho, rho_mean)
        ca_left, ca_right = calculate_contact_angle(rho, rho_mean)
        cm_x, _ = calculate_center_of_mass(rho, rho_mean)

        iterations.append(iteration)
        avg_u_x_list.append(avg_u_x)
        avg_x_location_list.append(avg_x_loc)
        contact_line_left_list.append(cl_left)
        contact_line_right_list.append(cl_right)
        contact_angle_left_list.append(ca_left)
        contact_angle_right_list.append(ca_right)
        cm_x_list.append(cm_x)

        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(npz_files)}")

    # Calculate derived quantities
    iterations = np.array(iterations)
    max_iteration = max(iterations) if len(iterations) > 0 else 1
    iterations_norm = iterations / max_iteration
    avg_x_loc_norm = np.array(avg_x_location_list) / r_zero

    v_left, v_right, v_cm = calculate_velocities(
        np.array(contact_line_left_list),
        np.array(contact_line_right_list),
        np.array(cm_x_list),
        plot_every
    )

    nu = (tau - 0.5) / 3  # kinematic viscosity
    Ca_CL_L = (v_left * nu) / sigma_lg
    Ca_CL_R = (v_right * nu) / sigma_lg

    # Create DataFrame
    df = pd.DataFrame({
        'Iteration': iterations,
        'Normalised iterations': iterations_norm,
        'Average X Location': avg_x_location_list,
        'Average X location normalised': avg_x_loc_norm,
        'Average X Velocity': avg_u_x_list,
        'Contact line left': contact_line_left_list,
        'Contact line right': contact_line_right_list,
        'Contact line velocity left': v_left,
        'Contact line velocity right': v_right,
        'Contact angle left': contact_angle_left_list,
        'Contact angle right': contact_angle_right_list,
        'Center of Mass X': cm_x_list,
        'Ca left contact line': Ca_CL_L,
        'Ca right contact line': Ca_CL_R,
    })

    # Save to CSV
    os.makedirs(analysis_dir, exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"Saved processed data to {csv_path}")

    return df, config


def get_npz_files_sorted(data_dir: str) -> List[str]:
    """Get sorted list of npz files"""
    npz_files = [f for f in os.listdir(data_dir) if f.endswith('.npz')]
    return sorted(npz_files, key=lambda x: extract_iteration_number(x) or 0)


def create_composite_frame(
    rho: np.ndarray,
    df: pd.DataFrame,
    frame_idx: int,
    config: Dict,
    output_path: str,
    x_limits: Tuple[float, float] = None,
    ca_limits: Tuple[float, float] = None,
    angle_limits: Tuple[float, float] = None
):
    """
    Create a composite frame with density visualization on top and plots below.

    Parameters:
    -----------
    rho : np.ndarray
        Density field for current timestep
    df : pd.DataFrame
        DataFrame with all processed data
    frame_idx : int
        Current frame index (0-based), determines how many points to plot
    config : Dict
        Configuration dictionary
    output_path : str
        Path to save the output image
    x_limits : tuple
        X-axis limits for the plot (min, max)
    ca_limits : tuple
        Y-axis limits for Ca (min, max)
    angle_limits : tuple
        Y-axis limits for contact angle (min, max)
    """
    # Get density for visualization
    if rho.ndim == 4:
        rho_2d = rho[:, :, 0, 0]
    elif rho.ndim == 2:
        rho_2d = rho
    else:
        rho_2d = rho

    # Create figure with gridspec - use constrained_layout for better handling of twin axes
    fig = plt.figure(figsize=(14, 10), constrained_layout=True)
    gs = GridSpec(2, 1, height_ratios=[1, 1], hspace=0.15, figure=fig)

    # Top: Density visualization
    ax_density = fig.add_subplot(gs[0])

    # Transpose for correct orientation (x horizontal, y vertical)
    im = ax_density.imshow(
        rho_2d.T,
        origin='lower',
        cmap='viridis',
        aspect='auto'
    )
    cbar = plt.colorbar(im, ax=ax_density, label='Density')
    ax_density.set_xlabel('X position', fontsize=14)
    ax_density.set_ylabel('Y position', fontsize=14)

    iteration = df['Iteration'].iloc[frame_idx] if frame_idx < len(df) else 0
    ax_density.set_title(f'Density Field - Iteration {int(iteration)}', fontsize=16)

    # Bottom: Dual-axis plot (similar to plot 12 in Data_plots.py)
    ax1 = fig.add_subplot(gs[1])

    # Get data up to current frame (inclusive)
    data_subset = df.iloc[:frame_idx + 1]

    x_data = data_subset['Average X location normalised']
    ca_left = data_subset['Ca left contact line']
    ca_right = data_subset['Ca right contact line']
    angle_left = data_subset['Contact angle left']
    angle_right = data_subset['Contact angle right']

    # Plot Ca contact lines on left axis
    ax1.scatter(
        x_data, ca_left,
        s=20, label='Trailing Edge (Ca)',
        facecolors='none', edgecolors='blue', linewidths=1.5
    )
    ax1.scatter(
        x_data, ca_right,
        s=20, label='Leading Edge (Ca)',
        facecolors='none', edgecolors='red', linewidths=1.5
    )

    # Mark current point
    if len(x_data) > 0:
        ax1.scatter(
            [x_data.iloc[-1]], [ca_left.iloc[-1]],
            s=100, facecolors='blue', edgecolors='darkblue', linewidths=2, zorder=5
        )
        ax1.scatter(
            [x_data.iloc[-1]], [ca_right.iloc[-1]],
            s=100, facecolors='red', edgecolors='darkred', linewidths=2, zorder=5
        )

    ax1.set_xlabel(r'$X_{avg}/R_0$', fontsize=16)
    ax1.set_ylabel('Ca', color='black', fontsize=16)
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.tick_params(axis='both', labelsize=12)

    if x_limits:
        ax1.set_xlim(x_limits)
    if ca_limits:
        ax1.set_ylim(ca_limits)

    ax1.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax1.grid(True, alpha=0.3)

    # Create second y-axis for contact angles
    ax2 = ax1.twinx()
    ax2.scatter(
        x_data, angle_left,
        color='skyblue', s=15, marker='x',
        label='Trailing Edge Contact Angle'
    )
    ax2.scatter(
        x_data, angle_right,
        color='coral', s=15, marker='x',
        label='Leading Edge Contact Angle'
    )

    # Mark current point for angles
    if len(x_data) > 0:
        ax2.scatter(
            [x_data.iloc[-1]], [angle_left.iloc[-1]],
            s=80, color='deepskyblue', marker='X', linewidths=2, zorder=5
        )
        ax2.scatter(
            [x_data.iloc[-1]], [angle_right.iloc[-1]],
            s=80, color='orangered', marker='X', linewidths=2, zorder=5
        )

    ax2.set_ylabel(r'$\theta$ (degrees)', color='black', fontsize=16)
    ax2.tick_params(axis='y', labelcolor='black')
    ax2.tick_params(axis='both', labelsize=12)

    if angle_limits:
        ax2.set_ylim(angle_limits)

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)

    ax1.set_title('Contact Line Capillary Number and Contact Angles', fontsize=16)

    # Save figure (constrained_layout handles spacing automatically)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_animation_frames(sim_dir: str, output_dir: str = None) -> str:
    """
    Create all animation frames for the simulation.

    Parameters:
    -----------
    sim_dir : str
        Path to simulation directory
    output_dir : str
        Optional output directory for frames. If None, creates 'Animation' folder in sim_dir

    Returns:
    --------
    str : Path to the frames directory
    """
    # Set up output directory
    if output_dir is None:
        output_dir = os.path.join(sim_dir, 'Animation')

    frames_dir = os.path.join(output_dir, 'frames')
    os.makedirs(frames_dir, exist_ok=True)

    # Process simulation data
    print("Processing simulation data...")
    df, config = process_simulation_data(sim_dir)

    # Get data directory and npz files
    data_dir = os.path.join(sim_dir, 'data')
    npz_files = get_npz_files_sorted(data_dir)

    if len(npz_files) == 0:
        raise FileNotFoundError(f"No npz files found in {data_dir}")

    print(f"Found {len(npz_files)} timesteps to process")

    # Get config values
    rho_l = config.get('rho_l', config.get('RHO_L', 1.0))
    rho_v = config.get('rho_v', config.get('RHO_V', 0.001))
    rho_mean = (rho_l + rho_v) / 2

    # Calculate global limits for consistent axes
    x_min = df['Average X location normalised'].min()
    x_max = df['Average X location normalised'].max()
    x_margin = (x_max - x_min) * 0.1
    x_limits = (x_min - x_margin, x_max + x_margin)

    ca_min = min(df['Ca left contact line'].min(), df['Ca right contact line'].min())
    ca_max = max(df['Ca left contact line'].max(), df['Ca right contact line'].max())
    ca_margin = (ca_max - ca_min) * 0.1
    ca_limits = (ca_min - ca_margin, ca_max + ca_margin)

    angle_min = min(df['Contact angle left'].min(), df['Contact angle right'].min())
    angle_max = max(df['Contact angle left'].max(), df['Contact angle right'].max())
    angle_margin = (angle_max - angle_min) * 0.1
    angle_limits = (angle_min - angle_margin, angle_max + angle_margin)

    print(f"Creating frames in {frames_dir}...")

    for i, npz_file in enumerate(npz_files):
        npz_path = os.path.join(data_dir, npz_file)
        rho, _, _ = extract_npz_data(npz_path)

        output_path = os.path.join(frames_dir, f'frame_{i:05d}.png')

        create_composite_frame(
            rho=rho,
            df=df,
            frame_idx=i,
            config=config,
            output_path=output_path,
            x_limits=x_limits,
            ca_limits=ca_limits,
            angle_limits=angle_limits
        )

        if (i + 1) % 10 == 0:
            print(f"  Created frame {i + 1}/{len(npz_files)}")

    print(f"All frames saved to {frames_dir}")
    return frames_dir


def create_video(frames_dir: str, output_path: str = None, video_duration: float = 10.0) -> str:
    """
    Create video from frames.

    Parameters:
    -----------
    frames_dir : str
        Path to directory containing frame images
    output_path : str
        Path for output video. If None, creates video in parent directory
    video_duration : float
        Desired video duration in seconds

    Returns:
    --------
    str : Path to the created video
    """
    if ImageSequenceClip is None:
        print("Error: moviepy not installed. Cannot create video.")
        print("Install with: pip install moviepy")
        return None

    # Get all frame images
    exts = {'.png', '.jpg', '.jpeg'}
    frames_path = Path(frames_dir)

    images = sorted(
        [p for p in frames_path.iterdir() if p.suffix.lower() in exts],
        key=lambda x: int(re.search(r'\d+', x.stem).group()) if re.search(r'\d+', x.stem) else 0
    )

    if not images:
        print(f"No images found in {frames_dir}")
        return None

    print(f"Found {len(images)} frames")

    # Calculate fps
    fps = len(images) / video_duration
    print(f"Using fps = {fps:.2f}")

    # Set output path and ensure it has a valid extension
    if output_path is None:
        output_path = os.path.join(os.path.dirname(frames_dir), 'animation.mp4')

    # Ensure output path has .mp4 extension
    output_path_obj = Path(output_path)
    if output_path_obj.suffix.lower() not in {'.mp4', '.avi', '.mov', '.mkv', '.webm'}:
        output_path = str(output_path_obj.with_suffix('.mp4'))
        print(f"Added .mp4 extension to output filename")

    # Create video
    print(f"Creating video: {output_path}")
    clip = ImageSequenceClip([str(p) for p in images], fps=fps)

    # Handle API differences between moviepy 1.x and 2.x
    try:
        # moviepy 2.x API
        clip.write_videofile(str(output_path), codec='libx264', logger=None)
    except TypeError:
        # moviepy 1.x API (has verbose parameter)
        clip.write_videofile(str(output_path), codec='libx264', verbose=False, logger=None)

    print(f"Video saved to {output_path}")
    return output_path


def main():
    """Main function - interactive mode"""
    print("=" * 60)
    print("Animation Generator for NWO Presentation")
    print("=" * 60)
    print()

    # Get simulation directory
    sim_dir = input("Enter the simulation directory path: ").strip()
    sim_dir = os.path.expanduser(sim_dir)

    if not os.path.isdir(sim_dir):
        print(f"Error: '{sim_dir}' is not a valid directory.")
        sys.exit(1)

    # Check for data directory
    data_dir = os.path.join(sim_dir, 'data')
    if not os.path.isdir(data_dir):
        print(f"Error: Data directory not found: {data_dir}")
        print("Expected structure: sim_dir/data/it*.npz")
        sys.exit(1)

    # Create frames
    print()
    print("Step 1: Creating animation frames...")
    frames_dir = create_animation_frames(sim_dir)

    # Ask about video creation
    print()
    create_vid = input("Create video from frames? (y/n) [y]: ").strip().lower()
    if create_vid in ('', 'y', 'yes'):
        try:
            duration = float(input("Enter desired video duration in seconds [10]: ").strip() or "10")
        except ValueError:
            duration = 10.0

        output_name = input("Enter output video filename [animation.mp4]: ").strip()
        if not output_name:
            output_name = "animation.mp4"

        # Ensure filename has a valid video extension
        if not any(output_name.lower().endswith(ext) for ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm']):
            output_name = output_name + '.mp4'
            print(f"Added .mp4 extension: {output_name}")

        output_path = os.path.join(os.path.dirname(frames_dir), output_name)

        print()
        print("Step 2: Creating video...")
        create_video(frames_dir, output_path, duration)

    print()
    print("Done!")


if __name__ == '__main__':
    main()

