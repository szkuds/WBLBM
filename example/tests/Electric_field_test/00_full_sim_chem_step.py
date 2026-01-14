import os
import jax
from datetime import datetime
import shutil

from wblbm.run import Run
from wblbm import GravityForceMultiphaseDroplet, visualise

jax.config.update("jax_enable_x64", True)

# ============================================================================
# SIMULATION PARAMETERS - Set all parameters here
# ============================================================================

# Grid and lattice
GRID_SHAPE = (201, 101)
LATTICE_TYPE = "D2Q9"

# Fluid properties
TAU = 0.99
KAPPA = 0.04
RHO_L = 1.0
RHO_V = 0.001
INTERFACE_WIDTH = 5
PHI_VALUE = 1.1
D_RHO_VALUE = 0.1
FORCE_G = 5e-7
INCLINATION_ANGLE = 45

# Iteration parameters
WETTING_INIT_NT = 300000
WETTING_INIT_SAVE = WETTING_INIT_NT/10

CHEM_STEP_RUN_NT = 50000
CHEM_STEP_RUN_SAVE = CHEM_STEP_RUN_NT/100

# Chemical step parameters
CHEMICAL_STEP_LOCATION = 0.5
CHEMICAL_STEP_EDGE = 'bottom'
CA_ADVANCING_PRE = 110.0
CA_RECEDING_PRE = 90.0
CA_ADVANCING_POST = 70.0
CA_RECEDING_POST = 60.0

# Hysteresis parameters
CA_ADVANCING = 90.0
CA_RECEDING = 80.0
LEARNING_RATE = 0.05
MAX_ITERATIONS = 10


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_latest_timestep(results_dir):
    """Extract final timestep from results directory."""
    data_dir = os.path.join(results_dir, "data")

    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    timesteps = sorted([int(f.split('_')[-1].split('.')[0])
                        for f in os.listdir(data_dir)
                        if f.startswith('timestep_')])

    if not timesteps:
        raise FileNotFoundError(f"No timestep files found in {data_dir}")

    final_timestep = timesteps[-1]
    return os.path.join(data_dir, f"timestep_{final_timestep}.npz")


def move_and_rename_results(src_dir, target_stage_name, pipeline_timestamp):
    """Move simulation results from default location to pipeline directory."""
    base_results = os.path.expanduser("~/TUD_LBM/results")
    date_part = src_dir.split('/')[-2]  # Extract DATE from path
    target_dir = os.path.join(base_results, date_part, pipeline_timestamp, target_stage_name)

    # If source and target are different, move it
    if os.path.abspath(src_dir) != os.path.abspath(target_dir):
        os.makedirs(os.path.dirname(target_dir), exist_ok=True)
        if os.path.exists(target_dir):
            shutil.rmtree(target_dir)
        os.rename(src_dir, target_dir)

    return target_dir


def visualize_stage(sim_obj, stage_name):
    """Visualize simulation results from Run object."""
    print(f"\nVisualizing {stage_name}...")
    visualise(sim_obj, f"{stage_name} Results")
    print(f"✓ Visualization complete for {stage_name}")


def move_results_to_pipeline(sim_obj, pipeline_timestamp, stage_name):
    """
    Move simulation results to pipeline structure using the simulation's own results directory.

    Args:
        sim_obj: The Run simulation object (has io_handler.run_dir)
        pipeline_timestamp: The pipeline timestamp for organizing results
        stage_name: Name of the stage (e.g., "run_wetting_init")

    Returns:
        Path to the moved stage directory
    """
    # Get the actual results directory from the simulation object
    src_dir = sim_obj.io_handler.run_dir

    if not os.path.isdir(src_dir):
        raise FileNotFoundError(f"Simulation results directory not found: {src_dir}")

    # Move to pipeline structure
    stage_dir = move_and_rename_results(src_dir, stage_name, pipeline_timestamp)
    print(f"✓ {stage_name} results saved to: {stage_dir}")

    return stage_dir


# ============================================================================
# STAGE 1: WETTING INITIALIZATION
# ============================================================================

def run_wetting_init(pipeline_timestamp):
    """Initialize with wetting only (no electric field, horizontal surface)."""
    print("\n" + "=" * 80)
    print("STAGE 1: WETTING INITIALIZATION")
    print("=" * 80)

    inclination_angle = 0
    gravity = GravityForceMultiphaseDroplet(FORCE_G, inclination_angle, GRID_SHAPE)

    bc_config = {
        'left': 'periodic',
        'bottom': 'wetting',
        'top': 'symmetry',
        'right': 'periodic',
        'wetting_params': {
            'rho_l': RHO_L,
            'rho_v': RHO_V,
            'phi_left': PHI_VALUE,
            'phi_right': PHI_VALUE,
            'd_rho_left': D_RHO_VALUE,
            'd_rho_right': D_RHO_VALUE,
            'width': INTERFACE_WIDTH
        }
    }

    sim = Run(
        simulation_type="multiphase",
        grid_shape=GRID_SHAPE,
        lattice_type=LATTICE_TYPE,
        tau=TAU,
        nt=WETTING_INIT_NT,
        kappa=KAPPA,
        rho_l=RHO_L,
        rho_v=RHO_V,
        interface_width=INTERFACE_WIDTH,
        save_interval=WETTING_INIT_SAVE,
        bc_config=bc_config,
        force_enabled=True,
        force_obj=[gravity],
        phi_value=PHI_VALUE,
        d_rho_value=D_RHO_VALUE,
        wetting_enabled=True,
        force_g=FORCE_G,
        inclination_angle=inclination_angle,
        init_type="wetting_chem_step",
    )

    sim.run(verbose=True)

    # Visualize before moving
    visualize_stage(sim, "Wetting Initialization")

    # Move the simulation results to pipeline structure
    return move_results_to_pipeline(sim, pipeline_timestamp, "run_wetting_init")


# ============================================================================
# STAGE 2: CHEMICAL STEP EXECUTION
# ============================================================================

def run_chem_step(pipeline_timestamp, wetting_results_dir):
    """Run chemical step with incline (45 degrees)."""
    print("\n" + "=" * 80)
    print("STAGE 3A: CHEMICAL STEP RUN (45° incline)")
    print("=" * 80)

    # Get init path from electric init output
    init_path = get_latest_timestep(wetting_results_dir)
    print(f"Using electric init output: {init_path}")

    inclination_angle = INCLINATION_ANGLE
    gravity = GravityForceMultiphaseDroplet(FORCE_G, inclination_angle, GRID_SHAPE)

    bc_config = {
        'left': 'periodic',
        'bottom': 'wetting',
        'top': 'symmetry',
        'right': 'periodic',
        'chemical_step': {
            'chemical_step_location': CHEMICAL_STEP_LOCATION,
            'chemical_step_edge': CHEMICAL_STEP_EDGE,
            'ca_advancing_pre_step': CA_ADVANCING_PRE,
            'ca_receding_pre_step': CA_RECEDING_PRE,
            'ca_advancing_post_step': CA_ADVANCING_POST,
            'ca_receding_post_step': CA_RECEDING_POST,
        },
        'wetting_params': {
            'rho_l': RHO_L,
            'rho_v': RHO_V,
            'phi_left': PHI_VALUE,
            'phi_right': PHI_VALUE,
            'd_rho_left': D_RHO_VALUE,
            'd_rho_right': D_RHO_VALUE,
            'width': INTERFACE_WIDTH
        },
        'hysteresis_params': {
            'ca_advancing': CA_ADVANCING,
            'ca_receding': CA_RECEDING,
            'learning_rate': LEARNING_RATE,
            'max_iterations': MAX_ITERATIONS
        }
    }

    sim = Run(
        simulation_type="multiphase",
        grid_shape=GRID_SHAPE,
        lattice_type=LATTICE_TYPE,
        tau=TAU,
        nt=CHEM_STEP_RUN_NT,
        kappa=KAPPA,
        rho_l=RHO_L,
        rho_v=RHO_V,
        interface_width=INTERFACE_WIDTH,
        save_interval=CHEM_STEP_RUN_SAVE,
        bc_config=bc_config,
        force_enabled=True,
        force_obj=[gravity],
        phi_value=PHI_VALUE,
        d_rho_value=D_RHO_VALUE,
        wetting_enabled=True,
        force_g=FORCE_G,
        inclination_angle=inclination_angle,
        init_type="init_from_file",
        init_dir=init_path,
    )

    sim.run(verbose=True)

    # Visualize before moving
    visualize_stage(sim, "Chemical Step")

    # Move the simulation results to pipeline structure
    return move_results_to_pipeline(sim, pipeline_timestamp, "run_chem_step")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("FULL SIMULATION PIPELINE")
    print("=" * 80)

    # Create pipeline timestamp that will be used for all stages
    pipeline_timestamp = datetime.now().strftime("%H-%M-%S")
    pipeline_dir = os.path.join(os.path.expanduser("~/TUD_LBM/results"),
                                datetime.now().strftime("%Y-%m-%d"),
                                pipeline_timestamp)
    os.makedirs(pipeline_dir, exist_ok=True)

    # Stage 1: Wetting initialization
    wetting_dir = run_wetting_init(pipeline_timestamp)

    # Stage 2: Run chemical step simulation
    print("\n" + "=" * 80)
    print("STAGE 2: RUNNING SIMULATION")
    print("=" * 80)

    chem_dir = run_chem_step(pipeline_timestamp, wetting_dir)

    print("\n" + "=" * 80)
    print("ALL SIMULATIONS COMPLETED AND VISUALIZED")
    print("=" * 80)
    print(f"\n✓ Pipeline root directory: {pipeline_dir}")
    print(f"✓ Wetting init results: {wetting_dir}")
    print(f"✓ Chemical step results: {chem_dir}")
    print(f"\nPlots have been saved to each stage's plots/ directory")
