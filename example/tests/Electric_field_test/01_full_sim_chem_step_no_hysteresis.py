import os
import jax
from datetime import datetime
import uuid

from wblbm.run import Run
from wblbm import GravityForceMultiphaseDroplet, visualise
from wblbm.utils.full_sim_util import (
    get_latest_timestep,
    move_results_to_pipeline,
    visualize_stage,
)

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
RHO_V = 0.33
INTERFACE_WIDTH = 5
PHI_VALUE = 1.1
D_RHO_VALUE = 0.1
FORCE_G = 5e-7
INCLINATION_ANGLE = 45

# Iteration parameters
WETTING_INIT_NT = 40000
WETTING_INIT_SAVE = WETTING_INIT_NT/10

CHEM_STEP_RUN_NT = 25000
CHEM_STEP_RUN_SAVE = CHEM_STEP_RUN_NT/200

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
    pipeline_timestamp = f"{datetime.now().strftime('%H-%M-%S')}_{uuid.uuid4().hex[:8]}"
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
