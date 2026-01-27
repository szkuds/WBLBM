import os
import shutil
from wblbm import visualise


def get_latest_timestep(results_dir: str) -> str:
    data_dir = os.path.join(results_dir, "data")
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    timesteps = sorted(
        int(f.split('_')[-1].split('.')[0])
        for f in os.listdir(data_dir)
        if f.startswith("timestep_")
    )
    if not timesteps:
        raise FileNotFoundError(f"No timestep files found in {data_dir}")
    final_timestep = timesteps[-1]
    return os.path.join(data_dir, f"timestep_{final_timestep}.npz")


def move_and_rename_results(src_dir: str, target_stage_name: str, pipeline_timestamp: str) -> str:
    base_results = os.path.expanduser("~/TUD_LBM/results")
    date_part = src_dir.split("/")[-2]
    target_dir = os.path.join(base_results, date_part, pipeline_timestamp, target_stage_name)
    if os.path.abspath(src_dir) != os.path.abspath(target_dir):
        os.makedirs(os.path.dirname(target_dir), exist_ok=True)
        if os.path.exists(target_dir):
            shutil.rmtree(target_dir)
        os.rename(src_dir, target_dir)
    return target_dir


def visualize_stage(sim_obj, stage_name: str) -> None:
    print(f"\nVisualizing {stage_name}...")
    visualise(sim_obj, f"{stage_name} Results")
    print(f"✓ Visualization complete for {stage_name}")


def move_results_to_pipeline(sim_obj, pipeline_timestamp: str, stage_name: str) -> str:
    src_dir = sim_obj.io_handler.run_dir
    if not os.path.isdir(src_dir):
        raise FileNotFoundError(f"Simulation results directory not found: {src_dir}")
    stage_dir = move_and_rename_results(src_dir, stage_name, pipeline_timestamp)
    print(f"✓ {stage_name} results saved to: {stage_dir}")
    return stage_dir


def get_correct_wetting_init_timestep(results_dir, wetting_init_nt: int, electric_init_nt: int) -> str:
    data_dir = os.path.join(results_dir, "data")
    return os.path.join(data_dir, f"timestep_{wetting_init_nt - electric_init_nt}.npz")
