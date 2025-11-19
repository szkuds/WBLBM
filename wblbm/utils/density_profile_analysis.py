import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import re

# Paths
folder = "results/2025-11-19/17-54-55_mrt_static_test"
base_dir = os.path.join("/Users/sbszkudlarek/PycharmProjects/WBLBM/example/tests/", folder)
data_dir = os.path.join(base_dir, "data")


def plot_density_profile(data_folder, y_value=200):
    npz_files = sorted(glob.glob(os.path.join(data_folder, '*.npz')))
    assert npz_files, "No .npz files found"
    parent_dir = os.path.abspath(os.path.join(data_folder, os.pardir))
    save_dir = os.path.join(parent_dir, 'density_profile')
    os.makedirs(save_dir, exist_ok=True)

    iter_nums = []
    ratio_rhos = []

    for npz_file in npz_files:
        f_name = os.path.basename(npz_file)
        match = re.search(r'(?:timestep_|step)(\d+)', f_name)
        if match:
            iter_tag = match.group(1).zfill(5)
            iter_num = int(match.group(1))
        else:
            generic_match = re.search(r'(\d+)', f_name)
            iter_tag = generic_match.group(1).zfill(5) if generic_match else '00000'
            iter_num = int(generic_match.group(1)) if generic_match else 0
        data = np.load(npz_file)
        rho = data['rho']
        assert rho.ndim == 4, f"rho shape unexpected: {rho.shape}"
        rho_line = rho[:, y_value, 0, 0]
        min_rho = np.min(rho_line)
        max_rho = np.max(rho_line)
        ratio_rho = max_rho / min_rho
        iter_nums.append(iter_num)
        ratio_rhos.append(ratio_rho)
        plt.figure()
        plt.scatter(np.arange(len(rho_line)), rho_line)
        plt.xlabel('x')
        plt.ylabel(f'density at y={y_value}')
        plt.yscale('log')
        plt.title(f'Density profile over x at y={y_value} (log), iter {iter_tag}')
        plt.legend([f'Min: {min_rho:.3g}, Max: {max_rho:.3g}, Ratio: {ratio_rho:.3g}'])
        f_name_save = f'density_profile_iter-{iter_tag}.png'
        plot_path = os.path.join(save_dir, f_name_save)
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved plot to: {plot_path}")

    # Now scatter plot density ratio vs iteration number
    plt.figure()
    plt.scatter(iter_nums, ratio_rhos)
    plt.xlabel('Iteration number')
    plt.ylabel('Density ratio (max/min)')
    plt.title('Density ratio vs iteration number')
    plt.grid(True)
    summary_path = os.path.join(save_dir, "density_ratio_vs_iter.png")
    plt.savefig(summary_path)
    plt.close()
    print(f"Saved density ratio summary plot to: {summary_path}")
    print(f"Iteration numbers: {iter_nums}")
    print(f"Density ratios: {ratio_rhos}")


if __name__ == "__main__":
    plot_density_profile(data_dir)
