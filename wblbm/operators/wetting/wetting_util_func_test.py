import json
import numpy as np
import glob
import os
from wblbm.operators.wetting.contact_angle import ContactAngle
from wblbm.operators.wetting.contact_line_location import ContactLineLocation

# Paths
folder = "results/2025-11-18/12-58-34_wetting_hysteresis_simulation_test"
base_dir = os.path.join("/Users/sbszkudlarek/PycharmProjects/WBLBM/example/tests/", folder)
data_dir = os.path.join(base_dir, "data")
config_path = os.path.join(base_dir, "config.json")
output_path = os.path.join(base_dir, "wetting_analysis.txt")

# Load config
with open(config_path) as f:
    config = json.load(f)
rho_l = float(config["rho_l"])
rho_v = float(config["rho_v"])
rho_mean = (rho_l + rho_v) / 2

angle_calc = ContactAngle(rho_mean)
line_loc_calc = ContactLineLocation(rho_mean)

# Load and sort all .npz result files
npz_files = glob.glob(f"{data_dir}/*.npz")
npz_files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))

# Process sorted files and save results to file
with open(output_path, "w") as out_f:
    out_f.write("Wetting analysis results\n")
    out_f.write("=======================\n")
    for npz_file in npz_files:
        data = np.load(npz_file)
        rho = data["rho"]
        left_angle, right_angle = angle_calc.compute(rho)
        left_line, right_line = line_loc_calc.compute(rho, left_angle, right_angle)
        line = f"{npz_file}: left_angle={left_angle}, right_angle={right_angle}, left_line={left_line}, right_line={right_line}"
        print(line)
        out_f.write(line + "\n")