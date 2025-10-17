import json
import numpy as np
import glob
from wblbm.operators.wetting.contact_angle import ContactAngle
from wblbm.operators.wetting.contact_line_location import ContactLineLocation

# Paths
data_dir = "/Users/sbszkudlarek/PycharmProjects/WBLBM/example/tests/results/2025-10-16/15-32-24_wetting_simulation_test/data"
config_path = "/Users/sbszkudlarek/PycharmProjects/WBLBM/example/tests/results/2025-10-16/15-32-24_wetting_simulation_test/config.json"

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
# Sort by timestep number extracted from filename
npz_files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))

# Process sorted files
for npz_file in npz_files:
    data = np.load(npz_file)
    rho = data["rho"]
    # Calculate contact angles
    left_angle, right_angle = angle_calc.compute(rho)
    # Calculate contact line locations
    left_line, right_line = line_loc_calc.compute(rho, left_angle, right_angle)
    print(f"{npz_file}: left_angle={left_angle}, right_angle={right_angle}, left_line={left_line}, right_line={right_line}")
