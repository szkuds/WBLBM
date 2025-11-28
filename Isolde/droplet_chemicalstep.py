import numpy as np
import matplotlib.pyplot as plt

from wblbm.run import Run
from wblbm.operators.wetting.contact_angle import ContactAngle
from wblbm import GravityForceMultiphaseDroplet
from wblbm import visualise

#Simulation parameters
grid_shape = (200,100) #nx, ny
tau = 0.99
nt = 20000
save_interval = 500
kappa = 0.04
rho_l = 1.0
rho_v = 0.001
interface_width = 5

#Wetting parameters
phi_value = 1.1
d_rho_value = 0.3

#Inclination angle constant
inclination_angle = 40.0 #degrees

#Gravity
force_g = 0.0000001
gravity = GravityForceMultiphaseDroplet(grid_shape[0], grid_shape[1], 2, force_g, inclination_angle)

#Boundary conditions with chem step and hysteresis
bc_config = {
        'left': 'periodic',
        'bottom': 'wetting',
        'top': 'symmetry',
        'right': 'periodic',
        'chemical_step': {
            'chemical_step_location': .5,
            'chemical_step_edge': 'bottom',
            'ca_advancing_pre_step': 110.0,
            'ca_receding_pre_step': 90.0,
            'ca_advancing_post_step': 70.0,
            'ca_receding_post_step': 60.0,
        },
        'wetting_params': {
            'rho_l': rho_l,
            'rho_v': rho_v,
            'phi_left': phi_value,
            'phi_right': phi_value,
            'd_rho_left': d_rho_value,
            'd_rho_right': d_rho_value,
            'width': interface_width
        },
        'hysteresis_params': {
            'ca_advancing': 90.0,
            'ca_receding': 80.0,
            'learning_rate': 0.05,
            'max_iterations': 10
        }
}

sim = Run(
    simulation_type="multiphase",
    grid_shape=grid_shape,
    lattice_type="D2Q9",
    tau=tau,
    nt=nt,
    kappa=kappa,
    rho_l=rho_l,
    rho_v=rho_v,
    interface_width=interface_width,
    save_interval=save_interval,
    bc_config=bc_config,
    force_enabled=True,
    force_obj=gravity,
    phi_value=phi_value,
    d_rho_value=d_rho_value,
    wetting_enabled=True,
    init_type="wetting_chem_step",  # droplet on surface
    force_g=force_g,
    inclination_angle=inclination_angle,
)
sim.run(verbose=True)

#Extract data for each timestep
save_times = []
theta_left_list = []
theta_right_list = []

rho_mean = 0.5 * (rho_l + rho_v)
angle_calc = ContactAngle(rho_mean)

#Loop through saved timesteps
data_dir = sim.io_handler.data_dir

for t in range (0,nt, save_interval):
    file = f"{data_dir}/timestep_{t}.npz"

    try:
        data = np.load(file)
    except FileNotFoundError:
        continue #skip missing files

    #Calculate CA's for each timestep
    rho_t = data["rho"]
    theta_L, theta_R = angle_calc.compute(rho_t)

    save_times.append(t)
    theta_left_list.append(theta_L)
    theta_right_list.append(theta_R)

#Save calculated CA's in arrays
save_times = np.array(save_times)
theta_left_list = np.array(theta_left_list)
theta_right_list = np.array(theta_right_list)

#Plot CA (receding and advancing) vs time
plt.plot(save_times, theta_left_list, label="θ_receding")
plt.plot(save_times, theta_right_list, label="θ_advancing")
plt.xlabel("Timestep")
plt.ylabel("Contact angle (°)")
plt.title("Droplet moving over a Chemical step: Contact angle vs Time ")
plt.axvline(len(save_times)/2, linestyle="--", color="red")
plt.legend()
plt.grid(True)
plt.show()

#Load saved results
#latest_result = "/Users/isoldeholweg/PycharmProjects/WBLBM/Isolde/results/2025-11-21/15-55-27/data/timestep_9999.npz"
#latest_result = sim.io_handler.data_dir + f"/timestep_{nt-1}.npz"
#data = np.load(latest_result)
#rho = data['rho']

#Compute contact angle
#rho_mean = 0.5 * (rho_l + rho_v)
#angle_calc = ContactAngle(rho_mean)
#theta_left, theta_right = angle_calc.compute(rho)

#Extract and squeeze density field to 2D
#rho_2d = rho[:,:,0,0]  # removes dimensions of size 1
#print("rho shape after squeeze:", rho_2d.shape)  # should print (100, 200)

# Check with a quick plot
#plt.imshow(rho_2d.T, origin="lower", cmap="jet")
#plt.colorbar(label="Density")
#plt.title("Droplet density field at equilibrium", weight="bold")

#Add calculated angles and parameters to the plot for info
#plt.text(1, rho_2d.shape[1]-10, f'θ_left = {theta_left:.1f}°', color='white', fontsize=12, weight='bold')
#plt.text(70, rho_2d.shape[1]-10, f'θ_right = {theta_right:.1f}°', color='white', fontsize=12, weight='bold')
#plt.text(1, rho_2d.shape[1]-20, f'Gravity= {force_g:.1e} N', color='white', fontsize=12, weight='bold')
#plt.text(100, rho_2d.shape[1]-20, f'Incline angle = {inclination_angle:.1f}°', color='white', fontsize=12, weight='bold')

#plt.show()

#Plot droplet density field
visualise(
    sim,
    title=f"Equilibrium droplet (E = 0)" #θ_left={theta_left:.1f}°, θ_right={theta_right:.1f}°, Gravity= {force_g:.1e} N, Incline= {inclination_angle:.1f}°)"
)
