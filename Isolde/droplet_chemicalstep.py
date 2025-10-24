import numpy as np
import matplotlib.pyplot as plt

from Isolde.droplet_homogenous import grid_shape, phi_value, d_rho_value, inclination_angle, latest_result
from wblbm.run import Run
from wblbm.operators.wetting.contact_angle import ContactAngle
from wblbm import GravityForceMultiphaseDroplet
from wblbm import visualise

#Simulation parameters
grid_shape = (200,100) #nx, ny
tau = 0.99
nt = 500
save_interval = 100
kappa = 0.01
rho_l = 1.0
rho_v = 0.001
interface_width = 5

#Wetting parameters
phi_value = 1.0
d_rho_value = 0.0

#Inclination angle constant
inclination_angle = 45.0 #degrees

#Gravity
force_g = 0.00001
gravity = GravityForceMultiphaseDroplet(grid_shape[0], grid_shape[1], 2, force_g, inclination_angle)

#Boundary conditions
bc_config = {
        'left': 'periodic',
        'bottom': 'wetting',
        'top': 'symmetry',
        'right': 'periodic',
        'wetting_params': {
            'rho_l': rho_l,
            'rho_v': rho_v,
            'phi_left': phi_value,
            'phi_right': phi_value,
            'd_rho_left': d_rho_value,
            'd_rho_right': d_rho_value,
            'width': interface_width,
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
    init_type="wetting",  # droplet on surface
)
sim.run(verbose=True)


#Load saved results
latest_result = "/Users/isoldeholweg/PycharmProjects/WBLBM/Isolde/results/2025-10-22/13-47-03/data/timestep_499.npz"
data = np.load(latest_result)
rho = data['rho']

#Compute contact angle
rho_mean = 0.5 * (rho_l + rho_v)
angle_calc = ContactAngle(rho_mean)
theta_left, theta_right = angle_calc.compute(rho)
#Advancing vs receding here ?

#Extract and squeeze density field to 2D
rho_2d = rho[:,:,0,0]  # removes dimensions of size 1
print("rho shape after squeeze:", rho_2d.shape)  # should print (100, 200)

# Check with a quick plot
plt.imshow(rho_2d.T, origin="lower", cmap="jet")
plt.colorbar(label="Density")
plt.title("Droplet density field at equilibrium", weight="bold")

#Add calculated angles and parameters to the plot for info
plt.text(1, rho_2d.shape[1]-10, f'θ_left = {theta_left:.1f}°', color='white', fontsize=12, weight='bold')
plt.text(70, rho_2d.shape[1]-10, f'θ_right = {theta_right:.1f}°', color='white', fontsize=12, weight='bold')
plt.text(1, rho_2d.shape[1]-20, f'Gravity= {force_g:.1e} N', color='white', fontsize=12, weight='bold')
plt.text(100, rho_2d.shape[1]-20, f'Incline angle = {inclination_angle:.1f}°', color='white', fontsize=12, weight='bold')

plt.show()

#Plot droplet density field
visualise(
    sim,
    title=f"Equilibrium droplet (θ_left={theta_left:.1f}°, θ_right={theta_right:.1f}°, Gravity= {force_g:.1e} N, Incline= {inclination_angle:.1f}°)"
)
