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
phi_value = 1.2
d_rho_value = 0.2

#Inclination angle constant
inclination_angle = 40.0 #degrees

#Gravity
force_g = 1e-5
gravity = GravityForceMultiphaseDroplet(
    grid_shape[0], grid_shape[1], 2, force_g, inclination_angle)

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
save_times = [] #timesteps
theta_left_list = [] #receding CA
theta_right_list = [] #advancing Ca
cm_x_list = [] #CM x
cm_y_list = [] #CM y

#Compute contact angles (CA's)
rho_mean = 0.5 * (rho_l + rho_v)
angle_calc = ContactAngle(rho_mean)

#Compute Center of Mass (CM)
def center_of_mass(rho):
    rho2d = rho[:, :, 0, 0]          # (nx, ny)
    nx, ny = rho2d.shape

    x = np.arange(nx)
    y = np.arange(ny)
    X, Y = np.meshgrid(x, y, indexing='ij')

    mass = rho2d.sum()
    cm_x = (rho2d * X).sum() / mass
    cm_y = (rho2d * Y).sum() / mass
    return cm_x, cm_y

#Loop through saved timesteps while calculating CA's and CM for each timestep
data_dir = sim.io_handler.data_dir

for t in range (0,nt, save_interval):
    file = f"{data_dir}/timestep_{t}.npz"

    try:
        data = np.load(file)
    except FileNotFoundError:
        continue #skip missing files

    rho_t = data["rho"]

    #Calculate CA's for each timestep
    theta_L, theta_R = angle_calc.compute(rho_t)

    save_times.append(t)
    theta_left_list.append(theta_L)
    theta_right_list.append(theta_R)

    #Calculate CM for each timestep
    cm_x, cm_y = center_of_mass(rho_t)
    cm_x_list.append(cm_x)
    cm_y_list.append(cm_y)

#Save timesteps in array
save_times = np.array(save_times)

#-------- Plotting CA's vs time ----------
#Save calculated CA's in arrays
theta_left_list = np.array(theta_left_list)
theta_right_list = np.array(theta_right_list)

#Compute velocity v(t) and acceleration a(t)
dt = save_interval

#Velocity as a derivative of CM position
vx = np.gradient(cm_x_list, dt) #x
vy = np.gradient(cm_y_list, dt) #y
v = np.sqrt(vx**2 + vy**2) #vector v(t)

#Acceleration as a derivative of velocity v(t)
ax = np.gradient(vx, dt)
ay = np.gradient(vy, dt)
a = np.sqrt(ax**2 + ay**2)

#Plotting CA's vs timesteps
plt.plot(save_times, theta_left_list, label="θ_receding")
plt.plot(save_times, theta_right_list, label="θ_advancing")
plt.xlabel("Timestep")
plt.ylabel("Contact angle (°)")
plt.title("Droplet moving over a Chemical step: Contact angle vs Time\n"
          f"E = 0,  g = {force_g},  Incline = {inclination_angle}°")
plt.legend()
plt.grid(True)
plt.show()

#--------- Plotting position x(t) vs time ---------
#Save calculated CM's in arrays
cm_x_list = np.array(cm_x_list) #x(t)
cm_y_list = np.array(cm_y_list)

#Plotting x(t) vs timesteps
plt.plot(save_times, cm_x_list, label='x(t)')
plt.xlabel("Timestep")
plt.ylabel("Center of mass position x(t)")
plt.title("Droplet Center of Mass vs Time\n"
          f"E = 0,  g = {force_g}, Incline = {inclination_angle}°")
plt.legend()
plt.grid(True)
plt.show()

#--------- Plotting velocity v(t) (and acceleration a(t) vs time ---------
plt.subplot(1, 2, 1)
plt.plot(save_times, vx, label='v_x(t)')
#plt.plot(save_times, vy, label='v_y(t)')
plt.plot(save_times, v, label='v(t)')
plt.xlabel("Timestep")
plt.ylabel("Velocity (lattice units / timestep)")
plt.title("Droplet Velocity vs Time\n"
          f"E = 0,  g = {force_g}, Incline = {inclination_angle}°")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(save_times, ax, label='a_x(t)')
#plt.plot(save_times, ay, label='a_y(t)')
#plt.plot(save_times, a, label='a(t)')
plt.xlabel("Timestep")
plt.ylabel("Acceleration")
plt.title("Droplet Acceleration vs Time\n"
          f"E = 0,  g = {force_g}, Incline = {inclination_angle}°")
plt.legend()

plt.grid(True)
plt.tight_layout()
plt.show()

#Calculate location of chem step
chem_step_x = grid_shape[0] / 2

#Plot droplet density field
visualise(
    sim,
    title=f"Equilibrium droplet (E = 0, Gravity= {force_g:.1e} N, Incline= {inclination_angle:.1f}°)"
)
plt.axvline(chem_step_x, color='red', linestyle="--", label="Chemical step")
