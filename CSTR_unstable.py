import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize
from phe import paillier
import matplotlib.pyplot as plt

# Define system dynamics for unstable steady-state CSTR
def cstr_dynamics_unstable(x, t, u):
    CA, T = x
    k0, E, R, delta_H, rho, Cp, V, F = 1.0, 5000, 8.314, -5000, 1.0, 4.18, 100.0, 1.0
    CA0, Q = u[0], u[1]
    dCAdt = F/V * (CA0 - CA) - k0 * np.exp(-E / (R * T)) * CA**2
    dTdt = F/V * (300 - T) + (-delta_H) / (rho * Cp) * k0 * np.exp(-E / (R * T)) * CA**2 + Q / (rho * Cp * V)
    return [dCAdt, dTdt]

# Define cost function for MPC
def cost_function_unstable(u, x0, N, dt):
    u = np.reshape(u, (N, 2))
    state = np.array(x0)
    total_cost = 0
    for i in range(N):
        state = odeint(cstr_dynamics_unstable, state, [0, dt], args=(u[i],))[-1]
        total_cost += np.sum((state - [1.95, 402])**2) + np.sum(u[i]**2)
    return total_cost

# Nonlinear MPC Optimization
def mpc_optimization_unstable(x0, N=10, umin=[0, 0], umax=[7.5, 80], dt=0.1):
    u0 = np.random.rand(N, 2) * (np.array(umax) - np.array(umin)) + np.array(umin)
    bounds = [(umin[i], umax[i]) for i in range(2)] * N
    result = minimize(cost_function_unstable, u0.flatten(), args=(x0, N, dt), bounds=bounds, method='SLSQP')
    return result.x[:2]

# Paillier Encryption Setup
public_key, private_key = paillier.generate_paillier_keypair()

# Encrypt and Decrypt functions
def encrypt_data(data):
    return [public_key.encrypt(val) for val in data]

def decrypt_data(encrypted_data):
    return np.array([private_key.decrypt(val) for val in encrypted_data])

# Simulation loop with MPC and encryption
def simulate_encrypted_mpc_unstable(initial_state, time_horizon=5.0, dt=0.1):
    times = np.arange(0, time_horizon + dt, dt)
    state = initial_state
    all_states = [state]
    controls = []
    for t in times[:-1]:
        encrypted_state = encrypt_data(state)
        decrypted_state = decrypt_data(encrypted_state)
        control_input = mpc_optimization_unstable(decrypted_state, dt=dt)
        controls.append(control_input)
        state = odeint(cstr_dynamics_unstable, state, [t, t + dt], args=(control_input,))[-1]
        all_states.append(state)
    return np.array(all_states), np.array(controls), times

# Plotting functions
def plot_state_trajectories(states, times):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(times, states[:, 0], label="Concentration (CA)")
    plt.xlabel("Time")
    plt.ylabel("Concentration (kmol/m³)")
    plt.title("State Trajectory: Concentration")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(times, states[:, 1], label="Temperature (T)", color="orange")
    plt.xlabel("Time")
    plt.ylabel("Temperature (K)")
    plt.title("State Trajectory: Temperature")
    plt.legend()
    plt.tight_layout()
    plt.savefig('Unstable_trajectory.png')

def plot_control_inputs(controls, times):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(times[:-1], controls[:, 0], label="Control Input CA0", color="green")
    plt.xlabel("Time")
    plt.ylabel("CA0 (kmol/m³)")
    plt.title("Control Input: Feed Concentration CA0")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(times[:-1], controls[:, 1], label="Control Input Q", color="red")
    plt.xlabel("Time")
    plt.ylabel("Heat Q (MJ/hr)")
    plt.title("Control Input: Heat Q")
    plt.legend()
    plt.tight_layout()
    plt.savefig('Unstable_control.png')

# Run simulation for unstable steady-state CSTR
initial_state_unstable = [1.0, 300.0]
states, controls, times = simulate_encrypted_mpc_unstable(initial_state_unstable)

# Plot state and control trajectories
plot_state_trajectories(states, times)
plot_control_inputs(controls, times)
