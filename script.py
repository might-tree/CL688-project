import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize
from phe import paillier
import matplotlib.pyplot as plt
import time

# Define system dynamics (Nonlinear CSTR ODEs)
def cstr_dynamics(x, t, u):
    CA, T = x  # Reactant concentration and temperature
    k0, E, R, delta_H, rho, Cp, V, F = 1.0, 5000, 8.314, -5000, 1.0, 4.18, 100.0, 1.0
    CA0, Q = u[0], u[1]
    
    dCAdt = F/V * (CA0 - CA) - k0 * np.exp(-E / (R * T)) * CA**2
    dTdt = F/V * (300 - T) + (-delta_H) / (rho * Cp) * k0 * np.exp(-E / (R * T)) * CA**2 + Q / (rho * Cp * V)
    
    return [dCAdt, dTdt]

# Quantization function
def quantize(value, resolution=0.1):
    return np.round(value / resolution) * resolution

# Define cost function for nonlinear MPC
def cost_function(u, x0, N, dt):
    u = np.reshape(u, (N, 2))  # Reshape control inputs
    state = np.array(x0)
    total_cost = 0
    
    for i in range(N):
        # Simulate system dynamics with control input u[i]
        state = odeint(cstr_dynamics, state, [0, dt], args=(u[i],))[-1]
        # Cost: deviation from desired setpoint [CA = 2.96, T = 320]
        total_cost += np.sum((state - [2.96, 320])**2) + np.sum(u[i]**2)
    
    return total_cost

# Nonlinear MPC Optimization using SciPy
def mpc_optimization(x0, N=10, umin=[0, 0], umax=[7.5, 80], dt=0.1):
    u0 = np.random.rand(N, 2) * (np.array(umax) - np.array(umin)) + np.array(umin)  # Initial guess for controls
    bounds = [(umin[i], umax[i]) for i in range(2)] * N  # Control bounds

    # Optimize the cost function using SciPy's minimize
    result = minimize(cost_function, u0.flatten(), args=(x0, N, dt), bounds=bounds, method='SLSQP')
    return result.x[:2]  # Return first control input

# Paillier Encryption Setup
public_key, private_key = paillier.generate_paillier_keypair()

# Encrypt and Decrypt function
def encrypt_data(data):
    return [public_key.encrypt(val) for val in data]

def decrypt_data(encrypted_data):
    return np.array([private_key.decrypt(val) for val in encrypted_data])

# Simulation loop with nonlinear MPC and encryption
def simulate_encrypted_mpc(initial_state, time_horizon=5.0, dt=0.1, resolution=0.1):
    times = np.arange(0, time_horizon, dt)
    state = initial_state
    all_states = [state]
    controls = []
    
    for t in times:
        encrypted_state = encrypt_data(state)
        decrypted_state = decrypt_data(encrypted_state)
        control_input = mpc_optimization(decrypted_state, dt=dt)
        quantized_control = [quantize(c, resolution) for c in control_input]
        controls.append(quantized_control)
        
        # Integrate system dynamics using odeint
        state = odeint(cstr_dynamics, state, [t, t + dt], args=(quantized_control,))[-1]
        all_states.append(state)
    
    return np.array(all_states), np.array(controls)

# Initial conditions and run the simulation
initial_state = [2.0, 300.0]  # Initial concentration and temperature

# Effect of Different Quantization Levels on State and Control Trajectories
quantization_levels = [0.5, 0.1, 0.05, 0.01]
state_profiles = []
control_profiles = []

for res in quantization_levels:
    states, controls = simulate_encrypted_mpc(initial_state, resolution=res)
    state_profiles.append(states)
    control_profiles.append(controls)

# Plot state and input profiles for different quantization levels
for i, res in enumerate(quantization_levels):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(state_profiles[i][:, 0], label=f"Quantization {res}")
    plt.xlabel("Time")
    plt.ylabel("Concentration (CA)")
    plt.legend()
    plt.title(f"State Profile - Concentration for Quantization {res}")

    plt.subplot(1, 2, 2)
    plt.plot(control_profiles[i][:, 0], label=f"Control Input Quantization {res}")
    plt.xlabel("Time")
    plt.ylabel("Control Input")
    plt.legend()
    plt.title(f"Control Input Profile for Quantization {res}")
    plt.savefig('quantization_levels.png')

# Error Analysis Between Encrypted and Non-Encrypted Systems
def simulate_non_encrypted_mpc(initial_state, time_horizon=5.0, dt=0.1, resolution=0.1):
    times = np.arange(0, time_horizon, dt)
    state = initial_state
    all_states = [state]
    controls = []
    
    for t in times:
        control_input = mpc_optimization(state, dt=dt)
        quantized_control = [quantize(c, resolution) for c in control_input]
        controls.append(quantized_control)
        
        # Integrate system dynamics using odeint
        state = odeint(cstr_dynamics, state, [t, t + dt], args=(quantized_control,))[-1]
        all_states.append(state)
    
    return np.array(all_states), np.array(controls)

# Run both encrypted and non-encrypted simulations for error analysis
states_enc, controls_enc = simulate_encrypted_mpc(initial_state)
states_non_enc, controls_non_enc = simulate_non_encrypted_mpc(initial_state)

errors_states = np.abs(states_enc - states_non_enc)
errors_controls = np.abs(controls_enc - controls_non_enc)

# Plot error analysis
plt.figure(figsize=(10, 5))
plt.plot(errors_states[:, 0], label="Error in Concentration (CA)")
plt.plot(errors_states[:, 1], label="Error in Temperature (T)")
plt.xlabel("Time")
plt.ylabel("Absolute Error")
plt.legend()
plt.title("Error Between Encrypted and Non-Encrypted System States")
plt.savefig('state_error.png')

plt.figure(figsize=(10, 5))
plt.plot(errors_controls[:, 0], label="Error in Control Input (CA0)")
plt.plot(errors_controls[:, 1], label="Error in Control Input (Q)")
plt.xlabel("Time")
plt.ylabel("Absolute Error in Control")
plt.legend()
plt.title("Error Between Encrypted and Non-Encrypted Control Inputs")
plt.savefig('control_error.png')

# Computational Cost Analysis for Different Quantization Levels
encryption_times = []
decryption_times = []

for res in quantization_levels:
    encryption_time = 0
    decryption_time = 0
    state = initial_state
    dt = 0.1
    for t in np.arange(0, 5.0, dt):
        control_input = mpc_optimization(state, dt=dt)
        
        # Measure encryption time
        start = time.time()
        encrypted_control = encrypt_data(control_input)
        encryption_time += time.time() - start
        
        # Measure decryption time
        start = time.time()
        decrypted_control = decrypt_data(encrypted_control)
        decryption_time += time.time() - start

    encryption_times.append(encryption_time)
    decryption_times.append(decryption_time)

# Plot computational cost analysis
plt.figure(figsize=(10, 5))
plt.plot(quantization_levels, encryption_times, label="Encryption Time")
plt.plot(quantization_levels, decryption_times, label="Decryption Time")
plt.xlabel("Quantization Resolution")
plt.ylabel("Time (s)")
plt.legend()
plt.title("Computational Cost of Encryption and Decryption")
plt.savefig('cost.png')
