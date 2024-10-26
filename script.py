import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize
from phe import paillier

# Define system dynamics (Nonlinear CSTR ODEs)
def cstr_dynamics(x, t, u):
    CA, T = x  # Reactant concentration and temperature
    k0, E, R, delta_H, rho, Cp, V, F = 1.0, 5000, 8.314, -5000, 1.0, 4.18, 100.0, 1.0
    CA0, Q = u[0], u[1]
    
    dCAdt = F/V * (CA0 - CA) - k0 * np.exp(-E / (R * T)) * CA**2
    dTdt = F/V * (300 - T) + (-delta_H) / (rho * Cp) * k0 * np.exp(-E / (R * T)) * CA**2 + Q / (rho * Cp * V)
    
    return [dCAdt, dTdt]

# Quantization function (to reduce precision and simulate encryption effects)
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
def simulate_encrypted_mpc(initial_state, time_horizon=5.0, dt=0.1):
    times = np.arange(0, time_horizon, dt)
    state = initial_state
    all_states = [state]
    
    for t in times:
        encrypted_state = encrypt_data(state)
        decrypted_state = decrypt_data(encrypted_state)
        control_input = mpc_optimization(decrypted_state, dt=dt)
        quantized_control = quantize(control_input)
        
        # Integrate system dynamics using odeint
        state = odeint(cstr_dynamics, state, [t, t + dt], args=(quantized_control,))[-1]
        all_states.append(state)
    
    return np.array(all_states)

# Initial conditions and run the simulation
initial_state = [2.0, 300.0]  # Initial concentration and temperature
results = simulate_encrypted_mpc(initial_state)

# Plot results
import matplotlib.pyplot as plt
import matplotlib
plt.plot(results[:, 0], label="Concentration (CA)")
plt.plot(results[:, 1], label="Temperature (T)")
plt.legend()
plt.savefig('plot.png')

