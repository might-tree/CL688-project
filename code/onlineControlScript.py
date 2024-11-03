# To generate results for the paper on Online Control with Adversarial Disturbance

import numpy as np
import matplotlib.pyplot as plt

def online_control_algorithm(step_size, K, params, T):   
    κ_B = params['κ_B']
    κ = params['κ']
    γ = params['γ']
    A = params['A']
    B = params['B']
    H = int(γ ** -1 * np.log(T * κ ** 2))
    M_t = {f'M[{i}]': np.zeros_like(K) for i in range(H)}
    actions = []
    states = []
    errors = []
    x_t = np.zeros_like(K[:, 0])
    desired_state = np.zeros_like(x_t)

    for t in range(T):
        u_t = -np.dot(K, x_t)
        for i in range(1, H + 1):
            u_t += np.dot(M_t[f'M[{i-1}]'], x_t)
        actions.append(u_t)
        w_t = np.random.randn(*x_t.shape)  # Simulating disturbance
        x_next = np.dot(A, x_t) + np.dot(B, u_t) + w_t
        error = x_next - desired_state
        errors.append(np.linalg.norm(error))
        states.append(x_next)
        x_t = x_next
        grad_f_t = compute_gradient(x_next, desired_state)
        M_t_updated = {}
        for key in M_t.keys():
            M_t_updated[key] = M_t[key] - step_size * grad_f_t
        M_t = project_to_set(M_t_updated, M_t, κ_B, κ, γ)
    actions = np.array(actions)
    states = np.array(states)
    errors = np.array(errors)
    plot_results(actions[:,0], states[:,0], errors)
    return actions, states, errors

def compute_gradient(x_t, desired_state):
    error = x_t - desired_state
    gradient = 2 * error  # Derivative of MSE = 2 * (x - desired)
    return gradient

def project_to_set(M_t, M, κ_B, κ, γ):
    # Projecting M_t back onto the constraints defined by ‖M[i-1]‖ ≤ κ3κB(1−γ)^i
    for i in range(len(M_t)):
        M_t[f'M[{i}]'] = np.clip(M_t[f'M[{i}]'], -κ_B * (1 - γ) ** i, κ_B * (1 - γ) ** i)
    return M_t

def plot_results(actions, states, errors):
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 3, 1)
    plt.plot(actions, label="Control Input (u_t)")
    plt.xlabel("Time Steps")
    plt.ylabel("Control Input")
    plt.title("Control Input Over Time")
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(states, label="State (x_t)")
    plt.xlabel("Time Steps")
    plt.ylabel("State")
    plt.title("System State Over Time")
    plt.legend()
    
    plt.subplot(1, 3, 3)
    plt.plot(errors, label="Error (||x_t - desired||)")
    plt.xlabel("Time Steps")
    plt.ylabel("Error")
    plt.title("Control Error Over Time")
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# Sample inputs for simulating code
step_size = 0.01 
T = 1000
K = np.array([[0.1, 0.0], [0.0, 0.1]])  # Control matrix
A = np.array([[0.02, 0.06], [0.1, 0.7]])  # Stable state dynamics A and B
B = np.array([[0.05, 0.02], [0.02, 0.05]])  
params = {
    'κ_B': 1.0,
    'κ': 1.0,
    'γ': 0.9,
    'A': A,
    'B': B
}
actions, states, errors = online_control_algorithm(step_size, K, params, T)
