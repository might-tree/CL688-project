import numpy as np
import matplotlib.pyplot as plt

def online_control_algorithm(step_size, K, params, T):    
    # Unpack the required parameters
    κ_B = params['κ_B']
    κ = params['κ']
    γ = params['γ']
    A = params['A']
    B = params['B']
    
    # Define H based on given formula
    H = int(γ ** -1 * np.log(T * κ ** 2))
    
    # Initialize M_t arbitrarily as a list of zeros
    M_t = {f'M[{i}]': np.zeros_like(K) for i in range(H)}
    
    # Lists to store actions, states, and errors
    actions = []
    states = []
    errors = []
    
    # Simulation of the system state
    x_t = np.zeros_like(K[:, 0])  # Starting state
    desired_state = np.zeros_like(x_t)  # Reference or desired state (e.g., zero state)

    # Begin the time loop
    for t in range(T):
        
        # Compute the action at time t using the provided formula
        u_t = -np.dot(K, x_t)
        for i in range(1, H + 1):
            u_t += np.dot(M_t[f'M[{i-1}]'], x_t)
        
        # Record the action
        actions.append(u_t)
        
        # Update state and compute disturbance
        w_t = np.random.randn(*x_t.shape)  # Simulating disturbance
        x_next = np.dot(A, x_t) + np.dot(B, u_t) + w_t
        
        # Compute control error as distance to desired state
        error = x_next - desired_state
        errors.append(np.linalg.norm(error))
        
        # Record the current state and error
        states.append(x_next)
        
        # Update the current state
        x_t = x_next
        
        # Calculate the gradient of the loss function
        grad_f_t = compute_gradient(x_next, desired_state)  # Compute the gradient based on current state
        
        # Update M_t based on gradient and project back into M
        M_t_updated = {}
        for key in M_t.keys():
            M_t_updated[key] = M_t[key] - step_size * grad_f_t  # Perform element-wise subtraction

        M_t = project_to_set(M_t_updated, M_t, κ_B, κ, γ)

    # Convert lists to numpy arrays for easier manipulation in plotting
    actions = np.array(actions)
    states = np.array(states)
    errors = np.array(errors)
    
    # Plotting
    plot_results(actions[:,0], states[:,0], errors)
    
    return actions, states, errors


def compute_gradient(x_t, desired_state):
    error = x_t - desired_state
    gradient = 2 * error  # Derivative of MSE = 2 * (x - desired)
    return gradient


def project_to_set(M_t, M, κ_B, κ, γ):
    # This function projects M_t back onto the constraints defined by ‖M[i-1]‖ ≤ κ3κB(1−γ)^i
    for i in range(len(M_t)):
        M_t[f'M[{i}]'] = np.clip(M_t[f'M[{i}]'], -κ_B * (1 - γ) ** i, κ_B * (1 - γ) ** i)
    return M_t

def plot_results(actions, states, errors):
    # Plot actions (control input) over time
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 3, 1)
    plt.plot(actions, label="Control Input (u_t)")
    plt.xlabel("Time Steps")
    plt.ylabel("Control Input")
    plt.title("Control Input Over Time")
    plt.legend()
    
    # Plot states (output) over time
    plt.subplot(1, 3, 2)
    plt.plot(states, label="State (x_t)")
    plt.xlabel("Time Steps")
    plt.ylabel("State")
    plt.title("System State Over Time")
    plt.legend()
    
    # Plot errors (difference from desired state) over time
    plt.subplot(1, 3, 3)
    plt.plot(errors, label="Error (||x_t - desired||)")
    plt.xlabel("Time Steps")
    plt.ylabel("Error")
    plt.title("Control Error Over Time")
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# Define sample parameters for the function call
step_size = 0.01  # Smaller step size for updates
T = 1000  # Total time steps

# Define a more stable control matrix K
K = np.array([[0.1, 0.0], [0.0, 0.1]])  # Lower control gains

# Define stable state dynamics matrices A and B
A = np.array([[0.02, 0.06], [0.1, 0.7]])  # Eigenvalues are less than 1
B = np.array([[0.05, 0.02], [0.02, 0.05]])  # Adjusted influence of control

# Parameters dictionary for the algorithm
params = {
    'κ_B': 1.0,
    'κ': 1.0,
    'γ': 0.9,
    'A': A,
    'B': B
}

# Run the online control algorithm with the adjusted parameters
actions, states, errors = online_control_algorithm(step_size, K, params, T)
