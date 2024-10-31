# Load necessary libraries
library(deSolve)

# Define system dynamics for the nonlinear CSTR model
cstr_dynamics <- function(time, state, parameters) {
  CA <- state[1]  # Reactant concentration
  T <- state[2]   # Temperature
  
  # System parameters
  k0 <- 1.0; E <- 5000; R <- 8.314; delta_H <- -5000
  rho <- 1.0; Cp <- 4.18; V <- 100.0; F <- 1.0
  CA0 <- parameters[1]  # Control input 1
  Q <- parameters[2]    # Control input 2
  
  # Nonlinear rate and ODEs
  reaction_rate <- k0 * exp(-E / (R * T)) * CA^2
  dCAdt <- F / V * (CA0 - CA) - reaction_rate
  dTdt <- F / V * (300 - T) + (-delta_H) / (rho * Cp) * reaction_rate + Q / (rho * Cp * V)
  
  list(c(dCAdt, dTdt))  # Return list of derivatives
}

# Quantization function to reduce precision
quantize <- function(value, resolution = 0.1) {
  round(value / resolution) * resolution
}

# Cost function for the MPC optimization
cost_function <- function(u, state0, N, dt) {
  state <- state0
  total_cost <- 0
  
  u <- matrix(u, nrow=N, ncol=2, byrow=TRUE)  # Reshape u to N x 2 matrix
  
  for (i in 1:N) {
    parameters <- c(u[i, 1], u[i, 2])
    time <- c(0, dt)  # Time interval for ODE solver
    
    # Solve the system dynamics over this time step
    result <- ode(y = state, times = time, func = cstr_dynamics, parms = parameters)
    state <- result[nrow(result), 2:3]  # Update state
    
    # Calculate the cost (deviation from setpoint [CA = 2.96, T = 320] and control effort)
    total_cost <- total_cost + sum((state - c(2.96, 320))^2) + sum(u[i, ]^2)
  }
  
  return(total_cost)
}

# Nonlinear MPC optimization using R's optim function
mpc_optimization <- function(state0, N = 10, umin = c(0, 0), umax = c(7.5, 80), dt = 0.1) {
  # Initial guess for control inputs (random within bounds)
  u0 <- matrix(runif(N * 2, umin[1], umax[1]), N, 2)
  
  # Bounds for optimization
  lower <- rep(umin, N)
  upper <- rep(umax, N)
  
  # Use optim to minimize the cost function
  result <- optim(par = as.vector(u0), fn = cost_function, state0 = state0, N = N, dt = dt,
                  method = "L-BFGS-B", lower = lower, upper = upper)
  
  # Return the first control input to be applied
  return(result$par[1:2])
}

# Placeholder for encryption (identity for now)
encrypt_data <- function(data) {
  return(data)  # Simply return data (no real encryption)
}

decrypt_data <- function(encrypted_data) {
  return(encrypted_data)  # Simply return data (no real decryption)
}

# Simulation loop for MPC with encrypted communication
simulate_encrypted_mpc <- function(initial_state, time_horizon = 5.0, dt = 0.1, N = 10) {
  times <- seq(0, time_horizon, by = dt)
  state <- initial_state
  all_states <- matrix(NA, nrow = length(times), ncol = 2)
  all_states[1, ] <- state
  
  for (t in 2:length(times)) {
    # Encrypt and decrypt the current state
    encrypted_state <- encrypt_data(state)
    decrypted_state <- decrypt_data(encrypted_state)
    
    # Get the control input using MPC
    control_input <- mpc_optimization(decrypted_state, N = N, dt = dt)
    
    # Quantize the control input
    quantized_control <- quantize(control_input)
    
    # Simulate system dynamics for one step
    result <- ode(y = state, times = c(0, dt), func = cstr_dynamics, parms = quantized_control)
    state <- result[nrow(result), 2:3]  # Update state
    
    all_states[t, ] <- state  # Store state
  }
  
  return(all_states)
}

# Initial conditions and run the simulation
initial_state <- c(2.0, 300.0)  # Initial concentration (CA) and temperature (T)
start_time <- Sys.time()
results <- simulate_encrypted_mpc(initial_state)
end_time <- Sys.time()
time_taken <- end_time - start_time
cat("Time taken for the simulation:", time_taken, "seconds\n")

# Save the plot as a PNG file
png("R_CSTR_simulation_results.png", width = 800, height = 600)  # Specify the filename and dimensions
# Plot results
plot(results[, 1], type = "l", col = "blue", ylim = range(results), ylab = "States", xlab = "Time step")
lines(results[, 2], col = "red")
legend("topright", legend = c("Concentration (CA)", "Temperature (T)"), col = c("blue", "red"), lty = 1)
# Close the PNG device
dev.off()
