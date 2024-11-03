# R script to simulate the paper Encrypted model predictive control design for security to cyberattacks
# Not used to generate results in report, but included as a learning

library(deSolve)  # For solving differential equations
library(gmp)  # For large integer operations

generate_paillier_keypair <- function(bit_length = 512) {
  p <- nextprime(as.bigz(sample(2^(bit_length/2-1):2^(bit_length/2), 1)))
  q <- nextprime(as.bigz(sample(2^(bit_length/2-1):2^(bit_length/2), 1)))
  n <- p * q
  n_sq <- n^2
  g <- n + 1  # In practice, g = n + 1 is often chosen for simplicity
  lambda <- lcm(p - 1, q - 1)
  L <- function(x) (x - 1) / n
  mu <- modinv(L(powm(g, lambda, n_sq)), n)
  list(public_key = list(n = n, g = g), private_key = list(lambda = lambda, mu = mu, n_sq = n_sq))
}

encrypt_data <- function(m, public_key) {
  n <- public_key$n
  g <- public_key$g
  n_sq <- n^2
  r <- as.bigz(sample(1:(n-1), 1))
  while (gcd(r, n) != 1) {
    r <- as.bigz(sample(1:(n-1), 1))
  }
  c <- (powm(g, m, n_sq) * powm(r, n, n_sq)) %% n_sq
  return(c)
}

decrypt_data <- function(c, private_key) {
  lambda <- private_key$lambda
  mu <- private_key$mu
  n <- private_key$n
  n_sq <- private_key$n_sq
  L <- function(x) (x - 1) / n
  m <- (L(powm(c, lambda, n_sq)) * mu) %% n
  return(as.numeric(m))
}

cstr_dynamics <- function(time, state, parameters) {
  CA <- state[1]
  T <- state[2]
  k0 <- 1.0; E <- 5000; R <- 8.314; delta_H <- -5000
  rho <- 1.0; Cp <- 4.18; V <- 100.0; F <- 1.0
  CA0 <- parameters[1]
  Q <- parameters[2]
  reaction_rate <- k0 * exp(-E / (R * T)) * CA^2
  dCAdt <- F / V * (CA0 - CA) - reaction_rate
  dTdt <- F / V * (300 - T) + (-delta_H) / (rho * Cp) * reaction_rate + Q / (rho * Cp * V)
  list(c(dCAdt, dTdt))
}

quantize <- function(value, resolution = 0.1) {
  round(value / resolution) * resolution
}

cost_function <- function(u, state0, N, dt) {
  state <- state0
  total_cost <- 0
  u <- matrix(u, nrow=N, ncol=2, byrow=TRUE)
  
  for (i in 1:N) {
    parameters <- c(u[i, 1], u[i, 2])
    time <- c(0, dt)
    result <- ode(y = state, times = time, func = cstr_dynamics, parms = parameters)
    state <- result[nrow(result), 2:3]
    total_cost <- total_cost + sum((state - c(2.96, 320))^2) + sum(u[i, ]^2)
  }
  return(total_cost)
}

mpc_optimization <- function(state0, N = 10, umin = c(0, 0), umax = c(7.5, 80), dt = 0.1) {
  u0 <- matrix(runif(N * 2, umin[1], umax[1]), N, 2)
  lower <- rep(umin, N)
  upper <- rep(umax, N)
  result <- optim(par = as.vector(u0), fn = cost_function, state0 = state0, N = N, dt = dt, method = "L-BFGS-B", lower = lower, upper = upper)
  return(result$par[1:2])
}

simulate_encrypted_mpc <- function(initial_state, time_horizon = 5.0, dt = 0.1, N = 10, public_key, private_key) {
  times <- seq(0, time_horizon, by = dt)
  state <- initial_state
  all_states <- matrix(NA, nrow = length(times), ncol = 2)
  all_states[1, ] <- state
  
  for (t in 2:length(times)) {
    encrypted_state <- sapply(state, function(x) encrypt_data(as.bigz(round(x * 10)), public_key))
    decrypted_state <- sapply(encrypted_state, function(x) decrypt_data(x, private_key) / 10)
    control_input <- mpc_optimization(decrypted_state, N = N, dt = dt)
    quantized_control <- quantize(control_input)
    result <- ode(y = state, times = c(0, dt), func = cstr_dynamics, parms = quantized_control)
    state <- result[nrow(result), 2:3]
    all_states[t, ] <- state
  }
  return(all_states)
}

keypair <- generate_paillier_keypair()
public_key <- keypair$public_key
private_key <- keypair$private_key

# Sample initial conditions and run the simulation
initial_state <- c(2.0, 300.0)
start_time <- Sys.time()
results <- simulate_encrypted_mpc(initial_state, public_key = public_key, private_key = private_key)
end_time <- Sys.time()
time_taken <- end_time - start_time
cat("Time taken for the simulation:", time_taken, "seconds\n")

png("R_CSTR_simulation_results.png", width = 800, height = 600)
plot(results[, 1], type = "l", col = "blue", ylim = range(results), ylab = "States", xlab = "Time step")
lines(results[, 2], col = "red")
legend("topright", legend = c("Concentration (CA)", "Temperature (T)"), col = c("blue", "red"), lty = 1)
dev.off()
