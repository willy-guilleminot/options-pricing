import numpy as np

# Use Monte Carlo to find the call price
def mc_call_price(S, K, T, r, sigma, n_simulations=10000):
    Z = np.random.standard_normal(n_simulations) # Provides a vector of n_simulations draw from the standard normal distribution
    S_T = S *  np.exp((r - sigma**2 / 2) * T + sigma * np.sqrt(T) * Z) # Compute the vector of spot prices at maturity date
    payoffs = np.maximum(S_T - K, 0) # Vector of payoffs

    return np.mean(payoffs) * np.exp(-r * T)

# Use Monte Carlo to find the put price
def mc_put_price(S, K, T, r, sigma, n_simulations=10000):
    Z = np.random.standard_normal(n_simulations)
    S_T = S *  np.exp((r - sigma**2 / 2) * T + sigma * np.sqrt(T) * Z)
    payoffs = np.maximum(K - S_T, 0)

    return np.mean(payoffs) * np.exp(-r * T)

# Use Monte Carlo to find the call price by minimizing the calculation cost using the...
def mc_call_price_antithetic(S, K, T, r, sigma, n_simulations=10000):
    Z = np.random.standard_normal(n_simulations // 2)

    S_T1 = S *  np.exp((r - sigma**2 / 2) * T + sigma * np.sqrt(T) * Z)
    S_T2 = S *  np.exp((r - sigma**2 / 2) * T + sigma * np.sqrt(T) * -Z)

    payoffs_1 = np.maximum(S_T1 - K, 0)
    payoffs_2 = np.maximum(S_T2 - K, 0)

    return np.mean((payoffs_1 + payoffs_2) / 2) * np.exp(-r * T)

# Use Monte Carlo to find the put price by minimizing the calculation cost using the...
def mc_put_price_antithetic(S, K, T, r, sigma, n_simulations=10000):
    Z = np.random.standard_normal(n_simulations // 2)

    S_T1 = S *  np.exp((r - sigma**2 / 2) * T + sigma * np.sqrt(T) * Z)
    S_T2 = S *  np.exp((r - sigma**2 / 2) * T + sigma * np.sqrt(T) * -Z)

    payoffs_1 = np.maximum(K - S_T1, 0)
    payoffs_2 = np.maximum(K - S_T2, 0)

    return np.mean((payoffs_1 + payoffs_2) / 2) * np.exp(-r * T)

# Provides a n_simulations x n_steps vector of paths 
def simulate_paths(S, T, r, sigma, n_simulations, n_steps):
    dt = T / n_steps
    paths = np.zeros((n_simulations, n_steps + 1)) # Create a n_simulations x n_steps vector filled with zeros
    paths[:, 0] = S # Initialize the spot price at t=0

    # Compute the spot price at each steps using Brownian Motion
    for t in range(1, n_steps + 1):
        Z = np.random.standard_normal(n_simulations)
        paths[:, t] = paths[:, t - 1] * np.exp((r - sigma**2 / 2) * dt + sigma * np.sqrt(dt) * Z)

    return paths

# Compute asian call price using Monte Carlo
def mc_asian_call(S, K, T, r, sigma, n_simulations=10000, n_steps=252):
    paths = simulate_paths(S, T, r, sigma, n_simulations, n_steps)
    payoffs = np.maximum(paths.mean(axis=1) - K, 0)

    return np.mean(payoffs) * np.exp(-r * T)

# Compute call up and out price using Monte Carlo
def mc_barrier_call(S, K, T, r, sigma, barrier, n_simulations=10000, n_steps=252):
    paths = simulate_paths(S, T, r, sigma, n_simulations, n_steps)
    barrier_crossed = np.max(paths, axis=1) >= barrier
    final_payoffs = np.maximum(paths[:, -1] - K, 0)
    payoffs = np.where(barrier_crossed, 0, final_payoffs)

    return np.mean(payoffs) * np.exp(-r * T)