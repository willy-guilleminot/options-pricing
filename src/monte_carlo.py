import numpy as np

def mc_call_price(S, K, T, r, sigma, n_simulations=10000):
    Z = np.random.standard_normal(n_simulations)
    S_T = S *  np.exp((r - sigma**2 / 2) * T + sigma * np.sqrt(T) * Z)
    payoffs = np.maximum(S_T - K, 0)

    return np.mean(payoffs) * np.exp(-r * T)

def mc_put_price(S, K, T, r, sigma, n_simulations=10000):
    Z = np.random.standard_normal(n_simulations)
    S_T = S *  np.exp((r - sigma**2 / 2) * T + sigma * np.sqrt(T) * Z)
    payoffs = np.maximum(K - S_T, 0)

    return np.mean(payoffs) * np.exp(-r * T)

def mc_call_price_antithetic(S, K, T, r, sigma, n_simulations=10000):
    Z = np.random.standard_normal(n_simulations // 2)

    S_T1 = S *  np.exp((r - sigma**2 / 2) * T + sigma * np.sqrt(T) * Z)
    S_T2 = S *  np.exp((r - sigma**2 / 2) * T + sigma * np.sqrt(T) * -Z)

    payoffs_1 = np.maximum(S_T1 - K, 0)
    payoffs_2 = np.maximum(S_T2 - K, 0)

    return np.mean((payoffs_1 + payoffs_2) / 2) * np.exp(-r * T)

def mc_put_price_antithetic(S, K, T, r, sigma, n_simulations=10000):
    Z = np.random.standard_normal(n_simulations // 2)

    S_T1 = S *  np.exp((r - sigma**2 / 2) * T + sigma * np.sqrt(T) * Z)
    S_T2 = S *  np.exp((r - sigma**2 / 2) * T + sigma * np.sqrt(T) * -Z)

    payoffs_1 = np.maximum(K - S_T1, 0)
    payoffs_2 = np.maximum(K - S_T2, 0)

    return np.mean((payoffs_1 + payoffs_2) / 2) * np.exp(-r * T)

def simulate_paths(S, T, r, sigma, n_simulations, n_steps):
    dt = T / n_steps
    paths = np.zeros((n_simulations, n_steps + 1))
    paths[:, 0] = S

    for t in range(1, n_steps + 1):
        Z = np.random.standard_normal(n_simulations)
        paths[:, t] = paths[:, t - 1] * np.exp((r - sigma**2 / 2) * dt + sigma * np.sqrt(dt) * Z)

    return paths

def mc_asian_call(S, K, T, r, sigma, n_simulations=10000, n_steps=252):
    paths = simulate_paths(S, T, r, sigma, n_simulations, n_steps)
    payoffs = np.maximum(paths.mean(axis=1) - K, 0)

    return np.mean(payoffs) * np.exp(-r * T)

def mc_barrier_call(S, K, T, r, sigma, barrier, n_simulations=10000, n_steps=252):
    paths = simulate_paths(S, T, r, sigma, n_simulations, n_steps)
    barrier_crossed = np.max(paths, axis=1) >= barrier
    final_payoffs = np.maximum(paths[:, -1] - K, 0)
    payoffs = np.where(barrier_crossed, 0, final_payoffs)

    return np.mean(payoffs) * np.exp(-r * T)
    '''
    barrier_crossed = False
    for row_index in range(n_simulations):
        barrier_crossed = False
        for col_index in range(n_steps - 1):
            if paths[row_index, col_index] >= barrier:
                barrier_crossed = True
        if barrier_crossed:
            payoffs[row_index] = 0
        else:
            if paths[row_index, -1] - K > 0:
                payoffs[row_index] = paths[row_index, -1] - K
            else:
                payoffs[row_index] = 0
    '''
  