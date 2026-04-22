import numpy as np

# Compute risk-neutral probability
def _u_d_p(r, sigma, dt):
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)

    return u, d, p

# Compute call price using the binomial model
def crr_call(S, K, T, r, sigma, n_steps=100):
    dt = T /n_steps
    u, d, p = _u_d_p(r, sigma, dt)

    j = np.arange(0, n_steps + 1)
    S_T = S * u**j * d**(n_steps - j)

    payoffs = np.maximum(S_T - K, 0)

    for _ in range(n_steps):
        payoffs = np.exp(-r * dt) * (p * payoffs[1:] + (1 - p) * payoffs[:-1])
    
    return payoffs[0]

# Compute put price using the binomial model
def crr_put(S, K, T, r, sigma, n_steps=100):
    dt = T /n_steps
    u, d, p = _u_d_p(r, sigma, dt)

    j = np.arange(0, n_steps + 1)
    S_T = S * u**j * d**(n_steps - j)

    payoffs = np.maximum(K - S_T, 0)

    for _ in range(n_steps):
        payoffs = np.exp(-r * dt) * (p * payoffs[1:] + (1 - p) * payoffs[:-1])
    
    return payoffs[0]

# Compute american call price using the binomial model
def crr_american_call(S, K, T, r, sigma, n_steps=100):
    dt = T /n_steps
    u, d, p = _u_d_p(r, sigma, dt)

    j = np.arange(0, n_steps + 1)
    S_T = S * u**j * d**(n_steps - j)

    payoffs = np.maximum(S_T - K, 0)

    for node in range(1, n_steps + 1):
        j = np.arange(0, n_steps - node + 1)
        S_node = S * u**j * d**(n_steps - node - j)
        payoffs = np.maximum(np.exp(-r * dt) * (p * payoffs[1:] + (1 - p) * payoffs[:-1]), np.maximum(S_node - K, 0))

    return payoffs[0]

# Compute american pur price using the binomial model
def crr_american_put(S, K, T, r, sigma, n_steps=100):
    dt = T /n_steps
    u, d, p = _u_d_p(r, sigma, dt)

    j = np.arange(0, n_steps + 1)
    S_T = S * u**j * d**(n_steps - j)

    payoffs = np.maximum(K - S_T, 0)

    for node in range(1, n_steps + 1):
        j = np.arange(0, n_steps - node + 1)
        S_node = S * u**j * d**(n_steps - node - j)
        payoffs = np.maximum(np.exp(-r * dt) * (p * payoffs[1:] + (1 - p) * payoffs[:-1]), np.maximum(K - S_node, 0))

    return payoffs[0]