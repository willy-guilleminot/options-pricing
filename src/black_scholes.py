import numpy as np
from scipy.stats import norm

# Compute d1 and d2
def _d1_d2(S: float, K: float, T: float, r: float, sigma: float) -> tuple[float, float]:
    d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2

# Compute Black-Sholes call price giving the spot, the strike, the maturity, the interest rate, and the volatility
def call_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    return norm.cdf(d1) * S - norm.cdf(d2) * K * np.exp(-r * T)

# Compute Black-Sholes put price giving the spot, the strike, the maturity, the interest rate, and the volatility
def put_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    return norm.cdf(-d2) * K * np.exp(-r * T) - norm.cdf(-d1) * S

# Compute the call delta (partial derivative of the call price with respect to the underlying price)
def delta_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    d1, _ = _d1_d2(S, K, T, r, sigma)
    return norm.cdf(d1)

# Compute the put delta (partial derivative of the put price with respect to the underlying price)
def delta_put(S: float, K: float, T: float, r: float, sigma: float) -> float:
    d1, _ = _d1_d2(S, K, T, r, sigma)
    return norm.cdf(d1) - 1

# Compute the put delta (second partial derivative of the option price with respect to the underlying price)
def gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
    d1, _ = _d1_d2(S, K, T, r, sigma)
    return norm.pdf(d1) / (np.sqrt(T) * S * sigma)

# Compute the vega (partial derivative of the option price with respect to the underlying volatility)
def vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
    d1, _ = _d1_d2(S, K, T, r, sigma)
    return S * norm.pdf(d1) * np.sqrt(T)

# Compute the call theta (partial derivative of the call price with respect of the ...)
def theta_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    return -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)

# Compute the put theta (partial derivative of the put price with respect of the ...)
def theta_put(S: float, K: float, T: float, r: float, sigma: float) -> float:
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    return -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(d2)

# Compute the call rho (partial derivative of the call price with respect of the interest rate)
def rho_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    _, d2 = _d1_d2(S, K, T, r, sigma)
    return K * T * np.exp(-r * T) * norm.cdf(d2)

# Compute the put rho (partial derivative of the put price with respect of the interest rate)
def rho_put(S: float, K: float, T: float, r: float, sigma: float) -> float:
    _, d2 = _d1_d2(S, K, T, r, sigma)
    return -K * T * np.exp(-r * T) * norm.cdf(-d2)

# Use the bisection method to the root of function's root (implie volatility)
def _bisection_method(price: float, S: float, K: float, T: float, r: float, pricer: function, epsilon: float, max_iter: int) -> float:
    volLower = 0.001
    volUpper = 10
    volMid = (volLower + volUpper) / 2

    for _ in range(max_iter):
        if abs(pricer(S, K, T, r, volMid) - price) > epsilon:
            if (pricer(S, K, T, r, volMid) - price) * (pricer(S, K, T, r, volLower) - price) < 0:
                volUpper = volMid
            else:
                volLower = volMid
            volMid = (volLower + volUpper) / 2
        else:
            return volMid
    
    raise ValueError("Bisection did not converge")

# Use the Newton method to find the function's root (implied volatility)
def _newton_raphson(price: float, S: float, K: float, T: float, r: float, pricer: function, epsilon: float, guess: float, max_iter: int) -> float:
    sigma = guess

    for _ in range(max_iter):
        if abs(pricer(S, K, T, r, sigma) - price) > epsilon:
            sigma = sigma - (pricer(S, K, T, r, sigma) - price) / vega(S, K, T, r, sigma)
        else:
            return sigma
    
    raise ValueError("Newton-Raphson did not converge")

# Giving the Black Scholes parameters and the option's price, finds the option's implied volatility 
# using either the newton method or the bisection method 
def implied_volatility(price: float, S: float, K: float, T: float, r: float, option_type: str = 'call', method: str = 'bisection', epsilon: float = 10**(-5), guess: float = 0.2, max_iter: int = 1000) -> float:
    if option_type == 'call':
        pricer = call_price
    elif option_type == 'put':
        pricer = put_price
    else:
        raise ValueError("Option type not recognized. Please use 'call' or 'put'.")

    if type(max_iter) != int:
        raise ValueError("max_iter type not valid. Please provide an integer.")
    
    if method == 'bisection':
        return _bisection_method(price, S, K, T, r, pricer, epsilon, max_iter)
    elif method == 'newton':
        return _newton_raphson(price, S, K, T, r, pricer, epsilon, guess, max_iter)
    else:
        raise ValueError("Wrong method provided. Please use 'bisection' or 'newton'.")