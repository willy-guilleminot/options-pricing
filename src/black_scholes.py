import numpy as np
from scipy.stats import norm

def _d1_d2(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2

def call_price(S, K, T, r, sigma):
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    return norm.cdf(d1) * S - norm.cdf(d2) * K * np.exp(-r * T)

def put_price(S, K, T, r, sigma):
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    return norm.cdf(-d2) * K * np.exp(-r * T) - norm.cdf(-d1) * S

def delta_call(S, K, T, r, sigma):
    d1, _ = _d1_d2(S, K, T, r, sigma)
    return norm.cdf(d1)

def delta_put(S, K, T, r, sigma):
    d1, _ = _d1_d2(S, K, T, r, sigma)
    return norm.cdf(d1) - 1

def gamma(S, K, T, r, sigma):
    d1, _ = _d1_d2(S, K, T, r, sigma)
    return norm.pdf(d1) / (np.sqrt(T) * S * sigma)

def vega(S, K, T, r, sigma):
    d1, _ = _d1_d2(S, K, T, r, sigma)
    return S * norm.pdf(d1) * np.sqrt(T)

def theta_call(S, K, T, r, sigma):
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    return -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)

def theta_put(S, K, T, r, sigma):
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    return -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(d2)

def rho_call(S, K, T, r, sigma):
    _, d2 = _d1_d2(S, K, T, r, sigma)
    return K * T * np.exp(-r * T) * norm.cdf(d2)

def rho_put(S, K, T, r, sigma):
    _, d2 = _d1_d2(S, K, T, r, sigma)
    return -K * T * np.exp(-r * T) * norm.cdf(-d2)

def _bisection_method(price, S, K, T, r, option_type, epsilon):
    volLower = 0.001
    volUpper = 1
    volMid = (volLower + volUpper) / 2

    if option_type=='call':
        pricer = call_price
    else:
        pricer = put_price

    while abs(pricer(S, K, T, r, volMid) - price) > epsilon:
        if (pricer(S, K, T, r, volMid) - price) * (pricer(S, K, T, r, volLower) - price) < 0:
            volUpper = volMid
        else:
            volLower = volMid
        volMid = (volLower + volUpper) / 2
    
    return volMid

def _newton_raphson(price, S, K, T, r, option_type, epsilon, guess, max_iter):
    sigma = guess

    if option_type == 'call':
        pricer = call_price
    else:
        pricer = put_price

    for _ in range(max_iter):
        if abs(pricer(S, K, T, r, sigma) - price) > epsilon:
            sigma = sigma - (pricer(S, K, T, r, sigma) - price) / vega(S, K, T, r, sigma)
        else:
            return sigma
    
    raise ValueError("Newton-Raphson does not converge")

def implied_volatility(price, S, K, T, r, option_type='call', method='bisection', epsilon=10**(-5), guess=0.2, max_iter=100):
    if method == 'bisection':
        return _bisection_method(price, S, K, T, r, option_type, epsilon)
    else:
        return _newton_raphson(price, S, K, T, r, option_type, epsilon, guess, max_iter)