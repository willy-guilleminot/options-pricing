# Options Pricing Engine

An interactive options pricing engine built in Python, featuring analytical and numerical pricing models, Greeks computation, implied volatility surface from real market data, and a live Streamlit dashboard.

## Live Demo
[Options Pricing Engine](https://options-pricing-5lduknrxdbk9qvradp5dre.streamlit.app/)

## Models Implemented

### Black-Scholes
- Analytical pricing of European calls and puts
- Greeks: Delta, Gamma, Vega, Theta, Rho
- Implied volatility via Bisection and Newton-Raphson methods

### Monte Carlo Simulation
- European options pricing with convergence analysis
- Variance reduction via antithetic variates
- Path-dependent exotic options: Asian and Barrier (knock-out)

### Binomial Tree (Cox-Ross-Rubinstein)
- European and American options pricing
- Early exercise premium analysis
- Convergence analysis vs Black-Scholes

### Implied Volatility Surface
- Real market data from Yahoo Finance
- OTM puts and calls convention
- Interactive 3D surface visualization with Plotly

## Project Structure
```
options-pricing/
├── src/
│   ├── black_scholes.py
│   ├── monte_carlo.py
│   └── binomial_tree.py
├── notebooks/
│   ├── 01_black_scholes.ipynb
│   ├── 02_monte_carlo.ipynb
│   ├── 03_binomial_tree.ipynb
│   └── 04_implied_vol_surface.ipynb
├── app.py
└── requirements.txt
```

## Installation
```bash
git clone https://github.com/willy-guilleminot/options-pricing.git
cd options-pricing
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
streamlit run app.py
```

## Technologies
Python, NumPy, SciPy, Pandas, Matplotlib, Plotly, Streamlit, yFinance