import streamlit as st
import matplotlib.pyplot as plt
from matplotlib import cm
import yfinance as yf
import pandas as pd
from scipy.interpolate import griddata
from datetime import date, datetime
import plotly.graph_objects as go

from src.black_scholes import *
from src.monte_carlo import *
from src.binomial_tree import *

def plot_options_price_spot(S, K, T, r, sigma, start_range, end_range, n_steps=800):
    spots = np.linspace(start_range, end_range, n_steps)
    calls = [call_price(spot, K, T, r, sigma) for spot in spots]
    puts = [put_price(spot, K, T, r, sigma) for spot in spots]

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(14, 5))

    ax[0].plot(spots, calls, color='tomato')
    ax[0].set_xlabel('Spot')
    ax[0].set_ylabel('Call price')
    ax[0].grid(True, alpha=0.4)

    ax[1].plot(spots, puts, color='steelblue')
    ax[1].set_xlabel('Spot')
    ax[1].set_ylabel('Put price')
    ax[1].grid(True, alpha=0.4)

    plt.tight_layout()

    return fig

def plot_greeks(S, K, T, r, sigma, start_range, end_range, n_steps=800):
    spots = np.linspace(start_range, end_range, n_steps)
    deltaCallList = [delta_call(spot, K, T, r, sigma) for spot in spots]
    deltaPutList = [delta_put(spot, K, T, r, sigma) for spot in spots]
    gammaList = [gamma(spot, K, T, r, sigma) for spot in spots]
    vegaList = [vega(spot, K, T, r, sigma) for spot in spots]
    thetaCallList = [theta_call(spot, K, T, r, sigma) for spot in spots]
    thetaPutList = [theta_put(spot, K, T, r, sigma) for spot in spots]
    rhoCallList = [rho_call(spot, K, T, r, sigma) for spot in spots]
    rhoPutList = [rho_put(spot, K, T, r, sigma) for spot in spots]

    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(16, 8))

    fig.suptitle('Greeks')

    ax[0,0].plot(spots, deltaCallList, color='tomato', label='Call delta')
    ax[0,0].plot(spots, deltaPutList, color='steelblue', label='Put Delta')
    ax[0,0].set_xlabel('Spot')
    ax[0,0].set_ylabel('Delta')
    ax[0,0].grid(True, alpha=0.4)
    ax[0,0].legend()

    ax[0,1].plot(spots, gammaList, color='blue', label='Call/Put gamma')
    ax[0,1].set_xlabel('Spot')
    ax[0,1].set_ylabel('Gamma')
    ax[0,1].grid(True, alpha=0.4)
    ax[0,1].legend()

    ax[0,2].plot(spots, vegaList, color='blue', label='Call/Put vega')
    ax[0,2].set_xlabel('Spot')
    ax[0,2].set_ylabel('Vega')
    ax[0,2].grid(True, alpha=0.4)
    ax[0,2].legend()

    ax[1,0].plot(spots, thetaCallList, color='tomato', label='Call theta')
    ax[1,0].plot(spots, thetaPutList, color='steelblue', label='Put theta')
    ax[1,0].set_xlabel('Spot')
    ax[1,0].set_ylabel('Theta')
    ax[1,0].grid(True, alpha=0.4)
    ax[1,0].legend()

    ax[1,1].plot(spots, rhoCallList, color='tomato', label='Call rho')
    ax[1,1].plot(spots, rhoPutList, color='steelblue', label='Put rho')
    ax[1,1].set_xlabel('Spot')
    ax[1,1].set_ylabel('Rho')
    ax[1,1].grid(True, alpha=0.4)
    ax[1,1].legend()

    ax[1,2].set_visible(False)

    plt.tight_layout()

    return fig

def plot_simulation_paths(S, T, r, sigma, n_simulations, n_steps):
    x = np.linspace(0, n_steps + 1, n_steps + 1)
    paths = simulate_paths(S, T, r, sigma, n_simulations, n_steps)

    fig = plt.figure(figsize=(7,5))

    for row_index in range(paths.shape[0]):
        plt.plot(x, paths[row_index, :], linewidth=0.9)

    plt.title(f'Monte Carlo with {n_simulations} simulations and {n_steps} steps')
    plt.xlabel('Time')
    plt.ylabel('Spot price')
    plt.grid(True, alpha=0.4)

    return fig

@st.cache_data
def plot_monte_carlo_conv(S, K, T, r, sigma, n_simulations):
    x = list(range(1, n_simulations + 1))
    monte_carlo_prices = [mc_call_price(S, K, T, r, sigma, n) for n in x]

    fig = plt.figure(figsize=(7, 5))
    
    plt.plot(x, monte_carlo_prices, linewidth=0.9, color='steelblue', label='Monte Carlo call price')
    plt.axhline(call_price(S, K, T, r, sigma), linestyle='--', linewidth=0.9, color='black', label=f'Black & Scholes call price {call_price(S, K, T, r, sigma):.4f}')
    plt.xlabel('Number of simulation')
    plt.ylabel('Call price')
    plt.grid(True, alpha=0.4)
    plt.legend()
    
    return fig

def load_spot_price(ticker_symbol):
    ticker = yf.Ticker(ticker_symbol)
    S = ticker.history()['Close'].iloc[-1]

    return S

@st.cache_data
def load_vol_surface_data(ticker_symbol, r):
    ticker = yf.Ticker(ticker_symbol)
    S = load_spot_price(ticker_symbol)

    call_prices = []
    call_strikes = []
    call_maturities = []
    call_type = []
    #call_volumes = []

    put_prices = []
    put_strikes = []
    put_maturities = []
    put_type = []
    #put_volumes = []

    for maturity in ticker.options:
        chain = ticker.option_chain(maturity)
        for _ in range(len(chain.calls)):
            call_maturities.append(maturity)
            call_prices.append((chain.calls['bid'][_] + chain.calls['ask'][_]) / 2)
            call_strikes.append(chain.calls['strike'][_])
            call_type.append('call')
            #call_volumes.append(chain.calls['volume'][_])
        for _ in range(len(chain.puts)):
            put_maturities.append(maturity)
            put_prices.append((chain.puts['bid'][_] + chain.puts['ask'][_]) / 2)
            put_strikes.append(chain.puts['strike'][_])
            put_type.append('put')
            #put_volumes.append(chain.puts['volume'][_])

    #calls = pd.DataFrame({'price': call_prices, 'strike': call_strikes, 'maturityDate': call_maturities, 'volume': call_volumes})
    #puts = pd.DataFrame({'price': put_prices, 'strike': put_strikes, 'maturityDate': put_maturities, 'volume': put_volumes})

    calls = pd.DataFrame({'price': call_prices, 'strike': call_strikes, 'maturityDate': call_maturities, 'optionType': call_type})
    puts = pd.DataFrame({'price': put_prices, 'strike': put_strikes, 'maturityDate': put_maturities, 'optionType': put_type})

    options = pd.concat([calls[calls['strike'] >= S], puts[puts['strike'] <= S]]).reset_index(drop=True)

    options['maturityDate'] = pd.to_datetime(options['maturityDate'])
    #calls['maturityDate'] = pd.to_datetime(calls['maturityDate'])
    #puts['maturityDate'] = pd.to_datetime(puts['maturityDate'])

    options_maturities = []
    # call_maturities = []
    # put_maturities = []
    for _ in range(len(options['maturityDate'])):
        diff = options['maturityDate'][_].to_pydatetime() - datetime.today()
        options_maturities.append(diff.days/252)
    # for _ in range(len(puts['maturityDate'])):
    #     diff = puts['maturityDate'][_].to_pydatetime() - datetime.today()
    #     put_maturities.append(diff.days/252)

    options['maturity'] = options_maturities
    #calls['maturity'] = call_maturities
    #puts['maturity'] = put_maturities

    options = options[options['maturity'] >= 7/252].reset_index(drop=True)
    # calls = calls[calls['maturity'] >= 7/252].reset_index(drop=True)
    # puts = puts[puts['maturity'] >= 7/252].reset_index(drop=True)

    options = options[options['price'] > 0].reset_index(drop=True)
    # calls = calls[calls['price'] > 0].reset_index(drop=True)
    # puts = puts[puts['price'] > 0].reset_index(drop=True)

    options = options[(options['strike'] >= 0.2 * S) & (options['strike'] <= 1.8 * S)].reset_index(drop=True)
    # calls = calls[(calls['strike'] >= 0.2 * S) & (calls['strike'] <= 1.8 * S)].reset_index(drop=True)
    # puts = puts[(puts['strike'] >= 0.2 * S) & (puts['strike'] <= 1.2 * S)].reset_index(drop=True)

    #calls = calls[calls['volume'] > 10].reset_index(drop=True)
    #puts = puts[puts['volume'] > 10].reset_index(drop=True)

    # calls_implied_vol = []
    # for _, row in calls.iterrows():
    #     try:
    #         implied_vol = implied_volatility(
    #             row['price'],
    #             S, 
    #             row['strike'], 
    #             row['maturity'], 
    #             r=0.05, 
    #             method='bisection'     
    #         )
    #         calls_implied_vol.append(implied_vol)
    #     except:
    #         calls_implied_vol.append(None)

    # calls['implied_vol'] = calls_implied_vol
    # print(f'{calls['implied_vol'].isna().sum()} calls without implied volatility')
    # print(calls.head())

    options_implied_vol = []
    for _, row in options.iterrows():
        try:
            implied_vol = implied_volatility(
                row['price'],
                S,
                row['strike'],
                row['maturity'],
                r,
                option_type=row['optionType'],
                method='bisection'
            )
            options_implied_vol.append(implied_vol)
        except:
            options_implied_vol.append(None)

    options['implied_vol'] = options_implied_vol

    # options = options[(options['implied_vol'] > 0.01) & (options['implied_vol'] < 1)].reset_index(drop=True)
    # calls = calls[(calls['implied_vol'] > 0.01) & (calls['implied_vol'] < 1)].reset_index(drop=True)
    # puts = puts[(puts['implied_vol'] > 0.01) & (puts['implied_vol'] < 1)].reset_index(drop=True)

    options.dropna(inplace=True)
    # calls.dropna(inplace=True)
    # puts.dropna(inplace=True)

    options['ticker'] = [ticker_symbol for _ in range(options.shape[0])]

    return options

def plot_vol_surf(options, method='linear'):
    strikes_grid = np.linspace(options['strike'].min(), options['strike'].max(), 100)
    maturities_grid = np.linspace(options['maturity'].min(), options['maturity'].max(), 100)
    K_grid, T_grid = np.meshgrid(strikes_grid, maturities_grid)

    vol_grid = griddata(
        points=(options['strike'], options['maturity']),
        values=options['implied_vol'],
        xi=(K_grid, T_grid),
        method=method
    )

    fig = go.Figure(data=[go.Surface(
        x=K_grid,
        y=T_grid,
        z=vol_grid
    )])

    fig.update_layout(
        title=f"{options['ticker'].iloc[0]} Implied Volatility Surface",
        scene=dict(
            xaxis_title='Strike',
            yaxis_title='Maturity',
            zaxis_title='Implied Vol'
        )
    )

    return fig
    
st.set_page_config(layout="wide")

# Title and text
st.title("Options pricing engine")
st.write("Visiualize Black Scholes, Monte Carlo, Binomial tree. Import real data and compute the implied volatility surface")

col1, col2 = st.columns(2)

# Sliders et inputs
S = col1.number_input("Spot price", value=100.0, min_value=0.0)
K = col1.slider("Strike", min_value=1, max_value=int(S * 2), value=int(S))
T = col1.slider("Maturity (in years)", min_value=7/252, max_value=5.0, value=1.0)
r = col1.slider("Interest rate", min_value=0.0, max_value=1.0, value=0.05)
sigma = col1.slider("Volatility", min_value=0.01, max_value=2.0, value=0.2)

# Afficher des valeurs
#st.metric("Black & Scholes call price", f"{call_price(S, K, T, r, sigma):.2f}")
#st.metric("Black & Scholes put price", f"{put_price(S, K, T, r, sigma):.2f}")
col1_1, col1_2 = col1.columns(2)
col1_1.metric("Black & Scholes call price", f"{call_price(S, K, T, r, sigma):.2f}")
col1_2.metric("Black & Scholes put price", f"{put_price(S, K, T, r, sigma):.2f}")

# Afficher un graphe matplotlib
#st.pyplot(fig)

# Onglets
tab1, tab2, tab3 = col2.tabs(["Black-Scholes", "Monte Carlo", "Volatility Surface"])
with tab1:
    start_range = st.slider("Lower bound", min_value=0, max_value=int(S), value=0)
    end_range = st.slider("Upper bound", min_value=int(S), max_value=int(S*4), value=int(S*2))

    options_price_spot_fig = plot_options_price_spot(S, K, T, r, sigma, start_range, end_range)
    greeks_fig = plot_greeks(S, K, T, r, sigma, start_range, end_range)

    tab1_1, tab1_2 = st.tabs(["Options price plot", "Greeks plot"])
    with tab1_1:
        st.pyplot(options_price_spot_fig)
    with tab1_2:
        st.pyplot(greeks_fig)

with tab2:
    n_simulations_1 = st.number_input("Number of simulations for paths simulation chart", value=30, min_value=1)
    n_simulations_2 = st.number_input("Number of simulations for Monte Carlo convergence", value=5000, min_value=1)
    n_steps = st.number_input("Number of steps", value=252, min_value=1)

    simulation_paths_fig = plot_simulation_paths(S, T, r, sigma, n_simulations_1, n_steps)
    monte_carlo_conv_fig = plot_monte_carlo_conv(S, K, T, r, sigma, n_simulations_2)

    st.pyplot(simulation_paths_fig)
    st.pyplot(monte_carlo_conv_fig)

with tab3:
    tab3_col1, tab3_col2 = st.columns(2)
    ticker_input = tab3_col1.text_input("Ticker", value="SPY")
    tab3_col2.metric(f"{ticker_input} spot price", f"{load_spot_price(ticker_input):.2f}")
    if tab3_col1.button("Load data"):
        with st.spinner("Loading data..."):
            options = load_vol_surface_data(ticker_input, r)
            st.session_state['options'] = options
        st.toast("Data loaded successfully!")

    if tab3_col2.button("Plot volatility surface"):
        if 'options' in st.session_state:
            vol_surf_fig = plot_vol_surf(st.session_state['options'])
            st.plotly_chart(vol_surf_fig, use_container_width=True)
        else:
            st.error("Please load data first!")