import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from scipy.stats import norm


# 🎯 **Main Show Function**
def show():
    st.title("📊 Options Pricing & Analysis")

    # 🔹 **User Input: Select Stock**
    ticker = st.text_input("Enter Stock Ticker", "AAPL")

    # 🔹 **Fetch Live Stock Data**
    if st.button("Fetch Latest Stock Price"):
        stock_data = get_stock_data(ticker)
        if stock_data:
            st.success(f"📈 Latest Price for {ticker}: ${stock_data['last_price']}")
        else:
            st.error("❌ Failed to fetch data. Check ticker or internet connection.")

    # 🔹 **User Input: Option Parameters**
    col1, col2 = st.columns(2)
    with col1:
        strike_price = st.number_input("Strike Price (K)", min_value=10.0, value=100.0, step=1.0)
        time_to_expiry = st.number_input("Time to Expiry (Years, T)", min_value=0.01, value=1.0, step=0.01)
    with col2:
        risk_free_rate = st.number_input("Risk-Free Rate (r)", min_value=0.0, value=0.05, step=0.01)
        volatility = st.number_input("Volatility (σ)", min_value=0.01, value=0.2, step=0.01)

    option_type = st.selectbox("Option Type", ["Call", "Put"])

    # 🔹 **Model Selection (Now allows multiple choices)**
    selected_models = st.multiselect(
        "Select Pricing Models", ["Black-Scholes", "Monte Carlo", "Binomial Tree", "Heston Model"],
        default=["Black-Scholes"]
    )

    # 🔹 **Calculate Option Prices**
    if st.button("Calculate Prices"):
        results = calculate_option_prices(ticker, strike_price, time_to_expiry, risk_free_rate, volatility, option_type, selected_models)
        st.subheader("📈 Model Comparison")
        st.table(results)

        # 🔹 **Visualizations**
        # if "Monte Carlo" in selected_models:
        #     visualize_monte_carlo(ticker, strike_price, time_to_expiry, risk_free_rate, volatility, option_type)
        # if "Binomial Tree" in selected_models:
        #     visualize_binomial_tree(ticker, strike_price, time_to_expiry, risk_free_rate, volatility, option_type)


# 📌 **Fetch Live Stock Data**
def get_stock_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        last_price = stock.history(period="1d")["Close"].iloc[-1]
        return {"last_price": round(last_price, 2)}
    except:
        return None


# 📌 **Calculate Option Prices for Selected Models**
def calculate_option_prices(ticker, K, T, r, sigma, option_type, selected_models):
    results = {"Model": [], "Option Price": []}
    stock_data = get_stock_data(ticker)
    if not stock_data:
        st.error("❌ Failed to fetch stock data. Ensure correct ticker.")
        return pd.DataFrame()

    S = stock_data["last_price"]
    
    for model in selected_models:
        if model == "Black-Scholes":
            price = black_scholes(S, K, T, r, sigma, option_type)
        elif model == "Monte Carlo":
            price = monte_carlo_simulation(S, K, T, r, sigma, option_type)
        elif model == "Binomial Tree":
            price = binomial_tree(S, K, T, r, sigma, option_type)
        elif model == "Heston Model":
            price = heston_model(S, K, T, r, sigma, option_type)
        else:
            price = None

        results["Model"].append(model)
        results["Option Price"].append(round(price, 2))

    return pd.DataFrame(results)


# 📌 **Black-Scholes Model**
def black_scholes(S, K, T, r, sigma, option_type):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == "Call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


# 📌 **Monte Carlo Simulation**
def monte_carlo_simulation(S, K, T, r, sigma, option_type, num_simulations=10000):
    np.random.seed(42)
    Z = np.random.standard_normal(num_simulations)
    ST = S * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * Z)
    
    if option_type == "Call":
        payoff = np.maximum(ST - K, 0)
    else:
        payoff = np.maximum(K - ST, 0)

    return np.exp(-r * T) * np.mean(payoff)


# 📌 **Binomial Tree Model**
def binomial_tree(S, K, T, r, sigma, option_type, steps=100):
    dt = T / steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)

    stock_price_tree = np.zeros((steps + 1, steps + 1))
    option_price_tree = np.zeros((steps + 1, steps + 1))

    for i in range(steps + 1):
        stock_price_tree[i, steps] = S * (u ** (steps - i)) * (d ** i)
    
    if option_type == "Call":
        option_price_tree[:, steps] = np.maximum(stock_price_tree[:, steps] - K, 0)
    else:
        option_price_tree[:, steps] = np.maximum(K - stock_price_tree[:, steps], 0)

    for j in range(steps - 1, -1, -1):
        for i in range(j + 1):
            option_price_tree[i, j] = np.exp(-r * dt) * (p * option_price_tree[i, j + 1] + (1 - p) * option_price_tree[i + 1, j + 1])

    return option_price_tree[0, 0]


# 📌 **Heston Model (Dummy Function)**
def heston_model(S, K, T, r, sigma, option_type):
    return black_scholes(S, K, T, r, sigma, option_type) * 1.05  # Adjusting Black-Scholes for volatility skew


# 📌 **Monte Carlo Visualization**
# def visualize_monte_carlo(S, K, T, r, sigma, option_type):
#     st.subheader("🎲 Monte Carlo Simulation")

#     paths = 100
#     time_steps = 50
#     dt = T / time_steps
#     S_t = np.zeros((time_steps, paths))
#     S_t[0] = S

#     for t in range(1, time_steps):
#         Z = np.random.standard_normal(paths)
#         S_t[t] = S_t[t - 1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z)

#     fig = go.Figure()
#     for i in range(paths):
#         fig.add_trace(go.Scatter(y=S_t[:, i], mode="lines", name=f"Path {i + 1}"))

#     fig.update_layout(title="Monte Carlo Price Paths", xaxis_title="Time Steps", yaxis_title="Stock Price")
#     st.plotly_chart(fig, use_container_width=True)


# 📌 **Run the App**
if __name__ == "__main__":
    show()
