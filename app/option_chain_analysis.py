import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
import scipy.stats as stats

# 🎯 **Main Show Function**
def show():
    st.title("📊 Advanced Option Chain Analysis")

    # 📌 **User Input for Stock Ticker**
    ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, TSLA, SPY)", "AAPL")

    # 📌 **Fetch & Display Option Chain Data**
    if st.button("Load Option Chain"):
        # Fetch current stock data
        stock = yf.Ticker(ticker)
        current_price = stock.history(period="1d")['Close'].iloc[-1]
        st.metric("Current Stock Price", f"${current_price:.2f}")

        option_chain_data = get_option_chain(ticker, current_price)
        if option_chain_data.empty:
            st.error("⚠️ No option chain data found. Try another ticker.")
        else:
            st.subheader(f"📜 Option Chain for {ticker}")
            st.dataframe(option_chain_data)

            # 📌 **Visualizations**
            visualize_bid_ask_spread(option_chain_data)
            visualize_open_interest(option_chain_data)
            visualize_volatility_surface(option_chain_data)
            visualize_greeks(option_chain_data)

# 📌 **Fetch Live Option Chain Data with Accurate Greeks**
def get_option_chain(ticker, current_price):
    try:
        stock = yf.Ticker(ticker)
        expirations = stock.options  

        if not expirations:
            return pd.DataFrame()

        # Get the nearest expiration
        expiry = expirations[0]  
        options = stock.option_chain(expiry)

        # Fetch stock-specific parameters
        stock_info = stock.info
        risk_free_rate = 0.05  # Approximating current risk-free rate
        
        # Prepare calls and puts
        calls = options.calls[['strike', 'bid', 'ask', 'openInterest', 'impliedVolatility']].copy()
        puts = options.puts[['strike', 'bid', 'ask', 'openInterest', 'impliedVolatility']].copy()

        calls['Type'] = 'Call'
        puts['Type'] = 'Put'
        
        # Handle missing bid/ask values
        calls[['bid', 'ask']] = calls[['bid', 'ask']].replace(0, np.nan)
        puts[['bid', 'ask']] = puts[['bid', 'ask']].replace(0, np.nan)

        # Use more accurate implied volatility estimation
        calls['impliedVolatility'] = calls['impliedVolatility'].fillna(
            calls['impliedVolatility'].median() if not calls['impliedVolatility'].median() == 0 
            else 0.3  # Default volatility if no data
        )
        puts['impliedVolatility'] = puts['impliedVolatility'].fillna(
            puts['impliedVolatility'].median() if not puts['impliedVolatility'].median() == 0 
            else 0.3  # Default volatility if no data
        )

        # Calculate accurate Greeks using Black-Scholes model
        def black_scholes_greeks(S, K, T, r, sigma, option_type):
            # Time to expiration (approximated to 30 days)
            T = 30 / 365  # Assuming first expiration is about a month out
            
            # Standard deviation of log returns
            d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            
            if option_type == 'Call':
                delta = stats.norm.cdf(d1)
                gamma = stats.norm.pdf(d1) / (S * sigma * np.sqrt(T))
                vega = S * stats.norm.pdf(d1) * np.sqrt(T)
                theta = -(S * stats.norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * stats.norm.cdf(d2)
                rho = K * T * np.exp(-r * T) * stats.norm.cdf(d2)
            else:  # Put
                delta = -stats.norm.cdf(-d1)
                gamma = stats.norm.pdf(d1) / (S * sigma * np.sqrt(T))
                vega = S * stats.norm.pdf(d1) * np.sqrt(T)
                theta = -(S * stats.norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * stats.norm.cdf(-d2)
                rho = -K * T * np.exp(-r * T) * stats.norm.cdf(-d2)
            
            return delta, gamma, vega, theta, rho

        # Calculate Greeks for each option
        calls[['Delta', 'Gamma', 'Vega', 'Theta', 'Rho']] = calls.apply(
            lambda row: black_scholes_greeks(
                current_price, row['strike'], 30/365, risk_free_rate, 
                row['impliedVolatility'], 'Call'
            ), axis=1, result_type='expand'
        )

        puts[['Delta', 'Gamma', 'Vega', 'Theta', 'Rho']] = puts.apply(
            lambda row: black_scholes_greeks(
                current_price, row['strike'], 30/365, risk_free_rate, 
                row['impliedVolatility'], 'Put'
            ), axis=1, result_type='expand'
        )

        # Combine and return data
        data = pd.concat([calls, puts]).reset_index(drop=True)
        return data

    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

# 📌 **Visualization for Bid-Ask Spread**
def visualize_bid_ask_spread(data):
    st.subheader("📈 Bid-Ask Spread")
    fig = go.Figure()
    
    # Filter out rows with None/NaN bid and ask
    valid_data = data[data['bid'].notna() & data['ask'].notna()]
    
    if not valid_data.empty:
        fig.add_trace(go.Scatter(x=valid_data["strike"], y=valid_data["bid"], mode="lines+markers", name="Bid", line=dict(color="green")))
        fig.add_trace(go.Scatter(x=valid_data["strike"], y=valid_data["ask"], mode="lines+markers", name="Ask", line=dict(color="red")))
        fig.update_layout(title="Bid-Ask Spread", xaxis_title="Strike Price", yaxis_title="Price")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No bid-ask data available for visualization")

# 📌 **Visualization for Open Interest**
def visualize_open_interest(data):
    st.subheader("📊 Open Interest Across Strikes")
    fig = go.Figure(go.Bar(x=data["strike"], y=data["openInterest"], marker_color="blue"))
    fig.update_layout(title="Open Interest", xaxis_title="Strike Price", yaxis_title="Open Interest")
    st.plotly_chart(fig, use_container_width=True)

# 📌 **Volatility Surface Plot**
def visualize_volatility_surface(data):
    st.subheader("🌎 Implied Volatility Surface")

    # Create a more meaningful 3D surface
    fig = go.Figure(data=[
        go.Surface(
            z=data["impliedVolatility"].values.reshape(-1, 1), 
            x=data["strike"], 
            y=data["openInterest"],
            colorscale='Viridis'
        )
    ])
    fig.update_layout(
        title="Implied Volatility Surface", 
        scene=dict(
            xaxis_title="Strike Price", 
            yaxis_title="Open Interest", 
            zaxis_title="Implied Volatility"
        )
    )
    st.plotly_chart(fig, use_container_width=True)

# 📌 **Visualization for Greeks**
def visualize_greeks(data):
    st.subheader("📉 Option Greeks Across Strikes")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data["strike"], y=data["Delta"], mode="lines+markers", name="Delta", line=dict(color="red")))
    fig.add_trace(go.Scatter(x=data["strike"], y=data["Gamma"], mode="lines+markers", name="Gamma", line=dict(color="green")))
    fig.add_trace(go.Scatter(x=data["strike"], y=data["Vega"], mode="lines+markers", name="Vega", line=dict(color="blue")))
    fig.add_trace(go.Scatter(x=data["strike"], y=data["Theta"], mode="lines+markers", name="Theta", line=dict(color="orange")))
    fig.add_trace(go.Scatter(x=data["strike"], y=data["Rho"], mode="lines+markers", name="Rho", line=dict(color="purple")))

    fig.update_layout(title="Greeks Analysis", xaxis_title="Strike Price", yaxis_title="Value")
    st.plotly_chart(fig, use_container_width=True)

# 📌 **Run the App**
if __name__ == "__main__":
    show()