# import streamlit as st
# import yfinance as yf
# import pandas as pd
# import plotly.graph_objects as go

# def show():  # ✅ This function allows navigation
#     # 📌 Page Configuration
#     st.title("📊 Stock Screener")

#     # 📌 Input for Stock Ticker
#     ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, TSLA):", "AAPL")

#     # Function to fetch stock data
#     def get_stock_data(ticker):
#         try:
#             stock = yf.Ticker(ticker)
#             stock_data = stock.history(period="6mo")  # Get last 6 months data
#             return stock_data
#         except Exception as e:
#             return None

#     # 📌 Function to Calculate Technical Indicators
#     def compute_technical_indicators(data):
#         data["SMA_50"] = data["Close"].rolling(window=50).mean()
#         data["SMA_200"] = data["Close"].rolling(window=200).mean()
#         data["EMA_20"] = data["Close"].ewm(span=20, adjust=False).mean()
#         data["RSI"] = 100 - (100 / (1 + data["Close"].pct_change().rolling(14).mean() /
#                                      data["Close"].pct_change().rolling(14).std()))
        
#         # Bollinger Bands
#         data["BB_Middle"] = data["Close"].rolling(window=20).mean()
#         data["BB_Upper"] = data["BB_Middle"] + (data["Close"].rolling(window=20).std() * 2)
#         data["BB_Lower"] = data["BB_Middle"] - (data["Close"].rolling(window=20).std() * 2)
        
#         return data

#     # 📌 Function to Compute Sentiment
#     def get_sentiment(data):
#         if data["RSI"].iloc[-1] > 70:
#             return "Bearish (Overbought)", "red"
#         elif data["RSI"].iloc[-1] < 30:
#             return "Bullish (Oversold)", "green"
#         else:
#             return "Neutral", "gray"

#     # 📌 Load Stock Data
#     if ticker:
#         stock_data = get_stock_data(ticker)

#         if stock_data is None or stock_data.empty:
#             st.warning("⚠ Stock data not found. Check the ticker symbol.")
#         else:
#             st.success(f"✅ Data loaded for {ticker}")

#             # 📌 Compute Technical Indicators
#             stock_data = compute_technical_indicators(stock_data)

#             # 📌 Display Stock Price Chart
#             st.subheader(f"{ticker} Closing Price Chart")
#             fig = go.Figure()
#             fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data["Close"], name="Close Price"))
#             fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data["SMA_50"], name="SMA 50", line=dict(color="blue")))
#             fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data["SMA_200"], name="SMA 200", line=dict(color="orange")))
#             fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data["BB_Upper"], name="BB Upper", line=dict(color="green", dash="dot")))
#             fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data["BB_Lower"], name="BB Lower", line=dict(color="red", dash="dot")))
#             st.plotly_chart(fig, use_container_width=True)

#             # 📌 Display Technical Indicators Table
#             st.subheader("📊 Technical Indicators")
#             st.dataframe(stock_data[["Close", "SMA_50", "SMA_200", "EMA_20", "RSI", "BB_Upper", "BB_Lower"]].tail(10))

#             # 📌 Display Sentiment Meter
#             sentiment, color = get_sentiment(stock_data)
#             st.subheader("📈 Stock Sentiment Meter")
#             st.markdown(f"""
#             <div style="text-align: center;">
#                 <span style="color: {color}; font-size: 24px; font-weight: bold;">{sentiment}</span>
#             </div>
#             """, unsafe_allow_html=True)
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import numpy as np

# 📌 Stock Screener Function
def show():
    # Page Title
    st.title("📊 Advanced Stock Screener")

    # User Input for Ticker
    ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, TSLA):", "AAPL")

    # Function to Fetch Stock Data
    def get_stock_data(ticker):
        try:
            stock = yf.Ticker(ticker)
            stock_data = stock.history(period="6mo")
            return stock_data
        except Exception as e:
            return None

    # Function to Compute Technical Indicators
    def compute_technical_indicators(data):
        # Moving Averages
        data["SMA_50"] = data["Close"].rolling(window=50).mean()
        data["SMA_200"] = data["Close"].rolling(window=200).mean()
        data["EMA_20"] = data["Close"].ewm(span=20, adjust=False).mean()
        
        # RSI (Relative Strength Index)
        delta = data["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        data["RSI"] = 100 - (100 / (1 + rs))

        # Bollinger Bands
        data["BB_Middle"] = data["Close"].rolling(window=20).mean()
        data["BB_Upper"] = data["BB_Middle"] + (data["Close"].rolling(window=20).std() * 2)
        data["BB_Lower"] = data["BB_Middle"] - (data["Close"].rolling(window=20).std() * 2)

        # MACD (Moving Average Convergence Divergence)
        data["MACD"] = data["Close"].ewm(span=12, adjust=False).mean() - data["Close"].ewm(span=26, adjust=False).mean()
        data["Signal_Line"] = data["MACD"].ewm(span=9, adjust=False).mean()

        # Stochastic Oscillator
        low_14 = data["Low"].rolling(window=14).min()
        high_14 = data["High"].rolling(window=14).max()
        data["Stochastic"] = 100 * (data["Close"] - low_14) / (high_14 - low_14)

        return data

    # Function to Compute Sentiment
    def get_sentiment(data):
        latest_rsi = data["RSI"].iloc[-1]
        latest_macd = data["MACD"].iloc[-1] - data["Signal_Line"].iloc[-1]  # MACD Crossover
        latest_price = data["Close"].iloc[-1]

        if latest_rsi > 70 and latest_macd < 0:
            return "Bearish (Overbought) - RSI is too high & MACD is signaling a downturn", "red"
        elif latest_rsi < 30 and latest_macd > 0:
            return "Bullish (Oversold) - RSI is low & MACD is trending upwards", "green"
        else:
            return "Neutral - No clear bullish or bearish signal", "gray"

    # Load Stock Data
    if ticker:
        stock_data = get_stock_data(ticker)

        if stock_data is None or stock_data.empty:
            st.warning("⚠ Stock data not found. Check the ticker symbol.")
        else:
            st.success(f"✅ Data loaded for {ticker}")

            # Compute Technical Indicators
            stock_data = compute_technical_indicators(stock_data)

            # 📌 Price Chart with Moving Averages & Bollinger Bands
            st.subheader(f"{ticker} Price Chart")
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(x=stock_data.index, y=stock_data["Close"], name="Close Price"))
            fig1.add_trace(go.Scatter(x=stock_data.index, y=stock_data["SMA_50"], name="SMA 50", line=dict(color="blue")))
            fig1.add_trace(go.Scatter(x=stock_data.index, y=stock_data["SMA_200"], name="SMA 200", line=dict(color="orange")))
            fig1.add_trace(go.Scatter(x=stock_data.index, y=stock_data["BB_Upper"], name="BB Upper", line=dict(color="green", dash="dot")))
            fig1.add_trace(go.Scatter(x=stock_data.index, y=stock_data["BB_Lower"], name="BB Lower", line=dict(color="red", dash="dot")))
            st.plotly_chart(fig1, use_container_width=True)

            # 📌 RSI Graph
            st.subheader("📊 RSI (Relative Strength Index)")
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=stock_data.index, y=stock_data["RSI"], name="RSI", line=dict(color="purple")))
            fig2.add_hline(y=70, line=dict(color="red", dash="dot"), annotation_text="Overbought")
            fig2.add_hline(y=30, line=dict(color="green", dash="dot"), annotation_text="Oversold")
            st.plotly_chart(fig2, use_container_width=True)

            # 📌 MACD Graph
            st.subheader("📊 MACD (Moving Average Convergence Divergence)")
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(x=stock_data.index, y=stock_data["MACD"], name="MACD", line=dict(color="blue")))
            fig3.add_trace(go.Scatter(x=stock_data.index, y=stock_data["Signal_Line"], name="Signal Line", line=dict(color="red")))
            st.plotly_chart(fig3, use_container_width=True)

            # 📌 Stochastic Oscillator Graph
            st.subheader("📊 Stochastic Oscillator")
            fig4 = go.Figure()
            fig4.add_trace(go.Scatter(x=stock_data.index, y=stock_data["Stochastic"], name="Stochastic Oscillator", line=dict(color="brown")))
            fig4.add_hline(y=80, line=dict(color="red", dash="dot"), annotation_text="Overbought")
            fig4.add_hline(y=20, line=dict(color="green", dash="dot"), annotation_text="Oversold")
            st.plotly_chart(fig4, use_container_width=True)

            # 📌 Display Technical Indicators Table
            st.subheader("📊 Technical Indicators Table")
            st.dataframe(stock_data[["Close", "SMA_50", "SMA_200", "EMA_20", "RSI", "MACD", "Signal_Line", "Stochastic"]].tail(10))

            # 📌 Display Sentiment Analysis with Explanation
            sentiment, color = get_sentiment(stock_data)
            st.subheader("📈 Stock Sentiment Analysis")
            st.markdown(f"""
            <div style="text-align: center;">
                <span style="color: {color}; font-size: 24px; font-weight: bold;">{sentiment}</span>
            </div>
            """, unsafe_allow_html=True)
