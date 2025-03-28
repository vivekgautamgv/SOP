import streamlit as st

def show():
    st.title("📈 Stock & Options Analysis")
    st.markdown("""
        Welcome to the **Stock & Options Analysis** tool.  
        - 📊 **Stock Screener**: Analyze stocks using technical indicators  
        - 🏦 **Options Pricing**: Black-Scholes & Monte Carlo models  
        - 🔗 **Option Chain Analysis**: Identify option pricing trends  
        
        Use the **sidebar navigation** to explore different sections.
    """)
