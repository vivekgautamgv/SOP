import streamlit as st
import home
import stock_screen
import options_pricing
import option_chain_analysis  # Ensure this file exists!

# ✅ Streamlit Page Configuration (Must be the FIRST command)
st.set_page_config(page_title="Stock & Options Analysis", layout="wide")

# ✅ Sidebar Navigation
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio("Select a Page:", ["Home", "Stock Screener", "Options Pricing", "Option Chain Analysis"])

# ✅ Page Routing
if page == "Home":
    home.show()  # Ensure home.py has a show() function
elif page == "Stock Screener":
    stock_screen.show()  # Ensure stock_screen.py has a show() function
elif page == "Options Pricing":
    options_pricing.show()  # Ensure options_pricing.py has a show() function
elif page == "Option Chain Analysis":
    option_chain_analysis.show()  # Ensure option_chain_analysis.py has a show() function
