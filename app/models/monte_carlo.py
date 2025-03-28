import numpy as np

def monte_carlo_option(S, K, T, r, sigma, num_simulations=10000, option_type="call"):
    """
    Monte Carlo simulation for option pricing.

    Parameters:
    S (float): Current stock price
    K (float): Strike price
    T (float): Time to expiration (in years)
    r (float): Risk-free interest rate
    sigma (float): Volatility of the stock
    num_simulations (int): Number of simulations
    option_type (str): "call" or "put"

    Returns:
    float: Estimated option price
    """
    np.random.seed(42)  # For reproducibility
    dt = T / 252  # Daily time step
    stock_prices = S * np.exp(np.cumsum((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * np.random.randn(num_simulations)))

    if option_type == "call":
        payoff = np.maximum(stock_prices - K, 0)
    elif option_type == "put":
        payoff = np.maximum(K - stock_prices, 0)
    else:
        raise ValueError("Invalid option type. Use 'call' or 'put'.")

    option_price = np.exp(-r * T) * np.mean(payoff)
    return option_price
