from flask import Flask, render_template, request
import yfinance as yf
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import datetime

app = Flask(__name__)

def get_risk_free_rate():
    # Symbol for 3-month US Treasury Bill
    treasury_symbol = '^IRX'

    try:
        # Fetch the latest data
        treasury_bill = yf.Ticker(treasury_symbol)
        hist = treasury_bill.history(period="1d")  # Get the latest available data

        # Calculate the latest rate (last close value)
        latest_rate = hist['Close'].iloc[-1] / 100
        return latest_rate
    except Exception as e:
        print(f"Error fetching risk-free rate: {e}")
        return None

def get_historical_data(stocks, start='2010-01-01', end=None):
    if end is None:
        end = pd.to_datetime('today').strftime('%Y-%m-%d')
    data = yf.download(stocks, start=start, end=end)['Adj Close']
    return data

def calculate_portfolio_stats(weights, mean_returns, cov_matrix, risk_free_rate):
    portfolio_return = np.sum(mean_returns * weights) * 252
    portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_std_dev
    return portfolio_return, portfolio_std_dev, sharpe_ratio

def maximize_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix, risk_free_rate)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for asset in range(num_assets))
    initial_guess = num_assets * [1. / num_assets,]
    opt_results = minimize(lambda w: -calculate_portfolio_stats(w, *args)[2], 
                           initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)
    return opt_results.x

def optimal_portfolio(stocks):
    data = get_historical_data(stocks)
    mean_returns = data.pct_change().mean()
    cov_matrix = data.pct_change().cov()
    
    # Fetch the current risk-free rate
    risk_free_rate = get_risk_free_rate()

    if risk_free_rate is None:
        raise ValueError("Failed to retrieve the risk-free rate.")

    max_sharpe_weights = maximize_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate)
    expected_return, expected_volatility, expected_sharpe = calculate_portfolio_stats(max_sharpe_weights, mean_returns, cov_matrix, risk_free_rate)

    return max_sharpe_weights, expected_return, expected_volatility, expected_sharpe

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        stocks = request.form.get('stocks').split(',')
        stocks = [stock.strip() for stock in stocks]
        
        try:
            opt_weights, exp_return, exp_volatility, exp_sharpe = optimal_portfolio(stocks)
            
            results = {
                'weights': [f"{stock}: {weight:.2%}" for stock, weight in zip(stocks, opt_weights)],
                'return': f"{exp_return:.2%}",
                'volatility': f"{exp_volatility:.2%}",
                'sharpe': f"{exp_sharpe:.2f}"
            }
            
            return render_template('result.html', results=results)
        except Exception as e:
            error_message = f"An error occurred: {str(e)}"
            return render_template('index.html', error=error_message)
    
    return render_template('index.html')

# if __name__ == '__main__':
#     app.run(debug=True)