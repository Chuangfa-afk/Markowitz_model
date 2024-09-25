from flask import Flask, render_template, request
import yfinance as yf
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import plotly

app = Flask(__name__)

def get_risk_free_rate():
    treasury_symbol = '^IRX'
    try:
        treasury_bill = yf.Ticker(treasury_symbol)
        hist = treasury_bill.history(period="1d")
        latest_rate = hist['Close'].iloc[-1] / 100
        return latest_rate
    except Exception as e:
        print(f"Error fetching risk-free rate: {e}")
        return 0.02  # Default to 2% if unable to fetch

def get_historical_data(stocks, start='2010-01-01', end=None):
    if end is None:
        end = pd.to_datetime('today').strftime('%Y-%m-%d')
    data = yf.download(stocks, start=start, end=end)['Adj Close']
    insufficient_data = [stock for stock in stocks if (data.index[-1] - data[stock].first_valid_index()).days < 3650]
    return data, insufficient_data

def calculate_portfolio_stats(weights, returns, risk_free_rate):
    portfolio_return = np.sum(returns.mean() * weights) * 252
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
    return portfolio_return, portfolio_volatility, sharpe_ratio

def negative_sharpe_ratio(weights, returns, risk_free_rate):
    return -calculate_portfolio_stats(weights, returns, risk_free_rate)[2]

def maximize_sharpe_ratio(returns, risk_free_rate):
    num_assets = returns.shape[1]
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    initial_weights = np.array([1.0 / num_assets] * num_assets)
    
    optimized = minimize(negative_sharpe_ratio, initial_weights, 
                         args=(returns, risk_free_rate),
                         method='SLSQP', bounds=bounds, constraints=constraints)
    
    return optimized.x

def optimal_portfolio(stocks):
    data, insufficient_data = get_historical_data(stocks)
    returns = data.pct_change().dropna()
    risk_free_rate = get_risk_free_rate()
    
    optimal_weights = maximize_sharpe_ratio(returns, risk_free_rate)
    expected_return, expected_volatility, expected_sharpe = calculate_portfolio_stats(optimal_weights, returns, risk_free_rate)
    
    return optimal_weights, expected_return, expected_volatility, expected_sharpe, insufficient_data

def create_comparison_chart(stocks, weights, start_date='2010-01-01'):
    data, _ = get_historical_data(stocks + ['^GSPC'], start_date)
    returns = data.pct_change().dropna()
    
    portfolio_returns = returns[stocks].dot(weights)
    cumulative_returns = (1 + returns).cumprod()
    portfolio_cumulative_returns = (1 + portfolio_returns).cumprod()
    
    ann_returns = (cumulative_returns.iloc[-1] ** (252 / len(cumulative_returns)) - 1)
    ann_volatilities = returns.std() * np.sqrt(252)
    
    portfolio_ann_return = (portfolio_cumulative_returns.iloc[-1] ** (252 / len(portfolio_cumulative_returns)) - 1)
    portfolio_ann_volatility = portfolio_returns.std() * np.sqrt(252)
    
    risk_free_rate = get_risk_free_rate()
    sharpe_ratios = (ann_returns - risk_free_rate) / ann_volatilities
    portfolio_sharpe = (portfolio_ann_return - risk_free_rate) / portfolio_ann_volatility
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.1, subplot_titles=('Cumulative Returns', 'Risk-Return Trade-off'))
    
    for stock in stocks:
        fig.add_trace(go.Scatter(x=cumulative_returns.index, y=cumulative_returns[stock], name=stock, mode='lines'), row=1, col=1)
    fig.add_trace(go.Scatter(x=portfolio_cumulative_returns.index, y=portfolio_cumulative_returns, name='Portfolio', mode='lines', line=dict(width=4)), row=1, col=1)
    fig.add_trace(go.Scatter(x=cumulative_returns.index, y=cumulative_returns['^GSPC'], name='S&P 500', mode='lines'), row=1, col=1)
    
    fig.add_trace(go.Scatter(x=ann_volatilities[stocks], y=ann_returns[stocks], mode='markers+text', 
                             text=stocks, textposition="top center", name='Individual Stocks'), row=2, col=1)
    fig.add_trace(go.Scatter(x=[portfolio_ann_volatility], y=[portfolio_ann_return], mode='markers+text', 
                             marker=dict(size=15), text=['Portfolio'], textposition="top center", name='Portfolio'), row=2, col=1)
    fig.add_trace(go.Scatter(x=[ann_volatilities['^GSPC']], y=[ann_returns['^GSPC']], mode='markers+text', 
                             marker=dict(size=15), text=['S&P 500'], textposition="top center", name='S&P 500'), row=2, col=1)
    
    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_xaxes(title_text="Annualized Volatility", row=2, col=1)
    fig.update_yaxes(title_text="Cumulative Returns", row=1, col=1)
    fig.update_yaxes(title_text="Annualized Return", row=2, col=1)
    
    fig.update_layout(height=1000, title_text="Portfolio Performance and Risk-Return Trade-off")
    
    for i, stock in enumerate(stocks + ['Portfolio', '^GSPC']):
        if stock == 'Portfolio':
            sharpe = portfolio_sharpe
        elif stock == '^GSPC':
            sharpe = sharpe_ratios['^GSPC']
        else:
            sharpe = sharpe_ratios[stock]
        fig.add_annotation(x=0.05, y=0.95 - i*0.05, xref="paper", yref="paper",
                           text=f"{stock} Sharpe Ratio: {sharpe:.2f}", showarrow=False, align="left")
    
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        stocks = request.form.get('stocks').split(',')
        stocks = [stock.strip() for stock in stocks]
        try:
            opt_weights, exp_return, exp_volatility, exp_sharpe, insufficient_data = optimal_portfolio(stocks)
            chart_json = create_comparison_chart(stocks, opt_weights)
            results = {
                'weights': [f"{stock}: {weight:.2%}" for stock, weight in zip(stocks, opt_weights)],
                'return': f"{exp_return:.2%}",
                'volatility': f"{exp_volatility:.2%}",
                'sharpe': f"{exp_sharpe:.2f}",
                'chart_json': chart_json,
                'insufficient_data': insufficient_data
            }
            return render_template('result.html', results=results)
        except Exception as e:
            error_message = f"An error occurred: {str(e)}"
            return render_template('index.html', error=error_message)
    return render_template('index.html')

# if __name__ == '__main__':
#     app.run(debug=True)
    # Test code
