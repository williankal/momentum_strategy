import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def load_data(file_path):
    """Loads data from a JSON file and converts to DataFrame"""
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.set_index('timestamp')
    return df

def load_all_data(data_folder='data'):
    """Loads all JSON files from the data folder"""
    data = {}
    for file in os.listdir(data_folder):
        if file.endswith('.json'):
            path = os.path.join(data_folder, file)
            # Extract cryptocurrency symbol from the filename
            symbol = file.split('_')[0].replace('usd', '')
            try:
                data[symbol] = load_data(path)
                print(f"Loaded: {symbol} - {len(data[symbol])} records")
            except Exception as e:
                print(f"Error loading {file}: {e}")
    return data

def top_performers_strategy(crypto_data, lookback_period=7, num_top_coins=5, initial_capital=10000, commission=0.001):
    """
    Implements a strategy that buys the top performing cryptocurrencies each week.
    
    Parameters:
    -----------
    crypto_data : dict
        Dictionary of cryptocurrency DataFrames
    lookback_period : int
        Number of days to look back for performance ranking
    num_top_coins : int
        Number of top performers to buy each period
    initial_capital : float
        Initial capital for the strategy
    commission : float
        Commission rate per trade
        
    Returns:
    --------
    tuple: (portfolio_df, metrics)
        portfolio_df: DataFrame with portfolio performance
        metrics: Dictionary with performance metrics
    """
    # Filter cryptocurrencies with sufficient data
    valid_cryptos = {}
    for symbol, data in crypto_data.items():
        if len(data) > lookback_period + 10:  # Ensure enough data
            valid_cryptos[symbol] = data
    
    print(f"Found {len(valid_cryptos)} cryptocurrencies with sufficient data")
    
    if len(valid_cryptos) == 0:
        print("Error: No cryptocurrencies with sufficient data found.")
        return None, None
    
    # Check for BTC data for benchmark
    if 'btc' not in valid_cryptos:
        print("Warning: Bitcoin data not found. Cannot use as benchmark.")
        btc_data = None
    else:
        btc_data = valid_cryptos['btc']
    
    # Find common date range
    start_dates = [data.index.min() for data in valid_cryptos.values()]
    end_dates = [data.index.max() for data in valid_cryptos.values()]

    print(start_dates, end_dates)
    
    common_start = max(start_dates)
    common_end = min(end_dates)
    
    print(f"Common date range: {common_start.date()} to {common_end.date()}")
    
    # Create weekly rebalance points (every Monday)
    start_monday = common_start + timedelta(days=(7 - common_start.weekday()) % 7)
    end_monday = common_end - timedelta(days=common_end.weekday())
    
    rebalance_dates = pd.date_range(start=start_monday, end=end_monday, freq='W-MON')
    print(f"Strategy will run from {start_monday.date()} to {end_monday.date()} with {len(rebalance_dates)} rebalance points")
    
    # Create dataframe for daily prices of all cryptocurrencies
    # Make sure to include the lookback period before the start date
    all_dates = pd.date_range(start=common_start - timedelta(days=lookback_period*2), end=common_end, freq='D')
    price_df = pd.DataFrame(index=all_dates)
    
    # Fill with prices
    for symbol, data in valid_cryptos.items():
        # Get close prices and handle missing dates with forward fill
        symbol_prices = pd.DataFrame({'close': data['close']})
        reindexed_prices = symbol_prices.reindex(all_dates)
        price_df[symbol] = reindexed_prices['close'].fillna(method='ffill')
    
    # Remove cryptocurrencies with too many missing values
    price_df = price_df.dropna(axis=1, thresh=len(all_dates)*0.95)  # Keep columns with at least 95% data
    print(f"After removing cryptocurrencies with insufficient data: {price_df.shape[1]} cryptocurrencies")
    
    if price_df.shape[1] == 0:
        print("Error: No cryptocurrencies with sufficient data found.")
        return None, None
    
    # Fill any remaining NaNs with forward fill followed by backward fill
    price_df = price_df.fillna(method='ffill').fillna(method='bfill')
    
    # Calculate returns for ranking
    returns_df = price_df.pct_change(lookback_period).dropna()
    
    # Create portfolio tracking dataframe (will be populated properly later)
    portfolio_df = pd.DataFrame(index=all_dates)
    portfolio_df['Cash'] = 0.0
    portfolio_df['Crypto_Value'] = 0.0
    portfolio_df['Total_Value'] = 0.0
    
    # Simulate weekly rebalancing
    print("Simulating weekly rebalancing strategy...")
    
    # We'll track our strategy more carefully now
    portfolio_history = []
    holdings_history = {}
    
    # Initialize portfolio tracking for the correct time period
    portfolio_df = portfolio_df.loc[start_monday:]
    portfolio_df['Cash'] = initial_capital
    portfolio_df['Crypto_Value'] = 0.0
    portfolio_df['Total_Value'] = initial_capital
    
    # Track portfolio value day by day
    current_holdings = {}
    current_cash = initial_capital
    
    # Apply the rebalancing decisions
    for i in range(len(rebalance_dates)):
        rebalance_date = rebalance_dates[i]
        
        # Get the next rebalance date or the end date
        next_rebalance = rebalance_dates[i+1] if i < len(rebalance_dates) - 1 else common_end
        next_day = min(next_rebalance, common_end)
        
        # Skip if we're beyond our date range
        if rebalance_date > common_end:
            break
            
        # Get top performers for this week
        if rebalance_date in returns_df.index:
            returns_at_date = returns_df.loc[rebalance_date]
            top_performers = returns_at_date.nlargest(num_top_coins).index.tolist()
            
            # Record holdings for this week
            week_key = rebalance_date.strftime('%Y-%W')
            holdings_history[week_key] = top_performers
            
            # Sell all current holdings
            for symbol, units in current_holdings.items():
                if symbol in price_df.columns:
                    price = price_df.loc[rebalance_date, symbol]
                    current_cash += units * price * (1 - commission)
            
            # Reset holdings
            current_holdings = {}
            
            # Buy new top performers
            allocation_per_coin = current_cash / num_top_coins if top_performers else 0
            
            for symbol in top_performers:
                if symbol in price_df.columns:
                    price = price_df.loc[rebalance_date, symbol]
                    if price > 0:
                        units = allocation_per_coin * (1 - commission) / price
                        current_holdings[symbol] = units
                        current_cash -= allocation_per_coin
            
            # Update portfolio for this rebalance period
            date_range = pd.date_range(rebalance_date, next_day - timedelta(days=1))
            for day in date_range:
                if day in portfolio_df.index:
                    crypto_value = 0
                    for symbol, units in current_holdings.items():
                        if symbol in price_df.columns and day in price_df.index:
                            price = price_df.loc[day, symbol]
                            crypto_value += units * price
                    
                    portfolio_df.loc[day, 'Cash'] = current_cash
                    portfolio_df.loc[day, 'Crypto_Value'] = crypto_value
                    portfolio_df.loc[day, 'Total_Value'] = current_cash + crypto_value
    
    # Fill NaN values
    portfolio_df = portfolio_df.fillna(method='ffill')
    
    # Calculate BTC buy & hold benchmark
    if btc_data is not None:
        # Make sure we're using the proper starting price from start_monday
        btc_start_price = price_df.loc[start_monday, 'btc'] if 'btc' in price_df.columns else btc_data.loc[start_monday:start_monday].iloc[0]['close']
        initial_btc_units = initial_capital * (1 - commission) / btc_start_price
        
        # Get BTC prices for our date range
        if 'btc' in price_df.columns:
            btc_values = initial_btc_units * price_df.loc[start_monday:, 'btc']
        else:
            btc_values = initial_btc_units * btc_data['close'].reindex(portfolio_df.loc[start_monday:].index).fillna(method='ffill')
        
        portfolio_df.loc[start_monday:, 'BTC_Value'] = btc_values
    
    # Calculate equal-weight buy & hold benchmark
    equal_weight_symbols = list(price_df.columns)
    allocation_per_coin_bh = initial_capital / len(equal_weight_symbols)
    
    portfolio_df['Equal_Weight_Value'] = 0.0
    for symbol in equal_weight_symbols:
        if pd.notna(price_df.loc[start_monday, symbol]) and price_df.loc[start_monday, symbol] > 0:
            initial_units = allocation_per_coin_bh * (1 - commission) / price_df.loc[start_monday, symbol]
            symbol_values = initial_units * price_df.loc[start_monday:, symbol]
            portfolio_df.loc[start_monday:, 'Equal_Weight_Value'] += symbol_values
    
    # Calculate returns and metrics
    strategy_data = portfolio_df.loc[start_monday:end_monday]
    
    # Calculate returns
    strategy_data['Strategy_Return'] = strategy_data['Total_Value'] / initial_capital - 1
    strategy_data['Equal_Weight_Return'] = strategy_data['Equal_Weight_Value'] / initial_capital - 1
    
    if btc_data is not None:
        strategy_data['BTC_Return'] = strategy_data['BTC_Value'] / initial_capital - 1
    
    # Calculate daily returns for risk metrics
    strategy_data['Strategy_Daily_Return'] = strategy_data['Total_Value'].pct_change().fillna(0)
    strategy_data['Equal_Weight_Daily_Return'] = strategy_data['Equal_Weight_Value'].pct_change().fillna(0)
    
    if btc_data is not None:
        strategy_data['BTC_Daily_Return'] = strategy_data['BTC_Value'].pct_change().fillna(0)
    
    # Calculate drawdowns
    strategy_data['Strategy_DD'] = 1 - strategy_data['Total_Value'] / strategy_data['Total_Value'].cummax()
    strategy_data['Equal_Weight_DD'] = 1 - strategy_data['Equal_Weight_Value'] / strategy_data['Equal_Weight_Value'].cummax()
    
    if btc_data is not None:
        strategy_data['BTC_DD'] = 1 - strategy_data['BTC_Value'] / strategy_data['BTC_Value'].cummax()
    
    # Calculate performance metrics
    total_days = (strategy_data.index[-1] - strategy_data.index[0]).days
    
    # Strategy metrics
    strategy_return = strategy_data['Strategy_Return'].iloc[-1]
    strategy_annual_return = (1 + strategy_return) ** (365 / total_days) - 1 if total_days > 0 and strategy_return > -1 else np.nan
    strategy_volatility = strategy_data['Strategy_Daily_Return'].std() * np.sqrt(252)
    strategy_sharpe = strategy_annual_return / strategy_volatility if not np.isnan(strategy_volatility) and strategy_volatility > 0 else 0
    strategy_max_dd = strategy_data['Strategy_DD'].max()
    
    # Equal-weight metrics
    equal_return = strategy_data['Equal_Weight_Return'].iloc[-1]
    equal_annual_return = (1 + equal_return) ** (365 / total_days) - 1 if total_days > 0 and equal_return > -1 else np.nan
    equal_volatility = strategy_data['Equal_Weight_Daily_Return'].std() * np.sqrt(252)
    equal_sharpe = equal_annual_return / equal_volatility if not np.isnan(equal_volatility) and equal_volatility > 0 else 0
    equal_max_dd = strategy_data['Equal_Weight_DD'].max()
    
    # BTC metrics
    if btc_data is not None:
        btc_return = strategy_data['BTC_Return'].iloc[-1]
        btc_annual_return = (1 + btc_return) ** (365 / total_days) - 1 if total_days > 0 and btc_return > -1 else np.nan
        btc_volatility = strategy_data['BTC_Daily_Return'].std() * np.sqrt(252)
        btc_sharpe = btc_annual_return / btc_volatility if not np.isnan(btc_volatility) and btc_volatility > 0 else 0
        btc_max_dd = strategy_data['BTC_DD'].max()
        alpha_vs_btc = strategy_return - btc_return
    else:
        btc_return = btc_annual_return = btc_volatility = btc_sharpe = btc_max_dd = alpha_vs_btc = np.nan
    
    # Calculate unique cryptos and average per week
    unique_cryptos_held = set()
    weekly_counts = []
    
    for holdings in holdings_history.values():
        unique_cryptos_held.update(holdings)
        weekly_counts.append(len(holdings))
    
    avg_cryptos_per_week = np.mean(weekly_counts) if weekly_counts else 0
    
    # Format metrics for display
    metrics = {
        'Strategy_Total_Return': f"{strategy_return:.2%}" if not np.isinf(strategy_return) else "N/A",
        'Strategy_Annual_Return': f"{strategy_annual_return:.2%}" if not np.isnan(strategy_annual_return) and not np.isinf(strategy_annual_return) else "N/A",
        'Strategy_Volatility': f"{strategy_volatility:.2%}" if not np.isnan(strategy_volatility) else "N/A",
        'Strategy_Sharpe': f"{strategy_sharpe:.2f}" if not np.isnan(strategy_sharpe) else "N/A",
        'Strategy_Max_DD': f"{strategy_max_dd:.2%}" if not np.isnan(strategy_max_dd) else "N/A",
        'Equal_Weight_Total_Return': f"{equal_return:.2%}" if not np.isinf(equal_return) else "N/A",
        'Equal_Weight_Annual_Return': f"{equal_annual_return:.2%}" if not np.isnan(equal_annual_return) and not np.isinf(equal_annual_return) else "N/A",
        'Equal_Weight_Sharpe': f"{equal_sharpe:.2f}" if not np.isnan(equal_sharpe) else "N/A",
        'Equal_Weight_Max_DD': f"{equal_max_dd:.2%}" if not np.isnan(equal_max_dd) else "N/A",
        'BTC_Total_Return': f"{btc_return:.2%}" if not np.isnan(btc_return) and not np.isinf(btc_return) else "N/A",
        'BTC_Annual_Return': f"{btc_annual_return:.2%}" if not np.isnan(btc_annual_return) and not np.isinf(btc_annual_return) else "N/A",
        'BTC_Sharpe': f"{btc_sharpe:.2f}" if not np.isnan(btc_sharpe) else "N/A",
        'BTC_Max_DD': f"{btc_max_dd:.2%}" if not np.isnan(btc_max_dd) else "N/A",
        'Alpha_vs_BTC': f"{alpha_vs_btc:.2%}" if not np.isnan(alpha_vs_btc) and not np.isinf(alpha_vs_btc) else "N/A",
        'Unique_Cryptos_Held': len(unique_cryptos_held),
        'Avg_Cryptos_Per_Week': f"{avg_cryptos_per_week:.1f}",
        'Total_Rebalances': len(holdings_history)
    }
    
    # Print performance summary
    print("\nTop Performers Strategy Performance Summary:")
    print(f"Total Return: {metrics['Strategy_Total_Return']}")
    print(f"Annualized Return: {metrics['Strategy_Annual_Return']}")
    print(f"Annualized Volatility: {metrics['Strategy_Volatility']}")
    print(f"Sharpe Ratio: {metrics['Strategy_Sharpe']}")
    print(f"Maximum Drawdown: {metrics['Strategy_Max_DD']}")
    print(f"BTC Total Return: {metrics['BTC_Total_Return']}")
    print(f"BTC Annualized Return: {metrics['BTC_Annual_Return']}")
    print(f"BTC Sharpe Ratio: {metrics['BTC_Sharpe']}")
    print(f"BTC Maximum Drawdown: {metrics['BTC_Max_DD']}")
    print(f"Alpha vs BTC: {metrics['Alpha_vs_BTC']}")
    print(f"Unique Cryptocurrencies Held: {metrics['Unique_Cryptos_Held']}")
    print(f"Avg Cryptos Per Week: {metrics['Avg_Cryptos_Per_Week']}")
    print(f"Total Rebalances: {metrics['Total_Rebalances']}")
    
    # Create visualization with Plotly
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=("Weekly Top Performers Strategy", "Drawdowns"),
        row_heights=[0.7, 0.3]
    )
    
    # Portfolio values
    fig.add_trace(
        go.Scatter(
            x=strategy_data.index,
            y=strategy_data['Total_Value'],
            mode='lines',
            name='Top 5 Performers Strategy',
            line=dict(color='blue', width=2)
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=strategy_data.index,
            y=strategy_data['Equal_Weight_Value'],
            mode='lines',
            name='Equal-Weight Buy & Hold',
            line=dict(color='green', width=2)
        ),
        row=1, col=1
    )
    
    if btc_data is not None:
        fig.add_trace(
            go.Scatter(
                x=strategy_data.index,
                y=strategy_data['BTC_Value'],
                mode='lines',
                name='BTC Buy & Hold',
                line=dict(color='orange', width=2)
            ),
            row=1, col=1
        )
    
    # Drawdowns
    fig.add_trace(
        go.Scatter(
            x=strategy_data.index,
            y=strategy_data['Strategy_DD'] * 100,
            mode='lines',
            name='Strategy Drawdown',
            line=dict(color='red', width=2),
            fill='tozeroy',
            fillcolor='rgba(255, 0, 0, 0.1)'
        ),
        row=2, col=1
    )
    
    if btc_data is not None:
        fig.add_trace(
            go.Scatter(
                x=strategy_data.index,
                y=strategy_data['BTC_DD'] * 100,
                mode='lines',
                name='BTC Drawdown',
                line=dict(color='orange', width=2, dash='dash'),
                fill='tozeroy',
                fillcolor='rgba(255, 165, 0, 0.1)'
            ),
            row=2, col=1
        )
    
    # Update layout
    fig.update_layout(
        title_text="Weekly Top 5 Performers Strategy",
        height=800,
        legend=dict(x=0, y=1, orientation='h'),
        template='plotly_white',
        yaxis_title="Portfolio Value ($)",
        yaxis2_title="Drawdown (%)"
    )
    
    fig.update_yaxes(ticksuffix="%", row=2, col=1)
    
    fig.show()
    
    # Create additional chart showing holdings over time
    if holdings_history:
        # Convert to DataFrame for better visualization
        holdings_df = pd.DataFrame(index=pd.to_datetime(list(holdings_history.keys()), format='%Y-%W').map(lambda x: x.to_pydatetime()))
        
        # Get all unique cryptos held
        all_cryptos = list(unique_cryptos_held)
        
        # Initialize columns with 0
        for crypto in all_cryptos:
            holdings_df[crypto] = 0
        
        # Fill in holdings
        for week, held_cryptos in holdings_history.items():
            week_date = pd.to_datetime(week, format='%Y-%W').to_pydatetime()
            for crypto in held_cryptos:
                if crypto in holdings_df.columns:
                    holdings_df.loc[week_date, crypto] = 1
        
        # Count how many cryptos were held each week
        holdings_df['Total_Held'] = holdings_df.sum(axis=1)
        
        # Visualize holdings distribution
        top_n = 20  # Show top 20 most frequently held cryptos
        holding_counts = holdings_df[all_cryptos].sum().sort_values(ascending=False).head(top_n)
        
        fig2 = go.Figure(data=[
            go.Bar(
                x=holding_counts.index,
                y=holding_counts.values,
                marker_color='blue'
            )
        ])
        
        fig2.update_layout(
            title=f"Top {top_n} Most Frequently Held Cryptocurrencies",
            xaxis_title="Cryptocurrency",
            yaxis_title="Weeks Held",
            template='plotly_white'
        )
        
        fig2.show()
    
    return strategy_data, metrics

def main():
    # Load all data
    crypto_data = load_all_data()
    
    # Run the top performers strategy with different parameters
    print("\n===== Testing Weekly Top 5 Strategy =====")
    portfolio_data, metrics = top_performers_strategy(
        crypto_data,
        lookback_period=7,    # Look back 7 days for performance ranking
        num_top_coins=5,      # Buy top 5 performers
        initial_capital=10000 # Starting with $10,000
    )
    
 
if __name__ == "__main__":
    main()