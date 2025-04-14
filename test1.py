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
    # Sort by timestamp to ensure data is in chronological order for hourly analysis
    df = df.sort_index()
    return df

def load_all_data(data_folder='data/hourly'):
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

def top_performers_strategy(crypto_data, lookback_period=7, num_top_coins=5, initial_capital=10000, 
                           commission=0.001, stop_loss_pct=0.05, check_interval_hours=1, 
                           start_date=None, end_date=None):
    """
    Implements a strategy that buys the top performing cryptocurrencies each week with stop loss.
    
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
    stop_loss_pct : float
        Stop loss percentage (e.g., 0.05 for 5%)
    check_interval_hours : int
        How often to check for stop loss conditions (in hours)
    start_date : str or datetime, optional
        Custom start date for the strategy (format: 'YYYY-MM-DD' or datetime object)
        If None, will use the latest common start date from all cryptos
    end_date : str or datetime, optional
        Custom end date for the strategy (format: 'YYYY-MM-DD' or datetime object)
        If None, will use the earliest common end date from all cryptos
        
    Returns:
    --------
    tuple: (portfolio_df, metrics)
        portfolio_df: DataFrame with portfolio performance
        metrics: Dictionary with performance metrics
    """
    # Filter cryptocurrencies with sufficient data
    valid_cryptos = {}
    for symbol, data in crypto_data.items():
        if len(data) > lookback_period * 24 + 10:  # Ensure enough data, accounting for hourly
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

    print(f"Data available range: {min(start_dates)} to {max(end_dates)}")
    
    # Determine start date - either custom or based on data
    if start_date is not None:
        # Convert string to datetime if needed
        if isinstance(start_date, str):
            custom_start = pd.to_datetime(start_date)
        else:
            custom_start = start_date
            
        # Make sure custom start date is within available data range
        earliest_possible = max(start_dates)
        if custom_start < earliest_possible:
            print(f"Warning: Requested start date {custom_start} is earlier than available data ({earliest_possible}).")
            print(f"Using earliest available date: {earliest_possible}")
            common_start = earliest_possible
        else:
            common_start = custom_start
            print(f"Using custom start date: {common_start}")
    else:
        common_start = max(start_dates)
        print(f"Using earliest common date: {common_start}")
    
    # Determine end date - either custom or based on data
    if end_date is not None:
        # Convert string to datetime if needed
        if isinstance(end_date, str):
            custom_end = pd.to_datetime(end_date)
        else:
            custom_end = end_date
            
        # Make sure custom end date is within available data range
        latest_possible = min(end_dates)
        if custom_end > latest_possible:
            print(f"Warning: Requested end date {custom_end} is later than available data ({latest_possible}).")
            print(f"Using latest available date: {latest_possible}")
            common_end = latest_possible
        else:
            common_end = custom_end
            print(f"Using custom end date: {common_end}")
    else:
        common_end = min(end_dates)
        print(f"Using latest common date: {common_end}")
    
    # Ensure the start date is before the end date
    if common_start >= common_end:
        print(f"Error: Start date {common_start} must be before end date {common_end}")
        return None, None
    
    print(f"Common date range: {common_start} to {common_end}")
    
    # Create weekly rebalance points (every Monday)
    start_monday = common_start + timedelta(days=(7 - common_start.weekday()) % 7)
    start_monday = start_monday.replace(hour=0, minute=0, second=0, microsecond=0)  # Start at midnight
    end_monday = common_end - timedelta(days=common_end.weekday())
    end_monday = end_monday.replace(hour=0, minute=0, second=0, microsecond=0)  # Start at midnight
    
    rebalance_dates = pd.date_range(start=start_monday, end=end_monday, freq='W-MON')
    print(f"Strategy will run from {start_monday} to {end_monday} with {len(rebalance_dates)} rebalance points")
    
    # Create hourly price dataframe for all cryptocurrencies
    # Include the lookback period before the start date
    all_dates = pd.date_range(
        start=common_start - timedelta(days=lookback_period*2), 
        end=common_end, 
        freq=f'{check_interval_hours}H'
    )
    price_df = pd.DataFrame(index=all_dates)
    
    # Fill with prices
    for symbol, data in valid_cryptos.items():
        # Get close prices and handle missing dates with forward fill
        symbol_prices = pd.DataFrame({'close': data['close']})
        # Resample to our desired frequency if data is higher frequency
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
    
    # Store daily prices for performance ranking
    daily_price_df = price_df.resample('D').last()
    
    # Calculate returns for ranking (using daily data)
    returns_df = daily_price_df.pct_change(lookback_period).dropna()
    
    # Create portfolio tracking dataframe
    portfolio_df = pd.DataFrame(index=all_dates)
    portfolio_df['Cash'] = 0.0
    portfolio_df['Crypto_Value'] = 0.0
    portfolio_df['Total_Value'] = 0.0
    
    # Track trade history for analysis
    trade_history = []
    
    # Initialize portfolio tracking for the correct time period
    portfolio_df = portfolio_df.loc[start_monday:]
    portfolio_df['Cash'] = initial_capital
    portfolio_df['Crypto_Value'] = 0.0
    portfolio_df['Total_Value'] = initial_capital
    
    # Track portfolio value hour by hour
    current_holdings = {}
    current_cash = initial_capital
    entry_prices = {}  # Track entry prices for stop loss calculation
    holdings_history = {}  # Track what we hold each week
    stop_loss_events = []  # Track stop loss events
    
    # Apply the rebalancing decisions
    print("Simulating strategy with hourly data and stop loss...")
    for i in range(len(rebalance_dates)):
        rebalance_date = rebalance_dates[i]
        
        # Get the next rebalance date or the end date
        next_rebalance = rebalance_dates[i+1] if i < len(rebalance_dates) - 1 else common_end
        
        # Skip if we're beyond our date range
        if rebalance_date > common_end:
            break
            
        # Get top performers for this week
        week_key = rebalance_date.strftime('%Y-%W')
        
        # Get the closest date in returns_df to our rebalance date
        closest_date = returns_df.index[returns_df.index <= rebalance_date][-1] if any(returns_df.index <= rebalance_date) else None
        
        if closest_date is not None:
            returns_at_date = returns_df.loc[closest_date]
            top_performers = returns_at_date.nlargest(num_top_coins).index.tolist()
            
            # Record holdings for this week
            holdings_history[week_key] = top_performers
            
            # Sell all current holdings
            for symbol, units in current_holdings.items():
                if symbol in price_df.columns:
                    closest_idx = price_df.index[price_df.index <= rebalance_date][-1]
                    price = price_df.loc[closest_idx, symbol]
                    sale_value = units * price * (1 - commission)
                    current_cash += sale_value
                    
                    # Record the trade
                    trade_history.append({
                        'date': rebalance_date,
                        'symbol': symbol,
                        'action': 'SELL',
                        'units': units,
                        'price': price,
                        'value': sale_value,
                        'reason': 'REBALANCE'
                    })
            
            # Reset holdings
            current_holdings = {}
            entry_prices = {}
            
            # Buy new top performers
            allocation_per_coin = current_cash / num_top_coins if top_performers else 0
            
            for symbol in top_performers:
                if symbol in price_df.columns:
                    closest_idx = price_df.index[price_df.index <= rebalance_date][-1]
                    price = price_df.loc[closest_idx, symbol]
                    if price > 0:
                        units = allocation_per_coin * (1 - commission) / price
                        current_holdings[symbol] = units
                        entry_prices[symbol] = price
                        current_cash -= allocation_per_coin
                        
                        # Record the trade
                        trade_history.append({
                            'date': rebalance_date,
                            'symbol': symbol,
                            'action': 'BUY',
                            'units': units,
                            'price': price,
                            'value': allocation_per_coin,
                            'reason': 'REBALANCE'
                        })
        
        # Create time range from this rebalance to next (hourly)
        time_range = pd.date_range(
            start=rebalance_date, 
            end=next_rebalance - timedelta(hours=check_interval_hours), 
            freq=f'{check_interval_hours}H'
        )
        
        # Iterate through each check point to apply stop losses
        for check_time in time_range:
            # Check if this date is in our price_df index
            if check_time not in price_df.index:
                continue
                
            # Check for stop losses
            symbols_to_sell = []
            
            for symbol, units in current_holdings.items():
                if symbol in price_df.columns and check_time in price_df.index:
                    current_price = price_df.loc[check_time, symbol]
                    entry_price = entry_prices.get(symbol, current_price)  # Fallback if no entry price
                    
                    # Calculate price drop percentage
                    price_drop = (entry_price - current_price) / entry_price
                    
                    # If drop exceeds stop loss threshold, mark for selling
                    if price_drop >= stop_loss_pct:
                        symbols_to_sell.append(symbol)
                        
                        # Record stop loss event
                        stop_loss_events.append({
                            'date': check_time,
                            'symbol': symbol,
                            'entry_price': entry_price,
                            'exit_price': current_price,
                            'drop_pct': price_drop
                        })
            
            # Execute stop loss sells
            for symbol in symbols_to_sell:
                units = current_holdings[symbol]
                current_price = price_df.loc[check_time, symbol]
                sale_value = units * current_price * (1 - commission)
                current_cash += sale_value
                
                # Record the trade
                trade_history.append({
                    'date': check_time,
                    'symbol': symbol,
                    'action': 'SELL',
                    'units': units,
                    'price': current_price,
                    'value': sale_value,
                    'reason': 'STOP_LOSS'
                })
                
                # Remove from holdings
                del current_holdings[symbol]
                if symbol in entry_prices:
                    del entry_prices[symbol]
            
            # Update portfolio value at this check point
            crypto_value = 0
            for symbol, units in current_holdings.items():
                if symbol in price_df.columns:
                    price = price_df.loc[check_time, symbol]
                    crypto_value += units * price
            
            portfolio_df.loc[check_time, 'Cash'] = current_cash
            portfolio_df.loc[check_time, 'Crypto_Value'] = crypto_value
            portfolio_df.loc[check_time, 'Total_Value'] = current_cash + crypto_value
    
    # Fill NaN values in portfolio tracking
    portfolio_df = portfolio_df.fillna(method='ffill')
    
    # Calculate BTC buy & hold benchmark
    if btc_data is not None:
        # Get the closest price to start_monday
        closest_start_idx = price_df.index[price_df.index >= start_monday][0]
        btc_start_price = price_df.loc[closest_start_idx, 'btc'] if 'btc' in price_df.columns else None
        
        if btc_start_price is not None and btc_start_price > 0:
            initial_btc_units = initial_capital * (1 - commission) / btc_start_price
            btc_values = initial_btc_units * price_df.loc[start_monday:, 'btc']
            portfolio_df.loc[start_monday:, 'BTC_Value'] = btc_values
    
    # Calculate equal-weight buy & hold benchmark
    equal_weight_symbols = list(price_df.columns)
    allocation_per_coin_bh = initial_capital / len(equal_weight_symbols)
    
    portfolio_df['Equal_Weight_Value'] = 0.0
    closest_start_idx = price_df.index[price_df.index >= start_monday][0]
    
    for symbol in equal_weight_symbols:
        if pd.notna(price_df.loc[closest_start_idx, symbol]) and price_df.loc[closest_start_idx, symbol] > 0:
            initial_units = allocation_per_coin_bh * (1 - commission) / price_df.loc[closest_start_idx, symbol]
            symbol_values = initial_units * price_df.loc[start_monday:, symbol]
            portfolio_df.loc[start_monday:, 'Equal_Weight_Value'] += symbol_values
    
    # Calculate returns and metrics
    strategy_data = portfolio_df.loc[start_monday:common_end]
    
    # Resample to daily data for cleaner metrics and visualization
    daily_strategy_data = strategy_data.resample('D').last()
    
    # Calculate returns
    daily_strategy_data['Strategy_Return'] = daily_strategy_data['Total_Value'] / initial_capital - 1
    daily_strategy_data['Equal_Weight_Return'] = daily_strategy_data['Equal_Weight_Value'] / initial_capital - 1
    
    if btc_data is not None and 'BTC_Value' in daily_strategy_data.columns:
        daily_strategy_data['BTC_Return'] = daily_strategy_data['BTC_Value'] / initial_capital - 1
    
    # Calculate daily returns for risk metrics
    daily_strategy_data['Strategy_Daily_Return'] = daily_strategy_data['Total_Value'].pct_change().fillna(0)
    daily_strategy_data['Equal_Weight_Daily_Return'] = daily_strategy_data['Equal_Weight_Value'].pct_change().fillna(0)
    
    if btc_data is not None and 'BTC_Value' in daily_strategy_data.columns:
        daily_strategy_data['BTC_Daily_Return'] = daily_strategy_data['BTC_Value'].pct_change().fillna(0)
    
    # Calculate drawdowns
    daily_strategy_data['Strategy_DD'] = 1 - daily_strategy_data['Total_Value'] / daily_strategy_data['Total_Value'].cummax()
    daily_strategy_data['Equal_Weight_DD'] = 1 - daily_strategy_data['Equal_Weight_Value'] / daily_strategy_data['Equal_Weight_Value'].cummax()
    
    if btc_data is not None and 'BTC_Value' in daily_strategy_data.columns:
        daily_strategy_data['BTC_DD'] = 1 - daily_strategy_data['BTC_Value'] / daily_strategy_data['BTC_Value'].cummax()
    
    # Calculate performance metrics
    total_days = (daily_strategy_data.index[-1] - daily_strategy_data.index[0]).days
    
    # Strategy metrics
    strategy_return = daily_strategy_data['Strategy_Return'].iloc[-1]
    strategy_annual_return = (1 + strategy_return) ** (365 / total_days) - 1 if total_days > 0 and strategy_return > -1 else np.nan
    strategy_volatility = daily_strategy_data['Strategy_Daily_Return'].std() * np.sqrt(252)
    strategy_sharpe = strategy_annual_return / strategy_volatility if not np.isnan(strategy_volatility) and strategy_volatility > 0 else 0
    strategy_max_dd = daily_strategy_data['Strategy_DD'].max()
    
    # Equal-weight metrics
    equal_return = daily_strategy_data['Equal_Weight_Return'].iloc[-1]
    equal_annual_return = (1 + equal_return) ** (365 / total_days) - 1 if total_days > 0 and equal_return > -1 else np.nan
    equal_volatility = daily_strategy_data['Equal_Weight_Daily_Return'].std() * np.sqrt(252)
    equal_sharpe = equal_annual_return / equal_volatility if not np.isnan(equal_volatility) and equal_volatility > 0 else 0
    equal_max_dd = daily_strategy_data['Equal_Weight_DD'].max()
    
    # BTC metrics
    if btc_data is not None and 'BTC_Return' in daily_strategy_data.columns:
        btc_return = daily_strategy_data['BTC_Return'].iloc[-1]
        btc_annual_return = (1 + btc_return) ** (365 / total_days) - 1 if total_days > 0 and btc_return > -1 else np.nan
        btc_volatility = daily_strategy_data['BTC_Daily_Return'].std() * np.sqrt(252)
        btc_sharpe = btc_annual_return / btc_volatility if not np.isnan(btc_volatility) and btc_volatility > 0 else 0
        btc_max_dd = daily_strategy_data['BTC_DD'].max()
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
    
    # Count stop loss events
    total_stop_losses = len(stop_loss_events)
    
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
        'Total_Rebalances': len(holdings_history),
        'Total_Stop_Losses': total_stop_losses,
        'Stop_Loss_Percentage': f"{stop_loss_pct:.1%}"
    }
    
    # Print performance summary
    print("\nTop Performers Strategy with Stop Loss Performance Summary:")
    print(f"Total Return: {metrics['Strategy_Total_Return']}")
    print(f"Annualized Return: {metrics['Strategy_Annual_Return']}")
    print(f"Annualized Volatility: {metrics['Strategy_Volatility']}")
    print(f"Sharpe Ratio: {metrics['Strategy_Sharpe']}")
    print(f"Maximum Drawdown: {metrics['Strategy_Max_DD']}")
    print(f"BTC Total Return: {metrics['BTC_Total_Return']}")
    print(f"Alpha vs BTC: {metrics['Alpha_vs_BTC']}")
    print(f"Unique Cryptocurrencies Held: {metrics['Unique_Cryptos_Held']}")
    print(f"Avg Cryptos Per Week: {metrics['Avg_Cryptos_Per_Week']}")
    print(f"Total Rebalances: {metrics['Total_Rebalances']}")
    print(f"Stop Loss Percentage: {metrics['Stop_Loss_Percentage']}")
    print(f"Total Stop Loss Events: {metrics['Total_Stop_Losses']}")
    
    # Create visualization with Plotly
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=("Weekly Top Performers Strategy with Stop Loss", "Drawdowns"),
        row_heights=[0.7, 0.3]
    )
    
    # Portfolio values
    fig.add_trace(
        go.Scatter(
            x=daily_strategy_data.index,
            y=daily_strategy_data['Total_Value'],
            mode='lines',
            name=f'Top {num_top_coins} Performers with {stop_loss_pct:.0%} Stop Loss',
            line=dict(color='blue', width=2)
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=daily_strategy_data.index,
            y=daily_strategy_data['Equal_Weight_Value'],
            mode='lines',
            name='Equal-Weight Buy & Hold',
            line=dict(color='green', width=2)
        ),
        row=1, col=1
    )
    
    if btc_data is not None and 'BTC_Value' in daily_strategy_data.columns:
        fig.add_trace(
            go.Scatter(
                x=daily_strategy_data.index,
                y=daily_strategy_data['BTC_Value'],
                mode='lines',
                name='BTC Buy & Hold',
                line=dict(color='orange', width=2)
            ),
            row=1, col=1
        )
    
    # Drawdowns
    fig.add_trace(
        go.Scatter(
            x=daily_strategy_data.index,
            y=daily_strategy_data['Strategy_DD'] * 100,
            mode='lines',
            name='Strategy Drawdown',
            line=dict(color='red', width=2),
            fill='tozeroy',
            fillcolor='rgba(255, 0, 0, 0.1)'
        ),
        row=2, col=1
    )
    
    if btc_data is not None and 'BTC_DD' in daily_strategy_data.columns:
        fig.add_trace(
            go.Scatter(
                x=daily_strategy_data.index,
                y=daily_strategy_data['BTC_DD'] * 100,
                mode='lines',
                name='BTC Drawdown',
                line=dict(color='orange', width=2, dash='dash'),
                fill='tozeroy',
                fillcolor='rgba(255, 165, 0, 0.1)'
            ),
            row=2, col=1
        )
    
    # Mark stop loss events on the chart
    if stop_loss_events:
        stop_loss_df = pd.DataFrame(stop_loss_events)
        stop_loss_df['date'] = pd.to_datetime(stop_loss_df['date'])
        
        # Add markers for stop loss events
        fig.add_trace(
            go.Scatter(
                x=stop_loss_df['date'],
                y=[daily_strategy_data.loc[daily_strategy_data.index >= date].iloc[0]['Total_Value'] 
                   if any(daily_strategy_data.index >= date) else np.nan 
                   for date in stop_loss_df['date']],
                mode='markers',
                name='Stop Loss Events',
                marker=dict(symbol='x', size=10, color='red'),
                hoverinfo='text',
                text=[f"Stop Loss: {row['symbol']} <br>Drop: {row['drop_pct']:.2%}" for _, row in stop_loss_df.iterrows()]
            ),
            row=1, col=1
        )
    
    # Update layout
    fig.update_layout(
        title_text=f"Weekly Top {num_top_coins} Performers Strategy with {stop_loss_pct:.0%} Stop Loss",
        height=800,
        legend=dict(x=0, y=1, orientation='h'),
        template='plotly_white',
        yaxis_title="Portfolio Value ($)",
        yaxis2_title="Drawdown (%)"
    )
    
    fig.update_yaxes(ticksuffix="%", row=2, col=1)
    
    fig.show()
    
    # Create a visualization of holdings over time
    if holdings_history:
        # Convert to DataFrame for better visualization
        # Fix the date conversion by adding a day component (Monday = day 1 of the week)
        holdings_dates = []
        for week_key in holdings_history.keys():
            year, week = week_key.split('-')
            # Create a proper date from year and week - adding day 1 (Monday)
            date = pd.to_datetime(f"{year}-{week}-1", format="%Y-%W-%w")
            holdings_dates.append(date)
            
        holdings_df = pd.DataFrame(index=holdings_dates)
        
        # Get all unique cryptos held
        all_cryptos = list(unique_cryptos_held)
        
        # Initialize columns with 0
        for crypto in all_cryptos:
            holdings_df[crypto] = 0
        
        # Fill in holdings
        for i, (week, held_cryptos) in enumerate(holdings_history.items()):
            week_date = holdings_dates[i]
            for crypto in held_cryptos:
                if crypto in holdings_df.columns:
                    holdings_df.loc[week_date, crypto] = 1
        
        # Count how many cryptos were held each week
        holdings_df['Total_Held'] = holdings_df.sum(axis=1)
        
    
    
    return daily_strategy_data, metrics

def main():
    # Load all data
    crypto_data = load_all_data()
    
    # Example custom date range - set for the year 2022
    custom_start_date = '2022-01-01'
    custom_end_date = '2022-12-31'
    
    # Run the top performers strategy with stop loss and custom date range
    print(f"\n===== Testing Weekly Top 5 Strategy with Stop Loss ({custom_start_date} to {custom_end_date}) =====")
    portfolio_data, metrics = top_performers_strategy(
        crypto_data,
        lookback_period=7,         # Look back 7 days for performance ranking
        num_top_coins=5,           # Buy top 5 performers
        initial_capital=10000,     # Starting with $10,000
        stop_loss_pct=0.05,        # 5% stop loss
        check_interval_hours=1,    # Check hourly for stop loss conditions
        start_date=custom_start_date,  # Use our custom start date
        end_date=custom_end_date    # Use our custom end date
    )
    
    # You can test different stop loss percentages with the same date range
    stop_loss_percentages = [0.03, 0.05, 0.07, 0.10]
    results = {}
    
    for stop_loss in stop_loss_percentages:
        print(f"\n===== Testing with {stop_loss:.0%} Stop Loss =====")
        _, test_metrics = top_performers_strategy(
            crypto_data,
            lookback_period=7,
            num_top_coins=5,
            initial_capital=10000,
            stop_loss_pct=stop_loss,
            check_interval_hours=1,
            start_date=custom_start_date,  # Use the same custom start date
            end_date=custom_end_date       # Use the same custom end date
        )
        results[f"{stop_loss:.0%}"] = test_metrics
    
    # Print comparison of different stop loss settings
    print("\n===== Stop Loss Percentage Comparison =====")
    print("Stop Loss\tTotal Return\tMax DD\tSharpe\tStop Loss Events")
    for stop_loss, result in results.items():
        print(f"{stop_loss}\t{result['Strategy_Total_Return']}\t{result['Strategy_Max_DD']}\t{result['Strategy_Sharpe']}\t{result['Total_Stop_Losses']}")
    
    # Test different date ranges to compare market cycles
    market_cycles = {
        "Bull Market": ("2020-03-15", "2021-11-15"),
        "Bear Market": ("2021-11-16", "2022-12-31"),
        "Recovery": ("2023-01-01", "2023-12-31")
    }
    
    cycle_results = {}
    
    print("\n===== Market Cycle Comparison (5% Stop Loss) =====")
    for cycle_name, (cycle_start, cycle_end) in market_cycles.items():
        print(f"\nTesting {cycle_name} period ({cycle_start} to {cycle_end})")
        try:
            _, cycle_metrics = top_performers_strategy(
                crypto_data,
                lookback_period=7,
                num_top_coins=5,
                initial_capital=10000,
                stop_loss_pct=0.05,
                check_interval_hours=1,
                start_date=cycle_start,
                end_date=cycle_end
            )
            cycle_results[cycle_name] = cycle_metrics
        except Exception as e:
            print(f"Error testing {cycle_name} period: {e}")
    
    # Print comparison of market cycles
    if cycle_results:
        print("\n===== Market Cycle Performance Comparison =====")
        print("Cycle\t\tTotal Return\tMax DD\tSharpe\tStop Loss Events")
        for cycle_name, result in cycle_results.items():
            print(f"{cycle_name}\t{result['Strategy_Total_Return']}\t{result['Strategy_Max_DD']}\t{result['Strategy_Sharpe']}\t{result['Total_Stop_Losses']}")
 
if __name__ == "__main__":
    main()
    
    # Print comparison of different stop loss settings
    print("\n===== Stop Loss Percentage Comparison =====")
    print("Stop Loss\tTotal Return\tMax DD\tSharpe\tStop Loss Events")
    for stop_loss, result in results.items():
        print(f"{stop_loss}\t{result['Strategy_Total_Return']}\t{result['Strategy_Max_DD']}\t{result['Strategy_Sharpe']}\t{result['Total_Stop_Losses']}")
 
if __name__ == "__main__":
    main()