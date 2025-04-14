import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

class DataLoader:
    """Handles loading and preprocessing cryptocurrency data"""
    
    @staticmethod
    def load_file(file_path):
        """Loads data from a JSON file and converts to DataFrame"""
        with open(file_path, 'r') as file:
            data = json.load(file)
        
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.set_index('timestamp')
        return df
    
    @staticmethod
    def load_all_data(data_folder='data'):
        """Loads all JSON files from the data folder"""
        data = {}
        for file in os.listdir(data_folder):
            if file.endswith('.json'):
                path = os.path.join(data_folder, file)
                # Extract cryptocurrency symbol from the filename
                symbol = file.split('_')[0].replace('usd', '')
                try:
                    data[symbol] = DataLoader.load_file(path)
                    print(f"Loaded: {symbol} - {len(data[symbol])} records")
                except Exception as e:
                    print(f"Error loading {file}: {e}")
        return data


class PriceDataProcessor:
    """Processes price data for strategy execution"""
    
    @staticmethod
    def create_price_dataframe(crypto_data, common_start, common_end, lookback_period):
        """Creates a dataframe with daily prices for all cryptocurrencies"""
        # Include lookback period before the start date
        all_dates = pd.date_range(
            start=common_start - timedelta(days=lookback_period*2),
            end=common_end,
            freq='D'
        )
        price_df = pd.DataFrame(index=all_dates)
        
        # Fill with prices
        for symbol, data in crypto_data.items():
            symbol_prices = pd.DataFrame({'close': data['close']})
            reindexed_prices = symbol_prices.reindex(all_dates)
            price_df[symbol] = reindexed_prices['close'].fillna(method='ffill')
        
        # Remove cryptocurrencies with too many missing values
        price_df = price_df.dropna(axis=1, thresh=len(all_dates)*0.95)
        
        # Fill any remaining NaNs with forward fill followed by backward fill
        price_df = price_df.fillna(method='ffill').fillna(method='bfill')
        
        return price_df


class Strategy:
    """Base class for trading strategies"""
    
    def __init__(self, initial_capital=10000, commission=0.001):
        self.initial_capital = initial_capital
        self.commission = commission
        self.metrics = {}
        self.holdings_history = {}
    
    def run(self, price_df, start_date, end_date):
        """Run the strategy (to be implemented by subclasses)"""
        raise NotImplementedError("Subclasses must implement the run method")
    
    def calculate_metrics(self, portfolio_df, start_date, end_date, btc_data=None):
        """Calculate performance metrics for the strategy"""
        strategy_data = portfolio_df.loc[start_date:end_date].copy()
        total_days = (strategy_data.index[-1] - strategy_data.index[0]).days
        if total_days == 0:
            # For hourly data, convert to days
            total_days = (strategy_data.index[-1] - strategy_data.index[0]).total_seconds() / 86400
        
        # Calculate returns
        strategy_data['Strategy_Return'] = (strategy_data['Total_Value'] / 
                                           self.initial_capital - 1)
        strategy_data['Equal_Weight_Return'] = (strategy_data['Equal_Weight_Value'] / 
                                               self.initial_capital - 1)
        
        # Calculate daily returns
        strategy_data['Strategy_Daily_Return'] = strategy_data['Total_Value'].pct_change().fillna(0)
        strategy_data['Equal_Weight_Daily_Return'] = strategy_data['Equal_Weight_Value'].pct_change().fillna(0)
        
        # Calculate drawdowns
        strategy_data['Strategy_DD'] = 1 - (strategy_data['Total_Value'] / 
                                          strategy_data['Total_Value'].cummax())
        strategy_data['Equal_Weight_DD'] = 1 - (strategy_data['Equal_Weight_Value'] / 
                                              strategy_data['Equal_Weight_Value'].cummax())
        
        # BTC metrics if available
        if btc_data is not None and 'BTC_Value' in strategy_data.columns:
            strategy_data['BTC_Return'] = strategy_data['BTC_Value'] / self.initial_capital - 1
            strategy_data['BTC_Daily_Return'] = strategy_data['BTC_Value'].pct_change().fillna(0)
            strategy_data['BTC_DD'] = 1 - strategy_data['BTC_Value'] / strategy_data['BTC_Value'].cummax()
        
        # Strategy metrics
        strategy_return = strategy_data['Strategy_Return'].iloc[-1]
        strategy_annual_return = ((1 + strategy_return) ** (365 / total_days) - 1 
                                 if total_days > 0 and strategy_return > -1 else np.nan)
        strategy_volatility = strategy_data['Strategy_Daily_Return'].std() * np.sqrt(252)
        strategy_sharpe = (strategy_annual_return / strategy_volatility 
                         if not np.isnan(strategy_volatility) and strategy_volatility > 0 else 0)
        strategy_max_dd = strategy_data['Strategy_DD'].max()
        
        # Equal-weight metrics
        equal_return = strategy_data['Equal_Weight_Return'].iloc[-1]
        equal_annual_return = ((1 + equal_return) ** (365 / total_days) - 1 
                              if total_days > 0 and equal_return > -1 else np.nan)
        equal_volatility = strategy_data['Equal_Weight_Daily_Return'].std() * np.sqrt(252)
        equal_sharpe = (equal_annual_return / equal_volatility 
                      if not np.isnan(equal_volatility) and equal_volatility > 0 else 0)
        equal_max_dd = strategy_data['Equal_Weight_DD'].max()
        
        # BTC metrics
        if btc_data is not None and 'BTC_Value' in strategy_data.columns:
            btc_return = strategy_data['BTC_Return'].iloc[-1]
            btc_annual_return = ((1 + btc_return) ** (365 / total_days) - 1 
                               if total_days > 0 and btc_return > -1 else np.nan)
            btc_volatility = strategy_data['BTC_Daily_Return'].std() * np.sqrt(252)
            btc_sharpe = (btc_annual_return / btc_volatility 
                        if not np.isnan(btc_volatility) and btc_volatility > 0 else 0)
            btc_max_dd = strategy_data['BTC_DD'].max()
            alpha_vs_btc = strategy_return - btc_return
        else:
            btc_return = btc_annual_return = btc_volatility = btc_sharpe = btc_max_dd = alpha_vs_btc = np.nan
        
        # Strategy statistics
        unique_cryptos_held = set()
        weekly_counts = []
        
        for holdings in self.holdings_history.values():
            unique_cryptos_held.update(holdings)
            weekly_counts.append(len(holdings))
        
        avg_cryptos_per_week = np.mean(weekly_counts) if weekly_counts else 0
        
        # Format metrics
        self.metrics = {
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
            'Total_Rebalances': len(self.holdings_history)
        }
        
        return strategy_data
    
    def print_metrics(self):
        """Print performance metrics"""
        print("\nStrategy Performance Summary:")
        print(f"Total Return: {self.metrics['Strategy_Total_Return']}")
        print(f"Annualized Return: {self.metrics['Strategy_Annual_Return']}")
        print(f"Annualized Volatility: {self.metrics['Strategy_Volatility']}")
        print(f"Sharpe Ratio: {self.metrics['Strategy_Sharpe']}")
        print(f"Maximum Drawdown: {self.metrics['Strategy_Max_DD']}")
        print(f"BTC Total Return: {self.metrics['BTC_Total_Return']}")
        print(f"BTC Annualized Return: {self.metrics['BTC_Annual_Return']}")
        print(f"BTC Sharpe Ratio: {self.metrics['BTC_Sharpe']}")
        print(f"BTC Maximum Drawdown: {self.metrics['BTC_Max_DD']}")
        print(f"Alpha vs BTC: {self.metrics['Alpha_vs_BTC']}")
        print(f"Unique Cryptocurrencies Held: {self.metrics['Unique_Cryptos_Held']}")
        print(f"Avg Cryptos Per Week: {self.metrics['Avg_Cryptos_Per_Week']}")
        print(f"Total Rebalances: {self.metrics['Total_Rebalances']}")
    
    def _add_equal_weight_benchmark(self, portfolio_df, price_df, start_date):
        """Add equal-weight buy & hold benchmark to portfolio dataframe"""
        equal_weight_symbols = list(price_df.columns)
        allocation_per_coin = self.initial_capital / len(equal_weight_symbols)
        
        portfolio_df['Equal_Weight_Value'] = 0.0
        for symbol in equal_weight_symbols:
            if pd.notna(price_df.loc[start_date, symbol]) and price_df.loc[start_date, symbol] > 0:
                initial_units = allocation_per_coin * (1 - self.commission) / price_df.loc[start_date, symbol]
                symbol_values = initial_units * price_df.loc[start_date:, symbol]
                portfolio_df.loc[start_date:, 'Equal_Weight_Value'] += symbol_values
    
    def _add_btc_benchmark(self, portfolio_df, price_df, start_date):
        """Add BTC buy & hold benchmark to portfolio dataframe"""
        if 'btc' in price_df.columns:
            btc_start_price = price_df.loc[start_date, 'btc']
            initial_btc_units = self.initial_capital * (1 - self.commission) / btc_start_price
            btc_values = initial_btc_units * price_df.loc[start_date:, 'btc']
            portfolio_df.loc[start_date:, 'BTC_Value'] = btc_values
    
    def _sell_holdings(self, holdings, price_df, date, cash):
        """Sell all current holdings and return updated cash"""
        for symbol, units in holdings.items():
            if symbol in price_df.columns:
                price = price_df.loc[date, symbol]
                cash += units * price * (1 - self.commission)
        return cash
    
    def _buy_new_holdings(self, symbols, price_df, date, cash):
        """Buy new holdings and return updated holdings dictionary"""
        new_holdings = {}
        allocation_per_coin = cash / len(symbols) if symbols else 0
        remaining_cash = cash
        
        for symbol in symbols:
            if symbol in price_df.columns:
                price = price_df.loc[date, symbol]
                if price > 0:
                    units = allocation_per_coin * (1 - self.commission) / price
                    new_holdings[symbol] = units
                    remaining_cash -= allocation_per_coin
        
        return new_holdings
    
    def _create_performance_chart(self, strategy_data):
        """Create performance visualization"""
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=("Strategy Performance", "Drawdowns"),
            row_heights=[0.7, 0.3]
        )
        
        # Portfolio values
        fig.add_trace(
            go.Scatter(
                x=strategy_data.index,
                y=strategy_data['Total_Value'],
                mode='lines',
                name='Strategy',
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
        
        if 'BTC_Value' in strategy_data.columns:
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
        
        if 'BTC_DD' in strategy_data.columns:
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
            title_text="Strategy Performance",
            height=800,
            legend=dict(x=0, y=1, orientation='h'),
            template='plotly_white',
            yaxis_title="Portfolio Value ($)",
            yaxis2_title="Drawdown (%)"
        )
        
        fig.update_yaxes(ticksuffix="%", row=2, col=1)
        
        fig.show()
    
    def _create_holdings_chart(self):
        """Create holdings distribution visualization"""
        if not self.holdings_history:
            return
            
        # Convert to DataFrame for better visualization
        holdings_df = pd.DataFrame(
            index=pd.to_datetime(
                list(self.holdings_history.keys()), format='%Y-%m-%d-%H'
            ).map(lambda x: x.to_pydatetime())
        )
        
        # Get all unique cryptos held
        unique_cryptos_held = set()
        for holdings in self.holdings_history.values():
            unique_cryptos_held.update(holdings)
        all_cryptos = list(unique_cryptos_held)
        
        # Initialize columns with 0
        for crypto in all_cryptos:
            holdings_df[crypto] = 0
        
        # Fill in holdings
        for period, held_cryptos in self.holdings_history.items():
            period_date = pd.to_datetime(period, format='%Y-%m-%d-%H').to_pydatetime()
            for crypto in held_cryptos:
                if crypto in holdings_df.columns:
                    holdings_df.loc[period_date, crypto] = 1
        
        # Count how many cryptos were held each period
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
            yaxis_title="Periods Held",
            template='plotly_white'
        )
        
        fig2.show()


class HourlyDataProcessor:
    """Process hourly cryptocurrency data"""
    
    @staticmethod
    def create_hourly_price_dataframe(crypto_data, common_start, common_end, lookback_period):
        """Creates a dataframe with hourly prices for all cryptocurrencies"""
        # Include lookback period before the start date
        all_hours = pd.date_range(
            start=common_start - pd.Timedelta(hours=lookback_period*2),
            end=common_end,
            freq='H'
        )
        
        price_df = pd.DataFrame(index=all_hours)
        
        # Fill with prices
        for symbol, data in crypto_data.items():
            symbol_prices = pd.DataFrame({'close': data['close']})
            reindexed_prices = symbol_prices.reindex(all_hours)
            price_df[symbol] = reindexed_prices['close'].fillna(method='ffill')
        
        # Remove cryptocurrencies with too many missing values
        price_df = price_df.dropna(axis=1, thresh=len(all_hours)*0.95)
        
        # Fill any remaining NaNs
        price_df = price_df.fillna(method='ffill').fillna(method='bfill')
        
        return price_df


class HourlyTopPerformersWithStopLoss(Strategy):
    """
    A strategy that selects top performing cryptocurrencies based on hourly data
    and implements a stop-loss mechanism to exit positions that drop by a specified percentage.
    """
    
    def __init__(self, lookback_period=24, num_top_coins=5, initial_capital=10000, 
                 commission=0.001, stop_loss_pct=0.05, stop_loss_window=4):
        """
        Initialize the strategy
        
        Parameters:
        -----------
        lookback_period : int
            Number of hours to look back for performance ranking
        num_top_coins : int
            Number of top performers to buy
        initial_capital : float
            Initial capital for the strategy
        commission : float
            Commission rate per trade
        stop_loss_pct : float
            Stop loss percentage threshold (e.g., 0.05 = 5%)
        stop_loss_window : int
            Number of hours to check for stop loss trigger
        """
        super().__init__(initial_capital, commission)
        self.lookback_period = lookback_period
        self.num_top_coins = num_top_coins
        self.stop_loss_pct = stop_loss_pct
        self.stop_loss_window = stop_loss_window
        self.stop_loss_events = []  # Track stop loss events
    
    def __str__(self):
        """String representation of the strategy"""
        return (f"Hourly Top {self.num_top_coins} Strategy with "
                f"{self.stop_loss_pct:.1%} Stop Loss ({self.lookback_period}-hour lookback)")
    
    def run(self, price_df, start_date, end_date, btc_data=None):
        """
        Run the hourly strategy with stop loss
        
        Parameters:
        -----------
        price_df : DataFrame
            DataFrame with cryptocurrency hourly prices
        start_date : datetime
            Start date for the strategy
        end_date : datetime
            End date for the strategy
        btc_data : DataFrame, optional
            Bitcoin data for benchmarking
            
        Returns:
        --------
        DataFrame: Portfolio performance data
        """
        print(f"\nRunning {self} from {start_date} to {end_date}")
        print(f"Initial capital: ${self.initial_capital:,.2f}")
        print(f"Stop Loss: {self.stop_loss_pct:.1%} over {self.stop_loss_window} hours")
        
        # Calculate returns for ranking
        returns_df = price_df.pct_change(self.lookback_period).dropna()
        
        # Create rebalance dates (e.g., every day at midnight)
        # Assuming hourly data, we'll rebalance every 24 hours
        rebalance_freq = '24H'  
        rebalance_dates = pd.date_range(start=start_date, end=end_date, freq=rebalance_freq)
        
        print(f"Strategy will run from {start_date} to {end_date} with {len(rebalance_dates)} rebalance points")
        
        # Initialize portfolio tracking
        all_dates = price_df.loc[start_date:end_date].index
        portfolio_df = pd.DataFrame(index=all_dates)
        portfolio_df['Cash'] = self.initial_capital
        portfolio_df['Crypto_Value'] = 0.0
        portfolio_df['Total_Value'] = self.initial_capital
        
        # Add equal-weight benchmark
        self._add_equal_weight_benchmark(portfolio_df, price_df, start_date)
        
        # Add BTC benchmark if available
        if btc_data is not None and 'btc' in price_df.columns:
            self._add_btc_benchmark(portfolio_df, price_df, start_date)
        
        # Run the strategy with stop loss
        self._run_strategy_with_stop_loss(portfolio_df, price_df, returns_df, rebalance_dates, end_date)
        
        # Calculate metrics
        result_df = self.calculate_metrics(portfolio_df, start_date, end_date, btc_data)
        
        # Print stop loss events
        if self.stop_loss_events:
            print(f"\nTriggered {len(self.stop_loss_events)} stop loss events:")
            for event in self.stop_loss_events[:10]:  # Show first 10 events
                print(f"  {event['timestamp']}: Sold {event['symbol']} at {event['price']:.4f} " 
                      f"(dropped {event['drop_pct']:.2%} in {self.stop_loss_window} hours)")
            
            if len(self.stop_loss_events) > 10:
                print(f"  ... and {len(self.stop_loss_events) - 10} more events")
        
        # Create visualizations
        self._create_performance_chart(result_df)
        self._create_holdings_chart()
        self._create_stop_loss_events_chart(price_df)
        
        return result_df
    
    def _run_strategy_with_stop_loss(self, portfolio_df, price_df, returns_df, rebalance_dates, end_date):
        """Execute the strategy with stop loss"""
        current_holdings = {}
        current_cash = self.initial_capital
        self.holdings_history = {}
        self.stop_loss_events = []
        
        # We need to check for stop loss between rebalance dates
        for i, rebalance_date in enumerate(rebalance_dates):
            # Get the next rebalance date or the end date
            next_rebalance = rebalance_dates[i+1] if i < len(rebalance_dates) - 1 else end_date
            
            # Skip if we're beyond our date range
            if rebalance_date > end_date:
                break
                
            # Get top performers for this period
            if rebalance_date in returns_df.index:
                returns_at_date = returns_df.loc[rebalance_date]
                top_performers = returns_at_date.nlargest(self.num_top_coins).index.tolist()
                
                # Record holdings for this period
                period_key = rebalance_date.strftime('%Y-%m-%d-%H')
                self.holdings_history[period_key] = top_performers
                
                # Sell current holdings
                current_cash = self._sell_holdings(current_holdings, price_df, rebalance_date, current_cash)
                
                # Buy new top performers
                current_holdings = self._buy_new_holdings(top_performers, price_df, rebalance_date, current_cash)
                current_cash = 0  # All cash is allocated
                
                # Update initial portfolio value after rebalance
                crypto_value = self._calculate_holdings_value(current_holdings, price_df, rebalance_date)
                portfolio_df.loc[rebalance_date, 'Cash'] = current_cash
                portfolio_df.loc[rebalance_date, 'Crypto_Value'] = crypto_value
                portfolio_df.loc[rebalance_date, 'Total_Value'] = current_cash + crypto_value
            
            # Check for stop loss between rebalance dates
            if rebalance_date < next_rebalance:
                hourly_dates = price_df.loc[rebalance_date:next_rebalance].index
                
                for hour_idx, current_hour in enumerate(hourly_dates):
                    # Skip the rebalance hour as we've already updated it
                    if current_hour == rebalance_date:
                        continue
                    
                    # Check if we need to trigger stop loss for any holding
                    coins_to_sell = []
                    
                    for symbol, units in list(current_holdings.items()):
                        # Only check stop loss if we have enough history after purchase
                        if (hour_idx >= self.stop_loss_window and 
                            symbol in price_df.columns and 
                            current_hour in price_df.index):
                            
                            # Calculate price change over the stop loss window
                            current_price = price_df.loc[current_hour, symbol]
                            lookback_hour = hourly_dates[hour_idx - self.stop_loss_window]
                            lookback_price = price_df.loc[lookback_hour, symbol]
                            
                            if lookback_price > 0:
                                price_change = (current_price / lookback_price) - 1
                                
                                # If price dropped more than stop loss threshold, sell the coin
                                if price_change <= -self.stop_loss_pct:
                                    coins_to_sell.append({
                                        'symbol': symbol,
                                        'units': units,
                                        'price': current_price,
                                        'drop_pct': price_change,
                                        'timestamp': current_hour
                                    })
                    
                    # Sell coins that triggered stop loss
                    for coin in coins_to_sell:
                        # Record stop loss event
                        self.stop_loss_events.append({
                            'timestamp': coin['timestamp'],
                            'symbol': coin['symbol'],
                            'price': coin['price'],
                            'drop_pct': coin['drop_pct']
                        })
                        
                        # Sell the coin
                        units = coin['units']
                        price = coin['price']
                        current_cash += units * price * (1 - self.commission)
                        del current_holdings[coin['symbol']]
                    
                    # Update portfolio value for this hour
                    crypto_value = self._calculate_holdings_value(current_holdings, price_df, current_hour)
                    portfolio_df.loc[current_hour, 'Cash'] = current_cash
                    portfolio_df.loc[current_hour, 'Crypto_Value'] = crypto_value
                    portfolio_df.loc[current_hour, 'Total_Value'] = current_cash + crypto_value
        
        # Fill NaN values in portfolio dataframe
        portfolio_df.fillna(method='ffill', inplace=True)
    
    def _calculate_holdings_value(self, holdings, price_df, timestamp):
        """Calculate the total value of current holdings at a given timestamp"""
        crypto_value = 0
        for symbol, units in holdings.items():
            if symbol in price_df.columns and timestamp in price_df.index:
                price = price_df.loc[timestamp, symbol]
                crypto_value += units * price
        return crypto_value