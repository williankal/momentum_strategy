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
    def load_all_data(data_folder='data/hourly'):
        """Loads all JSON files from the data folder"""
        data = {}
        for file in os.listdir(data_folder):
            if file.endswith('.json'):
                path = os.path.join(data_folder, file)
                # Extract cryptocurrency symbol from the filename
                symbol = file.split('_')[0].replace('usd', '')
                print(symbol)
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
        
        for i, rebalance_date in enumerate(rebalance_dates):
            # Get the next rebalance date or the end date
            next_rebalance = rebalance_dates[i+1] if i < len(rebalance_dates) - 1 else end_date
            
            if rebalance_date > end_date:
                break
                
            if rebalance_date in returns_df.index:
                returns_at_date = returns_df.loc[rebalance_date]
                top_performers = returns_at_date.nlargest(self.num_top_coins).index.tolist()
                
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
    
    def _create_stop_loss_events_chart(self, price_df):
        """Create visualization of stop loss events"""
        if not self.stop_loss_events:
            print("No stop loss events to visualize.")
            return
        
        # Get top coins with stop loss events
        symbol_count = {}
        for event in self.stop_loss_events:
            symbol = event['symbol']
            if symbol in symbol_count:
                symbol_count[symbol] += 1
            else:
                symbol_count[symbol] = 1
        
        # Get top 5 coins with most stop loss events
        top_symbols = sorted(symbol_count.items(), key=lambda x: x[1], reverse=True)[:5]
        top_symbol_names = [s[0] for s in top_symbols]
        
        print(f"\nTop cryptocurrencies with stop loss events:")
        for symbol, count in top_symbols:
            print(f"  {symbol.upper()}: {count} events")
        
        # Create figure for each of the top coins
        for symbol in top_symbol_names:
            # Get all events for this symbol
            symbol_events = [e for e in self.stop_loss_events if e['symbol'] == symbol]
            
            if not symbol_events:
                continue
                
            try:
                # Create price chart with stop loss markers
                fig = go.Figure()
                
                # Find the earliest and latest timestamps
                earliest_timestamp = min(e['timestamp'] for e in symbol_events)
                latest_timestamp = max(e['timestamp'] for e in symbol_events)
                
                # Add some padding before and after
                padding = pd.Timedelta(hours=24)
                chart_start = earliest_timestamp - padding
                chart_end = latest_timestamp + padding
                
                # Make sure these timestamps are in our price data
                chart_start = max(chart_start, price_df.index[0])
                chart_end = min(chart_end, price_df.index[-1])
                
                # Add price line
                if symbol in price_df.columns:
                    symbol_data = price_df[symbol].loc[chart_start:chart_end]
                    fig.add_trace(
                        go.Scatter(
                            x=symbol_data.index,
                            y=symbol_data.values,
                            mode='lines',
                            name=f'{symbol} Price',
                            line=dict(color='blue', width=1)
                        )
                    )
                
                # Add stop loss events as markers
                fig.add_trace(
                    go.Scatter(
                        x=[e['timestamp'] for e in symbol_events],
                        y=[e['price'] for e in symbol_events],
                        mode='markers',
                        name='Stop Loss Triggered',
                        marker=dict(color='red', size=8, symbol='x')
                    )
                )
                
                # Update layout
                fig.update_layout(
                    title=f"Stop Loss Events for {symbol.upper()} ({len(symbol_events)} events)",
                    xaxis_title="Date",
                    yaxis_title="Price",
                    template='plotly_white'
                )
                
                fig.show()
            except Exception as e:
                print(f"Error creating chart for {symbol}: {e}")
        
        try:
            # Create summary chart of stop loss events
            # Group by date (day)
            event_dates = [pd.to_datetime(e['timestamp']).date() for e in self.stop_loss_events]
            date_counts = {}
            for date in event_dates:
                date_str = date.strftime('%Y-%m-%d')
                if date_str in date_counts:
                    date_counts[date_str] += 1
                else:
                    date_counts[date_str] = 1
            
            # Convert to DataFrame
            events_df = pd.DataFrame(list(date_counts.items()), columns=['Date', 'Count'])
            events_df['Date'] = pd.to_datetime(events_df['Date'])
            events_df = events_df.sort_values('Date')
            
            fig = go.Figure()
            fig.add_trace(
                go.Bar(
                    x=events_df['Date'],
                    y=events_df['Count'],
                    marker_color='red'
                )
            )
            
            fig.update_layout(
                title="Daily Stop Loss Events",
                xaxis_title="Date",
                yaxis_title="Number of Stop Loss Events",
                template='plotly_white'
            )
            
            fig.show()
        except Exception as e:
            print(f"Error creating stop loss summary chart: {e}")


class TopPerformersStrategy(Strategy):
    """Implements a strategy that buys the top performing cryptocurrencies each period"""
    
    def __init__(self, lookback_period=7, num_top_coins=5, initial_capital=10000, commission=0.001):
        super().__init__(initial_capital, commission)
        self.lookback_period = lookback_period
        self.num_top_coins = num_top_coins
        
    def __str__(self):
        """String representation of the strategy"""
        return f"Top {self.num_top_coins} Performers Strategy ({self.lookback_period}-day lookback)"
        
    def run(self, price_df, start_date, end_date, btc_data=None):
        """
        Run the top performers strategy
        
        Parameters:
        -----------
        price_df : DataFrame
            DataFrame with cryptocurrency prices
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
        
        # Calculate returns for ranking
        returns_df = price_df.pct_change(self.lookback_period).dropna()
        
        # Create rebalance dates (every Monday)
        start_monday = start_date + timedelta(days=(7 - start_date.weekday()) % 7)
        end_monday = end_date - timedelta(days=end_date.weekday())
        rebalance_dates = pd.date_range(start=start_monday, end=end_monday, freq='W-MON')
        
        print(f"Strategy will run from {start_monday.date()} to {end_monday.date()} with {len(rebalance_dates)} rebalance points")
        
        # Initialize portfolio tracking
        portfolio_df = pd.DataFrame(index=price_df.loc[start_monday:end_date].index)
        portfolio_df['Cash'] = self.initial_capital
        portfolio_df['Crypto_Value'] = 0.0
        portfolio_df['Total_Value'] = self.initial_capital
        
        # Add equal-weight benchmark
        self._add_equal_weight_benchmark(portfolio_df, price_df, start_monday)
        
        # Add BTC benchmark if available
        if btc_data is not None and 'btc' in price_df.columns:
            self._add_btc_benchmark(portfolio_df, price_df, start_monday)
        
        # Run the strategy
        self._run_rebalancing_strategy(portfolio_df, price_df, returns_df, rebalance_dates, end_date)
        
        # Calculate metrics
        result_df = self.calculate_metrics(portfolio_df, start_monday, end_date, btc_data)
        
        # Create visualizations
        self._create_performance_chart(result_df)
        self._create_holdings_chart()
        
        return result_df
    
    def _run_rebalancing_strategy(self, portfolio_df, price_df, returns_df, rebalance_dates, end_date):
        """Execute the rebalancing strategy"""
        current_holdings = {}
        current_cash = self.initial_capital
        self.holdings_history = {}
        
        for i, rebalance_date in enumerate(rebalance_dates):
            # Get the next rebalance date or the end date
            next_rebalance = rebalance_dates[i+1] if i < len(rebalance_dates) - 1 else end_date
            next_day = min(next_rebalance, end_date)
            
            # Skip if we're beyond our date range
            if rebalance_date > end_date:
                break
                
            # Get top performers for this period
            if rebalance_date in returns_df.index:
                returns_at_date = returns_df.loc[rebalance_date]
                top_performers = returns_at_date.nlargest(self.num_top_coins).index.tolist()
                
                # Record holdings for this period
                week_key = rebalance_date.strftime('%Y-%W')
                self.holdings_history[week_key] = top_performers
                
                # Sell current holdings
                current_cash = self._sell_holdings(current_holdings, price_df, rebalance_date, current_cash)
                
                # Buy new top performers
                current_holdings = self._buy_new_holdings(top_performers, price_df, rebalance_date, current_cash)
                current_cash = 0  # All cash is allocated
                
                # Update portfolio for this period
                self._update_portfolio_values(
                    portfolio_df, current_holdings, current_cash, price_df, 
                    rebalance_date, next_day
                )
    
    def _update_portfolio_values(self, portfolio_df, holdings, cash, price_df, start_date, end_date):
        """Update portfolio values for the current period"""
        date_range = pd.date_range(start_date, end_date - timedelta(days=1))
        for day in date_range:
            if day in portfolio_df.index:
                crypto_value = 0
                for symbol, units in holdings.items():
                    if symbol in price_df.columns and day in price_df.index:
                        price = price_df.loc[day, symbol]
                        crypto_value += units * price
                
                portfolio_df.loc[day, 'Cash'] = cash
                portfolio_df.loc[day, 'Crypto_Value'] = crypto_value
                portfolio_df.loc[day, 'Total_Value'] = cash + crypto_value


class CryptoBacktester:
    """Main class to handle backtesting of crypto strategies"""
    
    def __init__(self, data_folder='data'):
        """Initialize the backtester"""
        self.data = DataLoader.load_all_data(data_folder)
    
    def find_common_date_range(self, valid_cryptos):
        """Find common date range for all cryptocurrencies"""
        start_dates = [data.index.min() for data in valid_cryptos.values()]
        end_dates = [data.index.max() for data in valid_cryptos.values()]
        
        common_start = max(start_dates)
        common_end = min(end_dates)
        
        print(f"Common date range: {common_start} to {common_end}")
        return common_start, common_end
    
    def filter_valid_cryptos(self, lookback_period):
        """Filter cryptocurrencies with sufficient data"""
        valid_cryptos = {}
        for symbol, data in self.data.items():
            print(data)
            if len(data) > lookback_period + 10:  # Ensure enough data
                valid_cryptos[symbol] = data
        
        print(f"Found {len(valid_cryptos)} cryptocurrencies with sufficient data")
        return valid_cryptos
    
    def run_backtest(self, strategy, lookback_period=7, custom_start_date=None, custom_end_date=None):
        """
        Run a backtest with the given strategy
        
        Parameters:
        -----------
        strategy : Strategy
            The strategy to backtest
        lookback_period : int
            Number of days to look back for performance calculation
        custom_start_date : str or datetime, optional
            Custom start date in 'YYYY-MM-DD' format or datetime object
        custom_end_date : str or datetime, optional
            Custom end date in 'YYYY-MM-DD' format or datetime object
        
        Returns:
        --------
        tuple: (result_df, metrics)
            result_df: DataFrame with portfolio performance
            metrics: Dictionary with performance metrics
        """
        # Filter cryptocurrencies with sufficient data
        valid_cryptos = self.filter_valid_cryptos(lookback_period)
        print(f"Valid cryptocurrencies: {list(valid_cryptos.keys())}")
        if len(valid_cryptos) == 0:
            print("Error: No cryptocurrencies with sufficient data found.")
            return None, None
        
        # Check for BTC data for benchmark
        btc_data = valid_cryptos.get('btc')
        if btc_data is None:
            print("Warning: Bitcoin data not found. Cannot use as benchmark.")
        
        # Find common date range
        common_start, common_end = self.find_common_date_range(valid_cryptos)
        
        # Apply custom date range if provided
        start_date = common_start
        end_date = common_end
        
        if custom_start_date is not None:
            if isinstance(custom_start_date, str):
                custom_start_date = pd.to_datetime(custom_start_date)
            start_date = max(common_start, custom_start_date)
            print(f"Using custom start date: {start_date}")
        
        if custom_end_date is not None:
            if isinstance(custom_end_date, str):
                custom_end_date = pd.to_datetime(custom_end_date)
            end_date = min(common_end, custom_end_date)
            print(f"Using custom end date: {end_date}")
        
        # Validate date range
        if start_date >= end_date:
            print("Error: Start date must be before end date.")
            return None, None
        
        # Make sure we have enough data before the start date for lookback calculations
        adjusted_start = start_date - timedelta(days=lookback_period*2)
        
        # Create price dataframe
        price_df = PriceDataProcessor.create_price_dataframe(
            valid_cryptos, adjusted_start, end_date, lookback_period
        )
        
        print(f"Price dataframe shape: {price_df.shape}")
        
        # Run the strategy
        result_df = strategy.run(price_df, start_date, end_date, btc_data)
        
        # Print metrics
        strategy.print_metrics()
        
        return result_df, strategy.metrics


class HourlyBacktester(CryptoBacktester):
    """Backtester for hourly cryptocurrency data"""
    
    def run_backtest(self, strategy, lookback_period=24, custom_start_date=None, custom_end_date=None):
        """
        Run a backtest with the given strategy on hourly data
        
        Parameters:
        -----------
        strategy : Strategy
            The strategy to backtest
        lookback_period : int
            Number of hours to look back for performance calculation
        custom_start_date : str or datetime, optional
            Custom start date in 'YYYY-MM-DD HH:MM:SS' format or datetime object
        custom_end_date : str or datetime, optional
            Custom end date in 'YYYY-MM-DD HH:MM:SS' format or datetime object
        """
        # Filter cryptocurrencies with sufficient data
        valid_cryptos = self.filter_valid_cryptos(lookback_period)
        
        if len(valid_cryptos) == 0:
            print("Error: No cryptocurrencies with sufficient data found.")
            return None, None
        
        # Check for BTC data for benchmark
        btc_data = valid_cryptos.get('btc')
        if btc_data is None:
            print("Warning: Bitcoin data not found. Cannot use as benchmark.")
        
        # Find common date range
        common_start, common_end = self.find_common_date_range(valid_cryptos)
        
        # Apply custom date range if provided
        start_date = common_start
        end_date = common_end
        
        if custom_start_date is not None:
            if isinstance(custom_start_date, str):
                custom_start_date = pd.to_datetime(custom_start_date)
            start_date = max(common_start, custom_start_date)
            print(f"Using custom start date: {start_date}")
        
        if custom_end_date is not None:
            if isinstance(custom_end_date, str):
                custom_end_date = pd.to_datetime(custom_end_date)
            end_date = min(common_end, custom_end_date)
            print(f"Using custom end date: {end_date}")
        
        # Validate date range
        if start_date >= end_date:
            print("Error: Start date must be before end date.")
            return None, None
        
        # Make sure we have enough data before the start date for lookback calculations
        adjusted_start = start_date - pd.Timedelta(hours=lookback_period*2)
        
        # Create hourly price dataframe
        price_df = HourlyDataProcessor.create_hourly_price_dataframe(
            valid_cryptos, adjusted_start, end_date, lookback_period
        )
        
        print(f"Price dataframe shape: {price_df.shape}")
        
        # Run the strategy
        result_df = strategy.run(price_df, start_date, end_date, btc_data)
        
        # Print metrics
        strategy.print_metrics()
        
        return result_df, strategy.metrics


def main():
    """Main function to run the backtester"""
    import argparse
    
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Cryptocurrency Trading Strategy Backtester')
    parser.add_argument('--start_date', type=str, help='Start date (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS)')
    parser.add_argument('--end_date', type=str, help='End date (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS)')
    parser.add_argument('--lookback', type=int, default=24, help='Lookback period in hours (default: 24)')
    parser.add_argument('--top_n', type=int, default=5, help='Number of top coins to select (default: 5)')
    parser.add_argument('--capital', type=float, default=10000, help='Initial capital (default: 10000)')
    parser.add_argument('--stop_loss', type=float, default=0.05, help='Stop loss percentage (default: 0.05)')
    parser.add_argument('--stop_window', type=int, default=4, help='Stop loss window in hours (default: 4)')
    parser.add_argument('--hourly', action='store_true', help='Use hourly data strategy with stop loss')
    parser.add_argument('--save_results', action='store_true', help='Save performance results to CSV')
    
    # Parse arguments
    args = parser.parse_args()
    
    if args.hourly:
        # Create hourly backtester instance
        backtester = HourlyBacktester()
        
        # Create hourly strategy with stop loss
        strategy = HourlyTopPerformersWithStopLoss(
            lookback_period=args.lookback,
            num_top_coins=args.top_n,
            initial_capital=args.capital,
            stop_loss_pct=args.stop_loss,
            stop_loss_window=args.stop_window
        )
        
        print(f"\n===== Testing Hourly Strategy with {args.lookback}-hour lookback and {args.stop_loss:.1%} stop loss =====")
        result_df, metrics = backtester.run_backtest(
            strategy,
            lookback_period=args.lookback,
            custom_start_date=args.start_date,
            custom_end_date=args.end_date
        )
    else:
        # Create regular backtester instance
        backtester = CryptoBacktester()
        
        # Create regular strategy
        strategy = TopPerformersStrategy(
            lookback_period=args.lookback,
            num_top_coins=args.top_n,
            initial_capital=args.capital
        )
        
        print(f"\n===== Testing Top {args.top_n} Strategy with {args.lookback}-day lookback =====")
        result_df, metrics = backtester.run_backtest(
            strategy,
            lookback_period=args.lookback,
            custom_start_date=args.start_date,
            custom_end_date=args.end_date
        )
    
    if result_df is not None:
        print("\nBacktest completed successfully!")
        
        # Save results if requested
        if args.save_results:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save metrics to CSV
            metrics_df = pd.DataFrame([metrics])
            metrics_file = f"metrics_{timestamp}.csv"
            metrics_df.to_csv(metrics_file, index=False)
            print(f"Metrics saved to {metrics_file}")
            
            # Save daily performance data
            results_file = f"performance_{timestamp}.csv"
            result_df.to_csv(results_file)
            print(f"Performance data saved to {results_file}")
            
            # Save summary of stop loss events for hourly strategy
            if args.hourly and hasattr(strategy, 'stop_loss_events') and strategy.stop_loss_events:
                stop_loss_data = pd.DataFrame(strategy.stop_loss_events)
                stop_loss_file = f"stop_loss_events_{timestamp}.csv"
                stop_loss_data.to_csv(stop_loss_file, index=False)
                print(f"Stop loss events saved to {stop_loss_file}")
    else:
        print("\nBacktest failed.")


if __name__ == "__main__":
    main()