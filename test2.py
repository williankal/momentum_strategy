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
    
    def _create_stop_loss_events_chart(self, price_df):
        """Create visualization of stop loss events"""
        if not self.stop_loss_events:
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
        
        # Create figure for each of the top coins
        for symbol in top_symbol_names:
            # Get all events for this symbol
            symbol_events = [e for e in self.stop_loss_events if e['symbol'] == symbol]
            
            # Create price chart with stop loss markers
            fig = go.Figure()
            
            # Add price line
            symbol_data = price_df[symbol].loc[symbol_events[0]['timestamp']:symbol_events[-1]['timestamp']]
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
    else:
        print("\nBacktest failed.")


if __name__ == "__main__":
    main()