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

import pandas as pd
import numpy as np
from datetime import timedelta
from plotly.subplots import make_subplots
import plotly.graph_objects as go

def top_performers_strategy(
    crypto_data,
    lookback_period=7,          # ainda usado p/ verificar histórico mínimo e alocar datas
    ema_span=7,                 
    num_top_coins=5,
    initial_capital=10_000,
    commission=0.001,
    stop_loss_pct=0.05,
    check_interval_hours=1,
    start_date=None,
    end_date=None
):
    """
    Estratégia semanal que compra as criptomoedas cujo preço está
    proporcionalmente mais acima da sua EMA de `ema_span` dias,
    com rebalanceamento toda segunda‑feira e stop‑loss horário.

    Parâmetros principais alterados:
    --------------------------------
    ema_span : int
        Janela (dias) usada para calcular a EMA.
    lookback_period : int
        Ainda usado para garantir que cada ativo tenha histórico suficiente
        e para estender o dataframe de preços.
    """

    # ------------------------------------------------------------------
    # 1. Filtra criptos com dados suficientes
    # ------------------------------------------------------------------
    valid_cryptos = {
        sym: df
        for sym, df in crypto_data.items()
        if len(df) > lookback_period * 24 + 10
    }
    print(f"Found {len(valid_cryptos)} cryptocurrencies with sufficient data")
    if not valid_cryptos:
        print("Error: No cryptocurrencies with sufficient data found.")
        return None, None

    btc_data = valid_cryptos.get("btc")  # pode ser None

    # ------------------------------------------------------------------
    # 2. Determina intervalo de datas comum
    # ------------------------------------------------------------------
    start_dates = [df.index.min() for df in valid_cryptos.values()]
    end_dates   = [df.index.max() for df in valid_cryptos.values()]
    print(f"Data available range: {min(start_dates)} to {max(end_dates)}")

    common_start = (
        max(start_dates) if start_date is None else pd.to_datetime(start_date)
    )
    common_end   = (
        min(end_dates)   if end_date   is None else pd.to_datetime(end_date)
    )
    if common_start < max(start_dates):
        common_start = max(start_dates)
        print(f"Start date adjusted to {common_start}")
    if common_end > min(end_dates):
        common_end = min(end_dates)
        print(f"End date adjusted to {common_end}")
    if common_start >= common_end:
        print("Error: start date must be before end date")
        return None, None
    print(f"Common date range: {common_start} to {common_end}")

    # ------------------------------------------------------------------
    # 3. Cria dataframe de preços horário
    # ------------------------------------------------------------------
    all_dates = pd.date_range(
        start=common_start - timedelta(days=lookback_period * 2),
        end=common_end,
        freq=f"{check_interval_hours}H",
    )
    price_df = pd.DataFrame(index=all_dates)
    for sym, df in valid_cryptos.items():
        symbol_prices = pd.DataFrame({"close": df["close"]})
        reindexed = symbol_prices.reindex(all_dates)
        price_df[sym] = reindexed["close"].ffill()
    price_df = price_df.dropna(axis=1, thresh=len(all_dates) * 0.95)
    if price_df.empty:
        print("Error: No cryptocurrencies with sufficient data after cleaning.")
        return None, None
    price_df = price_df.ffill().bfill()

    # ------------------------------------------------------------------
    # 4. Calcula EMA diária e "EMA‑score" (% acima da EMA)
    # ------------------------------------------------------------------
    daily_price_df = price_df.resample("D").last()
    ema_df = daily_price_df.ewm(span=ema_span, adjust=False).mean()
    ema_score_df = (daily_price_df - ema_df) / ema_df
    ema_score_df = ema_score_df.dropna()

    # ------------------------------------------------------------------
    # 5. Define datas de rebalanceamento (toda segunda)
    # ------------------------------------------------------------------
    start_monday = (common_start + timedelta(days=(7 - common_start.weekday()) % 7)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    end_monday = (common_end - timedelta(days=common_end.weekday())).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    rebalance_dates = pd.date_range(start=start_monday, end=end_monday, freq="W-MON")
    print(
        f"Strategy will run from {start_monday} to {end_monday} with "
        f"{len(rebalance_dates)} rebalance points"
    )

    # ------------------------------------------------------------------
    # 6. Inicializa portfólio
    # ------------------------------------------------------------------
    portfolio_df = pd.DataFrame(index=price_df.loc[start_monday:].index)
    portfolio_df[["Cash", "Crypto_Value", "Total_Value"]] = 0.0
    portfolio_df.loc[:, "Cash"] = initial_capital
    portfolio_df.loc[:, "Total_Value"] = initial_capital

    current_holdings, entry_prices = {}, {}
    current_cash = initial_capital
    trade_history, holdings_history, stop_loss_events = [], {}, []

    # ------------------------------------------------------------------
    # 7. Loop de rebalanceamento semanal
    # ------------------------------------------------------------------
    print("Simulating strategy with EMA ranking…")
    for i, rebalance_date in enumerate(rebalance_dates):
        next_rebalance = (
            rebalance_dates[i + 1] if i < len(rebalance_dates) - 1 else common_end
        )
        if rebalance_date > common_end:
            break

        # ----- 7.1 Seleciona top performers pelo EMA‑score -----
        week_key = rebalance_date.strftime("%Y-%W")
        closest_date = (
            ema_score_df.index[ema_score_df.index <= rebalance_date][-1]
            if any(ema_score_df.index <= rebalance_date)
            else None
        )
        if closest_date is not None:
            scores = ema_score_df.loc[closest_date]
            top_performers = scores.nlargest(num_top_coins).index.tolist()
            holdings_history[week_key] = top_performers

            # ----- 7.2 Vende posições antigas -----
            for sym, units in list(current_holdings.items()):
                price = price_df.loc[price_df.index[price_df.index <= rebalance_date][-1], sym]
                sale_val = units * price * (1 - commission)
                current_cash += sale_val
                trade_history.append(
                    dict(date=rebalance_date, symbol=sym, action="SELL",
                         units=units, price=price, value=sale_val, reason="REBALANCE")
                )
            current_holdings, entry_prices = {}, {}

            # ----- 7.3 Compra novas posições -----
            alloc = current_cash / num_top_coins if top_performers else 0
            for sym in top_performers:
                price = price_df.loc[price_df.index[price_df.index <= rebalance_date][-1], sym]
                if price > 0:
                    units = alloc * (1 - commission) / price
                    current_holdings[sym] = units
                    entry_prices[sym] = price
                    current_cash -= alloc
                    trade_history.append(
                        dict(date=rebalance_date, symbol=sym, action="BUY",
                             units=units, price=price, value=alloc, reason="REBALANCE")
                    )

        # ----- 7.4 Loop horário entre rebalanceamentos (stop‑loss) -----
        time_range = pd.date_range(
            start=rebalance_date,
            end=next_rebalance - timedelta(hours=check_interval_hours),
            freq=f"{check_interval_hours}H",
        )
        for check_time in time_range:
            if check_time not in price_df.index:
                continue

            # verifica stop‑loss
            to_sell = []
            for sym, units in current_holdings.items():
                price_now = price_df.loc[check_time, sym]
                drop = (entry_prices[sym] - price_now) / entry_prices[sym]
                if drop >= stop_loss_pct:
                    to_sell.append(sym)
                    stop_loss_events.append(
                        dict(date=check_time, symbol=sym, entry_price=entry_prices[sym],
                             exit_price=price_now, drop_pct=drop)
                    )
            for sym in to_sell:
                units = current_holdings.pop(sym)
                price_now = price_df.loc[check_time, sym]
                sale_val = units * price_now * (1 - commission)
                current_cash += sale_val
                entry_prices.pop(sym, None)
                trade_history.append(
                    dict(date=check_time, symbol=sym, action="SELL",
                         units=units, price=price_now, value=sale_val, reason="STOP_LOSS")
                )

            # atualiza valor da carteira
            crypto_val = sum(
                units * price_df.loc[check_time, sym]
                for sym, units in current_holdings.items()
            )
            portfolio_df.loc[check_time, "Cash"] = current_cash
            portfolio_df.loc[check_time, "Crypto_Value"] = crypto_val
            portfolio_df.loc[check_time, "Total_Value"] = current_cash + crypto_val

    portfolio_df = portfolio_df.ffill()

    # ------------------------------------------------------------------
    # 8. Benchmarks (BTC buy‑&‑hold e Equal‑Weight buy‑&‑hold)
    # ------------------------------------------------------------------
    if btc_data is not None and "btc" in price_df.columns:
        start_price = price_df.loc[price_df.index >= start_monday, "btc"].iloc[0]
        btc_units = initial_capital * (1 - commission) / start_price
        portfolio_df["BTC_Value"] = btc_units * price_df.loc[start_monday:, "btc"]

    equal_syms = list(price_df.columns)
    alloc_eq = initial_capital / len(equal_syms)
    portfolio_df["Equal_Weight_Value"] = 0.0
    start_idx = price_df.index[price_df.index >= start_monday][0]
    for sym in equal_syms:
        init_units = alloc_eq * (1 - commission) / price_df.loc[start_idx, sym]
        portfolio_df.loc[start_monday:, "Equal_Weight_Value"] += (
            init_units * price_df.loc[start_monday:, sym]
        )

    # ------------------------------------------------------------------
    # 9. Métricas de performance
    # ------------------------------------------------------------------
    daily = portfolio_df.resample("D").last()
    daily["Strategy_Return"] = daily["Total_Value"] / initial_capital - 1
    daily["Equal_Weight_Return"] = daily["Equal_Weight_Value"] / initial_capital - 1
    if "BTC_Value" in daily:
        daily["BTC_Return"] = daily["BTC_Value"] / initial_capital - 1

    daily["Strategy_Daily_Return"] = daily["Total_Value"].pct_change().fillna(0)
    daily["Equal_Weight_Daily_Return"] = daily["Equal_Weight_Value"].pct_change().fillna(0)
    if "BTC_Value" in daily:
        daily["BTC_Daily_Return"] = daily["BTC_Value"].pct_change().fillna(0)

    daily["Strategy_DD"] = 1 - daily["Total_Value"] / daily["Total_Value"].cummax()
    daily["Equal_Weight_DD"] = 1 - daily["Equal_Weight_Value"] / daily["Equal_Weight_Value"].cummax()
    if "BTC_Value" in daily:
        daily["BTC_DD"] = 1 - daily["BTC_Value"] / daily["BTC_Value"].cummax()

    tot_days = (daily.index[-1] - daily.index[0]).days
    strat_ret = daily["Strategy_Return"].iloc[-1]
    strat_ann = (1 + strat_ret) ** (365 / tot_days) - 1 if tot_days > 0 else np.nan
    strat_vol = daily["Strategy_Daily_Return"].std() * np.sqrt(252)
    strat_sharpe = strat_ann / strat_vol if strat_vol > 0 else 0
    strat_dd = daily["Strategy_DD"].max()

    eq_ret = daily["Equal_Weight_Return"].iloc[-1]
    eq_ann = (1 + eq_ret) ** (365 / tot_days) - 1 if tot_days > 0 else np.nan
    eq_vol = daily["Equal_Weight_Daily_Return"].std() * np.sqrt(252)
    eq_sharpe = eq_ann / eq_vol if eq_vol > 0 else 0
    eq_dd = daily["Equal_Weight_DD"].max()

    if "BTC_Return" in daily:
        btc_ret = daily["BTC_Return"].iloc[-1]
        btc_ann = (1 + btc_ret) ** (365 / tot_days) - 1 if tot_days > 0 else np.nan
        btc_vol = daily["BTC_Daily_Return"].std() * np.sqrt(252)
        btc_sharpe = btc_ann / btc_vol if btc_vol > 0 else 0
        btc_dd = daily["BTC_DD"].max()
        alpha_vs_btc = strat_ret - btc_ret
    else:
        btc_ret = btc_ann = btc_vol = btc_sharpe = btc_dd = alpha_vs_btc = np.nan

    # resumo
    metrics = {
        "Strategy_Total_Return": f"{strat_ret:.2%}",
        "Strategy_Annual_Return": f"{strat_ann:.2%}",
        "Strategy_Volatility": f"{strat_vol:.2%}",
        "Strategy_Sharpe": f"{strat_sharpe:.2f}",
        "Strategy_Max_DD": f"{strat_dd:.2%}",
        "Equal_Weight_Total_Return": f"{eq_ret:.2%}",
        "Equal_Weight_Annual_Return": f"{eq_ann:.2%}",
        "Equal_Weight_Sharpe": f"{eq_sharpe:.2f}",
        "Equal_Weight_Max_DD": f"{eq_dd:.2%}",
        "BTC_Total_Return": f"{btc_ret:.2%}" if not np.isnan(btc_ret) else "N/A",
        "BTC_Annual_Return": f"{btc_ann:.2%}" if not np.isnan(btc_ann) else "N/A",
        "BTC_Sharpe": f"{btc_sharpe:.2f}" if not np.isnan(btc_sharpe) else "N/A",
        "BTC_Max_DD": f"{btc_dd:.2%}" if not np.isnan(btc_dd) else "N/A",
        "Alpha_vs_BTC": f"{alpha_vs_btc:.2%}" if not np.isnan(alpha_vs_btc) else "N/A",
        "Total_Rebalances": len(holdings_history),
        "Total_Stop_Losses": len(stop_loss_events),
    }

    # ------------------------------------------------------------------
    # 10. Gráficos (Plotly)
    # ------------------------------------------------------------------
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=(
            f"Weekly Top {num_top_coins} EMA-Based Strategy (stop-loss {stop_loss_pct:.0%})",
            "Drawdowns",
        ),
        row_heights=[0.7, 0.3],
    )

    fig.add_trace(
        go.Scatter(
            x=daily.index,
            y=daily["Total_Value"],
            mode="lines",
            name="EMA Strategy",
            line=dict(width=2),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=daily.index,
            y=daily["Equal_Weight_Value"],
            mode="lines",
            name="Equal‑Weight B&H",
            line=dict(width=2),
        ),
        row=1,
        col=1,
    )
    if "BTC_Value" in daily:
        fig.add_trace(
            go.Scatter(
                x=daily.index,
                y=daily["BTC_Value"],
                mode="lines",
                name="BTC B&H",
                line=dict(width=2),
            ),
            row=1,
            col=1,
        )
    fig.add_trace(
        go.Scatter(
            x=daily.index,
            y=daily["Strategy_DD"] * 100,
            mode="lines",
            name="Strategy DD",
            line=dict(width=2, color="red"),
            fill="tozeroy",
            fillcolor="rgba(255,0,0,0.1)",
        ),
        row=2,
        col=1,
    )
    if "BTC_DD" in daily:
        fig.add_trace(
            go.Scatter(
                x=daily.index,
                y=daily["BTC_DD"] * 100,
                mode="lines",
                name="BTC DD",
                line=dict(width=2, dash="dash"),
                fill="tozeroy",
                fillcolor="rgba(255,165,0,0.1)",
            ),
            row=2,
            col=1,
        )

    # marca eventos de stop‑loss
    if stop_loss_events:
        sl_df = pd.DataFrame(stop_loss_events)
        sl_df["date"] = pd.to_datetime(sl_df["date"])
        fig.add_trace(
            go.Scatter(
                x=sl_df["date"],
                y=[
                    daily.loc[daily.index >= d].iloc[0]["Total_Value"]
                    if any(daily.index >= d)
                    else np.nan
                    for d in sl_df["date"]
                ],
                mode="markers",
                name="Stop‑Loss",
                marker=dict(symbol="x", size=10, color="red"),
                hoverinfo="text",
                text=[
                    f"{row.symbol}: drop {row.drop_pct:.2%}"
                    for _, row in sl_df.iterrows()
                ],
            ),
            row=1,
            col=1,
        )

    fig.update_layout(
        template="plotly_white",
        height=800,
        legend=dict(x=0, y=1, orientation="h"),
        yaxis_title="Portfolio Value ($)",
        yaxis2_title="Drawdown (%)",
    )
    fig.update_yaxes(ticksuffix="%", row=2, col=1)
    fig.show()

    # ------------------------------------------------------------------
    # 11. Retorna dados diários e métricas
    # ------------------------------------------------------------------
    return daily, metrics

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
        "Bull Market": ("2020-03-15", "2022-09-15"),
        "Recovery": ("2023-01-01", "2024-12-31")
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
