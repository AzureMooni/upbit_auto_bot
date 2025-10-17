
import pandas as pd
import numpy as np
import os
from universe_manager import get_top_10_coins
from dl_predictor import train_price_prediction_model, predict_win_probability
from market_regime_detector import precompute_regime_indicators, get_market_regime

# --- Configuration ---
INITIAL_CAPITAL = 1_000_000
MODEL_PATH = "data/v2_lightgbm_model.joblib"
TRANSACTION_FEE = 0.0005
TRAILING_STOP_PCT = 0.10

def run_multi_asset_backtest(start_date, end_date):
    print("--- Starting Multi-Asset Backtest for AI Commander v2.0 (with Defense Protocol) ---")
    
    local_data = pd.read_pickle("sample_data.pkl")
    trading_universe = local_data['ticker'].unique().tolist()
    
    all_daily_data = []
    for ticker in trading_universe:
        df_ticker = local_data[local_data['ticker'] == ticker].resample('D').agg(
            {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}
        ).dropna()
        df_ticker.columns = [f"{ticker.replace('KRW-','')}_{col}" for col in df_ticker.columns]
        all_daily_data.append(df_ticker)
    
    backtest_df = pd.concat(all_daily_data, axis=1).dropna()
    
    btc_df = pd.DataFrame({
        'open': backtest_df['BTC_open'], 'high': backtest_df['BTC_high'],
        'low': backtest_df['BTC_low'], 'close': backtest_df['BTC_close'],
    })
    regime_indicators = precompute_regime_indicators(btc_df)
    backtest_df = backtest_df.join(regime_indicators, how='inner')

    if not os.path.exists(MODEL_PATH):
        os.makedirs("data", exist_ok=True)
        train_price_prediction_model(local_data, MODEL_PATH)
    
    capital = INITIAL_CAPITAL
    portfolio_history = pd.Series(index=pd.to_datetime(pd.date_range(start=start_date, end=end_date, freq='D')), dtype=float)
    open_positions = {}

    for today, row in backtest_df.iterrows():
        if not (pd.to_datetime(start_date) <= today <= pd.to_datetime(end_date)): continue

        portfolio_value = capital + sum(p['amount'] * row[f"{t.replace('KRW-','')}_close"] for t, p in open_positions.items())
        portfolio_history[today] = portfolio_value

        current_regime = get_market_regime(row)
        
        if current_regime == 'BEARISH':
            if open_positions:
                for ticker in list(open_positions.keys()):
                    sell_price = row[f"{ticker.replace('KRW-','')}_close"]
                    capital += open_positions[ticker]['amount'] * sell_price * (1 - TRANSACTION_FEE)
                    del open_positions[ticker]
            continue

    # [FIX] Report generation with empty history check
    if portfolio_history.dropna().empty:
        print("\n[WARN] No trading activity during the backtest period. Cannot generate performance report.")
        return

    final_value = portfolio_history.dropna().iloc[-1]
    total_return = (final_value / INITIAL_CAPITAL - 1) * 100
    mdd = (portfolio_history / portfolio_history.cummax() - 1).min() * 100
    sharpe = (portfolio_history.pct_change().mean() / portfolio_history.pct_change().std()) * np.sqrt(365)

    print("\n--- 📊 Backtest Final Report ---")
    print(f"  - Final Value: {final_value:,.0f} KRW, Return: {total_return:.2f}%")
    print(f"  - MDD: {mdd:.2f}%, Sharpe: {sharpe:.2f}")

if __name__ == '__main__':
    # [FIX] Set backtest period to match the actual range of sample_data.pkl
    run_multi_asset_backtest(start_date="2025-08-28", end_date="2025-10-08")
