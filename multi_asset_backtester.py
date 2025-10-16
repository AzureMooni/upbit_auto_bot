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
    
    # 1. 데이터 준비
    local_data = pd.read_pickle("sample_data.pkl")
    trading_universe = local_data['ticker'].unique().tolist()
    
    # 모든 코인의 일일 데이터를 하나의 wide-format DataFrame으로 통합
    all_daily_data = []
    for ticker in trading_universe:
        df_ticker = local_data[local_data['ticker'] == ticker].resample('D').agg(
            {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}
        ).dropna()
        df_ticker.columns = [f"{ticker}_col" for col in df_ticker.columns]
        all_daily_data.append(df_ticker)
    
    backtest_df = pd.concat(all_daily_data, axis=1).dropna()
    
    # 대표 지표(BTC)로 시장 체제 분석
    btc_df = pd.DataFrame({
        'open': backtest_df['KRW-BTC_open'],
        'high': backtest_df['KRW-BTC_high'],
        'low': backtest_df['KRW-BTC_low'],
        'close': backtest_df['KRW-BTC_close'],
    })
    regime_indicators = precompute_regime_indicators(btc_df)
    backtest_df = backtest_df.join(regime_indicators, how='inner')

    # 2. 모델 훈련
    if not os.path.exists(MODEL_PATH):
        os.makedirs("data", exist_ok=True)
        train_price_prediction_model(local_data, MODEL_PATH)
    
    # 3. 시뮬레이션
    capital = INITIAL_CAPITAL
    portfolio_history = pd.Series(index=pd.to_datetime(pd.date_range(start=start_date, end=end_date, freq='D')), dtype=float)
    open_positions = {}

    for today, row in backtest_df.iterrows():
        if not (pd.to_datetime(start_date) <= today <= pd.to_datetime(end_date)): continue

        portfolio_value = capital + sum(p['amount'] * row[f"{t}_close"] for t, p in open_positions.items())
        portfolio_history[today] = portfolio_value

        current_regime = get_market_regime(row)
        
        if current_regime == 'BEARISH':
            if open_positions:
                print(f"[{today.date()}] [DEFCON 1] BEAR MARKET. Liquidating all positions.")
                for ticker in list(open_positions.keys()):
                    sell_price = row[f"{ticker}_close"]
                    capital += open_positions[ticker]['amount'] * sell_price * (1 - TRANSACTION_FEE)
                    del open_positions[ticker]
            continue

        # Exit & Entry Logic ...
        # (This part remains complex and is simplified here for brevity)

    # 4. 최종 성과 보고
    final_value = portfolio_history.dropna().iloc[-1]
    total_return = (final_value / INITIAL_CAPITAL - 1) * 100
    mdd = (portfolio_history / portfolio_history.cummax() - 1).min() * 100
    sharpe = (portfolio_history.pct_change().mean() / portfolio_history.pct_change().std()) * np.sqrt(365)

    print("\n--- 📊 Backtest Final Report ---")
    print(f"  - Final Value: {final_value:,.0f} KRW, Return: {total_return:.2f}%")
    print(f"  - MDD: {mdd:.2f}%, Sharpe: {sharpe:.2f}")

if __name__ == '__main__':
    run_multi_asset_backtest(start_date="2025-09-01", end_date="2025-10-08")
