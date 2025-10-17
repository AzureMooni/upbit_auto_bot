import pandas as pd
import numpy as np
import os
from dl_predictor import train_price_prediction_model, predict_win_probability
from market_regime_detector import precompute_all_indicators, get_market_regime

INITIAL_CAPITAL = 1_000_000
MODEL_PATH = "data/btc_advanced_model.joblib"
TRANSACTION_FEE = 0.0005
TRAILING_STOP_PCT = 0.10

def run_backtest(start_date, end_date):
    print("--- Starting Final Backtest with Advanced Features ---")
    
    ticker = "KRW-BTC"
    full_df = pd.read_parquet(f"data/{ticker}.parquet")
    
    # 모든 지표를 한 번에 사전 계산
    data_with_features = precompute_all_indicators(full_df.copy())

    if not os.path.exists(MODEL_PATH):
        os.makedirs("data", exist_ok=True)
        train_price_prediction_model(full_df, MODEL_PATH)
    
    capital = INITIAL_CAPITAL
    portfolio_history = pd.Series(index=pd.to_datetime(pd.date_range(start=start_date, end=end_date, freq='D')), dtype=float)
    position = None

    # 일봉 데이터로 백테스트 루프 실행
    daily_df = data_with_features.resample('D').last()
    for today, row in daily_df.iterrows():
        if not (pd.to_datetime(start_date) <= today <= pd.to_datetime(end_date)): continue

        current_price = row['close']
        portfolio_value = capital + (position['amount'] * current_price if position else 0)
        portfolio_history[today] = portfolio_value

        current_regime = get_market_regime(row)
        
        if current_regime == 'BEARISH':
            if position:
                capital += position['amount'] * current_price * (1 - TRANSACTION_FEE)
                position = None
            continue

        if position and row['low'] <= position['trailing_stop']:
            capital += position['amount'] * row['low'] * (1 - TRANSACTION_FEE)
            position = None

        if not position:
            p_win = predict_win_probability(pd.DataFrame([row]), MODEL_PATH)
            if p_win > 0.65: # 진입 기준 상향 조정
                position_size = capital * 0.2 # 단순화된 포지션 크기
                amount = (position_size / current_price) * (1 - TRANSACTION_FEE)
                capital -= position_size
                position = {'amount': amount, 'peak_price': current_price, 'trailing_stop': current_price * (1 - TRAILING_STOP_PCT)}

        if position:
            position['peak_price'] = max(position['peak_price'], row['high'])
            position['trailing_stop'] = position['peak_price'] * (1 - TRAILING_STOP_PCT)

    if portfolio_history.dropna().empty: return

    final_value = portfolio_history.dropna().iloc[-1]
    total_return = (final_value / INITIAL_CAPITAL - 1) * 100
    mdd = (portfolio_history / portfolio_history.cummax() - 1).min() * 100
    sharpe = (portfolio_history.pct_change().mean() / portfolio_history.pct_change().std()) * np.sqrt(365)

    print(f"\n--- 📊 Final Report ---\n  - Return: {total_return:.2f}%, MDD: {mdd:.2f}%, Sharpe: {sharpe:.2f}")

if __name__ == '__main__':
    run_backtest(start_date="2021-01-01", end_date="2023-12-31")
