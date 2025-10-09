import pandas as pd
import numpy as np
import os
from dl_predictor import train_price_prediction_model, predict_win_probability

# --- Configuration ---
INITIAL_CAPITAL = 1_000_000
MODEL_PATH = "data/v2_lightgbm_model.joblib" # 모델 경로 변경
TRANSACTION_FEE = 0.0005
TRAILING_STOP_PCT = 0.10

def run_multi_asset_backtest(start_date, end_date):
    print("--- Starting Multi-Asset Backtest for AI Commander v2.0 (LightGBM) ---")
    
    # 1. 로컬 샘플 데이터 로드
    all_data = pd.read_pickle("sample_data.pkl")
    trading_universe = all_data['ticker'].unique().tolist()
    print(f"[INFO] Backtest universe loaded from sample data: {trading_universe}")

    # 2. 모델 훈련
    if not os.path.exists(MODEL_PATH):
        os.makedirs("data", exist_ok=True)
        # 모든 코인 데이터를 사용하여 훈련
        train_price_prediction_model(all_data, MODEL_PATH)
    
    # 3. 백테스트 시뮬레이션
    print("\n[INFO] Starting backtest simulation loop...")
    capital = INITIAL_CAPITAL
    portfolio_history = pd.Series(index=pd.to_datetime(pd.date_range(start=start_date, end=end_date, freq='D')), dtype=float)
    open_positions = {}

    for today in portfolio_history.index:
        portfolio_value = capital + sum(p['value'] for p in open_positions.values())
        portfolio_history[today] = portfolio_value

        # Exit Logic
        for ticker in list(open_positions.keys()):
            position = open_positions[ticker]
            today_data = all_data[all_data['ticker'] == ticker]
            today_data = today_data[today_data.index.date == today.date()]
            if not today_data.empty:
                current_low = today_data['low'].min()
                position['peak_price'] = max(position['peak_price'], today_data['high'].max())
                trailing_stop_price = position['peak_price'] * (1 - TRAILING_STOP_PCT)
                if current_low <= trailing_stop_price:
                    sell_price = current_low
                    sell_value = position['amount'] * sell_price
                    capital += sell_value * (1 - TRANSACTION_FEE)
                    del open_positions[ticker]

        # Entry Logic
        for ticker in trading_universe:
            if ticker in open_positions:
                continue

            data = all_data[all_data['ticker'] == ticker]
            data = data[data.index <= today]
            if len(data) < 20:
                continue

            # 1. Get Signal (Win Probability)
            # 피처 생성 (dl_predictor와 동일한 수동 방식)
            features = data.copy()
            delta = features['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            features['RSI'] = 100 - (100 / (1 + rs))
            ema_fast = features['close'].ewm(span=12, adjust=False).mean()
            ema_slow = features['close'].ewm(span=26, adjust=False).mean()
            features['MACD_hist'] = ema_fast - ema_slow - (ema_fast - ema_slow).ewm(span=9, adjust=False).mean()
            mid_band = features['close'].rolling(window=20).mean()
            std_dev = features['close'].rolling(window=20).std()
            upper_band = mid_band + (std_dev * 2)
            lower_band = mid_band - (std_dev * 2)
            features['BBP'] = (features['close'] - lower_band) / (upper_band - lower_band)
            high_low = features['high'] - features['low']
            high_close = np.abs(features['high'] - features['close'].shift())
            low_close = np.abs(features['low'] - features['close'].shift())
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            features['ATR'] = tr.rolling(window=14).mean()
            features.dropna(inplace=True)
            
            if features.empty:
                continue

            live_features = features.tail(1)[['RSI', 'MACD_hist', 'BBP', 'ATR']]
            p_win = predict_win_probability(live_features, MODEL_PATH)

            # 2. Calculate Position Size
            if p_win > 0.55:
                kelly_fraction = p_win - (1 - p_win)
                position_size_ratio = min(kelly_fraction * 0.5, 0.25)
                position_size = portfolio_value * position_size_ratio
                
                # 3. Execute Trade
                if capital >= position_size:
                    buy_price = data.iloc[-1]['close']
                    amount = (position_size / buy_price) * (1 - TRANSACTION_FEE)
                    capital -= position_size
                    open_positions[ticker] = {'entry_price': buy_price, 'peak_price': buy_price, 'amount': amount, 'value': position_size}
    
    # 4. 최종 성과 보고
    final_value = portfolio_history.dropna().iloc[-1]
    total_return = (final_value / INITIAL_CAPITAL - 1) * 100
    mdd = (portfolio_history / portfolio_history.cummax() - 1).min() * 100
    sharpe = (portfolio_history.pct_change().mean() / portfolio_history.pct_change().std()) * np.sqrt(365)

    print("\n--- 📊 Multi-Asset LightGBM Backtest Final Report ---")
    print(f"  - Period: {start_date} ~ {end_date}")
    print(f"  - Final Portfolio Value: {final_value:,.0f} KRW")
    print(f"  - Total Return: {total_return:.2f}%")
    print(f"  - MDD: {mdd:.2f}%")
    print(f"  - Sharpe Ratio (Annualized): {sharpe:.2f}")

if __name__ == '__main__':
    # 샘플 데이터 기간에 맞춰 백테스트 기간 설정
    run_multi_asset_backtest(start_date="2025-09-08", end_date="2025-10-08")