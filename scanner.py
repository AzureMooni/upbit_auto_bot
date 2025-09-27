import ccxt
import pandas as pd
import time
import pandas_ta as ta
import os
from model_trainer import ModelTrainer

# Initialize ModelTrainer globally
model_trainer = ModelTrainer(model_path=os.path.join(os.path.dirname(__file__), '..', 'price_predictor.pkl'),
                             scaler_path=os.path.join(os.path.dirname(__file__), '..', 'price_scaler.pkl'))
model_trainer.load_model() # Load model if it exists

# Helper function for Pivot Points and Breakout Values (for backtesting)
def _calculate_breakout_levels(df_daily):
    if len(df_daily) < 2:
        return None, None, None, None, None, None # PP, R1, S1, R2, S2, Breakout_Value

    # Previous day's data
    prev_day = df_daily.iloc[-2]
    prev_high = prev_day['high']
    prev_low = prev_day['low']
    prev_close = prev_day['close']

    # Pivot Point (PP)
    pp = (prev_high + prev_low + prev_close) / 3

    # Resistance and Support levels (standard pivot point calculation)
    r1 = (2 * pp) - prev_low
    s1 = (2 * pp) - prev_high
    r2 = pp + (prev_high - prev_low)
    s2 = pp - (prev_high - prev_low)

    # Larry Williams' Breakout Value
    k = 0.5 # User defined k-factor
    breakout_value = (prev_high - prev_low) * k

    return pp, r1, s1, r2, s2, breakout_value

# Helper function for Pivot Points and Breakout Values (for live trading)
def _calculate_breakout_levels_live(ohlcv_daily):
    if not ohlcv_daily or len(ohlcv_daily) < 2:
        return None, None, None, None, None, None # PP, R1, S1, R2, S2, Breakout_Value

    # Previous day's data
    prev_day = ohlcv_daily[-2]
    prev_high = prev_day[2]
    prev_low = prev_day[3]
    prev_close = prev_day[4]

    # Pivot Point (PP)
    pp = (prev_high + prev_low + prev_close) / 3

    # Resistance and Support levels (standard pivot point calculation)
    r1 = (2 * pp) - prev_low
    s1 = (2 * pp) - prev_high
    r2 = pp + (prev_high - prev_low)
    s2 = pp - (prev_high - prev_low)

    # Larry Williams' Breakout Value
    k = 0.5 # User defined k-factor
    breakout_value = (prev_high - prev_low) * k

    return pp, r1, s1, r2, s2, breakout_value

def find_hot_coin(historical_data: dict):
    """
    과거 데이터를 기반으로 머신러닝 모델을 사용하여 미래 상승 확률이 가장 높은 코인을 찾습니다.
    """
    best_coin = None
    highest_buy_prob = -1

    if not model_trainer.model or not model_trainer.scaler:
        print("ML model not loaded. Please train and save the model first.")
        return []

    for symbol, df_1h in historical_data.items():
        if not symbol.endswith('/KRW'):
            continue

        # Ensure enough data for feature generation (e.g., 100 periods for some indicators)
        if len(df_1h) < 150: # A safe margin for various indicators
            continue
        
        try:
            # Predict buy probability for the latest data point
            buy_prob = model_trainer.predict(df_1h.copy())
            
            if buy_prob is not None and buy_prob > highest_buy_prob:
                highest_buy_prob = buy_prob
                best_coin = symbol
        except Exception as e:
            print(f"Error predicting for {symbol} in backtesting: {e}")
            continue

    if best_coin and highest_buy_prob > 0.5: # Only consider if buy probability is reasonably high
        print(f"ML model selected hot coin (Backtest): {best_coin} with buy probability: {highest_buy_prob:.4f}")
        return [best_coin]
    else:
        print("ML model found no hot coins with high buy probability (Backtest).")
        return []

def find_hot_coin_live(upbit_exchange: ccxt.Exchange):
    """
    실시간으로 머신러닝 모델을 사용하여 미래 상승 확률이 가장 높은 코인을 찾습니다.
    """
    best_coin = None
    highest_buy_prob = -1

    if not model_trainer.model or not model_trainer.scaler:
        print("ML model not loaded. Please train and save the model first.")
        return []

    try:
        markets = upbit_exchange.load_markets()
        krw_tickers = [m for m in markets if m.endswith('/KRW')]

        for symbol in krw_tickers:
            # Fetch 1-hour OHLCV data for feature generation
            ohlcv_1h = upbit_exchange.fetch_ohlcv(symbol, '1h', limit=150) # Need enough data for indicators
            if not ohlcv_1h or len(ohlcv_1h) < 150:
                continue
            
            df_1h = pd.DataFrame(ohlcv_1h, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df_1h['timestamp'] = pd.to_datetime(df_1h['timestamp'], unit='ms')
            df_1h.set_index('timestamp', inplace=True)

            try:
                buy_prob = model_trainer.predict(df_1h.copy())
                if buy_prob is not None and buy_prob > highest_buy_prob:
                    highest_buy_prob = buy_prob
                    best_coin = symbol
            except Exception as e:
                print(f"Error predicting for {symbol} in live mode: {e}")
                continue

    except Exception as e:
        print(f"Error in find_hot_coin_live: {e}")
        return []

    if best_coin and highest_buy_prob > 0.5: # Only consider if buy probability is reasonably high
        print(f"ML model selected hot coin (Live): {best_coin} with buy probability: {highest_buy_prob:.4f}")
        return [best_coin]
    else:
        print("ML model found no hot coins with high buy probability (Live).")
        return []

def get_dynamic_grid_prices(ticker: str, historical_data: dict):
    """
    코인 티커를 입력받아서 해당 코인의 최적 그리드 가격 범위를 볼린저 밴드를 이용해 계산합니다.
    상단 밴드 (SMA + 2 * 표준편차)를 upper_price로, 하단 밴드 (SMA - 2 * 표준편차)를 lower_price로 설정합니다.
    historical_data 딕셔너리에서 해당 티커의 과거 데이터를 사용하여 계산합니다.
    """
    df = None
    if ticker in historical_data:
        df = historical_data[ticker]

    if df is None or df.empty or len(df) < 20:
        print(f"Not enough OHLCV data for {ticker} to calculate Bollinger Bands. (Need 20, got {len(df) if df is not None else 0})")
        return None, None
    
    try:
        window = 20
        df['SMA'] = df['close'].rolling(window=window).mean()
        df['STD'] = df['close'].rolling(window=window).std()
        df['Upper_Band'] = df['SMA'] + (df['STD'] * 2)
        df['Lower_Band'] = df['SMA'] - (df['STD'] * 2)
        
        upper_price = df['Upper_Band'].iloc[-1]
        lower_price = df['Lower_Band'].iloc[-1]
        
        print(f"Dynamic Grid Prices for {ticker}: Lower Band = {lower_price:.2f}, Upper Band = {upper_price:.2f}")
        return lower_price, upper_price

    except Exception as e:
        print(f"Error in get_dynamic_grid_prices for {ticker}: {e}")
        return None, None

def get_dynamic_grid_prices_live(ticker: str, upbit_exchange: ccxt.Exchange):
    """
    실시간으로 코인 티커의 최적 그리드 가격 범위를 볼린저 밴드를 이용해 계산합니다.
    """
    try:
        ohlcv_1h = upbit_exchange.fetch_ohlcv(ticker, '1h', limit=20)
        if not ohlcv_1h or len(ohlcv_1h) < 20:
            print(f"Not enough live OHLCV data for {ticker} to calculate Bollinger Bands. (Need 20, got {len(ohlcv_1h) if ohlcv_1h else 0})")
            return None, None
        
        df = pd.DataFrame(ohlcv_1h, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)

        window = 20
        df['SMA'] = df['close'].rolling(window=window).mean()
        df['STD'] = df['close'].rolling(window=window).std()
        df['Upper_Band'] = df['SMA'] + (df['STD'] * 2)
        df['Lower_Band'] = df['SMA'] - (df['STD'] * 2)
        
        upper_price = df['Upper_Band'].iloc[-1]
        lower_price = df['Lower_Band'].iloc[-1]
        
        print(f"Dynamic Grid Prices for {ticker} (Live): Lower Band = {lower_price:.2f}, Upper Band = {upper_price:.2f}")
        return lower_price, upper_price

    except Exception as e:
        print(f"Error in get_dynamic_grid_prices_live for {ticker}: {e}")
        return None, None

def classify_market(ticker: str, historical_data: dict):
    """
    시장의 '성격'을 진단하는 함수.
    변동성 돌파 임박 상태를 최우선으로 진단하고, 그 다음 ADX를 기반으로
    'trending' (추세장), 'ranging' (횡보장), 'choppy' (혼조세)를 반환합니다.
    historical_data 딕셔너리에서 해당 티커의 과거 데이터를 사용하여 계산합니다.
    """
    df_1h = None
    if ticker in historical_data:
        df_1h = historical_data[ticker]

    if df_1h is None or df_1h.empty or len(df_1h) < 100: # ADX 계산을 위해 최소 14개 데이터 필요, 100개로 가정
        print(f"Not enough OHLCV data for {ticker} to classify market. (Need at least 100, got {len(df_1h) if df_1h is not None else 0})")
        return "unknown"

    try:
        # 1. 변동성 돌파 임박 상태 진단 (최우선)
        # 일봉 데이터 가져오기 (1시간 봉에서 일봉으로 리샘플링)
        df_daily = df_1h['close'].resample('1D').ohlc().dropna()
        if len(df_daily) >= 2:
            pp, r1, s1, breakout_value = _calculate_breakout_levels(df_daily)
            if pp is not None:
                current_price = df_1h['close'].iloc[-1]
                if current_price > (pp + breakout_value):
                    return "breakout_up"
                elif current_price < (pp - breakout_value):
                    return "breakout_down"

        # 2. ADX를 이용한 추세/횡보 진단
        df_1h.ta.adx(length=14, append=True, high='high', low='low', close='close')
        adx_values = df_1h['ADX_14']

        if len(adx_values) == 0:
            return "unknown"

        latest_adx = adx_values.iloc[-1] # Use iloc for Series

        if latest_adx >= 25:
            return "trending"
        elif latest_adx < 20:
            return "ranging"
        else:
            return "choppy"
    except Exception as e:
        print(f"Error in classify_market for {ticker}: {e}")
        return "unknown"

def classify_market_live(ticker: str, upbit_exchange: ccxt.Exchange):
    """
    실시간으로 시장의 '성격'을 진단하는 함수.
    변동성 돌파 임박 상태를 최우선으로 진단하고, 그 다음 ADX를 기반으로
    'trending' (추세장), 'ranging' (횡보장), 'choppy' (혼조세)를 반환합니다.
    """
    try:
        # 1. 변동성 돌파 임박 상태 진단 (최우선)
        # 일봉 데이터 가져오기
        ohlcv_daily = upbit_exchange.fetch_ohlcv(ticker, '1d', limit=2) # 전일 고가/저가/종가 필요
        if ohlcv_daily and len(ohlcv_daily) >= 2:
            pp, r1, s1, breakout_value = _calculate_breakout_levels_live(ohlcv_daily)
            if pp is not None:
                current_price = upbit_exchange.fetch_ticker(ticker)['last']
                if current_price > (pp + breakout_value):
                    return "breakout_up"
                elif current_price < (pp - breakout_value):
                    return "breakout_down"

        # 2. ADX를 이용한 추세/횡보 진단
        ohlcv_1h = upbit_exchange.fetch_ohlcv(ticker, '1h', limit=100)
        if not ohlcv_1h or len(ohlcv_1h) < 100:
            print(f"Not enough live OHLCV data for {ticker} to calculate ADX. (Need at least 100, got {len(ohlcv_1h) if ohlcv_1h else 0})")
            return "unknown"
        
        df = pd.DataFrame(ohlcv_1h, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)

        df.ta.adx(length=14, append=True, high='high', low='low', close='close')
        adx_values = df['ADX_14']

        if len(adx_values) == 0:
            return "unknown"

        latest_adx = adx_values.iloc[-1]

        if latest_adx >= 25:
            return "trending"
        elif latest_adx < 20:
            return "ranging"
        else:
            return "choppy"

    except Exception as e:
        print(f"Error in classify_market_live for {ticker}: {e}")
        return "unknown"

if __name__ == '__main__':
    print("--- Running scanner.py examples ---")

    # Dummy historical data for backtesting mode
    dummy_ohlcv_btc = [
        [1672531200000, 20000000, 20100000, 19900000, 20050000, 1000],
        [1672534800000, 20050000, 20200000, 20000000, 20150000, 1200],
        [1672538400000, 20150000, 20300000, 20100000, 20250000, 1500],
        [1672542000000, 20250000, 20400000, 20200000, 20350000, 1300],
        [1672545600000, 20350000, 20500000, 20300000, 20450000, 1100],
        [1672549200000, 20450000, 20600000, 20400000, 20550000, 1400],
        [1672552800000, 20550000, 20700000, 20500000, 20650000, 1600],
        [1672556400000, 20650000, 20800000, 20600000, 20750000, 1700],
        [1672560000000, 20750000, 20900000, 20700000, 20850000, 1800],
        [1672563600000, 20850000, 21000000, 20800000, 20950000, 1900],
        [1672567200000, 20950000, 21100000, 20900000, 21050000, 2000],
        [1672570800000, 21050000, 21200000, 21000000, 21150000, 2100],
        [1672574400000, 21150000, 21300000, 21100000, 21250000, 2200],
        [1672578000000, 21250000, 21400000, 21200000, 21350000, 2300],
        [1672581600000, 21350000, 21500000, 21300000, 21450000, 2400],
        [1672585200000, 21450000, 21600000, 21400000, 21550000, 2500],
        [1672588800000, 21550000, 21700000, 21500000, 21650000, 2600],
        [1672592400000, 21650000, 21800000, 21600000, 21750000, 2700],
        [1672596000000, 21750000, 21900000, 21700000, 21850000, 2800],
        [1672599600000, 21850000, 22000000, 21800000, 21950000, 2900],
        # Add more data to meet the 24-hour, 50-EMA, 100-ADX requirements
        # For simplicity, extending with similar trend
        *[
            [1672599600000 + (i * 3600000), 21950000 + (i * 10000), 22000000 + (i * 10000), 21800000 + (i * 10000), 21950000 + (i * 10000), 2900 + i]
            for i in range(20, 101)
        ]
    ]
    dummy_df_btc = pd.DataFrame(dummy_ohlcv_btc, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    dummy_df_btc['timestamp'] = pd.to_datetime(dummy_df_btc['timestamp'], unit='ms')
    dummy_df_btc.set_index('timestamp', inplace=True)

    dummy_historical_data = {
        'BTC/KRW': dummy_df_btc,
        # Add other dummy data if needed for testing multiple coins
    }

    # Live mode testing
    print("\n--- Live mode testing with ccxt ---")
    live_exchange = ccxt.upbit()

    # 핫 코인 찾기 예시 (Live)
    print("\nFinding hot coins (Live)...")
    hot_coins_live = find_hot_coin_live(live_exchange)
    if hot_coins_live:
        print(f"Found {len(hot_coins_live)} hot coins (Live).")
        first_hot_coin_live = hot_coins_live[0]
        
        # 핫 코인의 동적 그리드 가격 가져오기 예시 (Live)
        print(f"\nGetting dynamic grid prices for {first_hot_coin_live} (Live)...")
        lower_live, upper_live = get_dynamic_grid_prices_live(first_hot_coin_live, live_exchange)
        if lower_live and upper_live:
            print(f"Dynamic Grid Range for {first_hot_coin_live} (Live): Lower={lower_live:.2f}, Upper={upper_live:.2f}")
    else:
        print("Could not find any hot coins (Live) to get dynamic grid prices for.")

    # 시장 분류 예시 (Live)
    print("\nClassifying market for BTC/KRW (Live)...")
    market_type_live = classify_market_live('BTC/KRW', live_exchange)
    print(f"BTC/KRW Market Type (Live): {market_type_live}")

    print("\n--- Backtesting mode testing with historical_data ---")

    # 핫 코인 찾기 예시 (Backtest)
    print("\nFinding hot coins (Backtest)...")
    hot_coins_backtest = find_hot_coin(historical_data=dummy_historical_data)
    if hot_coins_backtest:
        print(f"Found {len(hot_coins_backtest)} hot coins (Backtest).")
        first_hot_coin_backtest = hot_coins_backtest[0]
        
        # 핫 코인의 동적 그리드 가격 가져오기 예시 (Backtest)
        print(f"\nGetting dynamic grid prices for {first_hot_coin_backtest} (Backtest)...")
        lower_backtest, upper_backtest = get_dynamic_grid_prices(first_hot_coin_backtest, historical_data=dummy_historical_data)
        if lower_backtest and upper_backtest:
            print(f"Dynamic Grid Range for {first_hot_coin_backtest} (Backtest): Lower={lower_backtest:.2f}, Upper={upper_backtest:.2f}")
    else:
        print("Could not find any hot coins (Backtest) to get dynamic grid prices for.")

    # 시장 분류 예시 (Backtest)
    print("\nClassifying market for BTC/KRW (Backtest)...")
    market_type_backtest = classify_market('BTC/KRW', historical_data=dummy_historical_data)
    print(f"BTC/KRW Market Type (Backtest): {market_type_backtest}")
