import ccxt
import pandas as pd
import time
import pandas_ta as ta

def find_hot_coin(historical_data: dict, ema_short_period: int = 20, ema_long_period: int = 60):
    """
    업비트의 모든 KRW 마켓 정보를 가져와서 거래대금 및 변동성 조건을 만족하는
    가장 거래대금이 높은 코인 1개의 티커를 반환합니다.
    historical_data 딕셔너리에서 해당 티커의 과거 데이터를 사용하여 계산합니다.
    """
    hot_coins = []

    # Backtesting mode: use provided historical data
    for symbol, df_1h in historical_data.items():
        if not symbol.endswith('/KRW'):
            continue

        if len(df_1h) < 240: # Need enough 1-hour data for 60 4-hour EMA and 14-period RSI
            continue
        
        df_24h = df_1h.iloc[-24:]

        volume_24h = (df_24h['volume'] * df_24h['close']).sum()
        if df_24h['open'].iloc[0] == 0:
            continue
        volatility_24h = ((df_24h['close'].iloc[-1] - df_24h['open'].iloc[0]) / df_24h['open'].iloc[0]) * 100
        
        if volume_24h >= 5_000_000_000 and abs(volatility_24h) >= 3:
            # Resample 1-hour data to 4-hour data for EMA calculation
            df_4h = df_1h['close'].resample('4h').ohlc().dropna()
            ema_short_values = ta.ema(close=df_4h['close'], length=ema_short_period)
            ema_long_values = ta.ema(close=df_4h['close'], length=ema_long_period)
            df_4h[f'EMA_{ema_short_period}'] = ema_short_values
            df_4h[f'EMA_{ema_long_period}'] = ema_long_values
            
            latest_ema_short_4h = df_4h[f'EMA_{ema_short_period}'].iloc[-1]
            latest_ema_long_4h = df_4h[f'EMA_{ema_long_period}'].iloc[-1]

            if pd.isna(latest_ema_short_4h) or pd.isna(latest_ema_long_4h):
                continue

            # RSI calculation remains on 1-hour data
            df_1h.ta.rsi(length=14, append=True, close='close')
            latest_rsi_1h = df_1h['RSI_14'].iloc[-1]

            if latest_ema_short_4h > latest_ema_long_4h and latest_rsi_1h < 70:
                hot_coins.append({
                    'symbol': symbol,
                    'volume_24h': volume_24h,
                    'volatility_24h': volatility_24h,
                    f'ema{ema_short_period}': latest_ema_short_4h,
                    f'ema{ema_long_period}': latest_ema_long_4h,
                    'rsi': latest_rsi_1h
                })
    hot_coins.sort(key=lambda x: x['volume_24h'], reverse=True)
    
    if hot_coins:
        print(f"Found hot coin: {hot_coins[0]['symbol']} (Volume: {hot_coins[0]['volume_24h']:.0f} KRW, Volatility: {hot_coins[0]['volatility_24h']:.2f}%)")
        return hot_coins[0]['symbol']
    else:
        print("No hot coins found matching the criteria.")
        return None

def find_hot_coin_live(upbit_exchange: ccxt.Exchange, ema_short_period: int = 30, ema_long_period: int = 100):
    """
    ccxt 거래소 객체를 사용하여 실시간으로 핫 코인을 찾습니다.
    """
    hot_coins = []
    try:
        tickers = upbit_exchange.fetch_tickers()
        krw_markets = {symbol: data for symbol, data in tickers.items() if symbol.endswith('/KRW')}

        for symbol, data in krw_markets.items():
            # 24시간 거래대금 (volume * last price)
            volume_24h = data['quoteVolume'] # Upbit's quoteVolume is KRW volume
            
            # 24시간 변동성
            if data['open'] is None or data['open'] == 0:
                continue
            volatility_24h = ((data['last'] - data['open']) / data['open']) * 100

            if volume_24h >= 5_000_000_000 and abs(volatility_24h) >= 3:
                # Fetch 4-hour OHLCV data for EMA calculation
                ohlcv_4h = upbit_exchange.fetch_ohlcv(symbol, '4h', limit=ema_long_period + 10) # Adjusted limit for safety
                if not ohlcv_4h or len(ohlcv_4h) < ema_long_period:
                    continue
                df_4h = pd.DataFrame(ohlcv_4h, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df_4h['timestamp'] = pd.to_datetime(df_4h['timestamp'], unit='ms')
                df_4h.set_index('timestamp', inplace=True)

                ema_short_values = ta.ema(close=df_4h['close'], length=ema_short_period)
                ema_long_values = ta.ema(close=df_4h['close'], length=ema_long_period)

                df_4h[f'EMA_{ema_short_period}'] = ema_short_values
                df_4h[f'EMA_{ema_long_period}'] = ema_long_values
                
                latest_ema_short_4h = df_4h[f'EMA_{ema_short_period}'].iloc[-1]
                latest_ema_long_4h = df_4h[f'EMA_{ema_long_period}'].iloc[-1]

                if pd.isna(latest_ema_short_4h) or pd.isna(latest_ema_long_4h):
                    continue

                # Fetch 1-hour OHLCV data for RSI calculation
                ohlcv_1h = upbit_exchange.fetch_ohlcv(symbol, '1h', limit=100) # Need enough for RSI14
                if not ohlcv_1h or len(ohlcv_1h) < 14: # RSI needs at least 14 periods
                    continue
                df_1h = pd.DataFrame(ohlcv_1h, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df_1h['timestamp'] = pd.to_datetime(df_1h['timestamp'], unit='ms')
                df_1h.set_index('timestamp', inplace=True)

                df_1h.ta.rsi(length=14, append=True, close='close')
                latest_rsi_1h = df_1h['RSI_14'].iloc[-1]

                if latest_ema_short_4h > latest_ema_long_4h and latest_rsi_1h < 70:
                    hot_coins.append({
                        'symbol': symbol,
                        'volume_24h': volume_24h,
                        'volatility_24h': volatility_24h,
                        f'ema{ema_short_period}': latest_ema_short_4h,
                        f'ema{ema_long_period}': latest_ema_long_4h,
                        'rsi': latest_rsi_1h
                    })
    except Exception as e:
        print(f"Error in find_hot_coin_live: {e}")
        return None

    hot_coins.sort(key=lambda x: x['volume_24h'], reverse=True)
    
    if hot_coins:
        print(f"Found hot coin (Live): {hot_coins[0]['symbol']} (Volume: {hot_coins[0]['volume_24h']:.0f} KRW, Volatility: {hot_coins[0]['volatility_24h']:.2f}%)")
        return hot_coins[0]['symbol']
    else:
        print("No hot coins found matching the criteria (Live).")
        return None

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
    1시간 봉 데이터 기준으로 14기간 ADX(Average Directional Index)를 계산하여
    'trending' (추세장), 'ranging' (횡보장), 'choppy' (혼조세)를 반환합니다.
    historical_data 딕셔너리에서 해당 티커의 과거 데이터를 사용하여 계산합니다.
    """
    df = None
    if ticker in historical_data:
        df = historical_data[ticker]

    if df is None or df.empty or len(df) < 100: # ADX 계산을 위해 최소 14개 데이터 필요, 100개로 가정
        print(f"Not enough OHLCV data for {ticker} to calculate ADX. (Need at least 100, got {len(df) if df is not None else 0})")
        return "unknown"

    try:
        df.ta.adx(length=14, append=True, high='high', low='low', close='close')
        adx_values = df['ADX_14']

        if len(adx_values) == 0:
            return "unknown"

        latest_adx = adx_values[-1]

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
    """
    try:
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

        latest_adx = adx_values[-1]

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
    print("\nFinding hot coin (Live)...")
    hot_coin_live = find_hot_coin_live(live_exchange)
    if hot_coin_live:
        print(f"The hottest coin (Live) is: {hot_coin_live}")
        
        # 핫 코인의 동적 그리드 가격 가져오기 예시 (Live)
        print(f"\nGetting dynamic grid prices for {hot_coin_live} (Live)...")
        lower_live, upper_live = get_dynamic_grid_prices_live(hot_coin_live, live_exchange)
        if lower_live and upper_live:
            print(f"Dynamic Grid Range for {hot_coin_live} (Live): Lower={lower_live:.2f}, Upper={upper_live:.2f}")
    else:
        print("Could not find a hot coin (Live) to get dynamic grid prices for.")

    # 시장 분류 예시 (Live)
    print("\nClassifying market for BTC/KRW (Live)...")
    market_type_live = classify_market_live('BTC/KRW', live_exchange)
    print(f"BTC/KRW Market Type (Live): {market_type_live}")

    print("\n--- Backtesting mode testing with historical_data ---")

    # 핫 코인 찾기 예시 (Backtest)
    print("\nFinding hot coin (Backtest)...")
    hot_coin_backtest = find_hot_coin(historical_data=dummy_historical_data)
    if hot_coin_backtest:
        print(f"""The hottest coin (Backtest) is: {hot_coin_backtest}\n""")
        
        # 핫 코인의 동적 그리드 가격 가져오기 예시 (Backtest)
        print(f"\nGetting dynamic grid prices for {hot_coin_backtest} (Backtest)...")
        lower_backtest, upper_backtest = get_dynamic_grid_prices(hot_coin_backtest, historical_data=dummy_historical_data)
        if lower_backtest and upper_backtest:
            print(f"Dynamic Grid Range for {hot_coin_backtest} (Backtest): Lower={lower_backtest:.2f}, Upper={upper_backtest:.2f}")
    else:
        print("Could not find a hot coin (Backtest) to get dynamic grid prices for.")

    # 시장 분류 예시 (Backtest)
    print("\nClassifying market for BTC/KRW (Backtest)...")
    market_type_backtest = classify_market('BTC/KRW', historical_data=dummy_historical_data)
    print(f"BTC/KRW Market Type (Backtest): {market_type_backtest}")
