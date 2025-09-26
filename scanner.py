import ccxt
import pandas as pd
import time

def find_hot_coin():
    """
    업비트의 모든 KRW 마켓 정보를 가져와서 거래대금 및 변동성 조건을 만족하는
    가장 거래대금이 높은 코인 1개의 티커를 반환합니다.
    """
    exchange = ccxt.upbit()
    
    try:
        # 모든 마켓 정보 가져오기
        markets = exchange.load_markets()
        krw_markets = [symbol for symbol in markets if symbol.endswith('/KRW')]
        
        hot_coins = []
        for symbol in krw_markets:
            try:
                # 티커 정보 가져오기 (거래대금, 변동성 포함)
                ticker_data = exchange.fetch_ticker(symbol)
                
                # 24시간 거래대금 (volume * close price)
                # Upbit의 fetch_ticker는 'quoteVolume' (KRW 기준 거래대금)을 제공합니다.
                # 또는 'baseVolume' * 'close'로 계산할 수도 있습니다.
                # 여기서는 'quoteVolume'을 사용합니다.
                volume_24h = ticker_data.get('quoteVolume', 0) # KRW 기준 거래대금
                
                # 24시간 가격 변동률 (change / open) * 100
                # 'percentage'는 24시간 동안의 가격 변동률을 나타냅니다.
                volatility_24h = ticker_data.get('percentage', 0)
                
                # 조건 확인: 거래대금 100억 이상, 변동률 5% 이상
                if volume_24h >= 10_000_000_000 and abs(volatility_24h) >= 5:
                    # 1시간 봉 데이터 가져오기 (추세 및 과매수 필터용)
                    ohlcv_1h = exchange.fetch_ohlcv(symbol, '1h', limit=50) # 50 EMA 계산을 위해 최소 50개 필요
                    if not ohlcv_1h or len(ohlcv_1h) < 50:
                        # print(f"Not enough 1h OHLCV data for {symbol} for trend/RSI analysis.")
                        continue

                    df_1h = pd.DataFrame(ohlcv_1h, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    
                    # 추세 필터 (EMA)
                    df_1h['EMA20'] = df_1h['close'].ewm(span=20, adjust=False).mean()
                    df_1h['EMA50'] = df_1h['close'].ewm(span=50, adjust=False).mean()
                    
                    # RSI 계산
                    delta = df_1h['close'].diff()
                    gain = delta.where(delta > 0, 0)
                    loss = -delta.where(delta < 0, 0)
                    
                    avg_gain = gain.ewm(span=14, adjust=False).mean()
                    avg_loss = loss.ewm(span=14, adjust=False).mean()
                    
                    rs = avg_gain / avg_loss
                    df_1h['RSI'] = 100 - (100 / (1 + rs))

                    # 최신 데이터 기준으로 필터링
                    latest_ema20 = df_1h['EMA20'].iloc[-1]
                    latest_ema50 = df_1h['EMA50'].iloc[-1]
                    latest_rsi = df_1h['RSI'].iloc[-1]

                    # 추세 필터: 단기 이평선이 장기 이평선보다 위에 있고 (정배열), 과매수 방지 필터: RSI 70 미만
                    if latest_ema20 > latest_ema50 and latest_rsi < 70:
                        hot_coins.append({
                            'symbol': symbol,
                            'volume_24h': volume_24h,
                            'volatility_24h': volatility_24h,
                            'ema20': latest_ema20,
                            'ema50': latest_ema50,
                            'rsi': latest_rsi
                        })
                
            except Exception as e:
                # print(f"Error fetching ticker for {symbol}: {e}")
                continue
        
        # 거래대금 기준으로 내림차순 정렬
        hot_coins.sort(key=lambda x: x['volume_24h'], reverse=True)
        
        if hot_coins:
            print(f"Found hot coin: {hot_coins[0]['symbol']} (Volume: {hot_coins[0]['volume_24h']:.0f} KRW, Volatility: {hot_coins[0]['volatility_24h']:.2f}%)")
            return hot_coins[0]['symbol']
        else:
            print("No hot coins found matching the criteria.")
            return None

    except Exception as e:
        print(f"Error in find_hot_coin: {e}")
        return None

def get_dynamic_grid_prices(ticker: str):
    """
    코인 티커를 입력받아서 해당 코인의 최적 그리드 가격 범위를 볼린저 밴드를 이용해 계산합니다.
    상단 밴드 (SMA + 2 * 표준편차)를 upper_price로, 하단 밴드 (SMA - 2 * 표준편차)를 lower_price로 설정합니다.
    """
    exchange = ccxt.upbit()
    
    try:
        # 1시간 봉(ohlcv) 데이터를 최근 20개 가져오기
        # fetch_ohlcv(symbol, timeframe, since, limit)
        # since를 지정하지 않으면 가장 최근 데이터부터 limit 개수만큼 가져옵니다.
        ohlcv = exchange.fetch_ohlcv(ticker, '1h', limit=20)
        
        if not ohlcv or len(ohlcv) < 20:
            print(f"Not enough OHLCV data for {ticker} to calculate Bollinger Bands. (Need 20, got {len(ohlcv)})")
            return None, None
        
        # pandas DataFrame으로 변환
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # 종가(close)를 기준으로 볼린저 밴드 계산
        window = 20 # 20기간 이동평균선 및 표준편차
        
        # 이동평균선 (SMA)
        df['SMA'] = df['close'].rolling(window=window).mean()
        
        # 표준편차 (Standard Deviation)
        df['STD'] = df['close'].rolling(window=window).std()
        
        # 상단 밴드 (Upper Bollinger Band)
        df['Upper_Band'] = df['SMA'] + (df['STD'] * 2)
        
        # 하단 밴드 (Lower Bollinger Band)
        df['Lower_Band'] = df['SMA'] - (df['STD'] * 2)
        
        # 가장 최근 데이터의 볼린저 밴드 값 사용
        upper_price = df['Upper_Band'].iloc[-1]
        lower_price = df['Lower_Band'].iloc[-1]
        
        print(f"Dynamic Grid Prices for {ticker}: Lower Band = {lower_price:.2f}, Upper Band = {upper_price:.2f}")
        return lower_price, upper_price

    except Exception as e:
        print(f"Error in get_dynamic_grid_prices for {ticker}: {e}")
        return None, None

if __name__ == '__main__':
    print("--- Running scanner.py examples ---")

    # 핫 코인 찾기 예시
    print("\nFinding hot coin...")
    hot_coin = find_hot_coin()
    if hot_coin:
        print(f"The hottest coin is: {hot_coin}")
        
        # 핫 코인의 동적 그리드 가격 가져오기 예시
        print(f"\nGetting dynamic grid prices for {hot_coin}...")
        lower, upper = get_dynamic_grid_prices(hot_coin)
        if lower and upper:
            print(f"Dynamic Grid Range for {hot_coin}: Lower={lower:.2f}, Upper={upper:.2f}")
    else:
        print("Could not find a hot coin to get dynamic grid prices for.")

    # 특정 코인의 동적 그리드 가격 가져오기 예시 (BTC/KRW)
    print("\nGetting dynamic grid prices for BTC/KRW...")
    btc_lower, btc_upper = get_dynamic_grid_prices('BTC/KRW')
    if btc_lower and btc_upper:
        print(f"Dynamic Grid Range for BTC/KRW: Lower={btc_lower:.2f}, Upper={btc_upper:.2f}")
