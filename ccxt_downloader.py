import ccxt
import pandas as pd
from datetime import datetime, timedelta
import os
import time
from dl_model_trainer import DLModelTrainer # Changed to DLModelTrainer

def download_ohlcv_data(start_date_str: str, end_date_str: str, tickers: list = None, timeframe: str = '1h', limit: int = 200):
    """
    지정된 기간 동안 각 코인의 OHLCV 데이터를 다운로드하여 CSV 파일로 저장합니다.
    """
    if tickers is None:
        tickers = DLModelTrainer.TARGET_COINS # Use DLModelTrainer.TARGET_COINS if not specified
    
    """
    지정된 기간 동안 각 코인의 OHLCV 데이터를 다운로드하여 CSV 파일로 저장합니다.
    """
    exchange = ccxt.upbit()
    data_dir = 'data'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    start_dt = datetime.strptime(start_date_str, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date_str, '%Y-%m-%d')

    for ticker in tickers:
        print(f"Downloading {ticker} {timeframe} data from {start_date_str} to {end_date_str}...")
        all_ohlcv = []
        
        # ccxt는 UTC 타임스탬프를 사용하므로, 시작 날짜를 UTC로 변환
        # since_timestamp = exchange.parse8601(start_date_str + 'T00:00:00Z') # Removed to fetch from earliest
        since_timestamp = None # Start from the earliest available data

        while True:
            try:
                ohlcv = exchange.fetch_ohlcv(ticker, timeframe, since=since_timestamp, limit=limit)
                if not ohlcv:
                    break # 더 이상 데이터가 없으면 중단
                
                all_ohlcv.extend(ohlcv)
                
                # 다음 요청을 위한 since_timestamp 업데이트
                # 마지막 데이터 포인트의 타임스탬프 + 1 (중복 방지)
                since_timestamp = ohlcv[-1][0] + 1 
                
                # 현재 다운로드된 데이터의 마지막 날짜가 종료 날짜를 초과하면 중단
                last_data_dt = datetime.fromtimestamp(ohlcv[-1][0] / 1000)
                if last_data_dt > end_dt:
                    break

                print(f"  Fetched {len(ohlcv)} data points for {ticker}. Last timestamp: {last_data_dt}")
                time.sleep(0.5) # Rate limit 방지를 위한 딜레이 (0.5초)

            except ccxt.RateLimitExceeded:
                print("  Rate limit exceeded. Waiting for 30 seconds...")
                time.sleep(30)
            except Exception as e:
                print(f"  Error downloading {ticker} data: {e}")
                break
        
        if all_ohlcv:
            df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # 지정된 종료 날짜까지의 데이터만 포함
            df = df[df.index <= end_dt + timedelta(days=1, microseconds=-1)] # 종료일의 마지막 시간까지 포함
            
            filename = ticker.replace('/', '_') + f'_{timeframe}.csv'
            filepath = os.path.join(data_dir, filename)
            df.to_csv(filepath)
            print(f"Successfully saved {len(df)} data points for {ticker} to {filepath}")
        else:
            print(f"No data downloaded for {ticker}.")

if __name__ == '__main__':
    # 예시 사용법
    start_date = '2023-01-01'
    end_date = '2023-01-31'
    # target_coins = ["BTC/KRW", "ETH/KRW", "XRP/KRW"] # 이제 ModelTrainer.TARGET_COINS를 기본으로 사용
    download_ohlcv_data(start_date, end_date) # tickers 인자 없이 호출
