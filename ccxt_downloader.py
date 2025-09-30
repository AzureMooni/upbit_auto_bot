import ccxt
import pandas as pd
from datetime import datetime, timedelta
import os
import time
import argparse # argparse 추가
from dl_model_trainer import DLModelTrainer

def download_ohlcv_data(start_date_str: str, end_date_str: str, tickers: list = None, timeframe: str = '1h', limit: int = 200):
    if tickers is None:
        tickers = DLModelTrainer.TARGET_COINS
    
    exchange = ccxt.upbit()
    data_dir = 'data'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    end_dt = datetime.strptime(end_date_str, '%Y-%m-%d')

    for ticker in tickers:
        print(f"Downloading {ticker} {timeframe} data from {start_date_str} to {end_date_str}...")
        all_ohlcv = []
        since_timestamp = exchange.parse8601(start_date_str + 'T00:00:00Z')

        while since_timestamp < exchange.parse8601(end_date_str + 'T00:00:00Z'):
            try:
                ohlcv = exchange.fetch_ohlcv(ticker, timeframe, since=since_timestamp, limit=limit)
                if not ohlcv:
                    break
                
                all_ohlcv.extend(ohlcv)
                since_timestamp = ohlcv[-1][0] + (3600000) # 1h
                last_data_dt = datetime.fromtimestamp(ohlcv[-1][0] / 1000)
                print(f"  Fetched {len(ohlcv)} data points for {ticker}. Last timestamp: {last_data_dt}")
                time.sleep(0.5)

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
            df = df[~df.index.duplicated(keep='first')]
            df = df[df.index <= end_dt + timedelta(days=1, microseconds=-1)]
            
            filename = ticker.replace('/', '_') + f'_{timeframe}.csv'
            filepath = os.path.join(data_dir, filename)
            df.to_csv(filepath)
            print(f"Successfully saved {len(df)} data points for {ticker} to {filepath}")
        else:
            print(f"No data downloaded for {ticker}.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Download OHLCV data from Upbit.")
    parser.add_argument("--start-date", type=str, required=True, help="Start date in YYYY-MM-DD format.")
    parser.add_argument("--end-date", type=str, required=True, help="End date in YYYY-MM-DD format.")
    parser.add_argument("--tickers", nargs='+', help="List of tickers to download (e.g., BTC/KRW ETH/KRW). Defaults to TARGET_COINS.")
    args = parser.parse_args()

    download_ohlcv_data(args.start_date, args.end_date, tickers=args.tickers)