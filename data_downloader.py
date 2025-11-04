
import ccxt
import pandas as pd
import os
from datetime import datetime
import time

from constants import SCALPING_TARGET_COINS

# --- Configuration ---
DATA_DIR = "data"
UNIVERSE = [f"KRW-{ticker.split('/')[0]}" for ticker in SCALPING_TARGET_COINS]
START_DATE = "2020-01-01T00:00:00Z"
TIMEFRAME = '1h' # 1-hour candles

def download_historical_data():
    """
    Downloads historical OHLCV data for the specified universe and saves it to Parquet files.
    """
    print("[INFO] Starting historical data download...")
    os.makedirs(DATA_DIR, exist_ok=True)
    
    upbit = ccxt.upbit() # No authentication needed for public data
    
    for ticker in UNIVERSE:
        try:
            print(f"  - Fetching data for {ticker}...")
            symbol = ticker.replace("KRW-", "") + "/KRW"
            all_ohlcv = []
            
            since = upbit.parse8601(START_DATE)
            end_timestamp = upbit.milliseconds()

            while since < end_timestamp:
                ohlcv = upbit.fetch_ohlcv(symbol, timeframe=TIMEFRAME, since=since)
                if len(ohlcv):
                    since = ohlcv[-1][0] + (upbit.parse_timeframe(TIMEFRAME) * 1000)
                    all_ohlcv.extend(ohlcv)
                    print(f"    Fetched {len(ohlcv)} candles, last timestamp: {upbit.iso8601(since)}")
                else:
                    break
                time.sleep(0.2) # Respect API rate limits

            if not all_ohlcv:
                print(f"  - No data found for {ticker}.")
                continue

            # Convert to DataFrame and save as Parquet
            df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            file_path = os.path.join(DATA_DIR, f"{ticker}.parquet")
            df.to_parquet(file_path)
            print(f"[SUCCESS] Saved {len(df)} data points for {ticker} to {file_path}")

        except Exception as e:
            print(f"[ERROR] Failed to download data for {ticker}: {e}")

if __name__ == "__main__":
    import time
    download_historical_data()
