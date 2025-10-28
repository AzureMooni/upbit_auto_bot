import pyupbit
import pandas as pd
import time
import os
from datetime import datetime, timedelta

class CCXTDataDownloader:
    """ Uses pyupbit to download OHLCV data from Upbit. """
    def __init__(self, limit: int = 200):
        self.data_dir = "data"
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        self.limit = limit

    def download_ohlcv(
        self, ticker: str, timeframe: str, start_date_str: str = None, end_date_str: str = None
    ) -> pd.DataFrame | None:
        print(f"  [pyupbit] Downloading {ticker} {timeframe} data...")
        try:
            # pyupbit uses interval='minute60' for 1h timeframe
            interval_map = {'1h': 'minute60'}
            pyupbit_interval = interval_map.get(timeframe, timeframe)
            
            print(f"  [pyupbit] Calling get_ohlcv with ticker={ticker}, interval={pyupbit_interval}, count={self.limit * 10})")
            df = pyupbit.get_ohlcv(ticker, interval=pyupbit_interval, count=self.limit * 10) # Fetch a large count of recent data
            
            if df is None:
                print(f"  [pyupbit] get_ohlcv returned None for {ticker}.")
                return None
            if df.empty:
                print(f"  [pyupbit] get_ohlcv returned empty DataFrame for {ticker}.")
                return None

            # Filter by date range if provided
            if start_date_str:
                start_dt = datetime.strptime(start_date_str, "%Y-%m-%d")
                df = df[df.index >= start_dt]
            if end_date_str:
                end_dt = datetime.strptime(end_date_str, "%Y-%m-%d")
                df = df[df.index <= end_dt + timedelta(days=1, microseconds=-1)]

            print(f"  [pyupbit] Successfully downloaded {len(df)} data points for {ticker}.")
            # pyupbit returns a dataframe with the correct index, so no need for timestamp conversion
            time.sleep(0.2) # Add a small delay to be respectful to the API
            return df

        except Exception as e:
            print(f"  [pyupbit] Error downloading {ticker} data: {e}")
            return None
