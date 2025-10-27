import pyupbit
import pandas as pd
import time
import os

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
            
            # fetch_ohlcv_by_date can be more reliable
            df = pyupbit.get_ohlcv_by_date(ticker, interval=pyupbit_interval, to=end_date_str, count=self.limit * 10) # Fetch more data
            
            if df is None or df.empty:
                print(f"  [pyupbit] No data returned for {ticker}.")
                return None

            print(f"  [pyupbit] Successfully downloaded {len(df)} data points for {ticker}.")
            # pyupbit returns a dataframe with the correct index, so no need for timestamp conversion
            time.sleep(0.2) # Add a small delay to be respectful to the API
            return df

        except Exception as e:
            print(f"  [pyupbit] Error downloading {ticker} data: {e}")
            return None
