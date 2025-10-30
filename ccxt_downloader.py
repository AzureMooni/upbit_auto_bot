from typing import Union

import ccxt
import pandas as pd
from datetime import datetime, timedelta


class CCXTDataDownloader:
    """
    ccxt 라이브러리를 사용하여 업비트의 OHLCV 데이터를 다운로드하는 클래스
    """

    def __init__(self):
        self.exchange = ccxt.upbit()

    def download_ohlcv(
        self, ticker="BTC/KRW", interval="1h", since=None, limit=1000
    ) -> Union[pd.DataFrame, None]:
        if self.exchange.has['fetchOHLCV'] == False:
            print(f"  [ccxt] {self.exchange.id} does not support fetchOHLCV.")
            return None

        print(f"  [ccxt] Downloading {ticker} {interval} data...")
        try:
            ohlcvs = []
            # If since is not provided, get the most recent data
            if since is None:
                ohlcvs = self.exchange.fetch_ohlcv(ticker, interval, limit=limit)
            else: # Fetch all data since the start date
                since_timestamp = self.exchange.parse8601(since)
                while since_timestamp < self.exchange.milliseconds():
                    limit = 1000 # max limit
                    ohlcv_segment = self.exchange.fetch_ohlcv(ticker, interval, since=since_timestamp, limit=limit)
                    if ohlcv_segment is None or len(ohlcv_segment) == 0:
                        break
                    ohlcvs.extend(ohlcv_segment)
                    since_timestamp = ohlcv_segment[-1][0] + 1

            if not ohlcvs:
                print(f"  [ccxt] No data returned for {ticker}.")
                return None

            df = pd.DataFrame(ohlcvs, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            print(f"  [ccxt] Successfully downloaded {len(df)} data points for {ticker}.")
            return df

        except Exception as e:
            print(f"  [ccxt] Error downloading {ticker} data: {e}")
            return None
