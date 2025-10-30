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
        self, ticker="BTC/KRW", interval="1h", since=None, limit=3000 # Target at least 3000 data points
    ) -> Union[pd.DataFrame, None]:
        if not self.exchange.has['fetchOHLCV']:
            print(f"  [ccxt] {self.exchange.id} does not support fetchOHLCV.")
            return None

        print(f"  [ccxt] Downloading {ticker} {interval} data...")
        try:
            all_ohlcvs = []
            now = self.exchange.milliseconds()
            since_timestamp = self.exchange.parse8601(since) if since else now - (limit * 3600000) # Default to fetching roughly `limit` hours ago

            while len(all_ohlcvs) < limit:
                ohlcv_segment = self.exchange.fetch_ohlcv(ticker, interval, since=since_timestamp, limit=1000) # Fetch in chunks
                if ohlcv_segment is None or len(ohlcv_segment) == 0:
                    break
                
                all_ohlcvs.extend(ohlcv_segment)
                since_timestamp = ohlcv_segment[-1][0] + (self.exchange.parse_timeframe(interval) * 1000)
                print(f"  [ccxt] Fetched {len(ohlcv_segment)} records. Total: {len(all_ohlcvs)}")

            if not all_ohlcvs:
                print(f"  [ccxt] No data returned for {ticker}.")
                return None

            df = pd.DataFrame(all_ohlcvs, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df.drop_duplicates(subset=['timestamp'], keep='first', inplace=True)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)
            
            print(f"  [ccxt] Successfully downloaded and de-duplicated {len(df)} data points for {ticker}.")
            return df

        except Exception as e:
            print(f"  [ccxt] Error downloading {ticker} data: {e}")
            return None
