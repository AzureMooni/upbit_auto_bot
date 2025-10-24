import ccxt
import pandas as pd
from datetime import datetime, timedelta
import os
import time
import argparse

# 고빈도 스캘핑을 위한 타겟 코인 목록
SCALPING_TARGET_COINS = ["BTC/KRW", "ETH/KRW", "XRP/KRW", "SOL/KRW", "DOGE/KRW"]

class CCXTDataDownloader:
    def __init__(self, limit: int = 200):
        self.exchange = ccxt.upbit()
        self.data_dir = "data"
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        self.limit = limit

    def download_ohlcv(
        self,
        ticker: str,
        timeframe: str,
        start_date_str: str = None,
        end_date_str: str = None,
    ) -> pd.DataFrame | None:
        """
        Downloads OHLCV data for a given ticker and timeframe.
        If start_date_str and end_date_str are not provided, it tries to download all available data.
        """
        if start_date_str is None:
            start_date_str = "2017-01-01" # Default to a very early date if not provided
        if end_date_str is None:
            end_date_str = datetime.now().strftime("%Y-%m-%d")

        print(
            f"Downloading {ticker} {timeframe} data from {start_date_str} to {end_date_str}..."
        )

        filename = ticker.replace("/", "_") + f"_{timeframe}.feather" # Changed to feather for performance
        filepath = os.path.join(self.data_dir, filename)

        since_timestamp = self.exchange.parse8601(start_date_str + "T00:00:00Z")
        timeframe_duration_ms = self.exchange.parse_timeframe(timeframe) * 1000

        all_ohlcv = []

        # Load existing data if available
        existing_df = pd.DataFrame()
        if os.path.exists(filepath):
            try:
                existing_df = pd.read_feather(filepath)
                existing_df["timestamp"] = pd.to_datetime(existing_df["timestamp"])
                existing_df.set_index("timestamp", inplace=True)
                if not existing_df.empty:
                    since_timestamp = (
                        int(existing_df.index[-1].timestamp() * 1000)
                        + timeframe_duration_ms
                    )
                    print(
                        f"  Existing data found. Last timestamp: {existing_df.index[-1]}. Resuming download..."
                    )
            except Exception as e:
                print(
                    f"  Error reading existing file({filepath}): {e}. Downloading from scratch."
                )

        end_dt = datetime.strptime(end_date_str, "%Y-%m-%d")

        while since_timestamp < self.exchange.parse8601(end_date_str + "T00:00:00Z"):
            try:
                ohlcv = self.exchange.fetch_ohlcv(
                    ticker, timeframe, since=since_timestamp, limit=self.limit
                )
                if not ohlcv:
                    break

                all_ohlcv.extend(ohlcv)
                since_timestamp = ohlcv[-1][0] + timeframe_duration_ms
                last_data_dt = datetime.fromtimestamp(ohlcv[-1][0] / 1000)
                print(
                    f"  Fetched {len(ohlcv)} data points for {ticker}. Last timestamp: {last_data_dt}"
                )
                time.sleep(0.5)  # Upbit API rate limit

            except ccxt.RateLimitExceeded:
                print("  Rate limit exceeded. Waiting for 30 seconds...")
                time.sleep(30)
            except Exception as e:
                print(f"  Error downloading {ticker} data: {e}")
                break

        if all_ohlcv:
            new_df = pd.DataFrame(
                all_ohlcv,
                columns=["timestamp", "open", "high", "low", "close", "volume"],
            )
            new_df["timestamp"] = pd.to_datetime(new_df["timestamp"], unit="ms")
            new_df.set_index("timestamp", inplace=True)
            new_df = new_df[~new_df.index.duplicated(keep="first")]
            new_df = new_df[new_df.index <= end_dt + timedelta(days=1, microseconds=-1)]

            if not existing_df.empty:
                df = pd.concat([existing_df, new_df])
                df = df[~df.index.duplicated(keep="last")]
                df.sort_index(inplace=True)
                print("  Merged with existing data.")
            else:
                df = new_df

            df.reset_index().to_feather(filepath) # Save as feather
            print(
                f"Successfully saved/updated {ticker} data to {filepath}. Total {len(df)} data points."
            )
            return df
        else:
            print(f"No new data downloaded for {ticker}.")
            return existing_df if not existing_df.empty else None # Return existing data if no new data