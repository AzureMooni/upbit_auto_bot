import ccxt
import pandas as pd
from datetime import datetime, timedelta
import os
import time
import argparse

# 고빈도 스캘핑을 위한 타겟 코인 목록
SCALPING_TARGET_COINS = ["BTC/KRW", "ETH/KRW", "XRP/KRW", "SOL/KRW", "DOGE/KRW"]


def download_ohlcv_data(
    start_date_str: str,
    end_date_str: str,
    tickers: list = None,
    timeframe: str = "1m",
    limit: int = 200,
):
    if tickers is None:
        tickers = SCALPING_TARGET_COINS

    exchange = ccxt.upbit()
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    end_dt = datetime.strptime(end_date_str, "%Y-%m-%d")
    timeframe_duration_ms = exchange.parse_timeframe(timeframe) * 1000

    for ticker in tickers:
        print(
            f"Downloading {ticker} {timeframe} data from {start_date_str} to {end_date_str}..."
        )

        filename = ticker.replace("/", "_") + f"_{timeframe}.csv"
        filepath = os.path.join(data_dir, filename)

        since_timestamp = exchange.parse8601(start_date_str + "T00:00:00Z")
        if os.path.exists(filepath):
            try:
                existing_df = pd.read_csv(
                    filepath, index_col="timestamp", parse_dates=True
                )
                if not existing_df.empty:
                    since_timestamp = (
                        int(existing_df.index[-1].timestamp() * 1000)
                        + timeframe_duration_ms
                    )
                    print(
                        f"  기존 데이터 발견. 마지막 시간: {existing_df.index[-1]}. 다운로드를 이어갑니다..."
                    )
            except Exception as e:
                print(
                    f"  기존 파일({filepath})을 읽는 중 오류 발생: {e}. 처음부터 다시 다운로드합니다."
                )

        all_ohlcv = []

        while since_timestamp < exchange.parse8601(end_date_str + "T00:00:00Z"):
            try:
                ohlcv = exchange.fetch_ohlcv(
                    ticker, timeframe, since=since_timestamp, limit=limit
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
            df = pd.DataFrame(
                all_ohlcv,
                columns=["timestamp", "open", "high", "low", "close", "volume"],
            )
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)
            df = df[~df.index.duplicated(keep="first")]
            df = df[df.index <= end_dt + timedelta(days=1, microseconds=-1)]

            if os.path.exists(filepath):
                try:
                    existing_df = pd.read_csv(
                        filepath, index_col="timestamp", parse_dates=True
                    )
                    df = pd.concat([existing_df, df])
                    df = df[~df.index.duplicated(keep="last")]
                    df.sort_index(inplace=True)
                    print("  기존 데이터와 병합 완료.")
                except Exception as e:
                    print(
                        f"  기존 파일과 병합 중 오류 발생: {e}. 새 데이터만 저장합니다."
                    )

            df.to_csv(filepath)
            print(
                f"성공적으로 {ticker} 데이터를 {filepath}에 저장/업데이트했습니다. 총 {len(df)}개의 데이터."
            )
        else:
            print(f"No new data downloaded for {ticker}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download OHLCV data from Upbit for scalping."
    )
    parser.add_argument(
        "--start-date", type=str, required=True, help="Start date in YYYY-MM-DD format."
    )
    parser.add_argument(
        "--end-date", type=str, required=True, help="End date in YYYY-MM-DD format."
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        help=f"List of tickers to download. Defaults to {SCALPING_TARGET_COINS}",
    )
    parser.add_argument(
        "--timeframe",
        type=str,
        default="1m",
        help="Timeframe to download (e.g., 1m, 5m, 1h, 1d).",
    )
    args = parser.parse_args()

    download_ohlcv_data(
        args.start_date, args.end_date, tickers=args.tickers, timeframe=args.timeframe
    )
