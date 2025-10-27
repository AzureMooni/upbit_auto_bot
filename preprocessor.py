import pandas as pd
import os
import pickle
from market_regime_detector import precompute_all_indicators, get_market_regime
from strategies.trend_follower import generate_v_recovery_signals
from strategies.mean_reversion_strategy import generate_sideways_signals
from ccxt_downloader import CCXTDataDownloader

class DataPreprocessor:
    def __init__(self, target_coins=None, interval="1h"):
        self.target_coins = target_coins if target_coins is not None else [
            "KRW-BTC", "KRW-ETH", "KRW-SOL", "KRW-XRP", "KRW-DOGE",
            "KRW-AVAX", "KRW-LINK", "KRW-ADA", "KRW-ETC", "KRW-LTC"
        ]
        self.interval = interval
        self.cache_dir = "cache"
        os.makedirs(self.cache_dir, exist_ok=True)
        self.data_downloader = CCXTDataDownloader()

    def _preprocess_single_ticker(self, ticker: str) -> pd.DataFrame | None:
        print(f"  [Preprocessor] Processing ticker: {ticker}")
        
        # 1. Download data using the new pyupbit-based downloader
        df = self.data_downloader.download_ohlcv(ticker, self.interval)
        if df is None or df.empty:
            print(f"  [Preprocessor] Failed to get data for {ticker}. Skipping.")
            return None

        # 2. Calculate all indicators manually for stability
        print(f"  [Preprocessor] Calculating indicators for {ticker}...")
        df_processed = precompute_all_indicators(df)
        
        # 3. Add market regime as a feature
        daily_regime = df_processed.apply(get_market_regime, axis=1).rename('regime')
        df_processed['regime'] = daily_regime.reindex(df_processed.index, method='ffill')
        
        # Map regime strings to integers
        regime_map = {'BEARISH': 0, 'NEUTRAL': 1} # Simplified regimes
        df_processed['regime'] = df_processed['regime'].map(regime_map).fillna(1) # Fill NaNs with Neutral

        # 4. Define final features based on what's reliably calculated
        final_features = [
            'open', 'high', 'low', 'close', 'volume',
            'SMA_50', 'SMA_200', 'RSI_14', 'MACDh_12_26_9', 'BBP_20_2.0', 'ATRr_14', 'regime'
        ]
        
        # Use only the features that were successfully calculated
        available_features = [col for col in final_features if col in df_processed.columns]
        
        df_final = df_processed[available_features].dropna()
        
        if df_final.empty:
            print(f"  [Preprocessor] No data left for {ticker} after processing. Skipping.")
            return None

        print(f"  [Preprocessor] Preprocessing complete for {ticker}. {len(df_final)} rows returned.")
        return df_final

    def run_and_save_to_pickle(self, save_path):
        """
        모든 타겟 코인에 대해 전처리를 실행하고, 결과를 하나의 피클 파일로 저장합니다.
        """
        print("모든 타겟 코인 데이터 전처리 시작...")
        all_data = {}
        for ticker in self.target_coins:
            df = self._preprocess_single_ticker(ticker)
            if df is not None:
                all_data[ticker] = df
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(all_data, f)
        print(f"모든 코인 데이터가 {save_path}에 저장되었습니다.")
        return all_data
