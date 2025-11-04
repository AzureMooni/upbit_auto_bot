import pandas as pd
import os
import pickle
import numpy as np
from market_regime_detector import precompute_all_indicators, get_market_regime
from strategies.trend_follower import generate_v_recovery_signals
from strategies.mean_reversion_strategy import generate_sideways_signals
from ccxt_downloader import CCXTDataDownloader

class DataPreprocessor:
    def __init__(self, target_coins=None, interval="1h"):
        self.target_coins = target_coins if target_coins is not None else [
            "BTC/KRW", "ETH/KRW", "XRP/KRW", "SOL/KRW", "DOGE/KRW",
            "AVAX/KRW", "ADA/KRW", "LINK/KRW", "ETC/KRW"
        ]
        self.interval = interval
        self.cache_dir = "cache"
        os.makedirs(self.cache_dir, exist_ok=True)
        self.data_downloader = CCXTDataDownloader()

    def _preprocess_single_ticker(self, ticker: str) -> pd.DataFrame | None:
        print(f"[{ticker}] 데이터 로딩...")
        file_path = os.path.join(self.cache_dir, f"{ticker.replace('/', '_')}_{self.interval}.feather") # Changed to replace '/' with '_'
        
        if not os.path.exists(file_path):
            print(f"[{ticker}] 캐시 파일 없음. 다운로드 시작...")
            df = self.data_downloader.download_ohlcv(ticker, self.interval)
            if df is None or df.empty:
                print(f"[ERROR] {ticker} 데이터를 다운로드할 수 없습니다.")
                return None
            df.reset_index().to_feather(file_path)
        else:
            print(f"[{ticker}] 캐시에서 데이터 로드.")
            df = pd.read_feather(file_path)
            df.set_index("timestamp", inplace=True)

        print(f"[{ticker}] 지표 및 시장 체제 계산...")
        df_processed = precompute_all_indicators(df)
        df_processed = generate_v_recovery_signals(df_processed)
        df_processed = generate_sideways_signals(df_processed)
        
        # Standardized regime detection
        df_processed = get_market_regime_dataframe(df_processed)
        
        regime_map = {name: i for i, name in enumerate(df_processed['market_regime'].dropna().unique())}
        df_processed['regime'] = df_processed['market_regime'].map(regime_map)
        
        final_features = [
            'open', 'high', 'low', 'close', 'volume',
            'ADX_14', 'NATR_14', 'BBP_20_2.0', 'EMA_20', 'EMA_50',
            'RSI_14', 'MACDh_12_26_9', 'regime'
        ]
        
        missing_cols = [col for col in final_features if col not in df_processed.columns]
        if missing_cols:
            print(f"[WARN] {ticker}에서 누락된 피처: {missing_cols}. 이 티커를 건너뜁니다.")
            return None
            
        df_final = df_processed[final_features].dropna()
        
        print(f"[{ticker}] 전처리 완료. {len(df_final)}개 데이터 반환.")
        return df_final

    def run_and_save_to_pickle(self, save_path):
        print("모든 타겟 코인 데이터 전처리 시작...")
        all_data = {}
        for ticker in self.target_coins:
            df = self._preprocess_single_ticker(ticker)
            if df is not None and not df.empty:
                all_data[ticker] = df
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(all_data, f)
        print(f"모든 코인 데이터가 {save_path}에 저장되었습니다.")
        return all_data