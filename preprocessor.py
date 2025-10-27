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
        print(f"데이터를 로드합니다: {ticker}")
        file_path = os.path.join(self.cache_dir, f"{ticker.replace('-', '_')}_{self.interval}.feather")
        
        if not os.path.exists(file_path):
            print(f"캐시 파일이 없습니다. {ticker} 데이터를 다운로드합니다.")
            df = self.data_downloader.download_ohlcv(ticker, self.interval)
            if df is None or df.empty:
                print(f"오류: {ticker} 데이터를 다운로드할 수 없습니다.")
                return None
            df.reset_index().to_feather(file_path)
        else:
            df = pd.read_feather(file_path)
            df.set_index("timestamp", inplace=True)

        print(f"{ticker} 모든 기술적 지표와 시장 체제를 계산합니다...")
        df_processed = precompute_all_indicators(df)
        df_processed = generate_v_recovery_signals(df_processed)
        df_processed = generate_sideways_signals(df_processed)
        
        daily_regime = df_processed.apply(get_market_regime, axis=1).rename('regime')
        df_processed['regime'] = daily_regime.reindex(df_processed.index.date).set_axis(df_processed.index)
        df_processed['regime'] = df_processed['regime'].ffill()
        
        regime_map = {name: i for i, name in enumerate(df_processed['regime'].unique())}
        df_processed['regime'] = df_processed['regime'].map(regime_map)
        
        final_features = [
            'open', 'high', 'low', 'close', 'volume',
            'ADX', 'Normalized_ATR', 'BBP', 'EMA_20', 'EMA_50',
            'RSI_14', 'MACD_hist', 'regime'
        ]
        
        # Check if all columns exist
        missing_cols = [col for col in final_features if col not in df_processed.columns]
        if missing_cols:
            print(f"[WARN] {ticker}에서 누락된 피처: {missing_cols}. 이 티커를 건너뜁니다.")
            return None
            
        df_final = df_processed[final_features].dropna()
        
        print(f"{ticker} 전처리가 완료되었습니다. {len(df_final)}개의 데이터 포인트를 반환합니다.")
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
