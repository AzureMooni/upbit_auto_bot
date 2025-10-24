import pandas as pd
import os
import pickle
from market_regime_detector import precompute_all_indicators, get_market_regime
from strategies.trend_follower import generate_v_recovery_signals
from strategies.mean_reversion_strategy import generate_sideways_signals
from ccxt_downloader import CCXTDataDownloader

class DataPreprocessor:
    def __init__(self, target_coins=None, interval="1h"):
        self.target_coins = target_coins if target_coins is not None else ["BTC/KRW", "ETH/KRW", "SOL/KRW", "XRP/KRW", "DOGE/KRW", "AVAX/KRW", "LINK/KRW"]
        self.interval = interval
        # self.data_downloader = CCXTDataDownloader() # Removed from here
        self.cache_dir = "cache"
        os.makedirs(self.cache_dir, exist_ok=True)

    def _preprocess_single_ticker(self, ticker: str) -> pd.DataFrame | None:
        """
        RL 훈련을 위해 모든 지표와 시장 체제를 계산하여 전처리된 데이터를 생성합니다.
        """
        print(f"데이터를 로드합니다: {ticker}")
        file_path = os.path.join(self.cache_dir, f"{ticker.replace('/', '_')}_{self.interval}.feather")

        # Check if feather file exists, if not, download and save it
        if not os.path.exists(file_path):
            print(f"캐시 파일이 없습니다. {ticker} 데이터를 다운로드합니다.")
            # Create downloader instance only when needed
            data_downloader = CCXTDataDownloader()
            df = data_downloader.download_ohlcv(ticker, self.interval)
            if df is None or df.empty:
                print(f"오류: {ticker} 데이터를 다운로드할 수 없습니다.")
                return None
            df.reset_index().to_feather(file_path)
        else:
            df = pd.read_feather(file_path)
            df.set_index("timestamp", inplace=True)

        print(f"{ticker} 모든 기술적 지표와 시장 체제를 계산합니다...")
        # 1. 시장 체제 분석에 필요한 모든 지표 계산
        df_processed = precompute_all_indicators(df)

        # 2. V-회복 및 횡보장 전략 신호에 필요한 지표 추가 계산
        df_processed = generate_v_recovery_signals(df_processed)
        df_processed = generate_sideways_signals(df_processed)

        # 3. 시장 체제 자체를 피처로 추가 (숫자로 변환)
        daily_regime = df_processed.apply(get_market_regime, axis=1).rename('regime')
        df_processed['regime'] = daily_regime.reindex(df_processed.index.date).set_axis(df_processed.index)
        df_processed['regime'] = df_processed['regime'].ffill()

        regime_map = {name: i for i, name in enumerate(df_processed['regime'].unique())}
        df_processed['regime'] = df_processed['regime'].map(regime_map)

        # 4. RL 환경에 필요한 최종 피처 선택
        final_features = [
            'open', 'high', 'low', 'close', 'volume',
            'ADX', 'Normalized_ATR', 'BBP', 'EMA_20', 'EMA_50',
            'RSI_14', 'MACD_hist', 'regime'
        ]
        df_final = df_processed[final_features].dropna()

        print(f"{ticker} 전처리가 완료되었습니다. {len(df_final)}개의 데이터 포인트를 반환합니다.")
        return df_final

    def run(self):
        """
        모든 타겟 코인에 대해 전처리를 실행하고, 결과를 캐시합니다.
        """
        print("모든 타겟 코인 데이터 전처리 시작...")
        all_data = {}
        for ticker in self.target_coins:
            df = self._preprocess_single_ticker(ticker)
            if df is not None:
                all_data[ticker] = df
        
        # 모든 코인의 전처리된 데이터를 하나의 pickle 파일로 저장 (foundational model용)
        all_coins_cache_path = os.path.join(self.cache_dir, "preprocessed_data.pkl")
        with open(all_coins_cache_path, "wb") as f:
            pickle.dump(all_data, f)
        print(f"모든 코인 데이터가 {all_coins_cache_path}에 저장되었습니다.")
