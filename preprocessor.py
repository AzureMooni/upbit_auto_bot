import pandas as pd
import pandas_ta as ta
import os
from dl_model_trainer import DLModelTrainer

class DataPreprocessor:
    """
    데이터 처리의 중앙 허브.
    원본 CSV를 읽어 기술적 지표, 시장 체제(Regime)를 모두 계산하고,
    빠른 입출력을 위해 Feather 형식으로 캐시 파일을 생성합니다.
    """
    def __init__(self, target_coins: list = None):
        self.target_coins = target_coins if target_coins else DLModelTrainer.TARGET_COINS
        self.data_dir = 'data'
        self.cache_dir = 'cache'
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

    def _classify_market_regime(self, df: pd.DataFrame, short_window=20, long_window=50) -> pd.DataFrame:
        """이동평균선을 사용하여 시장 상황을 분류합니다. (내부 헬퍼 함수)"""
        close_col = 'close' if 'close' in df.columns else 'Close'
        df['SMA_short'] = ta.sma(df[close_col], length=short_window)
        df['SMA_long'] = ta.sma(df[close_col], length=long_window)
        df['regime'] = 'Sideways'
        df.loc[df['SMA_short'] > df['SMA_long'] * 1.01, 'regime'] = 'Bullish'
        df.loc[df['SMA_short'] < df['SMA_long'] * 0.99, 'regime'] = 'Bearish'
        df.drop(columns=['SMA_short', 'SMA_long'], inplace=True)
        return df

    @staticmethod
    def generate_features(df: pd.DataFrame) -> pd.DataFrame:
        """모든 기술적 지표(feature)를 계산합니다."""
        df.ta.rsi(length=14, append=True)
        df.ta.bbands(length=20, std=2, append=True)
        df.ta.macd(fast=12, slow=26, signal=9, append=True)
        df.ta.adx(length=14, append=True)
        df.ta.atr(length=14, append=True)
        df.ta.obv(append=True)
        df.ta.stoch(k=14, d=3, smooth_k=3, append=True)
        return df

    def run(self):
        """전체 전처리 파이프라인을 실행합니다."""
        print("🚀 데이터 전처리 및 캐시 생성을 시작합니다...")
        
        # 1. BTC 데이터를 먼저 처리하여 시장 체제(Regime)의 기준을 마련합니다.
        btc_ticker = 'BTC/KRW'
        btc_df_raw = self._load_raw_data(btc_ticker)
        if btc_df_raw is None:
            print(f"오류: 시장 체제 기준이 되는 {btc_ticker} 데이터를 찾을 수 없습니다.")
            return

        print(f"- {btc_ticker}의 시장 체제를 분류합니다...")
        btc_regime_df = self._classify_market_regime(btc_df_raw.copy())

        # 2. 모든 타겟 코인에 대해 기술적 지표 추가 및 BTC 체제 병합
        for ticker in self.target_coins:
            print(f"- {ticker} 데이터를 처리합니다...")
            df_raw = self._load_raw_data(ticker)
            if df_raw is None or df_raw.empty:
                continue

            # 기술적 지표 생성
            df_featured = DataPreprocessor.generate_features(df_raw)

            # BTC 시장 체제 병합
            final_df = df_featured.join(btc_regime_df['regime'])
            final_df['regime'] = final_df['regime'].ffill() # 주말 등 비어있는 시간의 regime 채우기

            # NaN 값 제거 및 캐시 저장
            final_df.dropna(inplace=True)
            
            cache_path = os.path.join(self.cache_dir, f"{ticker.replace('/', '_')}_1h.feather")
            final_df.reset_index().to_feather(cache_path)
            print(f"  -> {ticker}의 전처리된 데이터 {len(final_df)}개를 '{cache_path}'에 저장했습니다.")

        print("✅ 모든 데이터 처리가 완료되었습니다.")

    def _load_raw_data(self, ticker: str) -> pd.DataFrame | None:
        """data 폴더에서 원본 CSV 파일을 로드합니다."""
        file_path = os.path.join(self.data_dir, f"{ticker.replace('/', '_')}_1h.csv")
        if not os.path.exists(file_path):
            print(f"  경고: {file_path}를 찾을 수 없습니다.")
            return None
        
        df = pd.read_csv(file_path, index_col='timestamp', parse_dates=True)
        # 컬럼명을 소문자로 통일
        df.columns = [col.lower() for col in df.columns]
        return df

if __name__ == '__main__':
    preprocessor = DataPreprocessor()
    preprocessor.run()