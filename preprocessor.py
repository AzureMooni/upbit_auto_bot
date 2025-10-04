import pandas as pd
import os

# 고빈도 스캘핑을 위한 타겟 코인 목록
SCALPING_TARGET_COINS = ['BTC/KRW', 'ETH/KRW', 'XRP/KRW', 'SOL/KRW', 'DOGE/KRW']

class DataPreprocessor:
    """
    고빈도 거래를 위한 데이터 전처리기.
    1분봉 데이터를 읽어, 스캘핑에 필요한 최소한의 기술적 지표를 계산하고 캐시를 생성합니다.
    """
    def __init__(self, target_coins: list = None):
        self.target_coins = target_coins if target_coins else SCALPING_TARGET_COINS
        self.data_dir = 'data'
        self.cache_dir = 'cache'
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

    @staticmethod
    def generate_features(df: pd.DataFrame) -> pd.DataFrame:
        """스캘핑에 필요한 핵심 기술적 지표(feature)를 계산합니다."""
        df.ta.rsi(length=14, append=True)
        df.ta.bbands(length=20, std=2, append=True)
        df.ta.macd(fast=12, slow=26, signal=9, append=True)
        return df

    def run(self):
        """전체 전처리 파이프라인을 실행합니다."""
        print("🚀 [Scalping] 1분봉 데이터 전처리 및 캐시 생성을 시작합니다...")
        
        for ticker in self.target_coins:
            print(f"- {ticker} 데이터를 처리합니다...")
            df_raw = self._load_raw_data(ticker)
            if df_raw is None or df_raw.empty:
                continue

            # 기술적 지표 생성
            df_featured = DataPreprocessor.generate_features(df_raw)

            # BTC 시장 체제 병합
            df_featured.dropna(inplace=True)
            
            cache_path = os.path.join(self.cache_dir, f"{ticker.replace('/', '_')}_1m.feather")
            df_featured.reset_index().to_feather(cache_path)
            print(f"  -> {ticker}의 전처리된 데이터 {len(df_featured)}개를 '{cache_path}'에 저장했습니다.")

        print("✅ 모든 데이터 처리가 완료되었습니다.")

    def _load_raw_data(self, ticker: str, timeframe: str = '1m') -> pd.DataFrame | None:
        """data 폴더에서 원본 CSV 파일을 로드합니다."""
        file_path = os.path.join(self.data_dir, f"{ticker.replace('/', '_')}_{timeframe}.csv")
        if not os.path.exists(file_path):
            print(f"  경고: {file_path}를 찾을 수 없습니다.")
            return None
        
        df = pd.read_csv(file_path, index_col='timestamp', parse_dates=True)
        df.columns = [col.lower() for col in df.columns]
        return df

if __name__ == '__main__':
    preprocessor = DataPreprocessor()
    preprocessor.run()
