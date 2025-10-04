import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# 고빈도 스캘핑을 위한 타겟 코인 목록
SCALPING_TARGET_COINS = ['BTC/KRW', 'ETH/KRW', 'XRP/KRW', 'SOL/KRW', 'DOGE/KRW']

class ModelTrainer:
    """
    XGBoost를 사용하여 초단기 가격 변동을 예측하는 모델을 훈련합니다.
    """
    def __init__(self, target_coins: list = None):
        self.target_coins = target_coins if target_coins else SCALPING_TARGET_COINS
        self.cache_dir = 'cache'
        self.model_path = 'price_predictor.pkl'
        self.scaler_path = 'price_scaler.pkl'
        self.model = None
        self.scaler = None

    def _create_labels(self, df: pd.DataFrame, look_forward_mins=5, threshold=0.003) -> pd.DataFrame:
        """향후 N분 내의 가격 변동을 기반으로 레이블을 생성합니다."""
        df['future_high'] = df['high'].shift(-look_forward_mins).rolling(window=look_forward_mins).max()
        df['future_low'] = df['low'].shift(-look_forward_mins).rolling(window=look_forward_mins).min()

        df['label'] = 0 # 관망
        df.loc[df['future_high'] >= df['close'] * (1 + threshold), 'label'] = 1 # 매수
        df.loc[df['future_low'] <= df['close'] * (1 - threshold), 'label'] = 2 # 매도
        
        return df.dropna()

    def train_model(self):
        """전체 모델 훈련 파이프라인을 실행합니다."""
        print("🚀 [XGBoost] 초단기 예측 모델 훈련을 시작합니다...")
        
        # 1. 데이터 로드
        all_data = []
        for ticker in self.target_coins:
            cache_path = os.path.join(self.cache_dir, f"{ticker.replace('/', '_')}_1m.feather")
            if os.path.exists(cache_path):
                df = pd.read_feather(cache_path)
                all_data.append(df)
            else:
                print(f"  경고: {ticker}의 캐시 파일을 찾을 수 없습니다. 전처리 단계를 먼저 실행하세요.")
        
        if not all_data:
            print("오류: 훈련할 데이터가 없습니다.")
            return

        df_combined = pd.concat(all_data, ignore_index=True)
        print(f"  총 {len(df_combined)}개의 1분봉 데이터로 훈련을 시작합니다.")

        # 2. 레이블 및 피처 생성
        df_labeled = self._create_labels(df_combined)
        
        features = [
            'RSI_14', 'BBL_20', 'BBM_20', 'BBU_20', 
            'MACD_12_26_9', 'MACDH_12_26_9', 'MACDS_12_26_9'
        ]
        target = 'label'

        X = df_labeled[features]
        y = df_labeled[target]

        # 3. 데이터 분할 및 스케일링
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # 4. XGBoost 모델 훈련
        self.model = xgb.XGBClassifier(
            objective='multi:softmax',
            num_class=3,
            eval_metric='mlogloss',
            use_label_encoder=False,
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        
        print("  XGBoost 모델 훈련 중...")
        self.model.fit(X_train_scaled, y_train)

        # 5. 모델 평가
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\n[모델 평가] 테스트 정확도: {accuracy:.4f}")
        print(classification_report(y_test, y_pred, target_names=['Hold(0)', 'Buy(1)', 'Sell(2)']))

        # 6. 모델 및 스케일러 저장
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.scaler, self.scaler_path)
        print(f"✅ 모델을 '{self.model_path}'에, 스케일러를 '{self.scaler_path}'에 저장했습니다.")

if __name__ == '__main__':
    trainer = ModelTrainer()
    trainer.train_model()
