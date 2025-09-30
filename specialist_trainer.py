import pandas as pd
import numpy as np
from stable_baselines3 import PPO
import os
from trading_env_simple import SimpleTradingEnv
from dl_model_trainer import DLModelTrainer

def train_specialists(foundational_model_path='foundational_agent.zip', 
                      ticker='BTC/KRW', 
                      total_timesteps_per_specialist=50000):
    """
    마스터 AI를 기반으로 각 시장 상황에 특화된 전문가 AI들을 훈련 (전이학습).
    전처리된 캐시 데이터를 사용합니다.
    """
    if not os.path.exists(foundational_model_path):
        print(f"오류: 마스터 AI 모델 '{foundational_model_path}'을 찾을 수 없습니다.")
        return

    print(f"{ticker}의 전처리된 캐시 데이터를 불러옵니다...")
    cache_path = f"cache/{ticker.replace('/', '_')}_1h.feather"
    if not os.path.exists(cache_path):
        print(f"오류: 캐시 파일 '{cache_path}'를 찾을 수 없습니다.")
        print("먼저 --mode preprocess를 실행하여 캐시를 생성해주세요.")
        return

    labeled_df = pd.read_feather(cache_path)
    labeled_df.set_index('timestamp', inplace=True)

    print("데이터셋 정보:")
    print(labeled_df['regime'].value_counts())

    regimes = ['Bullish', 'Bearish', 'Sideways']
    specialist_models = {}

    for regime in regimes:
        print(f"\n{'='*30}")
        print(f"{regime} 시장 전문가 AI 훈련 시작...")
        print(f"{'='*30}")

        regime_data = labeled_df[labeled_df['regime'] == regime]
        print(f"{regime} 데이터셋 크기: {len(regime_data)}")

        if len(regime_data) < 100:
            print(f"{regime} 시장 데이터가 너무 적어 훈련을 건너뜁니다.")
            continue

        env_data = regime_data.select_dtypes(include=np.number)
        env = SimpleTradingEnv(env_data)

        model = PPO.load(foundational_model_path, env=env, custom_objects={'learning_rate': 0.0001, 'n_steps': 2048})
        
        model.tensorboard_log = f"./{regime.lower()}_specialist_logs/"
        model.learn(total_timesteps=total_timesteps_per_specialist, tb_log_name="PPO")

        specialist_model_name = f"{regime.lower()}_market_agent.zip"
        model.save(specialist_model_name)
        specialist_models[regime] = specialist_model_name
        print(f"{regime} 전문가 AI를 '{specialist_model_name}'으로 저장했습니다.")

    print("\n모든 전문가 AI 훈련이 완료되었습니다.")
    print("생성된 모델:", specialist_models)

if __name__ == '__main__':
    train_specialists(total_timesteps_per_specialist=50000)