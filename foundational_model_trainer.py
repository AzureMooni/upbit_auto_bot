import pandas as pd
from stable_baselines3 import PPO
import os
from trading_env_simple import SimpleTradingEnv

def train_foundational_agent(ticker='BTC/KRW', model_save_path='foundational_agent.zip', total_timesteps=200000):
    """
    마스터 AI (Foundational Agent)를 훈련시키는 함수.
    전처리된 캐시 데이터를 사용합니다.
    """
    print("전처리된 캐시 데이터를 불러옵니다...")
    cache_path = f"cache/{ticker.replace('/', '_')}_1h.feather"
    if not os.path.exists(cache_path):
        print(f"오류: 캐시 파일 '{cache_path}'를 찾을 수 없습니다.")
        print("먼저 --mode preprocess를 실행하여 캐시를 생성해주세요.")
        return

    df = pd.read_feather(cache_path)
    df.set_index('timestamp', inplace=True)

    # 환경에 필요한 숫자형 데이터만 전달 (regime 등 문자열 컬럼 제외)
    env_df = df.select_dtypes(include=np.number)

    print("거래 환경을 설정합니다...")
    env = SimpleTradingEnv(env_df)

    print("PPO 모델을 설정하고 훈련을 시작합니다...")
    model = PPO('MlpPolicy', env, verbose=1, tensorboard_log="./foundational_rl_tensorboard_logs/")
    model.learn(total_timesteps=total_timesteps)

    print(f"훈련된 마스터 AI 모델을 '{model_save_path}'에 저장합니다.")
    model.save(model_save_path)

if __name__ == '__main__':
    import numpy as np # select_dtypes를 위해 추가
    train_foundational_agent()