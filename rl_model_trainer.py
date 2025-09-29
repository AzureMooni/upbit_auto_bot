import pandas as pd
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from rl_environment import TradingEnv
from preprocessor import DataPreprocessor
from dl_model_trainer import DLModelTrainer # To get TARGET_COINS

class RLModelTrainer:
    """
    강화학습 (RL) 에이전트를 훈련하고 관리하는 클래스.
    - PPO 알고리즘을 사용하여 에이전트를 훈련합니다.
    - 훈련된 에이전트를 파일로 저장하고 불러올 수 있습니다.
    """
    def __init__(self, model_path="trading_agent.zip"):
        self.model_path = model_path
        # 데이터 전처리를 위한 Preprocessor 인스턴스
        self.preprocessor = DataPreprocessor(DLModelTrainer.TARGET_COINS)

    def train_agent(self, total_timesteps=1_000_000):
        """
        강화학습 에이전트를 훈련시킵니다.

        Args:
            total_timesteps (int): 훈련을 위한 총 타임스텝 수.
        """
        print("강화학습(RL) 에이전트 훈련을 시작합니다...")

        # 1. 훈련 데이터 로드 및 전처리
        # 여러 코인 중 하나(예: BTC/KRW)의 데이터로 우선 훈련합니다.
        # 향후 모든 코인 데이터를 사용하거나, 여러 환경을 동시에 사용하는 방식으로 확장할 수 있습니다.
        ticker = 'BTC/KRW'
        df = self.preprocessor.load_and_preprocess_single_coin(ticker, '1h')

        if df is None or df.empty or len(df) < 200: # 데이터가 충분한지 확인
            print(f"{ticker}에 대한 훈련 데이터를 로드할 수 없거나 데이터가 부족합니다. 훈련을 중단합니다.")
            return

        # 2. 강화학습 거래 환경 생성
        # Stable Baselines3에서 여러 환경을 병렬로 처리하기 위해 DummyVecEnv로 래핑합니다.
        env = DummyVecEnv([lambda: TradingEnv(df)])

        # 3. PPO 모델 인스턴스화
        # 'MlpPolicy'는 표준 다층 퍼셉트론 정책입니다.
        # verbose=1은 훈련 진행 상황을 출력합니다.
        # n_steps를 환경의 길이보다 작거나 같게 설정해야 할 수 있습니다.
        model = PPO('MlpPolicy', env, verbose=1, n_steps=1024, tensorboard_log="./rl_tensorboard_logs/")

        # 4. 모델 훈련
        print(f"총 {total_timesteps} 타임스텝 동안 훈련을 진행합니다.")
        model.learn(total_timesteps=total_timesteps)

        # 5. 훈련된 모델 저장
        model.save(self.model_path)
        print(f"훈련된 RL 에이전트를 '{self.model_path}' 파일로 저장했습니다.")

    def load_agent(self):
        """저장된 RL 에이전트를 불러옵니다."""
        if os.path.exists(self.model_path):
            print(f"저장된 RL 에이전트('{self.model_path}')를 불러옵니다.")
            return PPO.load(self.model_path)
        else:
            print(f"저장된 에이전트 파일('{self.model_path}')을 찾을 수 없습니다. 먼저 에이전트를 훈련시켜야 합니다.")
            return None

if __name__ == '__main__':
    # 이 파일을 직접 실행하여 훈련을 시작할 수 있습니다.
    # 예: python rl_model_trainer.py
    trainer = RLModelTrainer()
    # 초기 테스트를 위해 타임스텝을 줄여서 실행
    trainer.train_agent(total_timesteps=100_000)