import os
import shutil
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from rl_environment import PortfolioTradingEnv


class RLModelTrainer:
    """
    강화학습 PPO 에이전트 훈련 및 로드를 담당하는 클래스
    """

    def __init__(
        self,
        model_path="trading_agent.zip",
        tensorboard_log_path="./rl_tensorboard_logs/",
    ):
        self.model_path = model_path
        self.tensorboard_log_path = tensorboard_log_path

    def train_agent(self, total_timesteps=100_000, ticker="BTC/KRW"):
        """
        PPO 알고리즘을 사용하여 강화학습 에이전트를 훈련합니다.
        """
        if os.path.exists(self.tensorboard_log_path):
            print(f"기존 로그 디렉토리 {self.tensorboard_log_path}를 삭제합니다.")
            shutil.rmtree(self.tensorboard_log_path)
        os.makedirs(self.tensorboard_log_path, exist_ok=True)

        print(f"🤖 {ticker}에 대한 강화학습 에이전트 훈련을 시작합니다...")

        # 1. 데이터 로드
        cache_dir = "cache"
        file_path = os.path.join(cache_dir, f"{ticker.replace('/', '_')}_1h.feather")
        if not os.path.exists(file_path):
            print(f"오류: {file_path}에서 훈련 데이터를 찾을 수 없습니다.")
            print("먼저 'preprocess' 모드를 실행하여 데이터를 준비해주세요.")
            return

        df = pd.read_feather(file_path)
        df.set_index("timestamp", inplace=True)

        df.drop(columns=["regime"], inplace=True, errors="ignore")
        df.dropna(inplace=True)
        df = df.astype(np.float32)

        if len(df) < 200:
            print(
                f"오류: {ticker}의 데이터가 너무 적어 훈련할 수 없습니다 ({len(df)} rows)."
            )
            return

        # 2. 거래 환경 생성
        env = TradingEnv(df)
        vec_env = DummyVecEnv([lambda: env])

        # 3. PPO 모델 정의
        model = PPO(
            "MlpPolicy", vec_env, verbose=1, tensorboard_log=self.tensorboard_log_path
        )

        # 4. 모델 훈련
        print(f"총 {total_timesteps} 타임스텝 동안 훈련을 진행합니다.")
        model.learn(total_timesteps=total_timesteps)

        # 5. 모델 저장
        model.save(self.model_path)
        print(f"✅ 훈련이 완료되었습니다. 모델이 '{self.model_path}'에 저장되었습니다.")

    def load_agent(self):
        """
        저장된 PPO 에이전트 모델을 로드합니다.
        """
        if os.path.exists(self.model_path):
            print(f"강화학습 에이전트를 '{self.model_path}'에서 로드합니다.")
            return PPO.load(self.model_path)
        else:
            print(
                f"경고: 저장된 에이전트 파일('{self.model_path}')을 찾을 수 없습니다."
            )
            return None


if __name__ == "__main__":
    # 이 스크립트를 직접 실행할 경우, 훈련을 시작합니다.
    trainer = RLModelTrainer()
    trainer.train_agent(total_timesteps=200_000)
