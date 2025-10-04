import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium.spaces import Box, Discrete


class TradingEnv(gym.Env):
    """
    강화학습 에이전트를 위한 가상 거래 환경
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, df: pd.DataFrame, initial_capital=1_000_000, window_size=60):
        super().__init__()

        self.df = df
        self.initial_capital = initial_capital
        self.window_size = window_size

        # Action Space: 0: Hold, 1: Buy, 2: Sell
        self.action_space = Discrete(3)

        # Observation Space: [cash, asset_holdings] + window_size * num_features
        # cash: 현재 현금
        # asset_holdings: 현재 보유 자산 가치
        # num_features: OHLCV + 기술적 지표 수
        self.observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(2 + window_size * len(self.df.columns),),
            dtype=np.float32,
        )

        self.current_step = 0
        self.cash = initial_capital
        self.asset_holdings = 0.0  # 보유 코인 수량
        self.total_asset = initial_capital
        self.start_step = window_size

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.start_step
        self.cash = self.initial_capital
        self.asset_holdings = 0.0
        self.total_asset = self.initial_capital

        obs = self._get_obs()
        info = self._get_info()

        return obs, info

    def step(self, action):
        self.current_step += 1

        # 현재 가격
        current_price = self.df.loc[self.df.index[self.current_step], "close"]

        # 이전 스텝의 총 자산
        prev_total_asset = self.total_asset

        # 행동 수행
        self._take_action(action, current_price)

        # 현재 총 자산 계산
        self.total_asset = self.cash + self.asset_holdings * current_price

        # 보상 계산
        reward = self.total_asset - prev_total_asset

        # 거래 미체결 시 작은 페널티
        if action == 0:  # Hold
            reward -= self.initial_capital * 0.00001

        # 종료 조건 확인
        terminated = self.current_step >= len(self.df) - 1
        truncated = False  # 시간 제한으로 인한 종료는 사용하지 않음

        obs = self._get_obs()
        info = self._get_info()

        return obs, reward, terminated, truncated, info

    def _take_action(self, action, current_price):
        # action: 0: Hold, 1: Buy, 2: Sell
        if action == 1:  # Buy
            # 현금의 10% 만큼 매수
            buy_amount = self.cash * 0.1
            if self.cash > buy_amount:
                self.asset_holdings += buy_amount / current_price
                self.cash -= buy_amount
        elif action == 2:  # Sell
            # 보유 자산의 10% 만큼 매도
            sell_amount = self.asset_holdings * 0.1
            if self.asset_holdings > sell_amount:
                self.cash += sell_amount * current_price
                self.asset_holdings -= sell_amount

    def _get_obs(self):
        # 현재 스텝의 관측 데이터
        obs_df = self.df.iloc[self.current_step - self.window_size : self.current_step]

        # 정규화 (가격 기반 지표들은 첫번째 close 값으로, 나머지는 그대로)
        # 간단한 구현을 위해 여기서는 정규화를 생략하고, 실제 훈련 시 데이터 전처리 단계에서 수행하는 것을 권장
        obs_values = obs_df.values.flatten()

        # 포트폴리오 상태 추가
        portfolio_state = np.array(
            [
                self.cash,
                self.asset_holdings
                * self.df.loc[self.df.index[self.current_step], "close"],
            ],
            dtype=np.float32,
        )

        return np.concatenate([portfolio_state, obs_values]).astype(np.float32)

    def _get_info(self):
        return {
            "total_asset": self.total_asset,
            "cash": self.cash,
            "asset_holdings": self.asset_holdings,
            "current_price": self.df.loc[self.df.index[self.current_step], "close"],
        }

    def render(self, mode="human", close=False):
        if mode == "human":
            info = self._get_info()
            print(
                f"Step: {self.current_step}, Total Asset: {info['total_asset']:.2f}, Cash: {info['cash']:.2f}, Holdings: {info['asset_holdings']:.6f}"
            )
