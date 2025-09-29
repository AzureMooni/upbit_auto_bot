import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class PortfolioTradingEnv(gym.Env):
    """
    여러 암호화폐 자산을 동시에 관리하는 포트폴리오 거래 환경입니다.
    AI 에이전트는 이 환경에서 전체 포트폴리오의 가치를 극대화하는 방법을 학습합니다.

    - Action Space: 각 코인에 대한 투자 비중 조절. (Box space)
        - `-1` ~ `0`: 매도 (가치의 0~100%)
        - `0`: 관망
        - `0` ~ `+1`: 매수 (가용 현금의 0~100%)
    - Observation Space: 포트폴리오 상태와 시장 데이터를 포함하는 복합 공간. (Dict space)
        - 'portfolio': [현금 비중, 각 코인 보유 비중, ...]
        - 'market': [모든 코인의 과거 가격 및 기술적 지표 데이터]
    - Reward Function: 포트폴리오 순자산(net worth)의 변화량. 샤프 지수 등을 추가하여 위험 조정 수익률로 고도화할 수 있습니다.
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, data_dict: dict, initial_balance: float = 1_000_000, lookback_window: int = 50):
        super().__init__()

        self.tickers = list(data_dict.keys())
        self.data_dict = data_dict
        self.initial_balance = initial_balance
        self.lookback_window = lookback_window
        self.n_coins = len(self.tickers)
        
        # 데이터프레임 정렬 및 길이 통일
        self._align_data()
        self.n_features = self.df.shape[1] // self.n_coins

        # Action Space: 각 코인에 대한 행동 (-1: All-Sell, 1: All-In Buy)
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.n_coins,), dtype=np.float32)

        # Observation Space
        self.observation_space = spaces.Dict({
            # 포트폴리오 상태: [현금비중, 코인1비중, ...]
            'portfolio': spaces.Box(low=0, high=1, shape=(self.n_coins + 1,), dtype=np.float32),
            # 시장 데이터: (코인 수, 과거 데이터 길이, 특징 수)
            'market': spaces.Box(low=-np.inf, high=np.inf, shape=(self.n_coins, self.lookback_window, self.n_features), dtype=np.float32)
        })

    def _align_data(self):
        """모든 코인 데이터의 길이를 맞추고 하나의 데이터프레임으로 결합합니다."""
        min_len = min(len(df) for df in self.data_dict.values())
        aligned_dfs = []
        for ticker in self.tickers:
            df = self.data_dict[ticker].iloc[-min_len:].copy()
            df.columns = [f"{ticker}_{col}" for col in df.columns] # 컬럼명에 티커 추가
            aligned_dfs.append(df)
        self.df = pd.concat(aligned_dfs, axis=1).ffill().bfill()
        self.end_step = len(self.df) - 1

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.balance = self.initial_balance
        self.shares_held = np.zeros(self.n_coins)
        self.net_worth = self.initial_balance
        self.current_step = self.lookback_window

        return self._get_observation(), {}

    def step(self, action):
        self.current_step += 1
        current_prices = self._get_current_prices()
        old_net_worth = self.net_worth

        self._take_action(action, current_prices)

        self.net_worth = self.balance + np.sum(self.shares_held * current_prices)
        reward = self.net_worth - old_net_worth

        # 샤프 지수 등을 여기에 추가하여 reward를 고도화할 수 있습니다.
        # 예: reward -= self.volatility * RISK_AVERSION_PARAM

        terminated = self.net_worth <= self.initial_balance * 0.5 or self.current_step >= self.end_step
        truncated = False

        return self._get_observation(), reward, terminated, truncated, {}

    def _get_current_prices(self):
        """모든 코인의 현재 가격을 가져옵니다."""
        prices = []
        for ticker in self.tickers:
            prices.append(self.df[f'{ticker}_close'].iloc[self.current_step])
        return np.array(prices)

    def _get_observation(self):
        """현재 관측(observation)을 생성합니다."""
        # 시장 데이터 관측
        market_obs = np.array([ 
            self.df.iloc[self.current_step - self.lookback_window : self.current_step, i*self.n_features:(i+1)*self.n_features].values
            for i in range(self.n_coins)
        ])

        # 포트폴리오 관측
        portfolio_values = self.shares_held * self._get_current_prices()
        portfolio_obs = np.concatenate(([self.balance], portfolio_values)) / self.net_worth

        return {'portfolio': portfolio_obs.astype(np.float32), 'market': market_obs.astype(np.float32)}

    def _take_action(self, action, current_prices):
        """주어진 행동(action)을 실행하여 잔고와 보유 주식 수를 업데이트합니다."""
        for i, act in enumerate(action):
            if act > 0: # 매수
                buy_amount = self.balance * act
                if buy_amount > 10: # 최소 거래 금액
                    self.shares_held[i] += buy_amount / current_prices[i]
                    self.balance -= buy_amount
            elif act < 0: # 매도
                sell_fraction = -act
                shares_to_sell = self.shares_held[i] * sell_fraction
                if shares_to_sell > 0:
                    self.balance += shares_to_sell * current_prices[i]
                    self.shares_held[i] -= shares_to_sell

    def render(self, mode='human'):
        profit = self.net_worth - self.initial_balance
        profit_percent = (profit / self.initial_balance) * 100
        print(f'Step: {self.current_step} | Net Worth: {self.net_worth:,.2f} KRW | Profit: {profit:,.2f} ({profit_percent:.2f}%)')
