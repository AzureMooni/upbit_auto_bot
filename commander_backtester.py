import pandas as pd
import numpy as np
import os


# --- Manual Indicator Implementations ---
def _manual_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def _manual_ema(prices, period):
    return prices.ewm(span=period, adjust=False).mean()


class CommanderBacktester:
    """
    AI 총사령관의 동적 자산 배분 전략을 시뮬레이션합니다.
    """

    def __init__(self, start_date: str, end_date: str, initial_capital: float):
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.initial_capital = initial_capital
        self.cache_dir = "cache"
        print("✅ AI 총사령관 백테스팅 시스템 초기화.")

    def _precompute_indicators(
        self, df: pd.DataFrame, volatility_days=30, rsi_period=14
    ) -> pd.DataFrame:
        """백테스팅에 필요한 모든 지표를 사전에 일괄 계산합니다."""
        print("  - 백테스팅에 필요한 거시 지표를 사전 계산 중...")
        df["daily_return"] = df["close"].pct_change()
        df["volatility"] = (
            df["daily_return"].rolling(window=volatility_days).std()
            * np.sqrt(365)
            * 100
        )
        df["rsi"] = _manual_rsi(df["close"], period=rsi_period)
        df.dropna(inplace=True)
        print("  - 지표 계산 완료.")
        return df

    def _simulate_main_squad_pnl(
        self, capital: float, market_return: float, alpha: float = 1.2
    ) -> float:
        """주력 부대의 일일 손익을 시장 수익률 기반 프록시로 계산합니다."""
        return capital * market_return * alpha

    def _simulate_scalping_squad_pnl(
        self, capital: float, intraday_data: pd.DataFrame, trade_amount: float
    ) -> float:
        """단기 부대의 일일 손익을 분봉 데이터 기반으로 정밀 시뮬레이션합니다."""
        if intraday_data.empty or len(intraday_data) < 10:
            return 0.0

        df = intraday_data.copy()
        df["EMA_5"] = _manual_ema(df["close"], period=5)
        df["EMA_10"] = _manual_ema(df["close"], period=10)
        df.dropna(inplace=True)

        position_held = False
        entry_price = 0.0
        pnl = 0.0
        take_profit_ratio = 1.02
        stop_loss_ratio = 0.99

        for i in range(1, len(df)):
            prev_row = df.iloc[i - 1]
            current_row = df.iloc[i]

            if not position_held:
                if (
                    current_row["EMA_5"] > current_row["EMA_10"]
                    and prev_row["EMA_5"] <= prev_row["EMA_10"]
                ):
                    position_held = True
                    entry_price = current_row["close"]
            else:
                if (
                    current_row["close"] >= entry_price * take_profit_ratio
                    or current_row["close"] <= entry_price * stop_loss_ratio
                ):
                    exit_price = current_row["close"]
                    pnl += (exit_price - entry_price) / entry_price * trade_amount
                    position_held = False
        return pnl

    def run_simulation(self):
        """
        AI 총사령관의 동적 자산배분 전략의 최종 성과를 시뮬레이션합니다.
        """
        print("🚀 AI 총사령관 전체 전략 시뮬레이션을 시작합니다...")

        # 1. 데이터 로드 및 지표 계산
        btc_ticker = "BTC/KRW"
        cache_path = os.path.join(
            self.cache_dir, f"{btc_ticker.replace('/', '_')}_1h.feather"
        )
        if not os.path.exists(cache_path):
            print(f"오류: {cache_path} 파일이 없습니다.")
            return

        df_btc_hourly = pd.read_feather(cache_path).set_index("timestamp")
        df_btc_daily = df_btc_hourly["close"].resample("D").last().to_frame()
        df_indicators = self._precompute_indicators(df_btc_daily)

        # 2. 시뮬레이션 루프
        portfolio_value = self.initial_capital
        portfolio_history = []

        timeline = pd.date_range(self.start_date, self.end_date, freq="D")

        for today in timeline:
            if today not in df_indicators.index:
                continue

            current_indicators = df_indicators.loc[today]
            volatility = current_indicators["volatility"]
            fear_greed = int(current_indicators["rsi"])
            market_return = current_indicators["daily_return"]

            # 3. 자산 배분 결정
            main_squad_ratio = 0.7
            scalping_squad_ratio = 0.1

            if volatility < 40 and fear_greed > 60:
                main_squad_ratio, scalping_squad_ratio = 0.9, 0.1
            elif volatility > 70 and fear_greed > 50:
                main_squad_ratio, scalping_squad_ratio = 0.5, 0.5
            elif fear_greed < 20 or fear_greed > 85 or volatility < 20:
                main_squad_ratio, scalping_squad_ratio = 0.1, 0.1

            # 4. 일일 손익 계산
            main_squad_capital = portfolio_value * main_squad_ratio
            scalping_squad_capital = portfolio_value * scalping_squad_ratio

            pnl_main = self._simulate_main_squad_pnl(main_squad_capital, market_return)

            intraday_data = df_btc_hourly[df_btc_hourly.index.date == today.date()]
            pnl_scalping = self._simulate_scalping_squad_pnl(
                scalping_squad_capital, intraday_data, trade_amount=50000
            )

            daily_pnl = pnl_main + pnl_scalping
            portfolio_value += daily_pnl
            portfolio_history.append(
                {"date": today, "portfolio_value": portfolio_value}
            )

        # 5. 최종 성과 보고
        report_df = pd.DataFrame(portfolio_history).set_index("date")
        total_return = (
            report_df["portfolio_value"].iloc[-1] / self.initial_capital - 1
        ) * 100

        rolling_max = report_df["portfolio_value"].cummax()
        daily_drawdown = report_df["portfolio_value"] / rolling_max - 1.0
        mdd = daily_drawdown.cummin().iloc[-1] * 100

        daily_returns = report_df["portfolio_value"].pct_change()
        sharpe_ratio = (
            (daily_returns.mean() / daily_returns.std()) * np.sqrt(365)
            if daily_returns.std() != 0
            else 0
        )

        print("\n--- 📊 AI 총사령관 전략 최종 성과 보고 ---")
        print(f"  - 시뮬레이션 기간: {self.start_date.date()} ~ {self.end_date.date()}")
        print(f"  - 최종 자산: {portfolio_value:,.0f} KRW")
        print(f"  - 총 수익률: {total_return:.2f}%")
        print(f"  - 최대 낙폭 (MDD): {mdd:.2f}%")
        print(f"  - 샤프 지수 (연율화): {sharpe_ratio:.2f}")
        print("-----------------------------------------")


if __name__ == "__main__":
    commander_backtester = CommanderBacktester(
        start_date="2023-01-01", end_date="2023-12-31", initial_capital=1_000_000
    )
    commander_backtester.run_simulation()
