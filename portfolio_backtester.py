import pandas as pd
import numpy as np
import os
import json
import pickle
from pandas.tseries.offsets import DateOffset

# TF_ENABLE_ONEDNN_OPTS=0 환경 변수 설정으로 mutex.cc 오류 방지
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from stable_baselines3 import PPO
from trading_env_simple import SimpleTradingEnv
from dl_model_trainer import DLModelTrainer
from foundational_model_trainer import train_foundational_agent
from specialist_trainer import train_specialist_agents


class PortfolioBacktester:
    def __init__(
        self, start_date: str, end_date: str, initial_capital: float = 10_000_000
    ):
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.initial_capital = initial_capital
        self.target_coins = DLModelTrainer.TARGET_COINS
        self.cache_dir = "cache"

        # Walk-forward results
        self.all_oos_trades = []
        self.all_oos_portfolio_history = []
        self.all_oos_specialist_stats = {
            regime: {
                "wins": 0,
                "losses": 0,
                "total_profit": 0.0,
                "total_loss": 0.0,
                "trades": 0,
            }
            for regime in ["Bullish", "Bearish", "Sideways"]
        }

    def _load_specialist_agents(self):
        agents = {}
        regimes = ["Bullish", "Bearish", "Sideways"]
        print("\n[WFO] 훈련된 전문가 AI 에이전트들을 로드합니다...")

        try:
            dummy_df = pd.read_feather(
                os.path.join(
                    self.cache_dir,
                    f"{self.target_coins[0].replace('/', '_')}_1h.feather",
                )
            )
            dummy_env = SimpleTradingEnv(dummy_df.select_dtypes(include=np.number))
        except Exception as e:
            print(f"오류: 에이전트 로드를 위한 더미 환경 생성 실패 - {e}")
            return None

        for regime in regimes:
            model_path = f"{regime.lower()}_market_agent.zip"
            if os.path.exists(model_path):
                print(f"  - [{regime}] 전문가 AI 로드 중...")
                agents[regime] = PPO.load(model_path, env=dummy_env)
            else:
                print(
                    f"  - 경고: [{regime}] 전문가 모델({model_path})을 찾을 수 없습니다."
                )

        if not agents:
            print("오류: 어떤 전문가 AI 모델도 로드할 수 없습니다.")
            return None
        return agents

    def _train_models_for_period(self, train_start, train_end):
        print(
            f"\n--- [WFO] 모델 훈련 시작 (기간: {train_start.date()} ~ {train_end.date()}) ---"
        )
        # 1. Foundational Agent 훈련
        train_foundational_agent(
            start_date=train_start,
            end_date=train_end,
            total_timesteps=100000,  # WFO에서는 타임스텝을 줄여서 빠르게 진행
        )
        # 2. Specialist Agents 훈련
        train_specialist_agents(
            start_date=train_start,
            end_date=train_end,
            total_timesteps_per_specialist=25000,  # WFO에서는 타임스텝을 줄여서 빠르게 진행
        )
        print("--- [WFO] 모델 훈련 완료 ---")

    def _simulate_on_period(
        self,
        agents,
        all_data,
        validation_start,
        validation_end,
        cash,
        holdings,
        purchase_info,
    ):
        print(
            f"\n--- [WFO] 검증 시뮬레이션 시작 (기간: {validation_start.date()} ~ {validation_end.date()}) ---"
        )

        timeline = pd.date_range(validation_start, validation_end, freq="h")
        period_trade_log = []
        period_portfolio_history = []

        for now in timeline:
            if "BTC/KRW" not in all_data or now not in all_data["BTC/KRW"].index:
                continue
            current_regime = all_data["BTC/KRW"].loc[now, "regime"]
            agent_to_use = agents.get(current_regime, agents.get("Sideways"))
            if agent_to_use is None:
                continue

            for ticker, df in all_data.items():
                if now not in df.index:
                    continue

                current_loc = df.index.get_loc(now)
                start_loc = max(0, current_loc - 50)
                observation_df = df.iloc[start_loc:current_loc]

                if len(observation_df) < 50:
                    continue

                env_data = observation_df.select_dtypes(include=np.number)
                action, _ = agent_to_use.predict(env_data, deterministic=True)
                action = int(action)

                current_price = df.loc[now, "close"]
                log_entry = {
                    "timestamp": now,
                    "ticker": ticker,
                    "regime": current_regime,
                    "action": action,
                    "price": current_price,
                }

                if action == 1:  # Buy
                    buy_amount_krw = cash * 0.05
                    if buy_amount_krw > 5000:
                        buy_amount_coin = buy_amount_krw / current_price
                        holdings[ticker] += buy_amount_coin
                        cash -= buy_amount_krw
                        purchase_info[ticker]["total_cost"] += buy_amount_krw
                        purchase_info[ticker]["total_amount"] += buy_amount_coin
                        log_entry.update({"trade": "BUY", "amount_krw": buy_amount_krw})
                        period_trade_log.append(log_entry)

                elif action == 2:  # Sell
                    if holdings[ticker] > 0:
                        sell_amount_coin = holdings[ticker]
                        sell_value_krw = sell_amount_coin * current_price
                        total_cost = purchase_info[ticker]["total_cost"]
                        total_amount = purchase_info[ticker]["total_amount"]

                        if total_amount > 0:
                            avg_purchase_price = total_cost / total_amount
                            profit_loss = (
                                current_price - avg_purchase_price
                            ) * sell_amount_coin
                            stats = self.all_oos_specialist_stats[current_regime]
                            stats["trades"] += 1
                            if profit_loss > 0:
                                stats["wins"] += 1
                                stats["total_profit"] += profit_loss
                            else:
                                stats["losses"] += 1
                                stats["total_loss"] += abs(profit_loss)

                        cash += sell_value_krw
                        holdings[ticker] = 0
                        purchase_info[ticker] = {"total_cost": 0.0, "total_amount": 0.0}
                        log_entry.update(
                            {"trade": "SELL", "amount_coin": sell_amount_coin}
                        )
                        period_trade_log.append(log_entry)

            current_net_worth = cash
            for t, amount in holdings.items():
                if amount > 0 and t in all_data and now in all_data[t].index:
                    current_net_worth += amount * all_data[t].loc[now, "close"]
            period_portfolio_history.append(
                {"timestamp": now, "net_worth": current_net_worth}
            )

        self.all_oos_trades.extend(period_trade_log)
        self.all_oos_portfolio_history.extend(period_portfolio_history)

        print("--- [WFO] 검증 시뮬레이션 완료 ---")
        return cash, holdings, purchase_info  # Return the state for the next fold

    def run_walk_forward_optimization(self, train_months=24, validation_months=6):
        print("\n=== 🤖 워크 포워드 최적화 백테스팅 시작 ===")
        print(f"  - 훈련 기간: {train_months}개월 / 검증 기간: {validation_months}개월")

        # 최적화: 모든 데이터를 처음에 한 번만 로드
        print("\n- 모든 기간의 데이터를 메모리로 사전 로딩합니다...")
        preprocessed_data_path = os.path.join("data", "preprocessed_data.pkl")
        if not os.path.exists(preprocessed_data_path):
            print(f"오류: 전처리된 데이터 파일을 찾을 수 없습니다: {preprocessed_data_path}")
            return

        with open(preprocessed_data_path, "rb") as f:
            full_market_data = pickle.load(f)
        
        if not full_market_data:
            print("오류: 백테스팅에 사용할 데이터가 없습니다.")
            return

        current_start = self.start_date
        fold = 1

        # Initialize portfolio state
        cash = self.initial_capital
        holdings = {ticker: 0.0 for ticker in self.target_coins}
        purchase_info = {
            ticker: {"total_cost": 0.0, "total_amount": 0.0}
            for ticker in self.target_coins
        }

        while True:
            train_start = current_start
            train_end = train_start + DateOffset(months=train_months)
            validation_start = train_end
            validation_end = validation_start + DateOffset(months=validation_months)

            if validation_end > self.end_date:
                print("\n남은 기간이 검증 기간보다 짧아 최적화를 종료합니다.")
                break

            print(f"\n================== FOLD {fold} ==================")

            # 1. Train models on the training period
            self._train_models_for_period(train_start, train_end)

            # 2. Load the newly trained agents
            current_agents = self._load_specialist_agents()
            if not current_agents:
                print("오류: 훈련된 모델을 로드할 수 없어 해당 Fold를 건너뜁니다.")
                current_start += DateOffset(months=validation_months)
                fold += 1
                continue

            # 3. Simulate on the validation (out-of-sample) period
            cash, holdings, purchase_info = self._simulate_on_period(
                current_agents,
                full_market_data,
                validation_start,
                validation_end,
                cash,
                holdings,
                purchase_info,
            )

            # 4. Slide the window for the next fold
            current_start += DateOffset(months=validation_months)
            fold += 1

        print("\n=== ✅ 모든 워크 포워드 검증 완료 ===")
        self._generate_final_report(
            self.all_oos_portfolio_history,
            self.all_oos_trades,
            self.all_oos_specialist_stats,
        )

    def _generate_final_report(self, portfolio_history, trade_log, specialist_stats):
        if not portfolio_history:
            print("성과를 분석할 데이터가 없습니다.")
            return

        print("\n--- 📊 최종 포트폴리오 성과 리포트 (Out-of-Sample 기준) ---")
        history_df = pd.DataFrame(portfolio_history).set_index("timestamp")

        final_net_worth = history_df["net_worth"].iloc[-1]
        total_return = (
            (final_net_worth - self.initial_capital) / self.initial_capital * 100
        )
        print(f"- 총 수익률: {total_return:.2f}%")
        print(f"- 초기 자본: {self.initial_capital:,.0f} KRW")
        print(f"- 최종 자산: {final_net_worth:,.0f} KRW")

        history_df["peak"] = history_df["net_worth"].cummax()
        history_df["drawdown"] = (
            history_df["net_worth"] - history_df["peak"]
        ) / history_df["peak"]
        max_drawdown = history_df["drawdown"].min() * 100
        print(f"- 최대 낙폭 (MDD): {max_drawdown:.2f}%")

        history_df["daily_return"] = history_df["net_worth"].pct_change()
        sharpe_ratio = (
            history_df["daily_return"].mean() / history_df["daily_return"].std()
        ) * np.sqrt(365 * 24)
        print(f"- 샤프 지수 (시간봉 기준): {sharpe_ratio:.2f}")

        print("\n--- 👨‍🏫 전문가 AI별 거래 분석 (Out-of-Sample 기준) ---")
        trade_df = pd.DataFrame(trade_log)
        if not trade_df.empty:
            print(
                trade_df.groupby(["regime", "trade"])["ticker"]
                .count()
                .unstack(fill_value=0)
            )
        else:
            print("거래 기록이 없습니다.")
        print("-------------------------------------")

        print("\n--- 📈 전문가 AI별 성과 지표 (Out-of-Sample 기준) ---")
        for regime, stats in specialist_stats.items():
            trades = stats["trades"]
            if trades > 0:
                win_rate = (stats["wins"] / trades) * 100 if trades > 0 else 0
                avg_profit = (
                    stats["total_profit"] / stats["wins"] if stats["wins"] > 0 else 0
                )
                avg_loss = (
                    stats["total_loss"] / stats["losses"] if stats["losses"] > 0 else 0
                )

                print(f"\n[{regime} 전문가]")
                print(f"  - 총 거래: {trades} 회")
                print(
                    f"  - 승률: {win_rate:.2f}% ({stats['wins']}승 / {stats['losses']}패)"
                )
                print(f"  - 평균 이익: {avg_profit:,.0f} KRW")
                print(f"  - 평균 손실: {avg_loss:,.0f} KRW")
            else:
                print(f"\n[{regime} 전문가] 거래 기록 없음")
        print("-------------------------------------\n")

        with open("specialist_stats.json", "w") as f:
            serializable_stats = {}
            for regime, stats in specialist_stats.items():
                serializable_stats[regime] = {
                    "wins": int(stats["wins"]),
                    "losses": int(stats["losses"]),
                    "total_profit": float(stats["total_profit"]),
                    "total_loss": float(stats["total_loss"]),
                    "trades": int(stats["trades"]),
                }
            json.dump(serializable_stats, f, indent=4)
        print(
            "\n--- 💾 최종 OOS 성과 지표를 'specialist_stats.json'에 저장했습니다. ---"
        )
