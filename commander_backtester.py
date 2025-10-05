import pandas as pd
import numpy as np
import os

# --- 의존성 임포트 ---
from market_regime_detector import precompute_regime_indicators, get_regime_from_indicators
from risk_manager import get_position_size_ratio
from strategies.trend_follower import generate_trend_signals


class CommanderBacktester:
    """
    AI 총사령관의 동적 자산 배분 전략을 시뮬레이션합니다.
    """

    def __init__(self, start_date: str, end_date: str, initial_capital: float):
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.initial_capital = initial_capital
        self.cache_dir = "cache"
        
        self.precompute_indicators = precompute_regime_indicators
        self.get_regime = get_regime_from_indicators
        self.get_size_ratio = get_position_size_ratio
        self.generate_trend_signals = generate_trend_signals
        
        print("✅ AI 총사령관 백테스팅 시스템 초기화 (추세 전략 추가).")

    def _simulate_scalping_squad_pnl(
        self, capital_for_day: float, intraday_data: pd.DataFrame
    ) -> tuple[float, int]:
        if intraday_data.empty or len(intraday_data) < 10 or capital_for_day <= 0:
            return 0.0, 0

        df = intraday_data.copy()
        df["EMA_5"] = df['close'].ewm(span=5, adjust=False).mean()
        df["EMA_10"] = df['close'].ewm(span=10, adjust=False).mean()
        df.dropna(inplace=True)

        position_held = False
        entry_price = 0.0
        pnl = 0.0
        trade_count = 0
        take_profit_ratio = 1.02
        stop_loss_ratio = 0.99
        trade_amount = capital_for_day

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
                    pnl += ((exit_price - entry_price) / entry_price) * trade_amount
                    position_held = False
                    trade_count += 1
        return pnl, trade_count

    def run_simulation(self, trailing_stop_pct: float = 0.10):
        """
        AI 총사령관의 동적 자산배분 전략의 최종 성과를 시뮬레이션합니다.
        (Trailing Stop-Loss 로직 추가)
        """
        print("🚀 AI 총사령관 전체 전략 시뮬레이션을 시작합니다...")

        # 1. 데이터 로드
        btc_ticker = "BTC/KRW"
        cache_path = os.path.join(self.cache_dir, f"{btc_ticker.replace('/', '_')}_1h.feather")
        if not os.path.exists(cache_path):
            print(f"오류: {cache_path} 파일이 없습니다.")
            return

        df_btc_hourly = pd.read_feather(cache_path).set_index("timestamp")
        df_btc_daily = df_btc_hourly.resample("D").agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
        }).dropna()
        df_btc_daily["daily_return"] = df_btc_daily["close"].pct_change()

        # 2. 모든 지표 및 신호 일괄 계산
        print("  - 모든 거시 지표 및 신호를 사전 계산 중...")
        df_indicators = self.precompute_indicators(df_btc_daily)
        df_indicators = self.generate_trend_signals(df_indicators)
        df_indicators['SMA_200'] = df_indicators['close'].rolling(window=200, min_periods=100).mean()
        df_indicators.dropna(inplace=True)
        print("  - 지표 및 신호 계산 완료.")

        # 3. 시뮬레이션 상태 변수 초기화
        cash = self.initial_capital
        holdings = 0.0
        portfolio_history = []
        total_trades = 0
        benchmark_value = self.initial_capital
        
        # 추세 추종 전략 상태 변수
        trend_position_active = False
        # Trailing Stop Logic Start
        peak_price_since_entry = 0
        trailing_stop_price = 0
        # Trailing Stop Logic End

        # 4. 메인 시뮬레이션 루프
        for today, row in df_indicators.iterrows():
            if not (self.start_date <= today <= self.end_date):
                continue

            current_price = row['close']
            portfolio_value = cash + holdings * current_price

            # 하락장 방어 로직
            is_bear_market = current_price < row['SMA_200']
            if is_bear_market:
                if trend_position_active: # 하락장 진입 시, 보유 중인 추세 포지션 즉시 청산
                    sell_value = holdings * current_price
                    cash += sell_value
                    holdings = 0
                    trend_position_active = False
                    total_trades += 1
                
                # 포트폴리오 기록 후, 당일 모든 거래 활동 중지
                portfolio_history.append({"date": today, "portfolio_value": portfolio_value})
                market_return = row["daily_return"]
                if not pd.isna(market_return):
                    benchmark_value *= (1 + market_return)
                continue # 다음 날로 넘어감

            # Trailing Stop Logic Start
            if trend_position_active:
                # 보유 중일 때, 매일 고점을 기준으로 피크 가격 업데이트
                peak_price_since_entry = max(peak_price_since_entry, row['high'])
                # 트레일링 스톱 가격 업데이트
                trailing_stop_price = peak_price_since_entry * (1 - trailing_stop_pct)
            # Trailing Stop Logic End

            # 시장 체제 기반 거래 로직
            current_regime = self.get_regime(
                adx=row['ADX'],
                normalized_atr=row['Normalized_ATR'],
                natr_ma=row['Normalized_ATR_MA']
            )
            
            active_capital_ratio = self.get_size_ratio(
                regime=current_regime, 
                normalized_atr=row['Normalized_ATR'], 
                natr_ma=row['Normalized_ATR_MA']
            )
            
            # 추세 전략 실행
            trend_signal = row['trend_signal']
            
            # --- EXIT Conditions ---
            # 1. 기본 매도 신호(MACD) 발생 or 2. 트레일링 스톱 가격 도달
            if trend_position_active and (trend_signal == -1.0 or row['low'] <= trailing_stop_price):
                sell_value = holdings * current_price
                cash += sell_value
                holdings = 0
                trend_position_active = False
                total_trades += 1

            # --- ENTRY Condition ---
            elif not trend_position_active and 'TREND' in current_regime and trend_signal == 1.0:
                capital_to_invest = portfolio_value * active_capital_ratio
                if cash >= capital_to_invest:
                    units_to_buy = capital_to_invest / current_price
                    cash -= capital_to_invest
                    holdings += units_to_buy
                    trend_position_active = True
                    total_trades += 1
                    # Trailing Stop Logic Start
                    peak_price_since_entry = current_price
                    trailing_stop_price = current_price * (1 - trailing_stop_pct)
                    # Trailing Stop Logic End
            
            # 횡보 전략 실행
            elif 'SIDEWAYS' in current_regime:
                if trend_position_active:
                    sell_value = holdings * current_price
                    cash += sell_value
                    holdings = 0
                    trend_position_active = False
                    total_trades += 1

                capital_to_invest = portfolio_value * active_capital_ratio
                intraday_data = df_btc_hourly[df_btc_hourly.index.date == today.date()]
                pnl_scalping, daily_trades = self._simulate_scalping_squad_pnl(capital_to_invest, intraday_data)
                cash += pnl_scalping
                total_trades += daily_trades

            # 일일 기록 업데이트
            portfolio_value = cash + holdings * current_price
            portfolio_history.append({"date": today, "portfolio_value": portfolio_value})
            
            market_return = row["daily_return"]
            if not pd.isna(market_return):
                benchmark_value *= (1 + market_return)

        # 5. 최종 성과 보고
        if not portfolio_history:
            print("데이터 부족으로 보고서를 생성할 수 없습니다.")
            return
            
        report_df = pd.DataFrame(portfolio_history).set_index("date")
        final_portfolio_value = report_df["portfolio_value"].iloc[-1]
        total_return = (final_portfolio_value / self.initial_capital - 1) * 100
        benchmark_return = (benchmark_value / self.initial_capital - 1) * 100

        rolling_max = report_df["portfolio_value"].cummax()
        daily_drawdown = report_df["portfolio_value"] / rolling_max - 1.0
        mdd = daily_drawdown.cummin().iloc[-1] * 100

        daily_returns = report_df["portfolio_value"].pct_change()
        sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(365) if daily_returns.std() > 0 else 0

        print("\n--- 📊 AI 총사령관 전략 최종 성과 보고 (추세 전략 추가) ---")
        print(f"  - 시뮬레이션 기간: {self.start_date.date()} ~ {self.end_date.date()}")
        print("-" * 50)
        print("  [AI Commander 전략 성과]")
        print(f"  - 최종 자산: {final_portfolio_value:,.0f} KRW")
        print(f"  - 총 수익률: {total_return:.2f}%")
        print(f"  - 총 거래 횟수: {total_trades} 회")
        print(f"  - 최대 낙폭 (MDD): {mdd:.2f}%")
        print(f"  - 샤프 지수 (연율화): {sharpe_ratio:.2f}")
        print("-" * 50)
        print("  [벤치마크 (Buy & Hold) 성과]")
        print(f"  - 최종 자산: {benchmark_value:,.0f} KRW")
        print(f"  - 총 수익률: {benchmark_return:.2f}%")
        print("-" * 50)


if __name__ == "__main__":
    commander_backtester = CommanderBacktester(
        start_date="2023-01-01", end_date="2023-12-31", initial_capital=1_000_000
    )
    commander_backtester.run_simulation()