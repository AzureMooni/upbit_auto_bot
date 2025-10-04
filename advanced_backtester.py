import pandas as pd
import joblib
import os

# 고빈도 스캘핑을 위한 타겟 코인 목록
SCALPING_TARGET_COINS = ['BTC/KRW', 'ETH/KRW', 'XRP/KRW', 'SOL/KRW', 'DOGE/KRW']

class AdvancedBacktester:
    """
    고빈도 퀀트 스캘핑 전략을 1분봉 데이터 기준으로 시뮬레이션합니다.
    """
    def __init__(self, start_date: str, end_date: str, initial_capital: float):
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.initial_capital = initial_capital
        self.cache_dir = 'cache'
        self.model = None
        self.scaler = None

    def _load_model(self, model_path='price_predictor.pkl', scaler_path='price_scaler.pkl'):
        try:
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            print("✅ XGBoost 모델 및 스케일러 로드 완료.")
        except FileNotFoundError:
            print("오류: 모델 또는 스케일러 파일을 찾을 수 없습니다. 모델 훈련을 먼저 실행하세요.")
            raise

    def _generate_report(self, trades: list, final_capital: float):
        if not trades:
            print("거래가 발생하지 않았습니다.")
            return

        df = pd.DataFrame(trades)
        total_trades = len(df)
        wins = df[df['pnl'] > 0]
        losses = df[df['pnl'] < 0]
        win_rate = len(wins) / total_trades if total_trades > 0 else 0
        
        avg_profit = wins['pnl'].mean() if not wins.empty else 0
        avg_loss = losses['pnl'].mean() if not losses.empty else 0
        profit_loss_ratio = abs(avg_profit / avg_loss) if avg_loss != 0 else float('inf')

        total_return = (final_capital / self.initial_capital - 1) * 100

        print("\n--- 📈 고빈도 스캘핑 백테스트 최종 성과 보고 ---")
        print(f"  - 시뮬레이션 기간: {self.start_date.date()} ~ {self.end_date.date()}")
        print(f"  - 초기 자본: {self.initial_capital:,.0f} KRW")
        print(f"  - 최종 자산: {final_capital:,.0f} KRW")
        print(f"  - 총 수익률: {total_return:.2f}%")
        print("-" * 40)
        print(f"  - 총 거래 횟수: {total_trades}")
        print(f"  - 승률: {win_rate:.2%}")
        print(f"  - 손익비: {profit_loss_ratio:.2f}")
        print(f"  - 평균 익절: {avg_profit:,.2f} KRW")
        print(f"  - 평균 손절: {avg_loss:,.2f} KRW")
        print("--------------------------------------------------")

    def run_simulation(self):
        self._load_model()
        print("🚀 고빈도 스캘핑 전략 시뮬레이션을 시작합니다...")

        # 1. 데이터 로드
        all_data = []
        for ticker in SCALPING_TARGET_COINS:
            cache_path = os.path.join(self.cache_dir, f"{ticker.replace('/', '_')}_1m.feather")
            if os.path.exists(cache_path):
                df = pd.read_feather(cache_path).set_index('timestamp')
                df = df[(df.index >= self.start_date) & (df.index <= self.end_date)]
                df['ticker'] = ticker
                all_data.append(df)
        
        if not all_data:
            print("오류: 시뮬레이션할 데이터가 없습니다.")
            return

        df_full = pd.concat(all_data).sort_index()
        print(f"  총 {len(df_full)}개의 1분봉 데이터로 시뮬레이션을 시작합니다.")

        # 2. 벡터화된 시뮬레이션
        features = [
            'RSI_14', 'BBL_20', 'BBM_20', 'BBU_20', 
            'MACD_12_26_9', 'MACDH_12_26_9', 'MACDS_12_26_9'
        ]
        
        # 일괄 예측
        scaled_features = self.scaler.transform(df_full[features])
        df_full['prediction'] = self.model.predict(scaled_features)
        
        # 매수 신호만 필터링
        buy_signals = df_full[df_full['prediction'] == 1].copy()

        trades = []
        capital = self.initial_capital
        last_exit_time = pd.Timestamp.min

        for index, row in buy_signals.iterrows():
            if index < last_exit_time:
                continue

            capital_for_trade = capital * 0.5
            if capital_for_trade < 5000:
                continue

            entry_time = index
            entry_price = row['close']
            ticker = row['ticker']
            
            take_profit_price = entry_price * 1.005
            stop_loss_price = entry_price * 0.996

            # 효율적인 매도 시점 탐색
            future_df = df_full.loc[entry_time:].query("ticker == @ticker")
            
            tp_hits = future_df[future_df['high'] >= take_profit_price]
            tp_time = tp_hits.index.min() if not tp_hits.empty else pd.Timestamp.max

            sl_hits = future_df[future_df['low'] <= stop_loss_price]
            sl_time = sl_hits.index.min() if not sl_hits.empty else pd.Timestamp.max

            exit_price = None
            exit_time = None

            if tp_time < sl_time:
                exit_price = take_profit_price
                exit_time = tp_time
            elif sl_time < tp_time:
                exit_price = stop_loss_price
                exit_time = sl_time
            
            if exit_time and exit_time != pd.Timestamp.max:
                pnl = (exit_price - entry_price) / entry_price * capital_for_trade * (1 - 0.0005 * 2)
                capital += pnl
                trades.append({
                    'entry_time': entry_time,
                    'exit_time': exit_time,
                    'ticker': ticker,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl': pnl
                })
                last_exit_time = exit_time

        # 3. 최종 리포트 생성
        self._generate_report(trades, capital)

if __name__ == '__main__':
    backtester = AdvancedBacktester(start_date='2023-01-01', end_date='2023-12-31', initial_capital=50000)
    backtester.run_simulation()