import pandas as pd
import numpy as np
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
                df['ticker'] = ticker # 어떤 코인인지 식별자 추가
                all_data.append(df)
        
        if not all_data:
            print("오류: 시뮬레이션할 데이터가 없습니다.")
            return

        df_full = pd.concat(all_data).sort_index()
        print(f"  총 {len(df_full)}개의 1분봉 데이터로 시뮬레이션을 시작합니다.")

        # 2. 시뮬레이션 루프
        capital = self.initial_capital
        trades = []
        features = [
            'RSI_14', 'BBL_20', 'BBM_20', 'BBU_20', 
            'MACD_12_26_9', 'MACDH_12_26_9', 'MACDS_12_26_9'
        ]
        
        i = 0
        while i < len(df_full):
            row = df_full.iloc[i]
            
            # 예측
            scaled_features = self.scaler.transform(row[features].to_frame().T)
            prediction = self.model.predict(scaled_features)[0]

            if prediction == 1: # 매수 신호
                capital_for_trade = capital * 0.5 # 가용 자본의 50% 사용
                if capital_for_trade < 5000:
                    i += 1
                    continue

                entry_price = row['close']
                entry_time = row.name
                ticker = row['ticker']
                
                take_profit_price = entry_price * 1.005
                stop_loss_price = entry_price * 0.996

                # OCO 시뮬레이션 (향후 데이터 탐색)
                exit_price = None
                exit_time = None
                for j in range(i + 1, len(df_full)):
                    future_row = df_full.iloc[j]
                    if future_row['ticker'] != ticker: continue # 다른 코인 데이터는 무시

                    if future_row['high'] >= take_profit_price:
                        exit_price = take_profit_price
                        exit_time = future_row.name
                        break
                    if future_row['low'] <= stop_loss_price:
                        exit_price = stop_loss_price
                        exit_time = future_row.name
                        break
                
                if exit_price is not None:
                    pnl = (exit_price - entry_price) / entry_price * capital_for_trade * (1 - 0.0005 * 2) # 수수료 2번
                    capital += pnl
                    trades.append({
                        'entry_time': entry_time,
                        'exit_time': exit_time,
                        'ticker': ticker,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'pnl': pnl
                    })
                    # 거래 종료 시점으로 인덱스 점프
                    loc = df_full.index.get_loc(exit_time)
                    if isinstance(loc, slice):
                        i = loc.stop
                    else:
                        i = loc
                else:
                    i += 1 # 거래가 종료되지 않으면 다음 분으로
            else:
                i += 1

        # 3. 최종 리포트 생성
        self._generate_report(trades, capital)

if __name__ == '__main__':
    backtester = AdvancedBacktester(start_date='2023-01-01', end_date='2023-12-31', initial_capital=50000)
    backtester.run_simulation()