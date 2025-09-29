import ccxt
import asyncio
import pandas as pd
import pandas_ta as ta
from core.exchange import UpbitService
from scanner import find_hot_coin # find_hot_coin은 추세 조건을 포함하므로, 여기서는 진입 신호로 활용

class TrendFollower:
    def __init__(self, upbit_service: UpbitService, ticker: str, allocated_capital: float, atr_multiplier: float = 3.0):
        self.upbit_service = upbit_service
        self.ticker = ticker
        self.allocated_capital = allocated_capital
        self.position_held = False
        self.entry_price = 0.0
        self.high_water_mark = 0.0
        self.trailing_stop_price = 0.0
        self.base_currency = ticker.split('/')[0]
        self.atr_period = 14 # ATR 기간 설정
        self.atr_multiplier = atr_multiplier

        print(f"TrendFollower initialized for {self.ticker} with dynamic trailing stop-loss (ATR period: {self.atr_period}). Allocated capital: {self.allocated_capital:,.0f} KRW.")

    async def _calculate_atr(self, ticker: str, exchange: ccxt.Exchange):
        """
        1시간 봉 데이터를 가져와 14기간 ATR 값을 계산합니다.
        """
        try:
            ohlcv_1h = await exchange.fetch_ohlcv(ticker, '1h', limit=self.atr_period + 10) # ATR 계산에 필요한 충분한 데이터
            if not ohlcv_1h or len(ohlcv_1h) < self.atr_period:
                print(f"Not enough OHLCV data for {ticker} to calculate ATR. (Need at least {self.atr_period}, got {len(ohlcv_1h) if ohlcv_1h else 0})")
                return None
            
            df = pd.DataFrame(ohlcv_1h, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)

            df.ta.atr(length=self.atr_period, append=True, high='high', low='low', close='close')
            atr_values = df[f'ATR_{self.atr_period}']
            
            if len(atr_values) == 0:
                return None

            return atr_values[-1]

        except Exception as e:
            print(f"Error calculating ATR for {ticker}: {e}")
            return None

    async def run(self, interval_seconds: int = 5):
        """
        추세 추종 전략을 실행합니다.
        """
        print(f"Starting TrendFollower for {self.ticker}...")
        while True:
            try:
                current_price = await self.upbit_service.get_current_price(self.ticker)
                if current_price is None:
                    print(f"Could not fetch current price for {self.ticker}. Retrying...")
                    await asyncio.sleep(interval_seconds)
                    continue

                print(f"Current price for {self.ticker}: {current_price:,.2f} KRW")

                if not self.position_held:
                    print(f"No position held. Attempting to buy {self.ticker} at market price.")
                    
                    order = await self.upbit_service.create_market_buy_order(self.ticker, self.allocated_capital)
                    
                    if order and order['status'] == 'closed': # 주문이 체결되었다면
                        self.position_held = True
                        self.entry_price = current_price
                        self.high_water_mark = current_price
                        
                        # 초기 트레일링 스탑 가격 설정 (ATR 기반)
                        atr_value = await self._calculate_atr(self.ticker, self.upbit_service.exchange)
                        if atr_value is not None:
                            self.trailing_stop_price = current_price - (self.atr_multiplier * atr_value)
                            print(f"🟢 매수 완료: {self.ticker} at {self.entry_price:,.2f} KRW. Dynamic Trailing Stop-Loss set at {self.trailing_stop_price:,.2f} KRW ({self.atr_multiplier} * ATR: {self.atr_multiplier * atr_value:,.2f}).")
                        else:
                            print(f"❌ ATR 계산 실패. 트레일링 스탑 설정 불가. {self.ticker} 재시도...")
                            self.position_held = False # ATR 계산 실패 시 포지션 잡지 않음

                    else:
                        print(f"❌ 매수 실패 또는 미체결. {self.ticker} 재시도...")

                else: # 포지션을 보유 중인 경우
                    # 고점 업데이트 및 트레일링 스탑 조정
                    if current_price > self.high_water_mark:
                        self.high_water_mark = current_price
                        atr_value = await self._calculate_atr(self.ticker, self.upbit_service.exchange)
                        if atr_value is not None:
                            new_trailing_stop = self.high_water_mark - (self.atr_multiplier * atr_value)
                            if new_trailing_stop > self.trailing_stop_price: # 트레일링 스탑은 위로만 움직임
                                self.trailing_stop_price = new_trailing_stop
                                print(f"📈 고점 업데이트: {self.high_water_mark:,.2f} KRW. Dynamic Trailing Stop-Loss: {self.trailing_stop_price:,.2f} KRW ({self.atr_multiplier} * ATR: {self.atr_multiplier * atr_value:,.2f}).")
                        else:
                            print(f"❌ ATR 계산 실패. 트레일링 스탑 조정 불가.")

                    # 트레일링 스탑 로스 발동 조건 확인
                    if current_price <= self.trailing_stop_price:
                        print(f"🚨 트레일링 스탑 로스 발동! {self.ticker} 전량 시장가 매도.")
                        
                        balances = await self.upbit_service.get_balance()
                        amount_to_sell = balances['coins'].get(self.base_currency, 0)

                        if amount_to_sell > 0:
                            order = await self.upbit_service.create_market_sell_order(self.ticker, amount_to_sell)
                            if order and order['status'] == 'closed':
                                print(f"🔴 매도 완료: {self.ticker} 전량 시장가 매도. 손실/수익 확정.")
                                self.position_held = False
                            else:
                                print(f"❌ 매도 실패. {self.ticker} 재시도...")
                        else:
                            print(f"Warning: No {self.base_currency} to sell for trailing stop-loss.")
                        
                        self.position_held = False # 포지션 종료
                        self.entry_price = 0.0
                        self.high_water_mark = 0.0
                        self.trailing_stop_price = 0.0
                        await asyncio.sleep(interval_seconds * 5) # 다음 진입 기회를 위해 잠시 대기
                        continue # 다음 루프에서 다시 진입 기회 탐색

            except Exception as e:
                print(f"An error occurred in TrendFollower run loop: {e}")
            
            await asyncio.sleep(interval_seconds)

if __name__ == '__main__':
    import os
    from dotenv import load_dotenv
    import asyncio

    env_path = os.path.join(os.path.dirname(__file__), '..', 'config', '.env')
    if not os.path.exists(env_path):
        with open(env_path, 'w') as f:
            f.write("""UPBIT_ACCESS_KEY=YOUR_ACCESS_KEY
UPBIT_SECRET_KEY=YOUR_SECRET_KEY""")
        print(f"Created a dummy .env file at {env_path}. Please replace YOUR_ACCESS_KEY and UPBIT_SECRET_KEY with actual values.")
    load_dotenv(env_path)

    async def test_trend_follower():
        # 이 부분은 실제 UpbitService 인스턴스와 연동하여 테스트해야 합니다.
        # .env 파일에 UPBIT_ACCESS_KEY와 UPBIT_SECRET_KEY가 설정되어 있어야 합니다.
        try:
            # UpbitService 인스턴스 생성 및 연결 (실제 API 키 필요)
            upbit_service = UpbitService()
            await upbit_service.connect()

            # TrendFollower 인스턴스 생성
            ticker = 'BTC/KRW' # 예시 티커
            order_amount_krw = 50000 # 5만원 매수

            trend_follower = TrendFollower(upbit_service, ticker, order_amount_krw)
            
            print("TrendFollower example setup complete. To run, integrate into main.py and ensure API keys are set.")
            # await trend_follower.run(interval_seconds=10) # 실제 실행 시 주석 해제

        except Exception as e:
            print(f"An unexpected error occurred during TrendFollower setup: {e}")

    asyncio.run(test_trend_follower())
