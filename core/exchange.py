import ccxt.async_support as ccxt
import asyncio
import pandas as pd
import logging
import traceback

logger = logging.getLogger(__name__)

class UpbitService:
    def __init__(self, access_key: str, secret_key: str):
        self.exchange = ccxt.upbit({
            'apiKey': access_key,
            'secret': secret_key,
            'enableRateLimit': True,
        })
        print("Upbit exchange connected successfully.")

    async def connect(self):
        try:
            await self.exchange.load_markets()
            print("✅ Upbit 주문 실행 엔진 활성화.")
        except Exception as e:
            logger.fatal(f"[FATAL] Upbit 연결 실패: {e}", exc_info=True)
            await self.close()
            raise

    async def close(self):
        if self.exchange:
            await self.exchange.close()

    async def get_balance(self, currency: str):
        try:
            balances = await self.exchange.fetch_balance()
            return balances.get(currency, {}).get('free', 0)
        except Exception as e:
            logger.error(f"[ERROR] 잔고 조회 실패: {e}")
            print(traceback.format_exc())
            return None

    async def get_all_balances(self):
        try:
            balances = await self.exchange.fetch_balance()
            # FIX: 'info' is the float, not info['free']
            return {
                ticker: {'balance': info} 
                for ticker, info in balances['free'].items() if info > 0
            }
        except Exception as e:
            logger.error(f"[ERROR] 전체 잔고 조회 실패: {e}")
            print(traceback.format_exc())
            return None

    async def get_current_price(self, ticker: str):
        try:
            # FIX: Use 'KRW-BTC' format directly
            return (await self.exchange.fetch_ticker(ticker))['last']
        except Exception as e:
            logger.warning(f"[WARN] {ticker} 현재가 조회 실패: {e}")
            return None

    async def get_ohlcv(self, ticker: str, timeframe='1h', limit=200):
        try:
            # FIX: Use 'KRW-BTC' format directly
            ohlcv = await self.exchange.fetch_ohlcv(ticker, timeframe=timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            return df
        except Exception as e:
            logger.warning(f"[WARN] {ticker} OHLCV 데이터 가져오기 실패: {e}")
            return None

    async def create_market_buy_order(self, ticker, amount_krw):
        print(f"  - [EXEC] {ticker} 시장가 매수 주문 (금액: {amount_krw:.0f} KRW)")
        try:
            order = await self.exchange.create_market_buy_order_with_cost(ticker, amount_krw)
            print(f"  - [SUCCESS] 매수 주문 성공, ID: {order.get('id')}")
            return order
        except Exception as e:
            print(f"  - [ERROR] 매수 주문 실패: {e}")
            return None

    async def create_market_sell_order(self, ticker, amount_coin):
        print(f"  - [EXEC] {ticker} 시장가 매도 주문 (수량: {amount_coin})")
        try:
            order = await self.exchange.create_market_sell_order(ticker, amount_coin)
            print(f"  - [SUCCESS] 매도 주문 성공, ID: {order.get('id')}")
            return order
        except Exception as e:
            print(f"  - [ERROR] 매도 주문 실패: {e}")
            return None

    async def liquidate_all_positions(self, holdings):
        print("🚨 [RCT] 서킷 브레이커 발동! 모든 포지션 청산 시작...")
        for ticker, amount in holdings.items():
            await self.create_market_sell_order(ticker, amount)
