import ccxt.async_support as ccxt
import asyncio
import pandas as pd

class UpbitService:
    """ Manages all API interactions with the Upbit exchange using ccxt. """
    def __init__(self, access_key: str, secret_key: str):
        self.exchange = ccxt.upbit({
            'apiKey': access_key,
            'secret': secret_key,
            'enableRateLimit': True,
        })
        print("Upbit exchange connected successfully.")

    async def connect(self):
        try:
            # Test connection by fetching markets (a lightweight call)
            await self.exchange.load_markets()
            print("✅ Upbit 주문 실행 엔진 활성화.")
        except Exception as e:
            print(f"[FATAL] Upbit 연결 실패: {e}")
            await self.close()
            raise

    async def close(self):
        if self.exchange:
            await self.exchange.close()
            print("Upbit exchange connection closed.")

    async def get_balance(self, currency: str):
        try:
            balances = await self.exchange.fetch_balance()
            return balances.get(currency, {}).get('free', 0)
        except Exception as e:
            print(f"[ERROR] 잔고 조회 실패: {e}")
            return None

    async def get_all_balances(self):
        try:
            balances = await self.exchange.fetch_balance()
            # 'free' 밸런스가 있는 모든 자산 반환
            return {
                ticker: {'balance': info['free']}
                for ticker, info in balances['free'].items() if info > 0
            }
        except Exception as e:
            print(f"[ERROR] 전체 잔고 조회 실패: {e}")
            return None

    async def get_current_price(self, ticker: str):
        try:
            # Ticker must be in 'BTC/KRW' format for ccxt
            ccxt_ticker = ticker.replace("KRW-", "") + "/KRW"
            return (await self.exchange.fetch_ticker(ccxt_ticker))['last']
        except Exception:
            return None

    async def get_ohlcv(self, ticker: str, timeframe='1h', limit=200):
        try:
            # Ticker must be in 'BTC/KRW' format for ccxt
            ccxt_ticker = ticker.replace("KRW-", "") + "/KRW"
            ohlcv = await self.exchange.fetch_ohlcv(ccxt_ticker, timeframe=timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            return df
        except Exception as e:
            print(f"[ERROR] {ticker} OHLCV 데이터 가져오기 실패: {e}")
            return None

    async def create_market_buy_order(self, ticker, amount_krw):
        ccxt_ticker = ticker.replace("KRW-", "") + "/KRW"
        print(f"  - [EXEC] {ccxt_ticker} 시장가 매수 주문 (금액: {amount_krw:.0f} KRW)")
        try:
            # ccxt uses 'cost' for market buy orders in KRW
            order = await self.exchange.create_market_buy_order(ccxt_ticker, None, params={'cost': amount_krw})
            print(f"  - [SUCCESS] 매수 주문 성공, ID: {order.get('id')}")
            return order
        except Exception as e:
            print(f"  - [ERROR] 매수 주문 실패: {e}")
            return None

    async def create_market_sell_order(self, ticker, amount_coin):
        ccxt_ticker = ticker.replace("KRW-", "") + "/KRW"
        print(f"  - [EXEC] {ccxt_ticker} 시장가 매도 주문 (수량: {amount_coin})")
        try:
            order = await self.exchange.create_market_sell_order(ccxt_ticker, amount_coin)
            print(f"  - [SUCCESS] 매도 주문 성공, ID: {order.get('id')}")
            return order
        except Exception as e:
            print(f"  - [ERROR] 매도 주문 실패: {e}")
            return None

    async def liquidate_all_positions(self, holdings):
        print("🚨 [RCT] 서킷 브레이커 발동! 모든 포지션 청산 시작...")
        for ticker, amount in holdings.items():
            await self.create_market_sell_order(ticker, amount)