import os
from dotenv import load_dotenv
import ccxt

class UpbitService:
    def __init__(self):
        load_dotenv(os.path.join(os.path.dirname(__file__), '..', 'config', '.env'))
        self.access_key = os.getenv('UPBIT_ACCESS_KEY')
        self.secret_key = os.getenv('UPBIT_SECRET_KEY')
        self.exchange = None

        if not self.access_key or not self.secret_key:
            raise ValueError("UPBIT_ACCESS_KEY and UPBIT_SECRET_KEY must be set in config/.env")

    def connect(self):
        """
        ccxt 라이브러리를 사용하여 업비트 거래소 객체를 초기화하고 인증을 완료합니다.
        """
        self.exchange = ccxt.upbit({
            'apiKey': self.access_key,
            'secret': self.secret_key,
            'options': {
                'defaultType': 'spot',
            },
        })
        print("Upbit exchange connected successfully.")
        return self.exchange

    def get_balance(self):
        """
        현재 계정의 전체 자산 잔고(KRW 및 보유 코인)를 조회하여 딕셔너리 형태로 반환합니다.
        """
        if not self.exchange:
            raise ConnectionError("Exchange not connected. Call connect() first.")
        
        balances = self.exchange.fetch_balance()
        
        # KRW 잔고
        krw_balance = balances['free']['KRW'] if 'KRW' in balances['free'] else 0
        
        # 보유 코인 잔고
        coin_balances = {}
        for currency, balance_info in balances['free'].items():
            if currency != 'KRW' and balance_info > 0:
                coin_balances[currency] = balance_info
        
        return {
            'KRW': krw_balance,
            'coins': coin_balances
        }

    def get_total_capital(self):
        """
        현재 보유한 원화(KRW)와 보유 중인 모든 코인의 현재 가치를 평가하여 '총자산'을 원화로 계산해서 반환합니다.
        """
        if not self.exchange:
            raise ConnectionError("Exchange not connected. Call connect() first.")
        
        balances = self.get_balance()
        total_krw = balances['KRW']
        
        for currency, amount in balances['coins'].items():
            ticker = f"{currency}/KRW"
            current_price = self.get_current_price(ticker)
            if current_price:
                total_krw += amount * current_price
            else:
                print(f"Warning: Could not get current price for {ticker}. Excluding from total capital calculation.")
        
        return total_krw

    def get_current_price(self, ticker: str):
        """
        특정 티커(예: 'BTC/KRW')의 현재 가격을 조회하여 숫자(float)로 반환합니다.
        """
        if not self.exchange:
            raise ConnectionError("Exchange not connected. Call connect() first.")
        
        try:
            ticker_data = self.exchange.fetch_ticker(ticker)
            return float(ticker_data['last'])
        except Exception as e:
            print(f"Error fetching price for {ticker}: {e}")
            return None

    def cancel_all_orders(self, symbol: str):
        """
        특정 심볼의 모든 미체결 주문을 취소합니다.
        """
        if not self.exchange:
            raise ConnectionError("Exchange not connected. Call connect() first.")
        
        try:
            orders = self.exchange.fetch_open_orders(symbol)
            for order in orders:
                self.exchange.cancel_order(order['id'], symbol)
                print(f"Cancelled order {order['id']} for {symbol}.")
            return True
        except Exception as e:
            print(f"Error cancelling all orders for {symbol}: {e}")
            return False

    def create_market_sell_order(self, symbol: str, amount: float):
        """
        특정 심볼의 코인을 시장가로 전량 매도합니다.
        """
        if not self.exchange:
            raise ConnectionError("Exchange not connected. Call connect() first.")
        
        try:
            order = self.exchange.create_market_sell_order(symbol, amount)
            print(f"Placed MARKET SELL order: {amount:.4f} {symbol.split('/')[0]} at market price. Order ID: {order['id']}")
            return order
        except Exception as e:
            print(f"Error placing MARKET SELL order for {symbol} with amount {amount}: {e}")
            return None

    def create_market_buy_order(self, symbol: str, amount_krw: float):
        """
        특정 심볼의 코인을 시장가로 매수합니다.
        amount_krw는 매수할 원화 금액입니다.
        """
        if not self.exchange:
            raise ConnectionError("Exchange not connected. Call connect() first.")
        
        try:
            # 시장가 매수는 보통 amount_krw를 지정하여 해당 금액만큼 매수합니다.
            # ccxt의 create_market_buy_order는 amount (코인 수량)를 인자로 받으므로,
            # 현재 가격을 조회하여 코인 수량을 계산해야 합니다.
            current_price = self.get_current_price(symbol)
            if current_price is None:
                print(f"Error: Could not get current price for market buy of {symbol}.")
                return None
            
            amount_coin = amount_krw / current_price
            
            order = self.exchange.create_market_buy_order(symbol, amount_coin)
            print(f"Placed MARKET BUY order: {amount_krw:,.0f} KRW worth of {symbol.split('/')[0]} at market price. Order ID: {order['id']}")
            return order
        except Exception as e:
            print(f"Error placing MARKET BUY order for {symbol} with amount_krkrw {amount_krw}: {e}")
            return None

if __name__ == '__main__':
    # .env 파일 생성 (테스트용)
    env_path = os.path.join(os.path.dirname(__file__), '..', 'config', '.env')
    if not os.path.exists(env_path):
        with open(env_path, 'w') as f:
            f.write("""UPBIT_ACCESS_KEY=YOUR_ACCESS_KEY
UPBIT_SECRET_KEY=YOUR_SECRET_KEY""")
        print(f"Created a dummy .env file at {env_path}. Please replace YOUR_ACCESS_KEY and YOUR_SECRET_KEY with actual values.")

    # 사용 예시 (실제 API 키 필요)
    try:
        upbit_service = UpbitService()
        upbit_service.connect()

        # 잔고 조회
        balance = upbit_service.get_balance()
        print(f"Current Balance: {balance}")

        # 특정 코인 가격 조회
        btc_price = upbit_service.get_current_price('BTC/KRW')
        if btc_price:
            print(f"Current BTC/KRW Price: {btc_price}")
        
        eth_price = upbit_service.get_current_price('ETH/KRW')
        if eth_price:
            print(f"Current ETH/KRW Price: {eth_price}")

    except ValueError as e:
        print(f"Configuration Error: {e}")
    except ConnectionError as e:
        print(f"Connection Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
