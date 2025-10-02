import ccxt
import pandas as pd

class Backtester:
    def __init__(self, ticker: str, start_date: str, end_date: str, initial_capital: float = 1_000_000):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.exchange = ccxt.upbit() # Public client for fetching OHLCV data
        self.ohlcv_data = []

    def _fetch_ohlcv_data(self):
        """
        ccxt를 이용해 업비트에서 특정 티커의 일봉(daily) OHLCV 데이터를 특정 기간만큼 가져옵니다.
        """
        print(f"Fetching OHLCV data for {self.ticker} from {self.start_date} to {self.end_date}...")
        
        since = self.exchange.parse8601(self.start_date + 'T00:00:00Z')
        end_timestamp = self.exchange.parse8601(self.end_date + 'T23:59:59Z')
        
        all_ohlcv = []
        while since < end_timestamp:
            try:
                # fetch_ohlcv는 limit만큼 데이터를 가져오므로, 반복적으로 호출하여 전체 기간 데이터를 가져옵니다.
                # Upbit의 경우, 200개씩 가져오는 것이 일반적입니다.
                ohlcv = self.exchange.fetch_ohlcv(self.ticker, '1d', since, limit=200)
                if not ohlcv:
                    break
                all_ohlcv.extend(ohlcv)
                since = ohlcv[-1][0] + 86400000 # 다음 날짜부터 시작 (1일 = 86400000 ms)
            except Exception as e:
                print(f"Error fetching OHLCV data: {e}")
                break
        
        # 중복 제거 및 시간 순 정렬
        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.drop_duplicates(subset=['datetime']).sort_values(by='datetime')
        
        # 시작일과 종료일 범위에 맞게 필터링
        df = df[(df['datetime'] >= self.start_date) & (df['datetime'] <= self.end_date + ' 23:59:59')]
        
        self.ohlcv_data = df.to_dict('records')
        print(f"Fetched {len(self.ohlcv_data)} daily OHLCV data points.")
        return self.ohlcv_data

    def _generate_grids(self, lower_price: float, upper_price: float, grid_count: int):
        """
        그리드 가격 라인을 생성합니다. (strategies/grid_trading.py의 로직 재사용)
        """
        price_range = upper_price - lower_price
        if grid_count <= 0:
            return []
        interval = price_range / (grid_count + 1)
        
        grids = []
        for i in range(1, grid_count + 1):
            grids.append(lower_price + i * interval)
        
        return sorted(grids) # 오름차순으로 정렬

    def _calculate_mdd(self, portfolio_values: list):
        """
        최대 낙폭 (Maximum Drawdown)을 계산합니다.
        """
        if not portfolio_values:
            return 0

        peak = portfolio_values[0]
        mdd = 0
        for value in portfolio_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            if drawdown > mdd:
                mdd = drawdown
        return mdd * 100

    def run_test(self, lower_price: float, upper_price: float, grid_count: int, order_amount_krw: float):
        """
        가져온 데이터를 바탕으로 그리드 전략을 시뮬레이션하고 결과를 출력합니다.
        """
        if not self.ohlcv_data:
            self._fetch_ohlcv_data()
            if not self.ohlcv_data:
                print("No OHLCV data available for backtesting.")
                return

        krw_balance = self.initial_capital
        coin_balance = 0.0
        bought_positions = [] # [{'price': buy_price, 'amount': coin_amount}]
        
        trade_count = 0
        win_trades = 0
        
        portfolio_values = [self.initial_capital]
        
        grids = self._generate_grids(lower_price, upper_price, grid_count)
        
        # 그리드 상태 추적: 각 그리드 라인에서 마지막으로 발생한 이벤트 (None, 'buy', 'sell')
        # 이를 통해 한 번 매수/매도된 그리드 라인에서 다시 같은 방향의 주문이 나가는 것을 방지
        grid_status = {grid_price: None for grid_price in grids} 
        
        print(f"Initial Capital: {self.initial_capital:,.0f} KRW")
        print(f"Grids: {grids}")

        last_close_price = None

        for i, data in enumerate(self.ohlcv_data):
            current_date = data['datetime'].strftime('%Y-%m-%d')
            current_close_price = data['close']
            
            if last_close_price is None:
                last_close_price = current_close_price
                portfolio_values.append(krw_balance + coin_balance * current_close_price)
                continue

            # Simulate Buy Orders
            for grid_price in grids:
                # 가격이 그리드 라인 아래로 하향 돌파했을 때 (매수 기회)
                if last_close_price >= grid_price and current_close_price < grid_price:
                    if grid_status[grid_price] != 'buy': # 해당 그리드에서 아직 매수하지 않았다면
                        if krw_balance >= order_amount_krw:
                            amount_to_buy = order_amount_krw / current_close_price
                            krw_balance -= order_amount_krw
                            coin_balance += amount_to_buy
                            bought_positions.append({'price': current_close_price, 'amount': amount_to_buy})
                            trade_count += 1
                            grid_status[grid_price] = 'buy' # 매수 완료 표시
                            print(f"[{current_date}] BUY at {current_close_price:,.2f} (Grid: {grid_price:,.2f}). Coin: {coin_balance:.4f}, KRW: {krw_balance:,.0f}")
                        # else:
                            # print(f"[{current_date}] Not enough KRW to buy at {current_close_price:,.2f} (Grid: {grid_price:,.2f})")
                # 가격이 그리드 라인 위로 상향 돌파했을 때 (매수 그리드 초기화)
                elif last_close_price < grid_price and current_close_price >= grid_price:
                    if grid_status[grid_price] == 'buy':
                        grid_status[grid_price] = None # 매수 그리드 초기화 (다시 매수 가능)

            # Simulate Sell Orders
            for grid_price in grids:
                # 가격이 그리드 라인 위로 상향 돌파했을 때 (매도 기회)
                if last_close_price <= grid_price and current_close_price > grid_price:
                    if grid_status[grid_price] != 'sell': # 해당 그리드에서 아직 매도하지 않았다면
                        if len(bought_positions) > 0:
                            # 가장 오래된 매수 포지션 (FIFO)을 매도
                            position_to_sell = bought_positions.pop(0)
                            sell_price = current_close_price
                            amount_to_sell = position_to_sell['amount']
                            
                            profit = (sell_price - position_to_sell['price']) * amount_to_sell
                            krw_balance += sell_price * amount_to_sell # 매도 금액 추가
                            coin_balance -= amount_to_sell
                            trade_count += 1
                            grid_status[grid_price] = 'sell' # 매도 완료 표시
                            if profit > 0:
                                win_trades += 1
                            print(f"[{current_date}] SELL at {sell_price:,.2f} (Grid: {grid_price:,.2f}). Profit: {profit:,.2f} KRW. Coin: {coin_balance:.4f}, KRW: {krw_balance:,.0f}")
                        # else:
                            # print(f"[{current_date}] No coin to sell at {current_close_price:,.2f} (Grid: {grid_price:,.2f})")
                # 가격이 그리드 라인 아래로 하향 돌파했을 때 (매도 그리드 초기화)
                elif last_close_price > grid_price and current_close_price <= grid_price:
                    if grid_status[grid_price] == 'sell':
                        grid_status[grid_price] = None # 매도 그리드 초기화 (다시 매도 가능)

            last_close_price = current_close_price
            portfolio_values.append(krw_balance + coin_balance * current_close_price)

        # 백테스트 종료 후 남은 코인 모두 매도
        final_price = self.ohlcv_data[-1]['close'] if self.ohlcv_data else last_close_price
        if coin_balance > 0 and final_price is not None:
            total_bought_cost = sum(pos['price'] * pos['amount'] for pos in bought_positions)
            total_bought_amount = sum(pos['amount'] for pos in bought_positions)
            
            if total_bought_amount > 0:
                avg_buy_price_remaining = total_bought_cost / total_bought_amount
                profit_remaining = (final_price - avg_buy_price_remaining) * coin_balance
                krw_balance += final_price * coin_balance
                coin_balance = 0.0
                trade_count += 1 # 마지막 정리 매도도 거래로 간주
                if profit_remaining > 0:
                    win_trades += 1
                print(f"[END] Final SELL at {final_price:,.2f}. Profit from remaining: {profit_remaining:,.2f} KRW. Final KRW: {krw_balance:,.0f}")
            else: # 남은 코인이 있지만, bought_positions에 기록되지 않은 경우 (예: 초기 자본으로 코인 보유 시작)
                krw_balance += final_price * coin_balance
                coin_balance = 0.0
                print(f"[END] Liquidated remaining coin at {final_price:,.2f}. Final KRW: {krw_balance:,.0f}")


        final_portfolio_value = krw_balance
        
        # 결과 계산
        total_return = ((final_portfolio_value - self.initial_capital) / self.initial_capital) * 100
        win_rate = (win_trades / trade_count) * 100 if trade_count > 0 else 0
        mdd = self._calculate_mdd(portfolio_values)

        print("\n--- Backtest Results ---")
        print(f"Initial Capital: {self.initial_capital:,.0f} KRW")
        print(f"Final Portfolio Value: {final_portfolio_value:,.0f} KRW")
        print(f"Total Return: {total_return:.2f}%")
        print(f"Total Trades: {trade_count}")
        print(f"Win Rate: {win_rate:.2f}%")
        print(f"Maximum Drawdown (MDD): {mdd:.2f}%")

if __name__ == '__main__':
    # 백테스터 사용 예시
    # 실제 데이터는 Upbit API를 통해 가져오므로, API 키 없이도 동작합니다.
    # 하지만 너무 많은 데이터를 요청하면 Rate Limit에 걸릴 수 있습니다.
    
    # 예시 파라미터 (실제 사용 시 조정 필요)
    ticker = 'BTC/KRW'
    start_date = '2023-01-01'
    end_date = '2023-12-31'
    initial_capital = 1_000_000 # 100만원

    # 그리드 전략 파라미터
    lower_price = 20_000_000.0 # 2천만원
    upper_price = 60_000_000.0 # 6천만원
    grid_count = 10
    order_amount_krw = 100_000.0 # 각 그리드 라인에서 10만원씩 주문

    backtester = Backtester(ticker, start_date, end_date, initial_capital)
    backtester.run_test(lower_price, upper_price, grid_count, order_amount_krw)
