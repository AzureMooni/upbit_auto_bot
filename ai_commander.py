import asyncio
from market_diagnostics import MarketDiagnostics
from portfolio_manager import PortfolioManager
from strategies.scalping_bot import ScalpingBot
from core.exchange import UpbitService
import scanner

class AICommander:
    """
    AI 퀀트 펀드의 총사령관.
    시장의 거시 지표를 분석하여, 각 전략(부대)에 자산을 동적으로 배분하고 임무를 지시합니다.
    """
    def __init__(self, total_capital: float):
        self.total_capital = total_capital
        self.upbit_service = UpbitService()
        self.market_diagnostics = MarketDiagnostics()
        self.active_squads = [] # 현재 실행 중인 부대(전략)들의 태스크를 관리

        print(f"🤖 AI 총사령관 부임. 총 운용 자산: {self.total_capital:,.0f} KRW")

    async def _stop_all_squads(self):
        """현재 진행 중인 모든 전략 태스크를 중지시킵니다."""
        if not self.active_squads:
            return
        print("\n- 모든 부대(전략)에 중지 명령 하달...")
        for task in self.active_squads:
            task.cancel()
        await asyncio.gather(*self.active_squads, return_exceptions=True)
        self.active_squads = []
        print("  - 모든 부대 임무 중지 완료.")

    async def run(self):
        """AI 총사령관의 메인 지휘 루프. 매일 자산 배분을 재조정합니다."""
        await self.upbit_service.connect()

        while True:
            try:
                # 1. 시장 거시 상황 분석
                market_summary = await self.market_diagnostics.get_market_summary()
                volatility = market_summary.get("volatility_index", 50) # 기본값 50
                fear_greed = market_summary.get("fear_greed_index", 50) # 기본값 50

                # 2. 자산 배분 결정
                main_squad_ratio = 0.7  # 주력 부대 기본 비율
                scalping_squad_ratio = 0.1 # 단기 부대 기본 비율

                if volatility < 40 and fear_greed > 60: # 안정적인 상승장
                    print("  - [진단] 안정적인 상승장. 주력 부대 비중 상향.")
                    main_squad_ratio = 0.9
                    scalping_squad_ratio = 0.1
                elif volatility > 70 and fear_greed > 50: # 변동성 높은 상승/중립장
                    print("  - [진단] 변동성 확대. 단기 부대 비중 상향.")
                    main_squad_ratio = 0.5
                    scalping_squad_ratio = 0.5
                elif fear_greed < 20 or fear_greed > 85 or volatility < 20:
                    print("  - [진단] 극단적 심리 또는 낮은 변동성. 투자 비중 축소.")
                    main_squad_ratio = 0.1 # 현금 보유 비중 확대
                    scalping_squad_ratio = 0.1
                
                cash_ratio = 1.0 - main_squad_ratio - scalping_squad_ratio
                print(f"\n- [자산 배분] 주력(중장기): {main_squad_ratio:.0%}, 단기(스캘핑): {scalping_squad_ratio:.0%}, 현금: {cash_ratio:.0%}")

                # 3. 기존 부대 임무 중지
                await self._stop_all_squads()

                # 4. 새로운 자산으로 부대 재창설 및 임무 부여
                main_squad_capital = self.total_capital * main_squad_ratio
                scalping_squad_capital = self.total_capital * scalping_squad_ratio

                squads_to_launch = []
                if main_squad_capital > 100000: # 최소 자본금
                    portfolio_manager = PortfolioManager(self.upbit_service, main_squad_capital, max_concurrent_trades=5)
                    squads_to_launch.append(portfolio_manager.run())
                
                if scalping_squad_capital > 50000: # 최소 자본금
                    # 스캘핑은 변동성이 큰 코인에 적합
                    hot_coins = await scanner.find_hot_coin_live(self.upbit_service.exchange, None, 'Bullish', top_n=1)
                    if hot_coins:
                        scalping_ticker = hot_coins[0]
                        scalping_bot = ScalpingBot(self.upbit_service, scalping_ticker, scalping_squad_capital, trade_amount=50000)
                        squads_to_launch.append(scalping_bot.run())

                if squads_to_launch:
                    print("\n- 전 부대, 임무 실행 개시!")
                    self.active_squads = [asyncio.create_task(s) for s in squads_to_launch]
                    # 다음 재배분까지 대기 (24시간)
                    await asyncio.sleep(86400)
                else:
                    print("\n- 투자 조건 불충족. 모든 부대 대기. 24시간 후 재분석.")
                    await asyncio.sleep(86400)

            except Exception as e:
                print(f"총사령관 지휘 루프 중 치명적 오류 발생: {e}")
                await asyncio.sleep(300) # 5분 후 재시도

async def main():
    commander = AICommander(total_capital=1_000_000)
    await commander.run()

if __name__ == "__main__":
    asyncio.run(main())