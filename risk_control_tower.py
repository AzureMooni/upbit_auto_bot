import pandas as pd
from risk_manager import RiskManager

class RiskControlTower:
    """
    AI 위험 관리 위원회. 모든 거래 결정을 최종 승인하고 자본을 배분하며,
    포트폴리오의 위험을 총괄하는 중앙 통제 모듈.
    """
    def __init__(self, mdd_threshold: float = -0.15):
        """
        Args:
            mdd_threshold (float): 서킷 브레이커가 발동하는 최대 낙폭 임계값. (기본값: -15%)
        """
        self.mdd_threshold = mdd_threshold
        self.risk_manager = RiskManager()
        print(f"✅ AI 위험 관리 위원회 활성화. MDD 임계값: {self.mdd_threshold:.2%}")

    def check_mdd_circuit_breaker(self, portfolio_history: pd.Series) -> bool:
        """
        실시간으로 포트폴리오의 최대 낙폭(MDD)을 계산하여 서킷 브레이커 발동 여부를 결정합니다.

        Args:
            portfolio_history (pd.Series): 포트폴리오 순자산 가치의 시계열 데이터.

        Returns:
            bool: MDD 임계값을 초과하면 True(서킷 브레이커 발동), 아니면 False를 반환.
        """
        if len(portfolio_history) < 2:
            return False

        peak = portfolio_history.cummax()
        drawdown = (portfolio_history - peak) / peak
        current_mdd = drawdown.min()

        if current_mdd < self.mdd_threshold:
            print(f"🚨 비상! 포트폴리오 최대 낙폭({current_mdd:.2%})이 임계값({self.mdd_threshold:.2%})을 초과했습니다!")
            print("🚨 서킷 브레이커를 발동하여 모든 거래를 중단하고 포지션을 청산합니다.")
            return True
        return False

    def determine_investment_size(self, win_rate: float, avg_profit: float, avg_loss: float, prediction_confidence: float, sentiment_score: float) -> float:
        """
        전문가 AI의 성과, 예측 확신도, 시장 감성 지수를 종합하여 최종 투자 비율을 결정합니다.

        Args:
            win_rate (float): 전문가 AI의 승률.
            avg_profit (float): 전문가 AI의 평균 이익.
            avg_loss (float): 전문가 AI의 평균 손실.
            prediction_confidence (float): AI 예측에 대한 확신도 (0.0 ~ 1.0).
            sentiment_score (float): 시장 감성 지수 (-1.0 ~ +1.0).

        Returns:
            float: 최종적으로 결정된 투자 자본 비율 (0.0 ~ 1.0).
        """
        # 1. 켈리 비율 계산
        kelly_fraction = self.risk_manager.calculate_kelly_fraction(win_rate, avg_profit, avg_loss)
        if kelly_fraction <= 0:
            print("  - [RCT] 켈리 비율 <= 0. 통계적 우위가 없어 투자를 중단합니다.")
            return 0.0

        # 2. 확신도와 감성 지수를 이용한 동적 조정
        # 감성 지수가 긍정적일수록(>0) 베팅 강도를 높이고, 부정적일수록(<0) 낮춤
        sentiment_factor = (1 + sentiment_score) / 2  # 0.0 ~ 1.0 사이로 정규화

        # 최종 투자 비율 = 기본 켈리 비율 * 예측 확신도 * 감성 지수 팩터
        final_fraction = kelly_fraction * prediction_confidence * sentiment_factor
        
        print(f"  - [RCT] 켈리 비율: {kelly_fraction:.4f}, 예측 확신도: {prediction_confidence:.4f}, 감성 팩터: {sentiment_factor:.4f}")
        print(f"  - [RCT] 최종 투자 비율: {final_fraction:.4f}")

        # 너무 작은 규모의 거래 방지 및 최대 비율 제한 (예: 켈리값의 50%까지만, 최대 25%)
        capped_fraction = min(final_fraction, kelly_fraction * 0.5, 0.25)
        
        if capped_fraction < 0.01: # 최소 투자 비율 (1%) 미만이면 거래하지 않음
            print(f"  - [RCT] 조정된 비율({capped_fraction:.4f})이 최소 투자 비율 미만이라 거래를 건너뜁니다.")
            return 0.0

        print(f"  - [RCT] 최종 적용 비율 (안전장치 적용): {capped_fraction:.4f}")
        return capped_fraction
