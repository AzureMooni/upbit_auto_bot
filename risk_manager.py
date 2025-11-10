import math

def get_position_size_ratio(regime: str, normalized_atr: float, natr_ma: float) -> float:
    """
    시장 상황에 따라 동적으로 자산 배분 비율을 결정합니다.
    이것은 플레이스홀더 함수이며, 실제 로직은 더 정교하게 구현되어야 합니다.
    """
    if "BULLISH" in regime:
        return 0.8 # 강세장에서는 더 공격적으로
    elif "BEARISH" in regime:
        return 0.2 # 약세장에서는 보수적으로
    elif "SIDEWAYS" in regime:
        return 0.5 # 횡보장에서는 중간 정도로
    else:
        return 0.0 # 그 외 (예: 불확실) 에는 투자하지 않음

class RiskManager:
    """ Calculates the optimal position size based on the Kelly Criterion, applying safety constraints. """
    def __init__(self, half_kelly=True, max_position_pct=0.25):
        self.half_kelly = half_kelly
        self.max_position_pct = max_position_pct

    def get_position_size_pct(self, p_win, win_loss_ratio=1.0):
        if p_win <= 0.5:
            return 0.0
        p_loss = 1.0 - p_win
        kelly_pct = p_win - (p_loss / win_loss_ratio)
        if kelly_pct <= 0:
            return 0.0
        if self.half_kelly:
            kelly_pct /= 2.0
        final_pct = min(kelly_pct, self.max_position_pct)
        return final_pct