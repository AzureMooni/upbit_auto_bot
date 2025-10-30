import math

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
