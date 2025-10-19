import math

class RiskManager:
    """
    Calculates the optimal position size based on the Kelly Criterion,
    applying safety constraints.
    """
    def __init__(self, half_kelly=True, max_position_pct=0.25):
        """
        Initializes the risk manager.

        Args:
            half_kelly (bool): Whether to use half-Kelly sizing for safety.
            max_position_pct (float): The absolute maximum percentage of capital
                                      to allocate to a single trade (e.g., 0.25 for 25%).
        """
        self.half_kelly = half_kelly
        self.max_position_pct = max_position_pct

    def get_position_size_pct(self, p_win, win_loss_ratio=1.0):
        """
        Calculates the percentage of capital to allocate for a trade.

        Args:
            p_win (float): The model's predicted probability of winning (e.g., 0.62).
            win_loss_ratio (float): The expected ratio of profit vs. loss. Default is 1.0.

        Returns:
            float: The percentage of capital to allocate (e.g., 0.06 for 6%).
        """
        if p_win <= 0.5:
            return 0.0  # No edge, do not trade

        # Calculate Kelly Criterion fraction
        # Kelly % = P(win) - (P(loss) / Win/Loss Ratio)
        p_loss = 1.0 - p_win
        kelly_pct = p_win - (p_loss / win_loss_ratio)

        if kelly_pct <= 0:
            return 0.0

        # Apply safety factor (Half-Kelly)
        if self.half_kelly:
            kelly_pct /= 2.0

        # Enforce the absolute maximum position size cap
        final_pct = min(kelly_pct, self.max_position_pct)

        return final_pct
