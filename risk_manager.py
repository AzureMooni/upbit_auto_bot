class RiskManager:
    """
    Manages risk for trades using strategies like the Kelly Criterion.
    """

    def calculate_kelly_fraction(
        self, win_rate: float, avg_profit: float, avg_loss: float
    ) -> float:
        """
        Calculates the optimal fraction of capital to bet using the Kelly Criterion.

        Args:
            win_rate (float): The probability of winning a trade (e.g., 0.6 for 60%).
            avg_profit (float): The average profit from a winning trade.
            avg_loss (float): The average loss from a losing trade (as a positive value).

        Returns:
            float: The Kelly fraction. Returns 0 if the edge is not positive.
        """
        if avg_loss == 0 or avg_profit <= 0:
            return 0.0

        win_loss_ratio = avg_profit / avg_loss

        # Kelly Criterion formula: K = W - ((1 - W) / R)
        # W = win_rate
        # R = win_loss_ratio
        kelly_fraction = win_rate - ((1 - win_rate) / win_loss_ratio)

        # Bet nothing if the edge is negative, and cap at 100%
        return max(0.0, min(kelly_fraction, 1.0))
