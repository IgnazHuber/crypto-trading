def calculate_position_size(capital, max_position_pct=0.10, max_risk_pct=0.03, atr=None):
    max_position = capital * max_position_pct
    max_risk_amount = capital * max_risk_pct
    if atr and atr > 0:
        volatility_adjusted = max_risk_amount / (atr * 1.5)
        return min(max_position, max(volatility_adjusted, 100))
    return max_position
