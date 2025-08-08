def volatility_filter(current_data):
    vol_ratio = current_data['atr'] / current_data['close']
    return 0.005 < vol_ratio < 0.05

def time_filter(timestamp):
    hour = timestamp.hour if hasattr(timestamp, 'hour') else 12
    return 4 <= hour <= 22

def regime_filter(current_data):
    return current_data['regime'] in ['trending', 'range']
