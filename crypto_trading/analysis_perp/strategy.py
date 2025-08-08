from .utils import volatility_filter, time_filter, regime_filter
from .risk_management import calculate_position_size

def simulate_advanced_strategy(df, patterns, initial_capital=10000,
                               max_position_pct=0.10, max_risk_pct=0.03):
    trades = []
    capital = initial_capital
    positions = []
    equity = []
    symbol = df.iloc[0].get('symbol', 'Unknown')
    trade_counter = 1

    for i in range(len(patterns)):
        if i >= len(df):
            continue
        row = patterns.iloc[i]
        current_price = df.iloc[i]['close']
        timestamp = row['timestamp']
        current_data = df.iloc[i]

        # Entry
        if (row['signal'] in ['Kaufen', 'Verkaufen']
                and len(positions) < 3
                and time_filter(timestamp)
                and volatility_filter(current_data)
                and regime_filter(current_data)):
            is_long = row['signal'] == 'Kaufen'
            max_loss_pct = 0.03
            max_gain_pct = 0.20
            stop_loss = current_price * (1 - max_loss_pct) if is_long else current_price * (1 + max_loss_pct)
            take_profit = current_price * (1 + max_gain_pct) if is_long else current_price * (1 - max_gain_pct)

            position_size_eur = calculate_position_size(capital, max_position_pct, max_risk_pct, current_data['atr'])
            position_size_eur = min(position_size_eur, capital)
            size_coins = position_size_eur / current_price if current_price > 0 else 0

            position = {
                'type': 'long' if is_long else 'short',
                'entry_price': current_price,
                'size_eur': position_size_eur,
                'size_coins': size_coins,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'max_loss': position_size_eur * max_loss_pct,
                'max_gain': position_size_eur * max_gain_pct,
                'timestamp': timestamp,
                'symbol': symbol,
                'entry_reason': row.get('pattern', row['signal']),
                'pyramid_levels': 0,
                'regime': current_data['regime'],
                'hour': timestamp.hour if hasattr(timestamp, 'hour') else 0,
                'weekday': timestamp.weekday() if hasattr(timestamp, 'weekday') else 0
            }
            positions.append(position)

        # Position Management and Exit
        for pos in positions[:]:
            if pos['type'] == 'long':
                current_pnl = (current_price - pos['entry_price']) * pos['size_coins']
            else:
                current_pnl = (pos['entry_price'] - current_price) * pos['size_coins']
            exit_price = current_price
            exit_reason = None
            profit = 0

            if pos['type'] == 'long':
                if current_price <= pos['stop_loss']:
                    exit_reason = 'stop_loss'
                    exit_price = pos['stop_loss']
                elif current_price >= pos['take_profit']:
                    exit_reason = 'take_profit'
                    exit_price = pos['take_profit']
            else:
                if current_price >= pos['stop_loss']:
                    exit_reason = 'stop_loss'
                    exit_price = pos['stop_loss']
                elif current_price <= pos['take_profit']:
                    exit_reason = 'take_profit'
                    exit_price = pos['take_profit']

            if exit_reason:
                if pos['type'] == 'long':
                    raw_profit = (exit_price - pos['entry_price']) * pos['size_coins']
                else:
                    raw_profit = (pos['entry_price'] - exit_price) * pos['size_coins']
                profit = min(raw_profit, pos['max_gain'])
                capital += profit
                trades.append({
                    'trade_nr': trade_counter,
                    'symbol': pos['symbol'],
                    'entry_time': pos['timestamp'],
                    'exit_time': timestamp,
                    'type': 'Kaufen' if pos['type'] == 'long' else 'Verkaufen',
                    'entry_price': pos['entry_price'],
                    'exit_price': exit_price,
                    'size_eur': pos['size_eur'],
                    'size_coins': pos['size_coins'],
                    'profit': profit,
                    'return_pct': (profit / pos['size_eur']) * 100 if pos['size_eur'] > 0 else 0,
                    'capital_before': capital - profit,
                    'capital_after': capital,
                    'duration_hours': (timestamp - pos['timestamp']).total_seconds() / 3600,
                    'entry_reason': pos.get('entry_reason', ''),
                    'exit_reason': exit_reason,
                    'max_loss': pos['max_loss'],
                    'max_gain': pos['max_gain'],
                    'risk_reward_ratio': pos['max_gain'] / pos['max_loss'] if pos['max_loss'] > 0 else None,
                    'regime': pos['regime'],
                    'pyramid_levels': pos['pyramid_levels'],
                    'hour': pos['hour'],
                    'weekday': pos['weekday']
                })
                trade_counter += 1
                positions.remove(pos)

        # Equity Tracking
        equity.append({
            'timestamp': timestamp,
            'symbol': symbol,
            'capital': capital,
            'open_positions': len(positions),
            'unrealized_pnl': sum([
                (current_price - p['entry_price']) * p['size_coins'] if p['type'] == 'long'
                else (p['entry_price'] - current_price) * p['size_coins'] for p in positions
            ])
        })

    import pandas as pd
    return pd.DataFrame(trades), pd.DataFrame(equity), capital
