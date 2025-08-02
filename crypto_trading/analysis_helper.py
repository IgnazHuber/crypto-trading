def bulletify(lines):
    return "\n".join([f"• {line}" for line in lines if line])

def generate_trade_analysis(trade):
    short_lines = []
    long_lines = []

    pnl = trade.get("pnl_abs", 0)
    pnl_pct = trade.get("pnl_pct", 0)
    entry_price = trade.get("entry_price", None)
    exit_price = trade.get("exit_price", None)
    symbol = trade.get("symbol", "")
    
    short_lines.append(f"Asset: {symbol}")
    if pnl > 0:
        short_lines.append(f"Gewinn-Trade (+{pnl:.1f}) ✔️")
    elif pnl < 0:
        short_lines.append(f"Verlust-Trade ({pnl:.1f}) ❌")
    else:
        short_lines.append("Break-Even")
    if entry_price and exit_price:
        short_lines.append(f"Einstieg: {entry_price:.2f}, Ausstieg: {exit_price:.2f}")
    
    long_lines.append(f"Trade für {symbol}: {'Profitabel' if pnl>0 else 'Verlust' if pnl<0 else 'Break-Even'}")
    if pnl != 0:
        long_lines.append(f"Absoluter {'Gewinn' if pnl>0 else 'Verlust'}: {abs(pnl):.2f}")
    if pnl_pct:
        long_lines.append(f"Prozentual: {pnl_pct:.2f} %")

    return bulletify(short_lines), bulletify(long_lines)
