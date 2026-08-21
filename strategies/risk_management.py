def calculate_atr(df, period=14):
    """Calculates the Average True Range (ATR) from a dataframe."""
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    
    ranges = high_low.combine(high_close, max).combine(low_close, max)
    return ranges.rolling(period).mean().iloc[-1]

def calculate_bracket_prices(entry_price, atr, stop_multiplier=1.5, target_multiplier=3.6, direction="LONG"):
    """Calculates dynamic GTC bracket targets based on ATR."""
    if direction == "LONG":
        stop_loss = entry_price - (atr * stop_multiplier)
        take_profit = entry_price + (atr * target_multiplier)
    else:
        stop_loss = entry_price + (atr * stop_multiplier)
        take_profit = entry_price - (atr * target_multiplier)
        
    return round(stop_loss, 2), round(take_profit, 2)