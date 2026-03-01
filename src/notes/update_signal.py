# Update app.py - Add cache to crypto_signal, cn_predictions, us_predictions

with open(r'E:\StockandCrypto\src\notes\app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Update crypto_signal to use cache and return complete signal info
old_crypto_signal = '''@app.get("/api/crypto/signal/<symbol>")
def crypto_signal(symbol: str) -> Any:
    """Get trading signal for a specific crypto symbol."""
    symbol = symbol.upper().strip()
    data_path = _get_data_path()
    current_prices = _get_current_prices()
    current_price = current_prices.get(symbol)

    # Get signal data from session forecast
    session_file = data_path / "session_forecast_blocks.csv"
    if session_file.exists():
        rows = _read_csv_with_headers(session_file, limit=500)
        symbol_rows = [row for row in rows if row.get("symbol", "").upper() == symbol]
        if symbol_rows:
            latest = symbol_rows[-1]
            normalized = _normalize_prediction_fields(latest, current_price)
            return jsonify({
                "ok": True,
                "symbol": symbol,
                "signal": normalized,
                "timestamp": _utcnow().isoformat(),
            })

    # Return default signal if no data found
    return jsonify({
        "ok": True,
        "symbol": symbol,
        "signal": {
            "action": "Flat",
            "confidence": None,
            "p_up": None,
            "p_down": None,
            "current_price": current_price,
            "target_price_q50": None,
        },
        "timestamp": _utcnow().isoformat(),
        "message": "No signal found for symbol"
    })'''

new_crypto_signal = '''@app.get("/api/crypto/signal/<symbol>")
def crypto_signal(symbol: str) -> Any:
    """Get trading signal for a specific crypto symbol.
    Returns complete signal information including:
    - action: Long/Short/Flat
    - p_up, p_down: probabilities
    - confidence: confidence score (0-1)
    - current_price: current price
    - target_price: target price (q50)
    - signal_strength: combined strength metric
    - trend_label: trend description
    - risk_level: risk assessment
    """
    symbol = symbol.upper().strip()
    
    # Check cache first
    cache_key = _cache_key("crypto_signal", symbol)
    cached, found = _cache_get(cache_key)
    if found:
        return jsonify(cached)
    
    data_path = _get_data_path()
    current_prices = _get_current_prices()
    current_price = current_prices.get(symbol)

    # Get signal data from session forecast
    session_file = data_path / "session_forecast_blocks.csv"
    if session_file.exists():
        rows = _read_csv_with_headers(session_file, limit=500)
        symbol_rows = [row for row in rows if row.get("symbol", "").upper() == symbol]
        if symbol_rows:
            latest = symbol_rows[-1]
            normalized = _normalize_prediction_fields(latest, current_price)
            result = {
                "ok": True,
                "symbol": symbol,
                "signal": normalized,
                "timestamp": _utcnow().isoformat(),
            }
            _cache_set(cache_key, result)
            return jsonify(result)

    # Return default signal if no data found
    default_signal = {
        "action": "Flat",
        "confidence": None,
        "p_up": None,
        "p_down": None,
        "current_price": current_price,
        "target_price": None,
        "target_price_q50": None,
        "signal_strength": None,
        "trend_label": "neutral",
        "risk_level": "medium",
    }
    result = {
        "ok": True,
        "symbol": symbol,
        "signal": default_signal,
        "timestamp": _utcnow().isoformat(),
        "message": "No signal found for symbol"
    }
    _cache_set(cache_key, result)
    return jsonify(result)'''

if old_crypto_signal in content:
    content = content.replace(old_crypto_signal, new_crypto_signal)
    print("Updated crypto_signal")
else:
    print("Could not find crypto_signal")

# Write back
with open(r'E:\StockandCrypto\src\notes\app.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("File saved")
