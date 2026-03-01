# Update crypto_signal function with cache and complete signal info

with open(r'E:\StockandCrypto\src\notes\app.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
i = 0

while i < len(lines):
    line = lines[i]
    
    # Find crypto_signal function
    if '@app.get("/api/crypto/signal/<symbol>")' in line:
        # Found the function, replace the entire function
        # Find the end of the function (next @app or # ===)
        func_start = i
        func_end = i + 1
        while func_end < len(lines):
            if lines[func_end].strip().startswith('# ==========') or lines[func_end].strip().startswith('@app.get'):
                break
            func_end += 1
        
        # Insert new function
        new_func = ''' @app.get("/api/crypto/signal/<symbol>")
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
     return jsonify(result)

'''
        new_lines.append(new_func)
        i = func_end
        print(f"Replaced crypto_signal function (lines {func_start} to {func_end})")
    else:
        new_lines.append(line)
        i += 1

# Write back
with open(r'E:\StockandCrypto\src\notes\app.py', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print("File saved successfully")
