import re

# Read the file
with open(r'E:\StockandCrypto\src\notes\app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Update crypto_predictions to use cache
old_section = '''    normalized = request.args.get("normalized", "true").lower() != "false"
    data_path = _get_data_path()
    current_prices = _get_current_prices()
    # Try session_forecast_blocks first (has all required fields)
    session_file = data_path / "session_forecast_blocks.csv"
    if session_file.exists():
        rows = _read_csv_with_headers(session_file, limit=limit * 2, symbol_filter=symbol_filter if symbol_filter else None)
        crypto_rows = [row for row in rows if row.get("symbol", "").upper() in ["BTCUSDT", "ETHUSDT", "SOLUSDT"]]
        if crypto_rows:
            if normalized:
                predictions = [_normalize_prediction_fields(row, current_prices.get(row.get("symbol", "").upper())) for row in crypto_rows[:limit]]
            else:
                predictions = crypto_rows[:limit]
            return jsonify({
                "ok": True,
                "predictions": predictions,
                "count": len(predictions),
                "timestamp": _utcnow().isoformat(),
                "source": "session_forecast_blocks"
            })'''

new_section = '''    normalized = request.args.get("normalized", "true").lower() != "false"
    
    # Check cache first
    cache_key = _cache_key("crypto_pred", symbol_filter or "all", limit, normalized)
    cached, found = _cache_get(cache_key)
    if found:
        return jsonify(cached)
    
    data_path = _get_data_path()
    current_prices = _get_current_prices()
    
    # Try session_forecast_blocks first (has all required fields)
    session_file = data_path / "session_forecast_blocks.csv"
    result = None
    if session_file.exists():
        rows = _read_csv_with_headers(session_file, limit=limit * 2, symbol_filter=symbol_filter if symbol_filter else None)
        crypto_rows = [row for row in rows if row.get("symbol", "").upper() in ["BTCUSDT", "ETHUSDT", "SOLUSDT"]]
        if crypto_rows:
            if normalized:
                predictions = [_normalize_prediction_fields(row, current_prices.get(row.get("symbol", "").upper())) for row in crypto_rows[:limit]]
            else:
                predictions = crypto_rows[:limit]
            result = {
                "ok": True,
                "predictions": predictions,
                "count": len(predictions),
                "timestamp": _utcnow().isoformat(),
                "source": "session_forecast_blocks"
            }
            _cache_set(cache_key, result)
            return jsonify(result)'''

if old_section in content:
    content = content.replace(old_section, new_section)
    print("Updated crypto_predictions with cache")
else:
    print("Could not find crypto_predictions section")

# Write back
with open(r'E:\StockandCrypto\src\notes\app.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("File saved")
