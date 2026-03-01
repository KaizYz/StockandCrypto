"""
Apply cache to API endpoints:
- crypto_predictions
- crypto_signal
- cn_predictions
- us_predictions
"""

src_path = r'E:\StockandCrypto\src\notes\app.py'

with open(src_path, 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Update crypto_predictions - add cache check after normalized line
old_crypto_pred = '''    normalized = request.args.get("normalized", "true").lower() != "false"
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

new_crypto_pred = '''    normalized = request.args.get("normalized", "true").lower() != "false"
    
    # Check cache first
    cache_key = _cache_key("crypto_pred", symbol_filter or "all", limit, normalized)
    cached, found = _cache_get(cache_key)
    if found:
        return jsonify(cached)
    
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
            result = {
                "ok": True,
                "predictions": predictions,
                "count": len(predictions),
                "timestamp": _utcnow().isoformat(),
                "source": "session_forecast_blocks"
            }
            _cache_set(cache_key, result)
            return jsonify(result)'''

if old_crypto_pred in content:
    content = content.replace(old_crypto_pred, new_crypto_pred)
    print("Updated crypto_predictions with cache")

# 2. Update crypto_signal
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

# 3. Update cn_predictions - add cache check after limit
old_cn_pred = '''    data_path = _get_data_path()
    current_prices = _get_current_prices()
    snapshot_file = data_path / "market_snapshot.json"
    if snapshot_file.exists():
        try:
            with open(snapshot_file, "r", encoding="utf-8") as f:
                snapshot = json.load(f)
            cn_assets = [row for row in snapshot.get("rows", []) if row.get("market") == "cn_equity"]
            if cn_assets:
                predictions = []
                for row in cn_assets:
                    sym = row.get("symbol", "").upper()
                    if symbol_filter and sym != symbol_filter:
                        continue
                    predictions.append({
                        "symbol": row.get("symbol", ""),
                        "name": row.get("name", ""),
                        "instrument_id": row.get("instrument_id", ""),
                        "current_price": _safe_float(row.get("current_price")),
                        "predicted_change_pct": _safe_float(row.get("predicted_change_pct")),
                        "q10_change_pct": _safe_float(row.get("q10_change_pct")),
                        "q50_change_pct": _safe_float(row.get("q50_change_pct")),
                        "q90_change_pct": _safe_float(row.get("q90_change_pct")),
                    })
                    if len(predictions) >= limit:
                        break
                return jsonify({
                    "ok": True,
                    "predictions": predictions,
                    "count": len(predictions),
                    "timestamp": _utcnow().isoformat()
                })'''

new_cn_pred = '''    # Check cache first
    cache_key = _cache_key("cn_pred", symbol_filter or "all", limit)
    cached, found = _cache_get(cache_key)
    if found:
        return jsonify(cached)
    
    data_path = _get_data_path()
    current_prices = _get_current_prices()
    snapshot_file = data_path / "market_snapshot.json"
    if snapshot_file.exists():
        try:
            with open(snapshot_file, "r", encoding="utf-8") as f:
                snapshot = json.load(f)
            cn_assets = [row for row in snapshot.get("rows", []) if row.get("market") == "cn_equity"]
            if cn_assets:
                predictions = []
                for row in cn_assets:
                    sym = row.get("symbol", "").upper()
                    if symbol_filter and sym != symbol_filter:
                        continue
                    price = _safe_float(row.get("current_price"))
                    q10 = _safe_float(row.get("q10_change_pct"))
                    q50 = _safe_float(row.get("q50_change_pct"))
                    q90 = _safe_float(row.get("q90_change_pct"))
                    target_price = round(price * (1 + q50), 4) if price and q50 else None
                    
                    predictions.append({
                        "symbol": row.get("symbol", ""),
                        "name": row.get("name", ""),
                        "instrument_id": row.get("instrument_id", ""),
                        "current_price": price,
                        "target_price": target_price,
                        "predicted_change_pct": _safe_float(row.get("predicted_change_pct")),
                        "q10_change_pct": q10,
                        "q50_change_pct": q50,
                        "q90_change_pct": q90,
                        "action": "Flat",
                        "confidence": None,
                        "signal_strength": None,
                        "trend_label": "neutral",
                        "risk_level": "medium",
                    })
                    if len(predictions) >= limit:
                        break
                result = {
                    "ok": True,
                    "predictions": predictions,
                    "count": len(predictions),
                    "timestamp": _utcnow().isoformat()
                }
                _cache_set(cache_key, result)
                return jsonify(result)'''

if old_cn_pred in content:
    content = content.replace(old_cn_pred, new_cn_pred)
    print("Updated cn_predictions")

# 4. Update us_predictions - add cache check after limit
old_us_pred = '''    data_path = _get_data_path()
    snapshot_file = data_path / "market_snapshot.json"
    if snapshot_file.exists():
        try:
            with open(snapshot_file, "r", encoding="utf-8") as f:
                snapshot = json.load(f)
            us_assets = [row for row in snapshot.get("rows", []) if row.get("market") == "us_equity"]
            if us_assets:
                predictions = []
                for row in us_assets:
                    sym = row.get("symbol", "").upper()
                    if symbol_filter and sym != symbol_filter:
                        continue
                    predictions.append({
                        "symbol": row.get("symbol", ""),
                        "name": row.get("name", ""),
                        "instrument_id": row.get("instrument_id", ""),
                        "current_price": _safe_float(row.get("current_price")),
                        "predicted_change_pct": _safe_float(row.get("predicted_change_pct")),
                        "q10_change_pct": _safe_float(row.get("q10_change_pct")),
                        "q50_change_pct": _safe_float(row.get("q50_change_pct")),
                        "q90_change_pct": _safe_float(row.get("q90_change_pct")),
                    })
                    if len(predictions) >= limit:
                        break
                return jsonify({
                    "ok": True,
                    "predictions": predictions,
                    "count": len(predictions),
                    "timestamp": _utcnow().isoformat()
                })'''

new_us_pred = '''    # Check cache first
    cache_key = _cache_key("us_pred", symbol_filter or "all", limit)
    cached, found = _cache_get(cache_key)
    if found:
        return jsonify(cached)
    
    data_path = _get_data_path()
    snapshot_file = data_path / "market_snapshot.json"
    if snapshot_file.exists():
        try:
            with open(snapshot_file, "r", encoding="utf-8") as f:
                snapshot = json.load(f)
            us_assets = [row for row in snapshot.get("rows", []) if row.get("market") == "us_equity"]
            if us_assets:
                predictions = []
                for row in us_assets:
                    sym = row.get("symbol", "").upper()
                    if symbol_filter and sym != symbol_filter:
                        continue
                    price = _safe_float(row.get("current_price"))
                    q10 = _safe_float(row.get("q10_change_pct"))
                    q50 = _safe_float(row.get("q50_change_pct"))
                    q90 = _safe_float(row.get("q90_change_pct"))
                    target_price = round(price * (1 + q50), 4) if price and q50 else None
                    
                    predictions.append({
                        "symbol": row.get("symbol", ""),
                        "name": row.get("name", ""),
                        "instrument_id": row.get("instrument_id", ""),
                        "current_price": price,
                        "target_price": target_price,
                        "predicted_change_pct": _safe_float(row.get("predicted_change_pct")),
                        "q10_change_pct": q10,
                        "q50_change_pct": q50,
                        "q90_change_pct": q90,
                        "action": "Flat",
                        "confidence": None,
                        "signal_strength": None,
                        "trend_label": "neutral",
                        "risk_level": "medium",
                    })
                    if len(predictions) >= limit:
                        break
                result = {
                    "ok": True,
                    "predictions": predictions,
                    "count": len(predictions),
                    "timestamp": _utcnow().isoformat()
                }
                _cache_set(cache_key, result)
                return jsonify(result)'''

if old_us_pred in content:
    content = content.replace(old_us_pred, new_us_pred)
    print("Updated us_predictions")

# Write back
with open(src_path, 'w', encoding='utf-8') as f:
    f.write(content)

print("All API endpoints updated with cache")
