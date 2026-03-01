# Patch file for API optimizations
# This will be applied to app.py

PATCHES = {
    "crypto_predictions": '''
@app.get("/api/crypto/predictions")
def crypto_predictions() -> Any:
    """ Get crypto predictions for BTC/ETH/SOL.
    Query params:
        symbol: Filter by symbol (e.g., BTCUSDT)
        limit: Max number of results (default 100, max 500)
        normalized: Return normalized format (default true)
    Returns:
        { "ok": true, "predictions": [...], "count": N, "timestamp": "..." }
    """
    symbol_filter = request.args.get("symbol", "").strip().upper()
    try:
        limit = min(int(request.args.get("limit", "100")), 500)
    except ValueError:
        limit = 100
    normalized = request.args.get("normalized", "true").lower() != "false"
    
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
            return jsonify(result)

    # Fallback to policy_signals_hourly
    policy_file = data_path / "policy_signals_hourly.csv"
    if policy_file.exists():
        rows = _get_latest_csv_rows(policy_file, limit * 2, symbol_filter if symbol_filter else None)
        crypto_rows = [row for row in rows if row.get("market", "").lower() == "crypto" or row.get("symbol", "").upper() in ["BTCUSDT", "ETHUSDT", "SOLUSDT"]]
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
                "source": "policy_signals_hourly"
            }
            _cache_set(cache_key, result)
            return jsonify(result)

    # Fallback to predictions_hourly
    hourly_file = data_path / "predictions_hourly.csv"
    if hourly_file.exists():
        rows = _get_latest_csv_rows(hourly_file, limit, symbol_filter if symbol_filter else None)
        if rows:
            result = {
                "ok": True,
                "predictions": rows,
                "count": len(rows),
                "timestamp": _utcnow().isoformat(),
                "source": "predictions_hourly"
            }
            _cache_set(cache_key, result)
            return jsonify(result)

    return jsonify({"ok": False, "error": "crypto_predictions_not_found"}), 503
''',
}
