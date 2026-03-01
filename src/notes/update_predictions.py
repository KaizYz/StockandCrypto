# Update cn_predictions and us_predictions with cache and complete signal info

with open(r'E:\StockandCrypto\src\notes\app.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
i = 0

while i < len(lines):
    line = lines[i]
    
    # Find cn_predictions function
    if '@app.get("/api/cn/predictions")' in line:
        # Find the end of the function
        func_start = i
        func_end = i + 1
        while func_end < len(lines):
            if lines[func_end].strip().startswith('@app.get("/api/cn/history'):
                break
            func_end += 1
        
        # Insert new function
        new_func = ''' @app.get("/api/cn/predictions")
 def cn_predictions() -> Any:
     """
     Get A-share (Chinese stock) predictions.
     Query params:
         symbol: Filter by symbol
         limit: Max results (default 100)
     Returns:
         { "ok": true, "predictions": [...], "count": N }
     """
     symbol_filter = request.args.get("symbol", "").strip().upper()
     try:
         limit = min(int(request.args.get("limit", "100")), 500)
     except ValueError:
         limit = 100
     
     # Check cache first
     cache_key = _cache_key("cn_pred", symbol_filter or "all", limit)
     cached, found = _cache_get(cache_key)
     if found:
         return jsonify(cached)
     
     data_path = _get_data_path()
     current_prices = _get_current_prices()
     snapshot_file = data_path / "market_snapshot.json"
     
     result = None
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
                     
                     # Calculate target prices
                     target_price = round(price * (1 + q50), 4) if price and q50 else None
                     target_price_q10 = round(price * (1 + q10), 4) if price and q10 else None
                     target_price_q90 = round(price * (1 + q90), 4) if price and q90 else None
                     
                     predictions.append({
                         "symbol": row.get("symbol", ""),
                         "name": row.get("name", ""),
                         "instrument_id": row.get("instrument_id", ""),
                         "current_price": price,
                         "target_price": target_price,
                         "target_price_q10": target_price_q10,
                         "target_price_q50": target_price,
                         "target_price_q90": target_price_q90,
                         "predicted_change_pct": _safe_float(row.get("predicted_change_pct")),
                         "q10_change_pct": q10,
                         "q50_change_pct": q50,
                         "q90_change_pct": q90,
                         "action": "Flat",  # Default for snapshot data
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
                 return jsonify(result)
         except Exception:
             pass
 
     # Fallback to predictions_daily
     daily_file = data_path / "predictions_daily.csv"
     if daily_file.exists():
         rows = _get_latest_csv_rows(daily_file, limit, symbol_filter if symbol_filter else None)
         result = {
             "ok": True,
             "predictions": rows,
             "count": len(rows),
             "timestamp": _utcnow().isoformat()
         }
         _cache_set(cache_key, result)
         return jsonify(result)
 
     return jsonify({"ok": False, "error": "cn_predictions_not_found"}), 503

'''
        new_lines.append(new_func)
        i = func_end
        print(f"Replaced cn_predictions function (lines {func_start} to {func_end})")
    elif '@app.get("/api/us/predictions")' in line:
        # Find the end of the function
        func_start = i
        func_end = i + 1
        while func_end < len(lines):
            if lines[func_end].strip().startswith('@app.get("/api/us/history'):
                break
            func_end += 1
        
        # Insert new function
        new_func = ''' @app.get("/api/us/predictions")
 def us_predictions() -> Any:
     """
     Get US stock predictions.
     Query params:
         symbol: Filter by symbol
         limit: Max results (default 100)
     Returns:
         { "ok": true, "predictions": [...], "count": N }
     """
     symbol_filter = request.args.get("symbol", "").strip().upper()
     try:
         limit = min(int(request.args.get("limit", "100")), 500)
     except ValueError:
         limit = 100
     
     # Check cache first
     cache_key = _cache_key("us_pred", symbol_filter or "all", limit)
     cached, found = _cache_get(cache_key)
     if found:
         return jsonify(cached)
     
     data_path = _get_data_path()
     snapshot_file = data_path / "market_snapshot.json"
     
     result = None
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
                     
                     # Calculate target prices
                     target_price = round(price * (1 + q50), 4) if price and q50 else None
                     target_price_q10 = round(price * (1 + q10), 4) if price and q10 else None
                     target_price_q90 = round(price * (1 + q90), 4) if price and q90 else None
                     
                     predictions.append({
                         "symbol": row.get("symbol", ""),
                         "name": row.get("name", ""),
                         "instrument_id": row.get("instrument_id", ""),
                         "current_price": price,
                         "target_price": target_price,
                         "target_price_q10": target_price_q10,
                         "target_price_q50": target_price,
                         "target_price_q90": target_price_q90,
                         "predicted_change_pct": _safe_float(row.get("predicted_change_pct")),
                         "q10_change_pct": q10,
                         "q50_change_pct": q50,
                         "q90_change_pct": q90,
                         "action": "Flat",  # Default for snapshot data
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
                 return jsonify(result)
         except Exception:
             pass
 
     # Fallback to predictions_daily
     daily_file = data_path / "predictions_daily.csv"
     if daily_file.exists():
         rows = _get_latest_csv_rows(daily_file, limit, symbol_filter if symbol_filter else None)
         result = {
             "ok": True,
             "predictions": rows,
             "count": len(rows),
             "timestamp": _utcnow().isoformat()
         }
         _cache_set(cache_key, result)
         return jsonify(result)
 
     return jsonify({"ok": False, "error": "us_predictions_not_found"}), 503

'''
        new_lines.append(new_func)
        i = func_end
        print(f"Replaced us_predictions function (lines {func_start} to {func_end})")
    else:
        new_lines.append(line)
        i += 1

# Write back
with open(r'E:\StockandCrypto\src\notes\app.py', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print("File saved successfully")
