import re

# Read the file
with open(r'E:\StockandCrypto\src\notes\app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find the crypto_predictions function and update it
# Look for specific patterns

# Pattern 1: Add cache check after "normalized = request.args.get"
old_pattern1 = 'normalized = request.args.get("normalized", "true").lower() != "false"\n    data_path = _get_data_path()'
new_pattern1 = '''normalized = request.args.get("normalized", "true").lower() != "false"
    
    # Check cache first
    cache_key = _cache_key("crypto_pred", symbol_filter or "all", limit, normalized)
    cached, found = _cache_get(cache_key)
    if found:
        return jsonify(cached)
    
    data_path = _get_data_path()'''

if old_pattern1 in content:
    content = content.replace(old_pattern1, new_pattern1)
    print("Added cache check to crypto_predictions")
else:
    print("Pattern 1 not found")

# Pattern 2: Add cache set before first return in crypto_predictions
old_return = '''return jsonify({
            "ok": True,
            "predictions": predictions,
            "count": len(predictions),
            "timestamp": _utcnow().isoformat(),
            "source": "session_forecast_blocks"
        })'''
        
new_return = '''result = {
            "ok": True,
            "predictions": predictions,
            "count": len(predictions),
            "timestamp": _utcnow().isoformat(),
            "source": "session_forecast_blocks"
        }
        _cache_set(cache_key, result)
        return jsonify(result)'''

if old_return in content:
    content = content.replace(old_return, new_return)
    print("Updated first return in crypto_predictions")
else:
    print("Return pattern not found")

# Write back
with open(r'E:\StockandCrypto\src\notes\app.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("File saved")
