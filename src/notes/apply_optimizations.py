"""
Apply all optimizations to app.py:
1. Add cache mechanism
2. Add signal_strength, target_price fields
3. Add cache to all prediction endpoints
"""

import os
import shutil

# Backup original
src_path = r'E:\StockandCrypto\src\notes\app.py'
backup_path = r'E:\StockandCrypto\src\notes\app.py.bak'
shutil.copy(src_path, backup_path)
print(f"Backed up to {backup_path}")

with open(src_path, 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Add cache imports and functions after "import re"
cache_code = '''
import time

# ==================== Simple Cache Mechanism ====================
_cache: dict[str, dict[str, Any]] = {}
CACHE_TTL_SECONDS = 45  # Cache TTL for API responses

def _cache_get(key: str) -> tuple[Any, bool]:
    """Get value from cache if not expired. Returns (value, found)."""
    if key not in _cache:
        return None, False
    entry = _cache[key]
    if time.time() - entry["timestamp"] > CACHE_TTL_SECONDS:
        del _cache[key]
        return None, False
    return entry["value"], True

def _cache_set(key: str, value: Any) -> None:
    """Set value in cache with current timestamp."""
    _cache[key] = {"value": value, "timestamp": time.time()}

def _cache_key(prefix: str, *args) -> str:
    """Generate a cache key from prefix and arguments."""
    return f"{prefix}:{':'.join(str(a) for a in args)}"
'''

# Insert after "import re"
if 'import time' not in content:
    content = content.replace(
        'import re\nfrom datetime',
        f'import re{cache_code}\nfrom datetime'
    )
    print("Added cache mechanism")

# 2. Update _normalize_prediction_fields to include signal_strength and target_price
old_normalize_return = '''    # Build normalized result
    return {
        "symbol": row.get("symbol", ""),
        "action": action,
        "p_up": round(p_up, 4) if p_up is not None else None,
        "p_down": round(p_down, 4) if p_down is not None else None,
        "confidence": round(confidence, 4) if confidence is not None else None,
        "current_price": price,
        "target_price_q10": target_price_q10,
        "target_price_q50": target_price_q50,
        "target_price_q90": target_price_q90,
        "horizon": row.get("horizon", ""),
        "session_name": row.get("session_name", ""),
        "trend_label": row.get("trend_label", ""),
        "volatility_score": _safe_float(row.get("volatility_score")),
        "risk_level": row.get("risk_level", ""),
        # Additional useful fields
        "q10_change_pct": _safe_float(row.get("q10_change_pct")),
        "q50_change_pct": _safe_float(row.get("q50_change_pct")),
        "q90_change_pct": _safe_float(row.get("q90_change_pct")),
        "exchange": row.get("exchange", ""),
        "market_type": row.get("market_type", ""),
        "forecast_generated_at": row.get("forecast_generated_at_bj") or row.get("timestamp_utc", ""),
        "policy_reason": row.get("policy_reason", ""),
        "sample_size": row.get("sample_size"),
    }'''

new_normalize_return = '''    # Calculate signal_strength based on p_up/p_down difference and confidence
    signal_strength = None
    if p_up is not None and p_down is not None:
        prob_diff = abs(p_up - p_down)
        if confidence is not None:
            # Combine probability difference with confidence
            signal_strength = round(prob_diff * confidence, 4)
        else:
            signal_strength = round(prob_diff, 4)

    # Calculate target_price (use q50 as the main target)
    target_price = target_price_q50 or target_price_q10 or target_price_q90

    # Build normalized result with all required fields
    return {
        "symbol": row.get("symbol", ""),
        "action": action,
        "p_up": round(p_up, 4) if p_up is not None else None,
        "p_down": round(p_down, 4) if p_down is not None else None,
        "confidence": round(confidence, 4) if confidence is not None else None,
        "current_price": price,
        "target_price": target_price,
        "target_price_q10": target_price_q10,
        "target_price_q50": target_price_q50,
        "target_price_q90": target_price_q90,
        "signal_strength": signal_strength,
        "trend_label": row.get("trend_label", "") or "neutral",
        "risk_level": row.get("risk_level", "") or "medium",
        "horizon": row.get("horizon", ""),
        "session_name": row.get("session_name", ""),
        "volatility_score": _safe_float(row.get("volatility_score")),
        # Additional useful fields
        "q10_change_pct": _safe_float(row.get("q10_change_pct")),
        "q50_change_pct": _safe_float(row.get("q50_change_pct")),
        "q90_change_pct": _safe_float(row.get("q90_change_pct")),
        "exchange": row.get("exchange", ""),
        "market_type": row.get("market_type", ""),
        "forecast_generated_at": row.get("forecast_generated_at_bj") or row.get("timestamp_utc", ""),
        "policy_reason": row.get("policy_reason", ""),
        "sample_size": row.get("sample_size"),
    }'''

if old_normalize_return in content:
    content = content.replace(old_normalize_return, new_normalize_return)
    print("Updated _normalize_prediction_fields")

# Write the modified content
with open(src_path, 'w', encoding='utf-8') as f:
    f.write(content)

print("Phase 1 complete - cache and normalize function updated")
