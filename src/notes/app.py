from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import time
from collections import defaultdict

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

from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer
from sqlalchemy import func, or_
from sqlalchemy.exc import IntegrityError
from werkzeug.security import check_password_hash, generate_password_hash

db = SQLAlchemy()

USERNAME_PATTERN = re.compile(r"^[A-Za-z0-9_]{3,32}$")
YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
YAHOO_QUOTE_URL = "https://query1.finance.yahoo.com/v7/finance/quote"
BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
BINANCE_TICKER_PRICE_URL = "https://api.binance.com/api/v3/ticker/price"
BINANCE_KLINES_FALLBACK_URLS = [
    "https://api.binance.com/api/v3/klines",
    "https://api.binance.us/api/v3/klines",
]
BINANCE_TICKER_FALLBACK_URLS = [
    "https://api.binance.com/api/v3/ticker/price",
    "https://api.binance.us/api/v3/ticker/price",
]
YAHOO_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )
}

SESSION_ORDER = {"asia": 0, "europe": 1, "us": 2}
SESSION_NAME_ZH = {"asia": "亚盘", "europe": "欧盘", "us": "美盘"}
SESSION_HOURS = {"asia": "08:00-15:59", "europe": "16:00-23:59", "us": "00:00-07:59"}

INDEX_SESSION_UNIVERSE: dict[str, dict[str, str]] = {
    "sse": {
        "name_zh": "上证指数",
        "name_en": "SSE Composite",
        "symbol": "000001.SS",
        "market": "cn_equity",
    },
    "djia": {
        "name_zh": "道琼斯指数",
        "name_en": "Dow Jones Industrial Average",
        "symbol": "^DJI",
        "market": "us_equity",
    },
    "nasdaq": {
        "name_zh": "纳斯达克指数",
        "name_en": "NASDAQ Composite",
        "symbol": "^IXIC",
        "market": "us_equity",
    },
    "sp500": {
        "name_zh": "标普500",
        "name_en": "S&P 500",
        "symbol": "^GSPC",
        "market": "us_equity",
    },
}
INDEX_ACTIVE_SESSION = {"sse": "asia", "djia": "us", "nasdaq": "us", "sp500": "us"}
RANK_OPTIONS = {
    "edge_score": "Edge",
    "edge_risk": "Edge / Risk",
    "signal_strength": "Signal Strength",
    "p_up": "P(up)",
    "confidence": "Confidence",
    "volatility": "Volatility",
    "q50": "Expected Change(q50)",
}


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _fmt_dt(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _notes_db_uri() -> str:
    root = Path(__file__).resolve().parents[2]
    raw_db_path = os.getenv("NOTES_DB_PATH", "").strip()
    if raw_db_path:
        db_path = Path(raw_db_path)
        if not db_path.is_absolute():
            db_path = (root / db_path).resolve()
    else:
        db_path = root / "data" / "notes" / "notes.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{db_path.as_posix()}"


class User(db.Model):
    __tablename__ = "users"
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(32), unique=True, index=True, nullable=False)
    email = db.Column(db.String(255), unique=True, index=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    is_active = db.Column(db.Boolean, nullable=False, default=True)
    created_at = db.Column(db.DateTime(timezone=True), nullable=False, default=_utcnow)
    updated_at = db.Column(
        db.DateTime(timezone=True), nullable=False, default=_utcnow, onupdate=_utcnow
    )
    last_login_at = db.Column(db.DateTime(timezone=True), nullable=True)

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "username": self.username,
            "email": self.email,
            "is_active": bool(self.is_active),
            "created_at": _fmt_dt(self.created_at),
            "updated_at": _fmt_dt(self.updated_at),
            "last_login_at": _fmt_dt(self.last_login_at),
        }


class Note(db.Model):
    __tablename__ = "notes"
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), index=True, nullable=False)
    title = db.Column(db.String(200), nullable=False, default="")
    content = db.Column(db.Text, nullable=False, default="")
    tags_csv = db.Column(db.String(500), nullable=False, default="")
    note_type = db.Column(db.String(16), nullable=False, default="NOTE")
    is_public = db.Column(db.Boolean, nullable=False, default=False, index=True)
    created_at = db.Column(db.DateTime(timezone=True), nullable=False, default=_utcnow)
    updated_at = db.Column(
        db.DateTime(timezone=True), nullable=False, default=_utcnow, onupdate=_utcnow
    )
    user = db.relationship("User", backref=db.backref("notes", lazy=True))

    def to_dict(self, include_author: bool = False) -> dict[str, Any]:
        data = {
            "id": self.id,
            "title": self.title,
            "content": self.content,
            "tags": _split_tags(self.tags_csv),
            "note_type": self.note_type,
            "is_public": bool(self.is_public),
            "created_at": _fmt_dt(self.created_at),
            "updated_at": _fmt_dt(self.updated_at),
        }
        if include_author:
            data["author"] = {
                "id": self.user.id if self.user else None,
                "username": self.user.username if self.user else None,
            }
        return data


def _err(message: str, status_code: int):
    return jsonify({"ok": False, "error": message}), status_code


def _serializer(app: Flask) -> URLSafeTimedSerializer:
    return URLSafeTimedSerializer(
        secret_key=app.config["NOTES_AUTH_SECRET"],
        salt="notes-auth-token",
    )


def _issue_token(app: Flask, user_id: int) -> str:
    return _serializer(app).dumps({"uid": int(user_id)})


def _read_token(app: Flask, token: str) -> int:
    data = _serializer(app).loads(
        token,
        max_age=int(app.config["NOTES_TOKEN_TTL_SECONDS"]),
    )
    return int(data["uid"])


def _json_payload() -> dict[str, Any]:
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return {}
    return payload


def _normalize_username(raw: Any) -> str:
    return str(raw or "").strip()


def _normalize_email(raw: Any) -> str:
    return str(raw or "").strip().lower()


def _validate_register_input(username: str, email: str, password: str) -> str | None:
    if not username or not email or not password:
        return "username_email_password_required"
    if not USERNAME_PATTERN.fullmatch(username):
        return "username_invalid"
    if "@" not in email or "." not in email.split("@")[-1]:
        return "email_invalid"
    if len(password) < 8:
        return "password_too_short"
    return None


def _split_tags(tags_csv: str | None) -> list[str]:
    raw = str(tags_csv or "").strip()
    if not raw:
        return []
    return [part.strip() for part in raw.split(",") if part.strip()]


def _normalize_tags(raw: Any) -> list[str]:
    values: list[str] = []
    if isinstance(raw, list):
        for part in raw:
            txt = str(part or "").strip()
            if txt:
                values.append(txt)
    else:
        for part in str(raw or "").split(","):
            txt = part.strip()
            if txt:
                values.append(txt)
    dedup: list[str] = []
    seen: set[str] = set()
    for item in values:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        dedup.append(item)
    return dedup[:30]


def _parse_page_size(raw: Any, default: int = 20, max_value: int = 100) -> int:
    try:
        value = int(str(raw or default))
    except Exception:
        value = default
    if value <= 0:
        return default
    return min(value, max_value)


def _require_auth_user(app: Flask) -> tuple[User | None, Any | None]:
    auth = request.headers.get("Authorization", "")
    if not auth.lower().startswith("bearer "):
        return None, _err("authorization_required", 401)
    token = auth.split(" ", 1)[1].strip()
    if not token:
        return None, _err("authorization_required", 401)
    try:
        user_id = _read_token(app, token)
    except SignatureExpired:
        return None, _err("token_expired", 401)
    except (BadSignature, KeyError, ValueError, TypeError):
        return None, _err("invalid_token", 401)
    user = db.session.get(User, user_id)
    if user is None:
        return None, _err("user_not_found", 401)
    if not user.is_active:
        return None, _err("account_disabled", 403)
    return user, None


# ==================== Data Helper Functions ====================

def _get_data_path() -> Path:
    """Get the data/processed directory path."""
    return Path(__file__).resolve().parents[2] / "data" / "processed"


def _safe_float(value: Any) -> float | None:
    """Safely convert value to float, return None if NaN or invalid."""
    if value is None:
        return None
    try:
        f = float(value)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    except (TypeError, ValueError):
        return None


def _normalize_prediction_fields(row: dict[str, Any], current_price: float | None = None) -> dict[str, Any]:
    """Normalize prediction row to standard format for frontend."""
    # Get p_up and p_down
    p_up = _safe_float(row.get("p_up"))
    p_down = _safe_float(row.get("p_down"))
    
    # Get confidence (normalize to 0-1 if percentage)
    confidence = _safe_float(row.get("confidence_score"))
    if confidence is not None and confidence > 1:
        confidence = confidence / 100
    
    # Get action
    action = row.get("policy_action") or row.get("action") or "Flat"
    
    # Get target prices
    target_price_q10 = _safe_float(row.get("target_price_q10"))
    target_price_q50 = _safe_float(row.get("target_price_q50"))
    target_price_q90 = _safe_float(row.get("target_price_q90"))
    
    # If target prices not directly available, calculate from change percentages
    price = current_price or _safe_float(row.get("current_price"))
    if price is not None:
        q10_change = _safe_float(row.get("q10_change_pct"))
        q50_change = _safe_float(row.get("q50_change_pct"))
        q90_change = _safe_float(row.get("q90_change_pct"))
        
        if target_price_q10 is None and q10_change is not None:
            target_price_q10 = round(price * (1 + q10_change), 4)
        if target_price_q50 is None and q50_change is not None:
            target_price_q50 = round(price * (1 + q50_change), 4)
        if target_price_q90 is None and q90_change is not None:
            target_price_q90 = round(price * (1 + q90_change), 4)
    
    # Calculate signal_strength based on p_up/p_down difference and confidence
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
    }


def _read_csv_with_headers(csv_path: Path, limit: int = 100, symbol_filter: str | None = None) -> list[dict[str, Any]]:
    """Read CSV file and return list of dicts with proper type conversion."""
    if not csv_path.exists():
        return []
    rows = []
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if symbol_filter and row.get("symbol", "").upper() != symbol_filter.upper():
                    continue
                if len(rows) >= limit:
                    break
                converted_row = {}
                for key, value in row.items():
                    if value == "" or value.lower() == "nan":
                        converted_row[key] = None
                    else:
                        try:
                            converted_row[key] = float(value) if "." in value else int(value)
                        except ValueError:
                            converted_row[key] = value
                rows.append(converted_row)
    except Exception:
        pass
    return rows


def _get_latest_csv_rows(csv_path: Path, count: int = 50, symbol_filter: str | None = None) -> list[dict[str, Any]]:
    """Get the latest N rows from a CSV file, optionally filtered by symbol."""
    if not csv_path.exists():
        return []
    rows = []
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            all_rows = list(reader)
            if symbol_filter:
                all_rows = [r for r in all_rows if r.get("symbol", "").upper() == symbol_filter.upper()]
            recent_rows = all_rows[-count:] if len(all_rows) > count else all_rows
            for row in recent_rows:
                converted_row = {}
                for key, value in row.items():
                    if value == "" or value.lower() == "nan":
                        converted_row[key] = None
                    else:
                        try:
                            converted_row[key] = float(value) if "." in value else int(value)
                        except ValueError:
                            converted_row[key] = value
                rows.append(converted_row)
    except Exception:
        pass
    return rows


def _get_current_prices() -> dict[str, float]:
    """Load current prices from market snapshot."""
    data_path = _get_data_path()
    current_prices: dict[str, float] = {}
    snapshot_file = data_path / "market_snapshot.json"
    if snapshot_file.exists():
        try:
            with open(snapshot_file, "r", encoding="utf-8") as f:
                snapshot = json.load(f)
            for row in snapshot.get("rows", []):
                sym = row.get("symbol", "").upper()
                price = _safe_float(row.get("current_price"))
                if price is not None:
                    current_prices[sym] = price
        except Exception:
            pass
    return current_prices


def _coalesce(*values: Any) -> Any:
    """Return the first non-empty value."""
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return None


def _to_confidence_pct(value: Any) -> float | None:
    conf = _safe_float(value)
    if conf is None:
        return None
    return conf * 100 if conf <= 1 else conf


def _normalize_market_key(raw_market: Any) -> str:
    market = str(raw_market or "").strip().lower()
    alias_map = {
        "cn": "cn_equity",
        "a": "cn_equity",
        "a_share": "cn_equity",
        "ashare": "cn_equity",
        "china": "cn_equity",
        "us": "us_equity",
        "usa": "us_equity",
        "stock": "us_equity",
        "stocks": "us_equity",
    }
    return alias_map.get(market, market)


def _symbol_key(raw_symbol: Any) -> str:
    symbol = str(raw_symbol or "").upper().strip()
    symbol = symbol.replace("/", "").replace("-", "").replace("_", "")
    if "." in symbol:
        head = symbol.split(".", 1)[0]
        if head.isdigit():
            return head
    for suffix in ("USDT", "USDC", "USD"):
        if symbol.endswith(suffix):
            return symbol[: -len(suffix)]
    return symbol


def _pick_latest_symbol_row(rows: list[dict[str, Any]], symbol: str) -> dict[str, Any] | None:
    want = _symbol_key(symbol)
    for row in reversed(rows):
        if _symbol_key(row.get("symbol")) == want:
            return row
    return None


def _to_yahoo_symbol(symbol: str, market: str) -> str:
    raw = str(symbol or "").upper().strip()
    if not raw:
        return raw

    if market == "us_equity":
        alias = {
            "DJI": "^DJI",
            "^DJI": "^DJI",
            "IXIC": "^IXIC",
            "^IXIC": "^IXIC",
            "SPX": "^GSPC",
            "GSPC": "^GSPC",
            "^GSPC": "^GSPC",
            "INX": "^GSPC",
            "^INX": "^GSPC",
            "NDX": "^NDX",
            "^NDX": "^NDX",
            "RUT": "^RUT",
            "^RUT": "^RUT",
        }
        return alias.get(raw, raw)

    if market == "cn_equity":
        code = raw
        suffix = ""
        if "." in raw:
            code, suffix = raw.split(".", 1)
            suffix = suffix.upper().strip()
        code = code.strip()
        if code.isdigit():
            code = code.zfill(6)
            if suffix in {"SZ"}:
                return f"{code}.SZ"
            if suffix in {"SH", "SS"}:
                return f"{code}.SS"
            return f"{code}.SS" if code.startswith("6") else f"{code}.SZ"
        return raw

    if market == "crypto":
        for suffix in ("USDT", "USDC", "USD"):
            if raw.endswith(suffix):
                base = raw[: -len(suffix)]
                if base:
                    return f"{base}-USD"
        if raw.isalpha() and len(raw) <= 10:
            return f"{raw}-USD"
        return raw

    return raw


def _fetch_yahoo_history_bars(
    symbol: str,
    market: str,
    interval: str,
    limit: int,
) -> list[dict[str, Any]]:
    interval_key = str(interval or "daily").lower().strip()
    if interval_key not in {"daily", "hourly"}:
        interval_key = "daily"
    yahoo_interval = "1d" if interval_key == "daily" else "60m"

    safe_limit = max(2, min(int(limit or 100), 1000))
    if interval_key == "hourly":
        # Yahoo intraday has lookback limits; keep window bounded.
        lookback_days = min(729, max(7, math.ceil((safe_limit * 2) / 24) + 2))
    else:
        lookback_days = min(3650, max(30, safe_limit * 3))

    period2 = int((_utcnow() + timedelta(minutes=1)).timestamp())
    period1 = int((_utcnow() - timedelta(days=lookback_days)).timestamp())
    yahoo_symbol = _to_yahoo_symbol(symbol, market)

    try:
        qs = urlencode(
            {
                "interval": yahoo_interval,
                "period1": period1,
                "period2": period2,
                "includePrePost": "false",
                "events": "div,splits",
            }
        )
        url = f"{YAHOO_CHART_URL.format(symbol=yahoo_symbol)}?{qs}"
        req = Request(url, headers=YAHOO_HEADERS)
        with urlopen(req, timeout=15) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return []

    result = (payload.get("chart") or {}).get("result") or []
    if not result:
        return []
    item = result[0] or {}
    timestamps = item.get("timestamp") or []
    quote_list = ((item.get("indicators") or {}).get("quote") or [])
    if not timestamps or not quote_list:
        return []

    quote = quote_list[0] or {}
    opens = quote.get("open") or []
    highs = quote.get("high") or []
    lows = quote.get("low") or []
    closes = quote.get("close") or []
    volumes = quote.get("volume") or []

    bars: list[dict[str, Any]] = []
    for i, ts in enumerate(timestamps):
        close = _safe_float(closes[i] if i < len(closes) else None)
        if close is None:
            continue
        ts_int = int(ts)
        bar: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(ts_int, tz=timezone.utc).isoformat().replace("+00:00", "Z"),
            "close": close,
        }
        open_price = _safe_float(opens[i] if i < len(opens) else None)
        high = _safe_float(highs[i] if i < len(highs) else None)
        low = _safe_float(lows[i] if i < len(lows) else None)
        volume = _safe_float(volumes[i] if i < len(volumes) else None)
        if open_price is not None:
            bar["open"] = open_price
        if high is not None:
            bar["high"] = high
        if low is not None:
            bar["low"] = low
        if volume is not None:
            bar["volume"] = volume
        bars.append(bar)

    if safe_limit and len(bars) > safe_limit:
        bars = bars[-safe_limit:]
    return bars


def _fetch_binance_history_bars(symbol: str, interval: str, limit: int) -> list[dict[str, Any]]:
    interval_key = str(interval or "hourly").lower().strip()
    binance_interval = "1h" if interval_key == "hourly" else "1d"
    safe_limit = max(2, min(int(limit or 100), 1000))
    payload: Any = None
    qs = urlencode(
        {
            "symbol": str(symbol or "").upper().strip(),
            "interval": binance_interval,
            "limit": safe_limit,
        }
    )
    for base_url in BINANCE_KLINES_FALLBACK_URLS:
        try:
            req = Request(f"{base_url}?{qs}", headers=YAHOO_HEADERS)
            with urlopen(req, timeout=15) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            if isinstance(payload, list):
                break
        except Exception:
            payload = None
            continue

    if not isinstance(payload, list):
        return []

    bars: list[dict[str, Any]] = []
    for row in payload:
        if not isinstance(row, list) or len(row) < 6:
            continue
        try:
            open_ms = int(row[0])
        except Exception:
            continue
        close = _safe_float(row[4])
        if close is None:
            continue
        bar: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(open_ms / 1000.0, tz=timezone.utc).isoformat().replace("+00:00", "Z"),
            "close": close,
        }
        open_price = _safe_float(row[1])
        high = _safe_float(row[2])
        low = _safe_float(row[3])
        volume = _safe_float(row[5])
        if open_price is not None:
            bar["open"] = open_price
        if high is not None:
            bar["high"] = high
        if low is not None:
            bar["low"] = low
        if volume is not None:
            bar["volume"] = volume
        bars.append(bar)
    return bars[-safe_limit:]


def _fetch_json(url: str, params: dict[str, Any], timeout: int = 12) -> Any:
    qs = urlencode(params)
    req = Request(f"{url}?{qs}", headers=YAHOO_HEADERS)
    with urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _chunked(items: list[str], chunk_size: int) -> list[list[str]]:
    if chunk_size <= 0:
        return [items]
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


def _put_live_price(price_map: dict[str, float], symbol: str, price: float | None) -> None:
    if price is None:
        return
    up = str(symbol or "").upper().strip()
    if not up:
        return
    price_map[up] = price
    key = _symbol_key(up)
    if key:
        price_map[key] = price


def _lookup_live_price(price_map: dict[str, float], symbol: str) -> float | None:
    up = str(symbol or "").upper().strip()
    if not up:
        return None
    if up in price_map:
        return _safe_float(price_map.get(up))
    key = _symbol_key(up)
    if key in price_map:
        return _safe_float(price_map.get(key))
    return None


def _fetch_binance_price_map(symbols: list[str]) -> dict[str, float]:
    uniq = sorted({str(s or "").upper().strip() for s in symbols if str(s or "").strip()})
    if not uniq:
        return {}
    cache_key = _cache_key("binance_ticker", ",".join(uniq))
    cached, found = _cache_get(cache_key)
    if found and isinstance(cached, dict):
        return dict(cached)

    price_map: dict[str, float] = {}
    for base_url in BINANCE_TICKER_FALLBACK_URLS:
        try:
            if len(uniq) == 1:
                payload = _fetch_json(base_url, {"symbol": uniq[0]})
                if isinstance(payload, dict):
                    _put_live_price(price_map, str(payload.get("symbol") or uniq[0]), _safe_float(payload.get("price")))
            else:
                payload = _fetch_json(base_url, {"symbols": json.dumps(uniq)})
                if isinstance(payload, list):
                    for row in payload:
                        if not isinstance(row, dict):
                            continue
                        _put_live_price(price_map, str(row.get("symbol") or ""), _safe_float(row.get("price")))
            if price_map:
                break
        except Exception:
            continue

    if not price_map:
        return {}

    _cache_set(cache_key, price_map)
    return price_map


def _fetch_yahoo_quote_rows(symbols: list[str]) -> dict[str, dict[str, Any]]:
    uniq = sorted({str(s or "").upper().strip() for s in symbols if str(s or "").strip()})
    if not uniq:
        return {}
    cache_key = _cache_key("yahoo_quote_rows", ",".join(uniq))
    cached, found = _cache_get(cache_key)
    if found and isinstance(cached, dict):
        return dict(cached)

    rows: dict[str, dict[str, Any]] = {}
    try:
        # Keep query string length bounded.
        for batch in _chunked(uniq, 40):
            payload = _fetch_json(YAHOO_QUOTE_URL, {"symbols": ",".join(batch)})
            result = ((payload or {}).get("quoteResponse") or {}).get("result") or []
            if not isinstance(result, list):
                continue
            for row in result:
                if not isinstance(row, dict):
                    continue
                sym = str(row.get("symbol") or "").upper().strip()
                if sym:
                    rows[sym] = row
    except Exception:
        return rows

    _cache_set(cache_key, rows)
    return rows


def _fetch_live_price_map(market: str, symbols: list[str]) -> dict[str, float]:
    market_key = _normalize_market_key(market)
    uniq = sorted({str(s or "").upper().strip() for s in symbols if str(s or "").strip()})
    if not uniq:
        return {}

    out: dict[str, float] = {}

    # Use per-symbol cache and reliable chart endpoints to avoid quote-api 401 issues.
    for symbol in uniq:
        cache_key = _cache_key("live_price", market_key, symbol)
        cached, found = _cache_get(cache_key)
        if found:
            _put_live_price(out, symbol, _safe_float(cached))
            continue

        price: float | None = None
        if market_key == "crypto":
            price = _lookup_live_price(_fetch_binance_price_map([symbol]), symbol)
            if price is None:
                bars = _fetch_yahoo_history_bars(symbol=symbol, market="crypto", interval="hourly", limit=2)
                if bars:
                    price = _safe_float((bars[-1] or {}).get("close"))
        elif market_key in {"cn_equity", "us_equity"}:
            bars = _fetch_yahoo_history_bars(symbol=symbol, market=market_key, interval="hourly", limit=2)
            if bars:
                price = _safe_float((bars[-1] or {}).get("close"))
            if price is None:
                bars = _fetch_yahoo_history_bars(symbol=symbol, market=market_key, interval="daily", limit=2)
                if bars:
                    price = _safe_float((bars[-1] or {}).get("close"))

        if price is not None:
            _cache_set(cache_key, price)
        _put_live_price(out, symbol, price)
    return out


def _apply_live_price_to_signal(signal: dict[str, Any], live_price: float | None) -> dict[str, Any]:
    if live_price is None:
        return signal
    signal["current_price"] = live_price
    q50 = _safe_float(signal.get("q50_change_pct"))
    if q50 is not None:
        target = round(live_price * (1 + q50), 6)
        signal["target_price"] = target
        signal["target_price_q50"] = target
    return signal


def _apply_live_price_to_prediction(item: dict[str, Any], live_price: float | None) -> dict[str, Any]:
    if live_price is None:
        return item
    item["current_price"] = live_price
    q50 = _safe_float(_coalesce(item.get("predicted_change_pct"), item.get("q50_change_pct")))
    if q50 is not None:
        target = round(live_price * (1 + q50), 6)
        item["predicted_price"] = target
        item["target_price"] = target
        item["support_level"] = live_price * (1 + min(q50, 0))
        item["resistance_level"] = live_price * (1 + max(q50, 0))
    sig = item.get("signal")
    if isinstance(sig, dict):
        item["signal"] = _apply_live_price_to_signal(sig, live_price)
    return item


def _refresh_predictions_live_prices(items: list[dict[str, Any]], market: str) -> list[dict[str, Any]]:
    if not items:
        return items
    symbols = [str(item.get("symbol") or "").upper().strip() for item in items]
    price_map = _fetch_live_price_map(market, symbols)
    if not price_map:
        return items
    out: list[dict[str, Any]] = []
    for item in items:
        symbol = str(item.get("symbol") or "").upper().strip()
        live_price = _lookup_live_price(price_map, symbol)
        out.append(_apply_live_price_to_prediction(item, live_price))
    return out


def _signal_strength_label(p_up: float | None) -> tuple[str, float]:
    if p_up is None:
        return "Weak", 0.0
    pp = abs((float(p_up) - 0.5) * 100.0)
    if pp >= 8:
        return "Strong", pp
    if pp >= 3:
        return "Medium", pp
    return "Weak", pp


def _clamp(value: float, min_value: float, max_value: float) -> float:
    return min(max_value, max(min_value, value))


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _as_prob(value: Any) -> float | None:
    num = _safe_float(value)
    if num is None:
        return None
    if abs(num) > 1:
        return num / 100.0
    return num


def _as_confidence_pct(value: Any) -> float | None:
    num = _safe_float(value)
    if num is None:
        return None
    if num <= 1:
        return num * 100.0
    return num


def _quantile(values: list[float], q: float) -> float:
    valid = sorted(v for v in values if isinstance(v, (int, float)) and math.isfinite(v))
    if not valid:
        return float("nan")
    if len(valid) == 1:
        return float(valid[0])
    qq = _clamp(float(q), 0.0, 1.0)
    pos = (len(valid) - 1) * qq
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(valid[lo])
    w = pos - lo
    return float(valid[lo] * (1.0 - w) + valid[hi] * w)


def _weighted_avg(values: list[float], weights: list[float]) -> float | None:
    total = 0.0
    acc = 0.0
    for v, w in zip(values, weights):
        if not (math.isfinite(v) and math.isfinite(w)):
            continue
        ww = max(0.0, w)
        total += ww
        acc += v * ww
    if total <= 0:
        valid = [v for v in values if math.isfinite(v)]
        return (sum(valid) / len(valid)) if valid else None
    return acc / total


def _session_name_cn(session_name: Any) -> str:
    return SESSION_NAME_ZH.get(str(session_name or "").strip().lower(), str(session_name or "-"))


def _session_hours_label(session_name: Any) -> str:
    return SESSION_HOURS.get(str(session_name or "").strip().lower(), "--")


def _session_from_hour_bj(hour: int) -> str:
    h = int(hour) % 24
    if 8 <= h <= 15:
        return "asia"
    if 16 <= h <= 23:
        return "europe"
    return "us"


def _direction_from_p_up(p_up: float | None) -> str:
    if p_up is None:
        return "震荡"
    if p_up >= 0.53:
        return "看涨"
    if p_up <= 0.47:
        return "看跌"
    return "震荡"


def _trend_from_values(p_up: float | None, q50: float | None) -> str:
    if p_up is None or q50 is None:
        return "sideways"
    if p_up >= 0.55 and q50 >= 0.002:
        return "bullish"
    if p_up <= 0.45 and q50 <= -0.002:
        return "bearish"
    return "sideways"


def _risk_from_vol(volatility: float | None, low_cut: float, high_cut: float) -> str:
    if volatility is None or not math.isfinite(volatility):
        return "medium"
    if volatility <= low_cut:
        return "low"
    if volatility <= high_cut:
        return "medium"
    return "high"


def _normalize_mode(raw_mode: Any) -> str:
    mode = str(raw_mode or "").strip().lower()
    if mode in {"forecast", "seasonality"}:
        return mode
    return "forecast"


def _mode_pair(mode: str) -> str:
    return "seasonality" if _normalize_mode(mode) == "forecast" else "forecast"


def _normalize_risk_profile(raw_profile: Any) -> str:
    profile = str(raw_profile or "").strip().lower()
    if profile in {"conservative", "保守"}:
        return "conservative"
    if profile in {"aggressive", "激进"}:
        return "aggressive"
    return "standard"


def _sort_rows_by_session(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda r: (
            SESSION_ORDER.get(str(r.get("session_name") or "").lower(), 9),
            str(r.get("session_name") or ""),
        ),
    )


def _sort_rows_by_hour(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda r: _safe_int(r.get("hour_bj"), 0))


def _pick_latest_forecast_id(
    rows: list[dict[str, Any]],
    *,
    symbol: str = "",
    exchange: str = "",
    market_type: str = "",
    mode: str = "",
    horizon: str = "",
) -> tuple[str | None, str]:
    if not rows:
        return None, _normalize_mode(mode)

    symbol_key = _symbol_key(symbol) if symbol else ""
    exchange_key = str(exchange or "").strip().lower()
    market_type_key = str(market_type or "").strip().lower()
    horizon_key = str(horizon or "").strip().lower()
    mode_key = _normalize_mode(mode)

    filtered = []
    for row in rows:
        if symbol_key and _symbol_key(row.get("symbol")) != symbol_key:
            continue
        if exchange_key and str(row.get("exchange") or "").strip().lower() != exchange_key:
            continue
        if market_type_key and str(row.get("market_type") or "").strip().lower() != market_type_key:
            continue
        if horizon_key and str(row.get("horizon") or "").strip().lower() != horizon_key:
            continue
        filtered.append(row)

    if not filtered:
        return None, mode_key

    mode_filtered = [r for r in filtered if _normalize_mode(r.get("mode")) == mode_key]
    if mode_filtered:
        filtered = mode_filtered

    def _row_rank(row: dict[str, Any]) -> tuple[str, str, str]:
        return (
            str(row.get("forecast_generated_at_bj") or ""),
            str(row.get("data_updated_at_bj") or ""),
            str(row.get("forecast_id") or ""),
        )

    latest = max(filtered, key=_row_rank)
    fid = str(latest.get("forecast_id") or "").strip()
    mode_actual = _normalize_mode(latest.get("mode") or latest.get("mode_requested") or mode_key)
    if fid:
        return fid, mode_actual
    fallback_key = str(latest.get("forecast_generated_at_bj") or "").strip()
    return (fallback_key or None), mode_actual


def _rows_for_forecast_id(rows: list[dict[str, Any]], forecast_id: str | None) -> list[dict[str, Any]]:
    if not rows or not forecast_id:
        return []
    key = str(forecast_id).strip()
    if "|" in key:
        out = [r for r in rows if str(r.get("forecast_id") or "").strip() == key]
        if out:
            return out
    return [r for r in rows if str(r.get("forecast_generated_at_bj") or "").strip() == key]


def _refresh_rows_live_price(rows: list[dict[str, Any]], market: str) -> list[dict[str, Any]]:
    if not rows:
        return []
    symbols = [str(r.get("symbol") or "").upper().strip() for r in rows if str(r.get("symbol") or "").strip()]
    if not symbols:
        return rows
    price_map = _fetch_live_price_map(market, symbols)
    if not price_map:
        return rows
    out: list[dict[str, Any]] = []
    for row in rows:
        new_row = dict(row)
        symbol = str(new_row.get("symbol") or "").upper().strip()
        live_price = _lookup_live_price(price_map, symbol)
        if live_price is not None:
            new_row["current_price"] = live_price
            q10 = _safe_float(new_row.get("q10_change_pct"))
            q50 = _safe_float(new_row.get("q50_change_pct"))
            q90 = _safe_float(new_row.get("q90_change_pct"))
            if q10 is not None:
                new_row["target_price_q10"] = live_price * (1.0 + q10)
            if q50 is not None:
                new_row["target_price_q50"] = live_price * (1.0 + q50)
            if q90 is not None:
                new_row["target_price_q90"] = live_price * (1.0 + q90)
        out.append(new_row)
    return out


def _signal_strength_fields(p_up: float | None) -> tuple[str, float, float]:
    label, pp = _signal_strength_label(p_up)
    score = _clamp(pp * 10.0, 0.0, 100.0)
    return label, pp, score


def _edge_metrics(row: dict[str, Any], cost_bps: float) -> dict[str, float | None]:
    q10 = _safe_float(row.get("q10_change_pct"))
    q50 = _safe_float(row.get("q50_change_pct"))
    q90 = _safe_float(row.get("q90_change_pct"))
    cost_pct = float(cost_bps) / 10000.0
    width = None
    if q10 is not None and q90 is not None:
        width = q90 - q10
    edge_score = (q50 - cost_pct) if q50 is not None else None
    edge_score_short = ((-q50) - cost_pct) if q50 is not None else None
    edge_risk = None
    edge_risk_short = None
    if width is not None and abs(width) > 1e-12:
        if edge_score is not None:
            edge_risk = edge_score / width
        if edge_score_short is not None:
            edge_risk_short = edge_score_short / width
    return {
        "edge_score": edge_score,
        "edge_score_short": edge_score_short,
        "edge_risk": edge_risk,
        "edge_risk_short": edge_risk_short,
        "volatility_width": width,
    }


def _enrich_session_row(row: dict[str, Any], cost_bps: float) -> dict[str, Any]:
    out = dict(row)
    p_up = _as_prob(out.get("p_up"))
    p_down = _as_prob(out.get("p_down"))
    if p_up is not None and p_down is None:
        p_down = 1.0 - p_up
    confidence = _as_confidence_pct(out.get("confidence_score"))
    if confidence is None:
        confidence = _as_confidence_pct(out.get("confidence"))
    signal_label, signal_pp, signal_score = _signal_strength_fields(p_up)
    out["p_up"] = p_up
    out["p_down"] = p_down
    out["confidence_score"] = confidence
    out["signal_strength_label"] = signal_label
    out["signal_strength_pp"] = signal_pp
    out["signal_strength_score"] = signal_score
    out["session_name"] = str(out.get("session_name") or "").lower()
    out["session_name_cn"] = _session_name_cn(out.get("session_name"))
    out["session_hours"] = _session_hours_label(out.get("session_name"))
    out["hour_bj"] = _safe_int(out.get("hour_bj"), 0)
    if not out.get("hour_label"):
        out["hour_label"] = f"{out['hour_bj']:02d}:00"
    metrics = _edge_metrics(out, cost_bps=cost_bps)
    out.update(metrics)
    return out


def _build_summary_from_blocks(blocks: list[dict[str, Any]]) -> dict[str, Any]:
    if not blocks:
        return {
            "overallTrend": "震荡",
            "bestSession": "-",
            "riskSession": "-",
            "confidence": 60.0,
        }
    work = [b for b in blocks if _safe_float(b.get("q50_change_pct")) is not None]
    if not work:
        work = blocks[:]
    best = max(work, key=lambda r: _safe_float(r.get("q50_change_pct")) or -999.0)
    risk = max(work, key=lambda r: abs(_safe_float(r.get("volatility_score")) or 0.0))
    avg_q50_values = [_safe_float(r.get("q50_change_pct")) for r in work]
    avg_q50 = (
        sum(v for v in avg_q50_values if v is not None) / max(1, len([v for v in avg_q50_values if v is not None]))
        if avg_q50_values
        else 0.0
    )
    conf_values = [_as_confidence_pct(r.get("confidence_score")) for r in work]
    conf_values = [v for v in conf_values if v is not None]
    confidence = (sum(conf_values) / len(conf_values)) if conf_values else 60.0
    if avg_q50 > 0.001:
        trend = "震荡偏多"
    elif avg_q50 < -0.001:
        trend = "震荡偏空"
    else:
        trend = "震荡"
    return {
        "overallTrend": trend,
        "bestSession": str(best.get("session_name") or "-"),
        "riskSession": str(risk.get("session_name") or "-"),
        "confidence": round(confidence, 2),
    }


def _bj_hour_now() -> int:
    return int((_utcnow() + timedelta(hours=8)).hour)


def _select_decision_row(hourly_rows: list[dict[str, Any]], active_session: str = "all") -> dict[str, Any] | None:
    if not hourly_rows:
        return None
    now_h = _bj_hour_now()
    tradable_rows: list[dict[str, Any]] = []
    for row in hourly_rows:
        session_name = str(row.get("session_name") or "").lower()
        tradable = True
        if active_session != "all" and session_name != active_session:
            tradable = False
        is_trading_hour = _safe_int(row.get("is_trading_hour"), 1)
        if is_trading_hour == 0:
            tradable = False
        if tradable:
            tradable_rows.append(row)
    if not tradable_rows:
        tradable_rows = hourly_rows[:]

    exact = [
        r
        for r in tradable_rows
        if _safe_int(r.get("hour_bj"), -1) == now_h and _as_prob(r.get("p_up")) is not None
    ]
    if exact:
        return exact[0]

    candidates: list[tuple[tuple[int, float], dict[str, Any]]] = []
    for row in tradable_rows:
        p_up = _as_prob(row.get("p_up"))
        hour = _safe_int(row.get("hour_bj"), 0) % 24
        delta = (hour - now_h) % 24
        strength = abs((p_up or 0.5) - 0.5)
        candidates.append(((delta, -strength), row))
    if not candidates:
        return tradable_rows[0]
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]


def _build_trade_plan_light(
    row: dict[str, Any],
    *,
    risk_profile: str = "standard",
    p_bull: float = 0.55,
    p_bear: float = 0.45,
    conf_min: float = 60.0,
    cost_bps: float = 20.0,
) -> dict[str, Any]:
    p_up = _as_prob(row.get("p_up"))
    p_down = (1.0 - p_up) if p_up is not None else None
    conf = _as_confidence_pct(row.get("confidence_score"))
    q10 = _safe_float(row.get("q10_change_pct"))
    q50 = _safe_float(row.get("q50_change_pct"))
    q90 = _safe_float(row.get("q90_change_pct"))
    price = _safe_float(row.get("current_price")) or _safe_float(row.get("target_price_q50")) or 0.0
    risk_level = str(row.get("risk_level") or "medium").lower()
    allow_short = bool(row.get("policy_allow_short", True))
    profile = _normalize_risk_profile(risk_profile)
    cost_pct = float(cost_bps) / 10000.0
    edge_long = (q50 - cost_pct) if q50 is not None else None
    edge_short = ((-q50) - cost_pct) if q50 is not None else None

    sl_scale = {"conservative": 0.85, "standard": 1.0, "aggressive": 1.2}.get(profile, 1.0)
    tp_scale = {"conservative": 0.9, "standard": 1.0, "aggressive": 1.15}.get(profile, 1.0)
    min_rr_required = 1.0

    def _rr_from_returns(tp_ret: float | None, sl_ret: float | None) -> float | None:
        if tp_ret is None or sl_ret is None:
            return None
        risk_ret = abs(float(sl_ret))
        reward_ret = abs(float(tp_ret))
        if risk_ret <= 1e-12:
            return None
        return reward_ret / risk_ret

    # LONG scenario
    long_sl_ret_raw = q10 if q10 is not None else -0.008
    long_tp1_ret_raw = q50 if q50 is not None else 0.004
    long_tp2_ret_raw = q90 if q90 is not None else max(long_tp1_ret_raw * 1.6, 0.008)
    long_sl_ret = min(long_sl_ret_raw * sl_scale, -0.001)
    # Enforce minimum TP1 reward: RR(TP1)>=1 plus cost buffer.
    long_min_reward_ret = abs(long_sl_ret) * min_rr_required + cost_pct
    long_tp1_ret = max(long_tp1_ret_raw * tp_scale, long_min_reward_ret, 0.001)
    long_tp2_ret = max(
        long_tp2_ret_raw * tp_scale,
        long_tp1_ret * 1.2,
        long_tp1_ret + max(cost_pct * 0.5, 0.001),
    )
    long_rr_tp1 = _rr_from_returns(long_tp1_ret, long_sl_ret)
    long_rr_tp2 = _rr_from_returns(long_tp2_ret, long_sl_ret)

    # SHORT scenario
    short_sl_ret_raw = q90 if q90 is not None else 0.008
    short_tp1_ret_raw = q50 if q50 is not None else -0.004
    short_tp2_ret_raw = q10 if q10 is not None else min(short_tp1_ret_raw * 1.6, -0.008)
    short_sl_ret = max(short_sl_ret_raw * sl_scale, 0.001)
    # Enforce minimum TP1 reward: RR(TP1)>=1 plus cost buffer.
    short_min_reward_ret = abs(short_sl_ret) * min_rr_required + cost_pct
    short_tp1_ret = min(short_tp1_ret_raw * tp_scale, -short_min_reward_ret, -0.001)
    short_tp2_ret = min(
        short_tp2_ret_raw * tp_scale,
        short_tp1_ret * 1.2,
        short_tp1_ret - max(cost_pct * 0.5, 0.001),
    )
    short_rr_tp1 = _rr_from_returns(short_tp1_ret, short_sl_ret)
    short_rr_tp2 = _rr_from_returns(short_tp2_ret, short_sl_ret)

    long_checks = [
        ("p_up >= 阈值", p_up is not None and p_up >= p_bull),
        ("edge_long > 0", edge_long is not None and edge_long > 0),
        ("confidence >= 最低阈值", conf is not None and conf >= conf_min),
        ("RR(TP1) >= 1.0 (不含TP2)", long_rr_tp1 is not None and long_rr_tp1 >= min_rr_required),
        ("风险非高位", risk_level not in {"high", "extreme"}),
    ]
    short_checks = [
        ("p_up <= 阈值", p_up is not None and p_up <= p_bear),
        ("edge_short > 0", edge_short is not None and edge_short > 0),
        ("confidence >= 最低阈值", conf is not None and conf >= conf_min),
        ("RR(TP1) >= 1.0 (不含TP2)", short_rr_tp1 is not None and short_rr_tp1 >= min_rr_required),
        ("允许做空", allow_short),
        ("风险非高位", risk_level not in {"high", "extreme"}),
    ]
    long_ok = all(ok for _, ok in long_checks)
    short_ok = all(ok for _, ok in short_checks)

    action = "WAIT"
    if long_ok and not short_ok:
        action = "LONG"
    elif short_ok and not long_ok:
        action = "SHORT"
    elif long_ok and short_ok:
        action = "LONG" if (edge_long or 0.0) >= (edge_short or 0.0) else "SHORT"

    if action == "LONG":
        plan_side = "LONG"
        sl_ret, tp1_ret, tp2_ret = long_sl_ret, long_tp1_ret, long_tp2_ret
        rr_tp1, rr_tp2 = long_rr_tp1, long_rr_tp2
    elif action == "SHORT":
        plan_side = "SHORT"
        sl_ret, tp1_ret, tp2_ret = short_sl_ret, short_tp1_ret, short_tp2_ret
        rr_tp1, rr_tp2 = short_rr_tp1, short_rr_tp2
    else:
        plan_side = "LONG" if (q50 or 0.0) >= 0 else "SHORT"
        if plan_side == "LONG":
            sl_ret, tp1_ret, tp2_ret = long_sl_ret, long_tp1_ret, long_tp2_ret
            rr_tp1, rr_tp2 = long_rr_tp1, long_rr_tp2
        else:
            sl_ret, tp1_ret, tp2_ret = short_sl_ret, short_tp1_ret, short_tp2_ret
            rr_tp1, rr_tp2 = short_rr_tp1, short_rr_tp2

    entry = price
    stop_loss = entry * (1.0 + sl_ret)
    take_profit = entry * (1.0 + tp1_ret)
    take_profit_2 = entry * (1.0 + tp2_ret)
    risk_abs = abs(entry - stop_loss)
    reward1_abs = abs(take_profit - entry)
    reward2_abs = abs(take_profit_2 - entry)
    rr = rr_tp1 if rr_tp1 is not None else ((reward1_abs / risk_abs) if risk_abs > 1e-12 else None)
    rr_tp2 = rr_tp2 if rr_tp2 is not None else ((reward2_abs / risk_abs) if risk_abs > 1e-12 else None)

    trade_status = "READY" if action in {"LONG", "SHORT"} else "WAIT_RULES"
    trade_status_text = {"READY": "可执行", "WAIT_RULES": "规则未通过"}.get(trade_status, trade_status)
    trade_status_note = (
        "方向概率、edge 和置信度通过，允许执行。"
        if trade_status == "READY"
        else "规则过滤未完全通过，建议继续观察。"
    )
    execution_state = "EXECUTABLE" if trade_status == "READY" else "WAIT_ENTRY"
    execution_state_text = "可执行" if execution_state == "EXECUTABLE" else "未到价"
    execution_state_icon = "✅" if execution_state == "EXECUTABLE" else "⏳"

    selected_checks_raw = long_checks if (action == "LONG" or (action == "WAIT" and plan_side == "LONG")) else short_checks
    failed_checks = [label for label, ok in selected_checks_raw if not ok]
    long_checks_struct = [{"name": label, "pass": bool(ok)} for label, ok in long_checks]
    short_checks_struct = [{"name": label, "pass": bool(ok)} for label, ok in short_checks]
    selected_checks_struct = [{"name": label, "pass": bool(ok)} for label, ok in selected_checks_raw]

    signal_label, signal_pp, signal_score = _signal_strength_fields(p_up)
    risk_norm = {"low": 1.0, "medium": 0.75, "high": 0.45, "extreme": 0.1}.get(risk_level, 0.4)
    edge_active = edge_long if plan_side == "LONG" else edge_short
    edge_norm = _clamp(((edge_active or 0.0) * 10000.0 + 10.0) / 40.0, 0.0, 1.0)
    conf_norm = _clamp((conf or 0.0) / 100.0, 0.0, 1.0)
    score = _clamp(100.0 * (0.35 * (signal_score / 100.0) + 0.30 * edge_norm + 0.20 * conf_norm + 0.15 * risk_norm), 0.0, 100.0)

    return {
        "action": action,
        "action_cn": {"LONG": "做多", "SHORT": "做空", "WAIT": "观望"}.get(action, "观望"),
        "entry": entry,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "take_profit_2": take_profit_2,
        "rr": rr,
        "rr_tp1": rr,
        "rr_tp2": rr_tp2,
        "plan_side": plan_side,
        "plan_side_text": "做多预案" if plan_side == "LONG" else "做空预案",
        "entry_band_pct": 0.0,
        "entry_gap_pct": 0.0,
        "entry_touched": True,
        "entry_touched_at": "",
        "trade_status": trade_status,
        "trade_status_text": trade_status_text,
        "trade_status_note": trade_status_note,
        "execution_state": execution_state,
        "execution_state_text": execution_state_text,
        "execution_state_icon": execution_state_icon,
        "signal_time_utc": _utcnow().isoformat(),
        "valid_until": (_utcnow() + timedelta(hours=4)).isoformat(),
        "price_source": str(row.get("data_source_actual") or row.get("price_source") or "-"),
        "price_timestamp_market": str(row.get("data_updated_at_bj") or "-"),
        "price_timestamp_utc": _utcnow().isoformat(),
        "consistency_reason_codes": [],
        "consistency_reason_text": "-",
        "failed_checks": failed_checks,
        "gate_reason_codes": [],
        "horizon_label": str(row.get("horizon") or "4h"),
        "risk_level": risk_level,
        "confidence_score": conf,
        "p_up": p_up,
        "p_down": p_down,
        "q10": q10,
        "q50": q50,
        "q90": q90,
        "edge_long": edge_long,
        "edge_short": edge_short,
        "cost_bps": cost_bps,
        "risk_profile": profile,
        "model_health": "中",
        "event_risk": False,
        "long_checks": long_checks_struct,
        "short_checks": short_checks_struct,
        "selected_checks": selected_checks_struct,
        "checks_passed": len([1 for _, ok in selected_checks_raw if ok]),
        "checks_total": len(selected_checks_raw),
        "signal_strength": signal_label,
        "signal_strength_text": signal_label,
        "signal_strength_score": signal_score,
        "signal_score": score,
        "policy_reason": str(row.get("policy_reason") or ""),
        "news_risk_level": str(row.get("news_risk_level") or "low"),
        "news_gate_pass": bool(row.get("news_gate_pass", True)),
        "news_event_risk": bool(row.get("news_event_risk", False)),
    }


def _session_consensus_light(
    main_row: dict[str, Any] | None,
    compare_blocks: list[dict[str, Any]],
) -> dict[str, Any]:
    if not main_row or not compare_blocks:
        return {
            "aligned": None,
            "badge": "未启用对照",
            "detail": "未计算 Forecast vs Seasonality 分歧。",
        }
    session_name = str(main_row.get("session_name") or "").strip().lower()
    target = None
    for row in compare_blocks:
        if str(row.get("session_name") or "").strip().lower() == session_name:
            target = row
            break
    if target is None:
        return {
            "aligned": None,
            "badge": "无对照数据",
            "detail": "对应时段缺少对照数据。",
        }
    p_main = _as_prob(main_row.get("p_up"))
    p_cmp = _as_prob(target.get("p_up"))
    q_main = _safe_float(main_row.get("q50_change_pct"))
    q_cmp = _safe_float(target.get("q50_change_pct"))
    if p_main is None or p_cmp is None:
        return {"aligned": None, "badge": "无对照数据", "detail": "概率字段缺失，无法计算一致性。"}
    aligned = (p_main >= 0.5 and p_cmp >= 0.5) or (p_main < 0.5 and p_cmp < 0.5)
    dp = (p_main - p_cmp) if (p_main is not None and p_cmp is not None) else None
    dq = (q_main - q_cmp) if (q_main is not None and q_cmp is not None) else None
    dp_txt = f"{dp * 100:+.2f}%" if dp is not None else "-"
    dq_txt = f"{dq * 100:+.2f}%" if dq is not None else "-"
    if aligned:
        return {
            "aligned": True,
            "badge": "✅ 同向（2/2）",
            "detail": f"Forecast 与 Seasonality 同向；Δp_up={dp_txt}，Δq50={dq_txt}。",
        }
    return {
        "aligned": False,
        "badge": "⚠️ 分歧（1/2）",
        "detail": f"Forecast 与 Seasonality 方向冲突；Δp_up={dp_txt}，Δq50={dq_txt}。建议降仓。",
    }


def _build_compare_rows(
    main_blocks: list[dict[str, Any]],
    compare_blocks: list[dict[str, Any]],
    *,
    main_mode: str,
    compare_mode: str,
) -> list[dict[str, Any]]:
    if not main_blocks or not compare_blocks:
        return []
    cmp_by_session = {
        str(r.get("session_name") or "").strip().lower(): r
        for r in compare_blocks
    }
    rows: list[dict[str, Any]] = []
    for row in _sort_rows_by_session(main_blocks):
        key = str(row.get("session_name") or "").strip().lower()
        cmp = cmp_by_session.get(key)
        if not cmp:
            continue
        p_main = _as_prob(row.get("p_up"))
        p_cmp = _as_prob(cmp.get("p_up"))
        q_main = _safe_float(row.get("q50_change_pct"))
        q_cmp = _safe_float(cmp.get("q50_change_pct"))
        rows.append(
            {
                "session_name": key,
                "session_name_cn": _session_name_cn(key),
                f"{main_mode}_p_up": p_main,
                f"{compare_mode}_p_up": p_cmp,
                f"{main_mode}_q50": q_main,
                f"{compare_mode}_q50": q_cmp,
                "delta_p_up": (p_main - p_cmp) if (p_main is not None and p_cmp is not None) else None,
                "delta_q50": (q_main - q_cmp) if (q_main is not None and q_cmp is not None) else None,
            }
        )
    return rows


def _rank_score(row: dict[str, Any], side: str, rank_key: str) -> float:
    key = str(rank_key or "edge_score").lower()
    side_key = str(side or "up").lower()
    p_up = _as_prob(row.get("p_up")) or 0.0
    p_down = _as_prob(row.get("p_down")) or (1.0 - p_up)
    q50 = _safe_float(row.get("q50_change_pct")) or 0.0
    vol = abs(_safe_float(row.get("volatility_score")) or 0.0)
    conf = _as_confidence_pct(row.get("confidence_score")) or 0.0
    signal_score = _safe_float(row.get("signal_strength_score")) or 0.0
    edge = _safe_float(row.get("edge_score")) or 0.0
    edge_short = _safe_float(row.get("edge_score_short")) or 0.0
    edge_risk = _safe_float(row.get("edge_risk")) or 0.0
    edge_risk_short = _safe_float(row.get("edge_risk_short")) or 0.0

    if key == "signal_strength":
        return signal_score
    if key == "p_up":
        return p_up if side_key != "down" else p_down
    if key == "confidence":
        return conf
    if key == "volatility":
        return vol
    if key == "q50":
        if side_key == "down":
            return -q50
        if side_key == "vol":
            return abs(q50)
        return q50
    if key == "edge_risk":
        if side_key == "down":
            return edge_risk_short
        if side_key == "vol":
            return vol + abs(edge_risk) * 0.01
        return edge_risk
    if side_key == "down":
        return edge_short
    if side_key == "vol":
        return vol + abs(edge) * 0.01
    return edge


def _build_top_tables(
    rows: list[dict[str, Any]],
    *,
    top_n: int,
    rank_key: str,
    cost_bps: float,
) -> dict[str, list[dict[str, Any]]]:
    if not rows:
        return {"up": [], "down": [], "vol": []}
    enriched = [_enrich_session_row(r, cost_bps=cost_bps) for r in rows]
    enriched = [r for r in enriched if _as_prob(r.get("p_up")) is not None]
    if not enriched:
        return {"up": [], "down": [], "vol": []}

    def _top(side: str) -> list[dict[str, Any]]:
        ranked = sorted(
            enriched,
            key=lambda r: _rank_score(r, side=side, rank_key=rank_key),
            reverse=True,
        )
        return ranked[: max(1, int(top_n))]

    return {"up": _top("up"), "down": _top("down"), "vol": _top("vol")}


def _build_sim_path(
    daily_rows: list[dict[str, Any]],
    current_price: float | None,
    *,
    lookforward_days: int = 14,
    anchor_q10: float | None = None,
    anchor_q50: float | None = None,
    anchor_q90: float | None = None,
) -> list[dict[str, Any]]:
    price = _safe_float(current_price)
    if price is None or price <= 0:
        return []
    rows = sorted(daily_rows, key=lambda r: _safe_int(r.get("day_index"), 0))
    n_steps = max(1, int(lookforward_days or 14))
    if rows:
        n_steps = max(1, min(n_steps, len(rows)))

    def _ret_from_row(row: dict[str, Any], key: str, target_key: str) -> float | None:
        ret = _safe_float(row.get(key))
        target = _safe_float(row.get(target_key))
        if target is not None and target > 0 and price > 0:
            ret = (target / price) - 1.0
        return ret

    # Prefer decision-row quantiles (same口径 as dashboard projection chart),
    # fallback to daily tail quantiles when anchors are absent.
    q10_end = _safe_float(anchor_q10)
    q50_end = _safe_float(anchor_q50)
    q90_end = _safe_float(anchor_q90)
    if (q10_end is None or q50_end is None or q90_end is None) and rows:
        tail_row = rows[min(n_steps, len(rows)) - 1]
        if q10_end is None:
            q10_end = _ret_from_row(tail_row, "q10_change_pct", "target_price_q10")
        if q50_end is None:
            q50_end = _ret_from_row(tail_row, "q50_change_pct", "target_price_q50")
        if q90_end is None:
            q90_end = _ret_from_row(tail_row, "q90_change_pct", "target_price_q90")

    q10_end = float(q10_end or 0.0)
    q50_end = float(q50_end or 0.0)
    q90_end = float(q90_end or 0.0)

    # Keep projection stable and comparable with dashboard.
    q10_end = _clamp(q10_end, -0.20, 0.20)
    q50_end = _clamp(q50_end, -0.20, 0.20)
    q90_end = _clamp(q90_end, -0.20, 0.20)
    q10_end, q50_end, q90_end = sorted([q10_end, q50_end, q90_end])
    eps = 5e-4
    if q50_end <= q10_end:
        q50_end = min(0.20, q10_end + eps)
    if q90_end <= q50_end:
        q90_end = min(0.20, q50_end + eps)

    out: list[dict[str, Any]] = [
        {"label": "Now", "date_bj": "", "q10": price, "q50": price, "q90": price}
    ]
    for day_idx in range(1, n_steps + 1):
        # Smooth interpolation from Now -> terminal quantile anchors.
        t = float(day_idx) / float(n_steps)
        ease = 1.0 - (1.0 - t) ** 1.35
        q10 = q10_end * ease
        q50 = q50_end * ease
        q90 = q90_end * ease
        date_bj = ""
        if day_idx - 1 < len(rows):
            date_bj = str(rows[day_idx - 1].get("date_bj") or "")
        out.append(
            {
                "label": f"D{day_idx}",
                "date_bj": date_bj,
                "q10": price * (1.0 + q10),
                "q50": price * (1.0 + q50),
                "q90": price * (1.0 + q90),
            }
        )
    return out


def _hourly_stats_by_hour(
    bars: list[dict[str, Any]],
    *,
    horizon_hours: int,
    recent_days: int,
    mode: str,
) -> tuple[list[dict[str, Any]], str]:
    mode_key = _normalize_mode(mode)
    if len(bars) < max(4, horizon_hours + 2):
        return [], mode_key
    parsed: list[tuple[datetime, float]] = []
    for bar in bars:
        ts_txt = str(bar.get("timestamp") or "").strip()
        close = _safe_float(bar.get("close"))
        if not ts_txt or close is None:
            continue
        ts_norm = ts_txt.replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(ts_norm)
        except Exception:
            continue
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        parsed.append((dt, close))
    parsed.sort(key=lambda x: x[0])
    if len(parsed) < max(4, horizon_hours + 2):
        return [], mode_key

    all_samples: dict[int, list[float]] = defaultdict(list)
    recent_samples: dict[int, list[float]] = defaultdict(list)
    all_rets: list[float] = []
    recent_cut = _utcnow() - timedelta(days=max(14, int(recent_days)))
    h = max(1, int(horizon_hours))

    for idx in range(0, len(parsed) - h):
        dt0, p0 = parsed[idx]
        _, p1 = parsed[idx + h]
        if p0 <= 0:
            continue
        ret = (p1 / p0) - 1.0
        if not math.isfinite(ret):
            continue
        hour_bj = (dt0 + timedelta(hours=8)).hour
        all_samples[int(hour_bj)].append(ret)
        all_rets.append(ret)
        if dt0 >= recent_cut:
            recent_samples[int(hour_bj)].append(ret)

    if not all_rets:
        return [], mode_key
    g = {
        "p_up": sum(1 for v in all_rets if v > 0) / len(all_rets),
        "q10": _quantile(all_rets, 0.1),
        "q50": _quantile(all_rets, 0.5),
        "q90": _quantile(all_rets, 0.9),
        "sample_size": len(all_rets),
    }
    profile: list[dict[str, Any]] = []
    has_recent = any(len(v) >= 12 for v in recent_samples.values())
    mode_actual = mode_key
    if mode_key == "forecast" and not has_recent:
        mode_actual = "seasonality"
    for hour in range(24):
        base_vals = all_samples.get(hour, [])
        if base_vals:
            base = {
                "p_up": sum(1 for v in base_vals if v > 0) / len(base_vals),
                "q10": _quantile(base_vals, 0.1),
                "q50": _quantile(base_vals, 0.5),
                "q90": _quantile(base_vals, 0.9),
                "sample_size": len(base_vals),
            }
        else:
            base = dict(g)
            base["sample_size"] = 0

        if mode_key == "forecast" and has_recent:
            rec_vals = recent_samples.get(hour, [])
            if len(rec_vals) >= 12:
                recent = {
                    "p_up": sum(1 for v in rec_vals if v > 0) / len(rec_vals),
                    "q10": _quantile(rec_vals, 0.1),
                    "q50": _quantile(rec_vals, 0.5),
                    "q90": _quantile(rec_vals, 0.9),
                    "sample_size": len(rec_vals),
                }
                p_up = 0.65 * recent["p_up"] + 0.35 * base["p_up"]
                q10 = 0.65 * recent["q10"] + 0.35 * base["q10"]
                q50 = 0.65 * recent["q50"] + 0.35 * base["q50"]
                q90 = 0.65 * recent["q90"] + 0.35 * base["q90"]
                sample_size = recent["sample_size"]
            else:
                p_up, q10, q50, q90, sample_size = (
                    base["p_up"],
                    base["q10"],
                    base["q50"],
                    base["q90"],
                    base["sample_size"],
                )
        else:
            p_up, q10, q50, q90, sample_size = (
                base["p_up"],
                base["q10"],
                base["q50"],
                base["q90"],
                base["sample_size"],
            )

        profile.append(
            {
                "hour_bj": hour,
                "p_up": p_up,
                "q10_change_pct": q10,
                "q50_change_pct": q50,
                "q90_change_pct": q90,
                "sample_size": sample_size,
            }
        )
    return profile, mode_actual


def _build_hourly_rows_from_profile(
    profile: list[dict[str, Any]],
    *,
    current_price: float,
    active_session: str = "all",
) -> list[dict[str, Any]]:
    if not profile:
        return []
    vols = [
        (_safe_float(r.get("q90_change_pct")) or 0.0) - (_safe_float(r.get("q10_change_pct")) or 0.0)
        for r in profile
    ]
    valid_vols = [v for v in vols if math.isfinite(v)]
    low_cut = _quantile(valid_vols, 0.35) if valid_vols else 0.01
    high_cut = _quantile(valid_vols, 0.75) if valid_vols else 0.02
    vmin = min(valid_vols) if valid_vols else 0.0
    vmax = max(valid_vols) if valid_vols else 1.0
    width = max(1e-9, vmax - vmin)

    rows: list[dict[str, Any]] = []
    for row in profile:
        p_up = _as_prob(row.get("p_up"))
        q10 = _safe_float(row.get("q10_change_pct"))
        q50 = _safe_float(row.get("q50_change_pct"))
        q90 = _safe_float(row.get("q90_change_pct"))
        vol = (q90 - q10) if (q10 is not None and q90 is not None) else None
        conf = None
        if p_up is not None and vol is not None:
            conf_prob = _clamp(abs(p_up - 0.5) * 2.0, 0.0, 1.0)
            vol_norm = _clamp((vol - vmin) / width, 0.0, 1.0)
            conf = 100.0 * (0.6 * conf_prob + 0.4 * (1.0 - vol_norm))
        hour_bj = _safe_int(row.get("hour_bj"), 0) % 24
        session_name = _session_from_hour_bj(hour_bj)
        is_trading_hour = 1 if active_session == "all" or session_name == active_session else 0
        out = {
            "hour_bj": hour_bj,
            "hour_label": f"{hour_bj:02d}:00",
            "session_name": session_name,
            "session_name_cn": _session_name_cn(session_name),
            "p_up": p_up if is_trading_hour == 1 else None,
            "p_down": (1.0 - p_up) if (p_up is not None and is_trading_hour == 1) else None,
            "q10_change_pct": q10 if is_trading_hour == 1 else None,
            "q50_change_pct": q50 if is_trading_hour == 1 else None,
            "q90_change_pct": q90 if is_trading_hour == 1 else None,
            "volatility_score": vol if is_trading_hour == 1 else None,
            "target_price_q10": (current_price * (1.0 + q10)) if (q10 is not None and is_trading_hour == 1) else None,
            "target_price_q50": (current_price * (1.0 + q50)) if (q50 is not None and is_trading_hour == 1) else None,
            "target_price_q90": (current_price * (1.0 + q90)) if (q90 is not None and is_trading_hour == 1) else None,
            "trend_label": _trend_from_values(p_up, q50) if is_trading_hour == 1 else "-",
            "risk_level": _risk_from_vol(vol, low_cut, high_cut) if is_trading_hour == 1 else "-",
            "confidence_score": conf if is_trading_hour == 1 else None,
            "sample_size": _safe_int(row.get("sample_size"), 0),
            "is_trading_hour": is_trading_hour,
        }
        rows.append(out)
    return _sort_rows_by_hour(rows)


def _aggregate_blocks(hourly_rows: list[dict[str, Any]], *, active_session: str = "all") -> list[dict[str, Any]]:
    if not hourly_rows:
        return []
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in hourly_rows:
        s = str(row.get("session_name") or "").lower()
        if not s:
            continue
        if active_session != "all" and s != active_session:
            continue
        if _as_prob(row.get("p_up")) is None:
            continue
        groups[s].append(row)
    if not groups and active_session != "all":
        return _aggregate_blocks(hourly_rows, active_session="all")

    out: list[dict[str, Any]] = []
    for s, rows in groups.items():
        weights = [float(max(1, _safe_int(r.get("sample_size"), 1))) for r in rows]
        p_up = _weighted_avg([_as_prob(r.get("p_up")) or float("nan") for r in rows], weights)
        q10 = _weighted_avg([_safe_float(r.get("q10_change_pct")) or float("nan") for r in rows], weights)
        q50 = _weighted_avg([_safe_float(r.get("q50_change_pct")) or float("nan") for r in rows], weights)
        q90 = _weighted_avg([_safe_float(r.get("q90_change_pct")) or float("nan") for r in rows], weights)
        conf = _weighted_avg([_as_confidence_pct(r.get("confidence_score")) or float("nan") for r in rows], weights)
        vol = None
        if q10 is not None and q90 is not None:
            vol = q90 - q10
        current_price = _safe_float(rows[-1].get("current_price"))
        out.append(
            {
                "session_name": s,
                "session_name_cn": _session_name_cn(s),
                "session_hours": _session_hours_label(s),
                "p_up": p_up,
                "p_down": (1.0 - p_up) if p_up is not None else None,
                "q10_change_pct": q10,
                "q50_change_pct": q50,
                "q90_change_pct": q90,
                "volatility_score": vol,
                "confidence_score": conf,
                "sample_size": int(sum(weights)),
                "current_price": current_price,
                "target_price_q10": (current_price * (1.0 + q10)) if (current_price is not None and q10 is not None) else None,
                "target_price_q50": (current_price * (1.0 + q50)) if (current_price is not None and q50 is not None) else None,
                "target_price_q90": (current_price * (1.0 + q90)) if (current_price is not None and q90 is not None) else None,
                "trend_label": _trend_from_values(p_up, q50),
                "risk_level": _risk_from_vol(vol, _quantile([abs(_safe_float(r.get("volatility_score")) or 0.0) for r in rows], 0.35), _quantile([abs(_safe_float(r.get("volatility_score")) or 0.0) for r in rows], 0.75)),
            }
        )
    return _sort_rows_by_session(out)


def _next_weekdays(start_day: datetime, n: int) -> list[datetime]:
    out: list[datetime] = []
    cur = start_day
    while len(out) < max(1, int(n)):
        cur = cur + timedelta(days=1)
        if cur.weekday() >= 5:
            continue
        out.append(cur)
    return out


def _build_daily_rows_from_bars(
    bars: list[dict[str, Any]],
    *,
    lookforward_days: int,
    mode: str,
    current_price: float,
) -> tuple[list[dict[str, Any]], str]:
    mode_key = _normalize_mode(mode)
    if len(bars) < 20:
        return [], mode_key
    parsed: list[tuple[datetime, float]] = []
    for bar in bars:
        ts_txt = str(bar.get("timestamp") or "").strip()
        close = _safe_float(bar.get("close"))
        if not ts_txt or close is None:
            continue
        ts_norm = ts_txt.replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(ts_norm)
        except Exception:
            continue
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        parsed.append((dt, close))
    parsed.sort(key=lambda x: x[0])
    if len(parsed) < 20:
        return [], mode_key

    all_by_dow: dict[str, list[float]] = defaultdict(list)
    recent_by_dow: dict[str, list[float]] = defaultdict(list)
    all_rets: list[float] = []
    recent_cut = _utcnow() - timedelta(days=180)

    for idx in range(0, len(parsed) - 1):
        dt0, p0 = parsed[idx]
        _, p1 = parsed[idx + 1]
        if p0 <= 0:
            continue
        ret = (p1 / p0) - 1.0
        if not math.isfinite(ret):
            continue
        day_bj = (dt0 + timedelta(hours=8)).strftime("%A")
        all_by_dow[day_bj].append(ret)
        all_rets.append(ret)
        if dt0 >= recent_cut:
            recent_by_dow[day_bj].append(ret)

    if not all_rets:
        return [], mode_key
    g = {
        "p_up": sum(1 for v in all_rets if v > 0) / len(all_rets),
        "q10": _quantile(all_rets, 0.1),
        "q50": _quantile(all_rets, 0.5),
        "q90": _quantile(all_rets, 0.9),
        "sample_size": len(all_rets),
    }
    has_recent = any(len(v) >= 8 for v in recent_by_dow.values())
    mode_actual = mode_key
    if mode_key == "forecast" and not has_recent:
        mode_actual = "seasonality"

    today_bj = (_utcnow() + timedelta(hours=8)).replace(hour=0, minute=0, second=0, microsecond=0)
    future_days = _next_weekdays(today_bj, lookforward_days)
    rows: list[dict[str, Any]] = []
    for i, day_dt in enumerate(future_days, start=1):
        dow = day_dt.strftime("%A")
        vals = all_by_dow.get(dow, [])
        if vals:
            base = {
                "p_up": sum(1 for v in vals if v > 0) / len(vals),
                "q10": _quantile(vals, 0.1),
                "q50": _quantile(vals, 0.5),
                "q90": _quantile(vals, 0.9),
                "sample_size": len(vals),
            }
        else:
            base = dict(g)
            base["sample_size"] = 0

        if mode_key == "forecast" and has_recent and len(recent_by_dow.get(dow, [])) >= 8:
            rec_vals = recent_by_dow[dow]
            rec = {
                "p_up": sum(1 for v in rec_vals if v > 0) / len(rec_vals),
                "q10": _quantile(rec_vals, 0.1),
                "q50": _quantile(rec_vals, 0.5),
                "q90": _quantile(rec_vals, 0.9),
                "sample_size": len(rec_vals),
            }
            p_up = 0.6 * rec["p_up"] + 0.4 * base["p_up"]
            q10 = 0.6 * rec["q10"] + 0.4 * base["q10"]
            q50 = 0.6 * rec["q50"] + 0.4 * base["q50"]
            q90 = 0.6 * rec["q90"] + 0.4 * base["q90"]
            sample_size = rec["sample_size"]
        else:
            p_up, q10, q50, q90, sample_size = (
                base["p_up"],
                base["q10"],
                base["q50"],
                base["q90"],
                base["sample_size"],
            )
        vol = q90 - q10
        conf_prob = _clamp(abs(p_up - 0.5) * 2.0, 0.0, 1.0)
        conf = 100.0 * (0.6 * conf_prob + 0.4 * _clamp(1.0 - abs(vol) / 0.2, 0.0, 1.0))
        rows.append(
            {
                "day_index": i,
                "date_bj": day_dt.strftime("%Y-%m-%d"),
                "day_of_week": dow,
                "p_up": p_up,
                "p_down": 1.0 - p_up,
                "q10_change_pct": q10,
                "q50_change_pct": q50,
                "q90_change_pct": q90,
                "volatility_score": vol,
                "target_price_q10": current_price * (1.0 + q10),
                "target_price_q50": current_price * (1.0 + q50),
                "target_price_q90": current_price * (1.0 + q90),
                "trend_label": _trend_from_values(p_up, q50),
                "risk_level": _risk_from_vol(vol, 0.03, 0.08),
                "confidence_score": conf,
                "sample_size": sample_size,
                "start_window_top1": "W1",
            }
        )
    return rows, mode_actual


def _model_health_text(hourly_rows: list[dict[str, Any]]) -> tuple[str, str]:
    conf_values = [_as_confidence_pct(r.get("confidence_score")) for r in hourly_rows]
    conf_values = [v for v in conf_values if v is not None]
    avg_conf = (sum(conf_values) / len(conf_values)) if conf_values else 50.0
    if avg_conf >= 70:
        level = "良"
    elif avg_conf >= 50:
        level = "中"
    else:
        level = "弱"
    return level, f"{level}（Confidence {avg_conf:.1f}%）"


def _to_bool_query(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    txt = str(value).strip().lower()
    if txt in {"1", "true", "yes", "y", "on"}:
        return True
    if txt in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _build_signal_payload(row: dict[str, Any], symbol_override: str | None = None) -> dict[str, Any]:
    symbol = str(symbol_override or row.get("symbol") or "").upper().strip()
    current_price = _safe_float(row.get("current_price"))
    q10 = _safe_float(row.get("q10_change_pct"))
    q50 = _safe_float(_coalesce(row.get("q50_change_pct"), row.get("predicted_change_pct")))
    q90 = _safe_float(row.get("q90_change_pct"))
    target_price = _safe_float(_coalesce(row.get("target_price_q50"), row.get("predicted_price")))
    if target_price is None and current_price is not None and q50 is not None:
        target_price = round(current_price * (1 + q50), 6)

    p_up = _safe_float(_coalesce(row.get("p_up"), row.get("policy_p_up_used")))
    p_down = _safe_float(row.get("p_down"))
    if p_down is None and p_up is not None:
        p_down = max(0.0, 1.0 - p_up)

    confidence_pct = _to_confidence_pct(_coalesce(row.get("confidence_score"), row.get("confidence")))
    action = str(_coalesce(row.get("policy_action"), row.get("action"), "Flat"))
    trend_label = str(_coalesce(row.get("trend_label"), "neutral"))
    risk_level = str(_coalesce(row.get("risk_level"), "medium"))
    volatility_score = _safe_float(row.get("volatility_score"))
    position_size = _safe_float(_coalesce(row.get("policy_position_size"), row.get("position_size")))
    if position_size is None or action.lower() == "flat":
        position_size = 0.0
    strength_label, strength_pp = _signal_strength_label(p_up)

    return {
        "symbol": symbol,
        "name": row.get("name"),
        "action": action,
        "policy_action": action,
        "p_up": p_up,
        "p_down": p_down,
        "confidence": confidence_pct,
        "confidence_score": confidence_pct,
        "current_price": current_price,
        "target_price": target_price,
        "target_price_q50": target_price,
        "q10_change_pct": q10,
        "q50_change_pct": q50,
        "q90_change_pct": q90,
        "volatility_score": volatility_score,
        "signal_strength": strength_label,
        "signal_strength_pp": strength_pp,
        "trend_label": trend_label,
        "risk_level": risk_level,
        "position_size": position_size,
        "policy_position_size": position_size,
        "policy_reason": row.get("policy_reason", ""),
        "forecast_generated_at": _coalesce(row.get("forecast_generated_at_bj"), row.get("timestamp_utc"), ""),
    }


def _prediction_from_row(row: dict[str, Any]) -> dict[str, Any]:
    signal = _build_signal_payload(row)
    current_price = _safe_float(signal.get("current_price"))
    q10 = _safe_float(signal.get("q10_change_pct"))
    q50 = _safe_float(signal.get("q50_change_pct"))
    q90 = _safe_float(signal.get("q90_change_pct"))
    support = _safe_float(row.get("target_price_q10"))
    target = _safe_float(_coalesce(signal.get("target_price"), row.get("target_price_q50"), row.get("predicted_price")))
    resistance = _safe_float(row.get("target_price_q90"))
    if current_price is not None:
        if support is None and q10 is not None:
            support = current_price * (1 + q10)
        if target is None and q50 is not None:
            target = current_price * (1 + q50)
        if resistance is None and q90 is not None:
            resistance = current_price * (1 + q90)
    change_percent = None
    if q50 is not None:
        change_percent = q50 * 100
    return {
        "symbol": signal.get("symbol"),
        "name": _coalesce(row.get("name"), row.get("instrument_id"), signal.get("symbol")),
        "current_price": current_price,
        "change_percent": change_percent,
        "predicted_change_pct": q50,
        "predicted_price": target,
        "target_price": target,
        "support_level": support,
        "resistance_level": resistance,
        "p_up": signal.get("p_up"),
        "p_down": signal.get("p_down"),
        "confidence_score": signal.get("confidence_score"),
        "policy_action": signal.get("policy_action"),
        "signal": signal,
    }


def _load_tracking_rows(market: str | None = None) -> list[dict[str, Any]]:
    data_path = _get_data_path()
    candidates = [
        data_path / "tracking" / "tracking_snapshot.csv",
        data_path / "tracking" / "policy_signals_multi_market.csv",
    ]
    rows: list[dict[str, Any]] = []
    for csv_path in candidates:
        rows = _read_csv_with_headers(csv_path, limit=20000)
        if rows:
            break
    if not rows:
        return []
    if market:
        market_key = _normalize_market_key(market)
        rows = [r for r in rows if _normalize_market_key(r.get("market")) == market_key]
    return rows


def _load_snapshot_rows() -> list[dict[str, Any]]:
    snapshot_path = _get_data_path() / "market_snapshot.json"
    if not snapshot_path.exists():
        return []
    try:
        with open(snapshot_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        rows = payload.get("rows", [])
        if isinstance(rows, list):
            return [r for r in rows if isinstance(r, dict)]
    except Exception:
        return []
    return []


def _get_market_predictions(market: str, limit: int = 100, symbol_filter: str = "") -> list[dict[str, Any]]:
    market_key = _normalize_market_key(market)
    symbol_filter = str(symbol_filter or "").strip().upper()
    out: list[dict[str, Any]] = []

    if market_key == "crypto":
        blocks_file = _get_data_path() / "session_forecast_blocks.csv"
        rows = _get_latest_csv_rows(blocks_file, 5000)
        latest_by_symbol: dict[str, dict[str, Any]] = {}
        for row in rows:
            symbol = str(row.get("symbol") or "").upper()
            if not symbol:
                continue
            latest_by_symbol[symbol] = row
        preferred = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
        ordered_symbols = preferred + [s for s in latest_by_symbol.keys() if s not in preferred]
        for symbol in ordered_symbols:
            if symbol_filter and _symbol_key(symbol) != _symbol_key(symbol_filter):
                continue
            row = latest_by_symbol.get(symbol)
            if row is None:
                continue
            out.append(_prediction_from_row(row))
            if len(out) >= limit:
                break
        return _refresh_predictions_live_prices(out, market_key)

    rows = _load_tracking_rows(market=market_key)
    for row in rows:
        symbol = str(row.get("symbol") or "").upper()
        if not symbol:
            continue
        if symbol_filter and _symbol_key(symbol) != _symbol_key(symbol_filter):
            continue
        out.append(_prediction_from_row(row))
        if len(out) >= limit:
            break

    if out:
        return _refresh_predictions_live_prices(out, market_key)

    # fallback: market_snapshot.json usually has at least one symbol per market
    for row in _load_snapshot_rows():
        row_market = _normalize_market_key(row.get("market"))
        if row_market != market_key:
            continue
        symbol = str(row.get("symbol") or "").upper()
        if symbol_filter and _symbol_key(symbol) != _symbol_key(symbol_filter):
            continue
        out.append(_prediction_from_row(row))
        if len(out) >= limit:
            break
    return _refresh_predictions_live_prices(out, market_key)


def _find_market_signal(market: str, symbol: str) -> dict[str, Any] | None:
    market_key = _normalize_market_key(market)
    symbol = str(symbol or "").upper().strip()
    if not symbol:
        return None

    if market_key == "crypto":
        rows = _get_latest_csv_rows(_get_data_path() / "session_forecast_blocks.csv", 5000)
        row = _pick_latest_symbol_row(rows, symbol)
        signal = _build_signal_payload(row, symbol_override=symbol) if row else None
        if signal:
            live_price = _lookup_live_price(_fetch_live_price_map(market_key, [symbol]), symbol)
            signal = _apply_live_price_to_signal(signal, live_price)
        return signal

    rows = _load_tracking_rows(market=market_key)
    row = _pick_latest_symbol_row(rows, symbol)
    if row:
        signal = _build_signal_payload(row, symbol_override=symbol)
        live_price = _lookup_live_price(_fetch_live_price_map(market_key, [symbol]), symbol)
        return _apply_live_price_to_signal(signal, live_price)

    # fallback to snapshot row
    for snapshot_row in _load_snapshot_rows():
        if _normalize_market_key(snapshot_row.get("market")) != market_key:
            continue
        if _symbol_key(snapshot_row.get("symbol")) == _symbol_key(symbol):
            signal = _build_signal_payload(snapshot_row, symbol_override=symbol)
            live_price = _lookup_live_price(_fetch_live_price_map(market_key, [symbol]), symbol)
            return _apply_live_price_to_signal(signal, live_price)
    return None


def _fallback_index_payload(market: str) -> dict[str, dict[str, Any]]:
    if market == "cn":
        return {
            "sh": {"price": 3085.24, "change": 0.68, "open": 3065.0, "high": 3092.0, "low": 3060.0, "volume": "3250亿"},
            "sz": {"price": 10256.78, "change": 1.12, "open": 10200.0, "high": 10280.0, "low": 10150.0, "volume": "4120亿"},
            "cyb": {"price": 2085.36, "change": -0.45, "open": 2095.0, "high": 2100.0, "low": 2080.0, "volume": "1850亿"},
        }
    return {
        "dji": {"price": 38675.68, "change": 0.85, "open": 38500.0, "high": 38750.0, "low": 38400.0, "volume": "320M"},
        "ixic": {"price": 16156.33, "change": 1.24, "open": 16000.0, "high": 16200.0, "low": 15950.0, "volume": "5.2B"},
        "spx": {"price": 5123.41, "change": -0.32, "open": 5140.0, "high": 5150.0, "low": 5110.0, "volume": "2.1B"},
    }


def _live_index_payload(market: str) -> dict[str, dict[str, Any]]:
    fallback = _fallback_index_payload(market)
    if market == "cn":
        symbol_map = {
            "sh": "000001.SS",
            "sz": "399001.SZ",
            "cyb": "399006.SZ",
        }
    else:
        symbol_map = {
            "dji": "^DJI",
            "ixic": "^IXIC",
            "spx": "^GSPC",
        }

    out: dict[str, dict[str, Any]] = {}
    for key, yahoo_symbol in symbol_map.items():
        base = dict(fallback.get(key, {}))
        market_key = "cn_equity" if market == "cn" else "us_equity"
        bars = _fetch_yahoo_history_bars(
            symbol=yahoo_symbol,
            market=market_key,
            interval="hourly",
            limit=2,
        )
        if not bars:
            bars = _fetch_yahoo_history_bars(
                symbol=yahoo_symbol,
                market=market_key,
                interval="daily",
                limit=2,
            )
        if bars:
            last = bars[-1]
            prev = bars[-2] if len(bars) >= 2 else None
            price = _safe_float(last.get("close"))
            open_price = _safe_float(last.get("open"))
            high = _safe_float(last.get("high"))
            low = _safe_float(last.get("low"))
            volume = _safe_float(last.get("volume"))
            if price is not None:
                base["price"] = price
            if open_price is not None:
                base["open"] = open_price
            if high is not None:
                base["high"] = high
            if low is not None:
                base["low"] = low
            if volume is not None:
                base["volume"] = int(volume)
            prev_close = _safe_float((prev or {}).get("close"))
            if price is not None and prev_close is not None and prev_close != 0:
                base["change"] = ((price / prev_close) - 1.0) * 100.0
        out[key] = base
    return out


def _to_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    txt = str(value).strip().lower()
    if txt in {"1", "true", "yes", "y", "on"}:
        return True
    if txt in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _normalize_policy_action(value: Any) -> str:
    raw = str(value or "").strip().lower()
    if raw in {"long", "buy", "keep/open", "open", "bullish"}:
        return "Long"
    if raw in {"short", "sell", "reduce", "monitor/reduce", "bearish"}:
        return "Short"
    if raw in {"wait", "flat", "hold", "neutral"}:
        return "Flat"
    if "long" in raw or "buy" in raw or "open" in raw:
        return "Long"
    if "short" in raw or "sell" in raw or "reduce" in raw:
        return "Short"
    return "Flat"


def _normalize_risk_level(value: Any) -> str:
    raw = str(value or "").strip().lower()
    if raw in {"low", "medium", "high"}:
        return raw
    if raw in {"mid", "med"}:
        return "medium"
    return "medium"


def _risk_rank(value: Any) -> int:
    risk = _normalize_risk_level(value)
    if risk == "low":
        return 1
    if risk == "medium":
        return 2
    if risk == "high":
        return 3
    return 4


def _market_label(market: Any) -> str:
    market_key = _normalize_market_key(market)
    if market_key == "crypto":
        return "Crypto"
    if market_key == "cn_equity":
        return "A股"
    if market_key == "us_equity":
        return "美股"
    return market_key


def _normalize_pct_value(value: Any) -> float | None:
    num = _safe_float(value)
    if num is None:
        return None
    if abs(num) <= 1:
        return num * 100.0
    return num


def _compute_tracking_rule_status(
    action: str,
    confidence_pct: float | None,
    risk_level: str,
    edge_score: float | None,
    hard_filter_pass: bool,
) -> str:
    conf = confidence_pct if confidence_pct is not None else 0.0
    edge = edge_score if edge_score is not None else 0.0
    if hard_filter_pass and action in {"Long", "Short"} and conf >= 70.0 and edge > 0 and _risk_rank(risk_level) <= 2:
        return "executable"
    if (not hard_filter_pass) or conf < 45.0 or _risk_rank(risk_level) >= 3:
        return "paused"
    return "watch"


def _tracking_items(cost_bps: float = 8.0, market_filter: str = "") -> list[dict[str, Any]]:
    rows = _load_tracking_rows(market=market_filter or None)
    if not rows:
        return []

    # Keep only the latest record per (market, symbol).
    latest_by_key: dict[str, dict[str, Any]] = {}
    for row in reversed(rows):
        market_key = _normalize_market_key(row.get("market"))
        symbol = str(row.get("symbol") or "").upper().strip()
        if not symbol:
            continue
        key = f"{market_key}:{_symbol_key(symbol)}"
        if key in latest_by_key:
            continue
        latest_by_key[key] = row

    items: list[dict[str, Any]] = []
    for key, row in latest_by_key.items():
        market_key = _normalize_market_key(row.get("market"))
        symbol = str(row.get("symbol") or "").upper().strip()
        if not symbol:
            continue

        name = str(_coalesce(row.get("name"), row.get("instrument_id"), symbol))
        display = str(_coalesce(row.get("display"), row.get("display_name"), f"{symbol} {name}")).strip()
        current_price = _safe_float(row.get("current_price"))
        predicted_change_pct = _safe_float(_coalesce(row.get("policy_expected_edge_pct"), row.get("predicted_change_pct"), row.get("q50_change_pct")))
        if predicted_change_pct is None:
            predicted_change_pct = 0.0
        predicted_price = _safe_float(row.get("predicted_price"))
        if predicted_price is None and current_price is not None:
            predicted_price = current_price * (1.0 + predicted_change_pct)

        edge_score = _safe_float(row.get("policy_expected_edge_pct"))
        if edge_score is None:
            edge_score = predicted_change_pct - (float(cost_bps) / 10000.0)

        confidence_pct = _to_confidence_pct(_coalesce(row.get("confidence_score"), row.get("confidence")))
        risk_level = _normalize_risk_level(row.get("risk_level"))
        action = _normalize_policy_action(_coalesce(row.get("policy_action"), row.get("recommended_action")))
        hard_filter_pass = _to_bool(row.get("hard_filter_pass"), default=True)
        rule_status = _compute_tracking_rule_status(
            action=action,
            confidence_pct=confidence_pct,
            risk_level=risk_level,
            edge_score=edge_score,
            hard_filter_pass=hard_filter_pass,
        )

        volatility_score = _safe_float(row.get("volatility_score"))
        vol_for_score = max(0.01, abs(volatility_score) if volatility_score is not None else 0.01)
        edge_risk = edge_score / vol_for_score if edge_score is not None else 0.0

        items.append(
            {
                "track_key": key,
                "market": market_key,
                "market_label": _market_label(market_key),
                "symbol": symbol,
                "name": name,
                "display_name": display,
                "current_price": current_price,
                "predicted_price": predicted_price,
                "predicted_change_pct": predicted_change_pct,
                "action": action,
                "rule_status": rule_status,
                "confidence_score": confidence_pct,
                "risk_level": risk_level,
                "edge_score": edge_score,
                "edge_risk": edge_risk,
                "position_size": _safe_float(row.get("policy_position_size")),
                "reason": str(_coalesce(row.get("policy_reason"), row.get("recommended_action"), row.get("status"), "")),
                "alerts": str(row.get("alerts") or ""),
                "status": str(row.get("status") or ""),
                "recommended_action": str(row.get("recommended_action") or ""),
                "history_missing_rate": _safe_float(row.get("history_missing_rate")),
                "total_score": _safe_float(row.get("total_score")),
                "liquidity_score": _safe_float(row.get("liquidity_score")),
                "data_quality_score": _safe_float(row.get("data_quality_score")),
                "factor_support_count": _safe_float(row.get("factor_support_count")),
                "hard_filter_pass": hard_filter_pass,
                "timestamp_utc": str(_coalesce(row.get("timestamp_utc"), row.get("generated_at_utc"), "")),
            }
        )

    # Refresh with latest market prices.
    for mkt in ("crypto", "cn_equity", "us_equity"):
        subset = [item for item in items if item.get("market") == mkt]
        if not subset:
            continue
        symbols = [str(item.get("symbol") or "").upper() for item in subset]
        price_map = _fetch_live_price_map(mkt, symbols)
        for item in subset:
            sym = str(item.get("symbol") or "").upper()
            live_price = _lookup_live_price(price_map, sym)
            if live_price is None:
                continue
            item["current_price"] = live_price
            change = _safe_float(item.get("predicted_change_pct"))
            if change is not None:
                item["predicted_price"] = live_price * (1.0 + change)

    return items


def _read_jsonl_tail(path: Path, limit: int = 200) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return []
    rows: list[dict[str, Any]] = []
    for line in lines[-max(1, int(limit)):]:
        txt = str(line).strip()
        if not txt:
            continue
        try:
            obj = json.loads(txt)
        except Exception:
            continue
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def _execution_output_dir() -> Path:
    return _get_data_path() / "execution"


def _execution_stats(positions: list[dict[str, Any]], daily_pnl: list[dict[str, Any]]) -> dict[str, Any]:
    open_positions = [r for r in positions if str(r.get("status") or "").strip().lower() == "open"]
    closed_positions = [r for r in positions if str(r.get("status") or "").strip().lower() == "closed"]

    net_values = []
    wins = 0
    for row in closed_positions:
        net_pct = _normalize_pct_value(row.get("net_pnl_pct"))
        if net_pct is None:
            continue
        net_values.append(net_pct)
        if net_pct > 0:
            wins += 1

    win_rate = (wins / len(net_values) * 100.0) if net_values else 0.0
    avg_net_pnl_pct = (sum(net_values) / len(net_values)) if net_values else 0.0
    total_net_pnl_pct = sum(net_values) if net_values else 0.0

    # If there are no closed positions, fallback to cumulative daily pnl.
    if not net_values and daily_pnl:
        latest = daily_pnl[-1]
        cumulative = _normalize_pct_value(latest.get("cumulative_realized_net_pnl_pct"))
        if cumulative is not None:
            total_net_pnl_pct = cumulative

    return {
        "open_positions": len(open_positions),
        "closed_positions": len(closed_positions),
        "win_rate": win_rate,
        "avg_net_pnl_pct": avg_net_pnl_pct,
        "total_net_pnl_pct": total_net_pnl_pct,
    }


def _build_crypto_session_payload(
    *,
    symbol: str,
    exchange: str,
    market_type: str,
    mode: str,
    horizon_hours: int,
    lookforward_days: int,
    risk_profile: str,
    rank_key: str,
    cost_bps: float,
    top_n: int,
) -> tuple[dict[str, Any] | None, str | None]:
    data_path = _get_data_path()
    all_blocks = _get_latest_csv_rows(data_path / "session_forecast_blocks.csv", 50000)
    all_hourly = _get_latest_csv_rows(data_path / "session_forecast_hourly.csv", 50000)
    all_daily = _get_latest_csv_rows(data_path / "session_forecast_daily.csv", 50000)
    if not all_blocks and not all_hourly and not all_daily:
        return None, "crypto_session_data_not_found"

    preferred_symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    symbol_options = sorted({str(r.get("symbol") or "").upper() for r in all_blocks if str(r.get("symbol") or "").strip()})
    symbol_options = preferred_symbols + [s for s in symbol_options if s not in preferred_symbols]
    symbol_options = [s for s in symbol_options if s]
    if not symbol_options:
        symbol_options = preferred_symbols

    exchange_options = sorted({str(r.get("exchange") or "").lower() for r in all_blocks if str(r.get("exchange") or "").strip()})
    market_type_options = sorted({str(r.get("market_type") or "").lower() for r in all_blocks if str(r.get("market_type") or "").strip()})
    mode_options = sorted({_normalize_mode(r.get("mode")) for r in all_blocks if str(r.get("mode") or "").strip()})
    horizon_options = sorted(
        {
            max(1, _safe_int(str(r.get("horizon") or "4h").replace("h", ""), 4))
            for r in all_blocks
            if str(r.get("horizon") or "").strip()
        }
    )
    if not mode_options:
        mode_options = ["forecast", "seasonality"]
    if not horizon_options:
        horizon_options = [4]
    if not exchange_options:
        exchange_options = ["binance", "bybit"]
    if not market_type_options:
        market_type_options = ["perp", "spot"]

    symbol = str(symbol or (symbol_options[0] if symbol_options else "BTCUSDT")).upper()
    if symbol not in symbol_options:
        symbol = symbol_options[0]
    exchange = str(exchange or exchange_options[0]).lower()
    if exchange not in exchange_options:
        exchange = exchange_options[0]
    market_type = str(market_type or market_type_options[0]).lower()
    if market_type not in market_type_options:
        market_type = market_type_options[0]
    mode = _normalize_mode(mode)
    if mode not in mode_options:
        mode = mode_options[0]
    horizon_hours = max(1, int(horizon_hours or 4))
    if horizon_hours not in horizon_options:
        horizon_hours = horizon_options[0]
    lookforward_days = max(7, min(30, int(lookforward_days or 14)))
    top_n = max(3, min(12, int(top_n or 5)))
    rank_key = str(rank_key or "edge_score").strip().lower()
    if rank_key not in RANK_OPTIONS:
        rank_key = "edge_score"
    cost_bps = _clamp(float(cost_bps or 8.0), 0.0, 200.0)
    risk_profile = _normalize_risk_profile(risk_profile)

    horizon_key = f"{horizon_hours}h"
    forecast_id, mode_actual = _pick_latest_forecast_id(
        all_blocks,
        symbol=symbol,
        exchange=exchange,
        market_type=market_type,
        mode=mode,
        horizon=horizon_key,
    )
    if not forecast_id:
        return None, "crypto_session_bundle_not_found"

    blocks = _rows_for_forecast_id(all_blocks, forecast_id)
    hourly = _rows_for_forecast_id(all_hourly, forecast_id)
    daily = _rows_for_forecast_id(all_daily, forecast_id)
    blocks = [r for r in blocks if _symbol_key(r.get("symbol")) == _symbol_key(symbol)]
    hourly = [r for r in hourly if _symbol_key(r.get("symbol")) == _symbol_key(symbol)]
    daily = [r for r in daily if _symbol_key(r.get("symbol")) == _symbol_key(symbol)]
    blocks = _sort_rows_by_session(blocks)
    hourly = _sort_rows_by_hour(hourly)
    daily = sorted(daily, key=lambda r: _safe_int(r.get("day_index"), 0))
    if not blocks and not hourly:
        return None, "crypto_session_rows_not_found"

    blocks = _refresh_rows_live_price(blocks, "crypto")
    hourly = _refresh_rows_live_price(hourly, "crypto")
    daily = _refresh_rows_live_price(daily, "crypto")

    meta_row = blocks[0] if blocks else (hourly[0] if hourly else daily[0])
    current_price = _safe_float(meta_row.get("current_price"))

    compare_mode = _mode_pair(mode_actual)
    compare_id, compare_mode_actual = _pick_latest_forecast_id(
        all_blocks,
        symbol=symbol,
        exchange=exchange,
        market_type=market_type,
        mode=compare_mode,
        horizon=horizon_key,
    )
    compare_blocks = _rows_for_forecast_id(all_blocks, compare_id)
    compare_blocks = [r for r in compare_blocks if _symbol_key(r.get("symbol")) == _symbol_key(symbol)]
    compare_blocks = _sort_rows_by_session(_refresh_rows_live_price(compare_blocks, "crypto"))

    decision_row = _select_decision_row(hourly, active_session="all")
    decision_plan = _build_trade_plan_light(
        decision_row or {},
        risk_profile=risk_profile,
        p_bull=0.55,
        p_bear=0.45,
        conf_min=60.0,
        cost_bps=20.0,
    )

    sessions = [_enrich_session_row(r, cost_bps=cost_bps) for r in blocks]
    session_map = {str(r.get("session_name") or "").lower(): r for r in sessions}
    summary = _build_summary_from_blocks(sessions)
    top_hourly = _build_top_tables(hourly, top_n=top_n, rank_key=rank_key, cost_bps=cost_bps)
    top_daily = _build_top_tables(daily, top_n=top_n, rank_key=rank_key, cost_bps=cost_bps)
    sim_path = _build_sim_path(
        daily,
        current_price,
        lookforward_days=lookforward_days,
        anchor_q10=_safe_float(decision_plan.get("q10")),
        anchor_q50=_safe_float(decision_plan.get("q50")),
        anchor_q90=_safe_float(decision_plan.get("q90")),
    )
    consensus = _session_consensus_light(decision_row, compare_blocks)
    model_health_level, model_health_text = _model_health_text(hourly)
    compare_rows = _build_compare_rows(sessions, compare_blocks, main_mode=mode_actual, compare_mode=compare_mode_actual)

    data_version_seed = (
        f"{symbol}|{exchange}|{market_type}|{meta_row.get('data_updated_at_bj','-')}|"
        f"{meta_row.get('forecast_generated_at_bj','-')}|{meta_row.get('data_source_actual','-')}"
    )
    data_version = hashlib.sha1(data_version_seed.encode("utf-8")).hexdigest()[:12]

    payload = {
        "ok": True,
        "market": "crypto",
        "controls": {
            "symbol_options": symbol_options,
            "exchange_options": exchange_options,
            "market_type_options": market_type_options,
            "mode_options": mode_options,
            "horizon_options": horizon_options,
            "risk_profile_options": ["standard", "conservative", "aggressive"],
            "rank_options": RANK_OPTIONS,
            "selected": {
                "symbol": symbol,
                "exchange": exchange,
                "market_type": market_type,
                "mode": mode_actual,
                "horizon_hours": horizon_hours,
                "lookforward_days": lookforward_days,
                "risk_profile": risk_profile,
                "rank_key": rank_key,
                "top_n": top_n,
                "cost_bps": cost_bps,
            },
        },
        "meta": {
            "symbol": symbol,
            "exchange": exchange,
            "exchange_actual": meta_row.get("exchange_actual"),
            "market_type": market_type,
            "mode_requested": mode,
            "mode_actual": mode_actual,
            "horizon_hours": horizon_hours,
            "lookforward_days": lookforward_days,
            "current_price": current_price,
            "forecast_generated_at_bj": meta_row.get("forecast_generated_at_bj"),
            "data_updated_at_bj": meta_row.get("data_updated_at_bj"),
            "model_version": meta_row.get("model_version"),
            "data_source_actual": meta_row.get("data_source_actual"),
            "data_version": data_version,
        },
        "sessions": sessions,
        "hourly": hourly,
        "daily": daily[:lookforward_days],
        "summary": summary,
        "decision": {
            "plan": decision_plan,
            "consensus": consensus,
            "model_health": model_health_level,
            "model_health_text": model_health_text,
            "threshold_text": "阈值 p_bull=0.55, p_bear=0.45, conf>=60, RR(TP1)>=1.0(不含TP2), cost=20.0bps",
        },
        "compare": {
            "main_mode": mode_actual,
            "compare_mode": compare_mode_actual,
            "rows": compare_rows,
        },
        "top": {
            "rank_key": rank_key,
            "top_n": top_n,
            "cost_bps": cost_bps,
            "hourly": top_hourly,
            "daily": top_daily,
        },
        "sim_path": sim_path,
        "notes": [
            "小时级语义：从该小时开始的未来窗口回报，不代表该小时内必涨/必跌。",
            "Forecast 与 Seasonality 分歧较大时，建议降低仓位。"
        ],
        # Backward compatibility
        "blocks": blocks,
        "asian": session_map.get("asia"),
        "european": session_map.get("europe"),
        "american": session_map.get("us"),
        "timestamp": _utcnow().isoformat(),
    }
    return payload, None


def _build_index_session_payload(
    *,
    index_key: str,
    mode: str,
    horizon_hours: int,
    lookforward_days: int,
    risk_profile: str,
    rank_key: str,
    cost_bps: float,
    top_n: int,
) -> tuple[dict[str, Any] | None, str | None]:
    key = str(index_key or "").strip().lower()
    if key not in INDEX_SESSION_UNIVERSE:
        key = "sse"
    inst = INDEX_SESSION_UNIVERSE[key]
    symbol = str(inst.get("symbol"))
    market = str(inst.get("market"))
    mode = _normalize_mode(mode)
    horizon_hours = max(1, min(6, int(horizon_hours or 4)))
    lookforward_days = max(7, min(30, int(lookforward_days or 14)))
    top_n = max(3, min(12, int(top_n or 5)))
    rank_key = str(rank_key or "edge_score").strip().lower()
    if rank_key not in RANK_OPTIONS:
        rank_key = "edge_score"
    cost_bps = _clamp(float(cost_bps or 8.0), 0.0, 200.0)
    risk_profile = _normalize_risk_profile(risk_profile)
    active_session = INDEX_ACTIVE_SESSION.get(key, "us")

    hourly_bars = _fetch_yahoo_history_bars(symbol=symbol, market=market, interval="hourly", limit=4000)
    daily_bars = _fetch_yahoo_history_bars(symbol=symbol, market=market, interval="daily", limit=3000)
    if not hourly_bars:
        return None, "index_hourly_data_not_found"
    profile_main, mode_actual = _hourly_stats_by_hour(
        hourly_bars,
        horizon_hours=horizon_hours,
        recent_days=180,
        mode=mode,
    )
    if not profile_main:
        return None, "index_hourly_profile_empty"
    current_price = _safe_float((hourly_bars[-1] or {}).get("close"))
    if current_price is None:
        return None, "index_price_not_found"

    hourly_main = _build_hourly_rows_from_profile(profile_main, current_price=current_price, active_session=active_session)
    blocks_main = _aggregate_blocks(hourly_main, active_session=active_session)
    daily_main, daily_mode_actual = _build_daily_rows_from_bars(
        daily_bars,
        lookforward_days=lookforward_days,
        mode=mode_actual,
        current_price=current_price,
    )
    if mode_actual == "forecast" and daily_mode_actual != "forecast":
        mode_actual = daily_mode_actual

    compare_mode = _mode_pair(mode_actual)
    profile_cmp, compare_mode_actual = _hourly_stats_by_hour(
        hourly_bars,
        horizon_hours=horizon_hours,
        recent_days=180,
        mode=compare_mode,
    )
    hourly_cmp = _build_hourly_rows_from_profile(profile_cmp, current_price=current_price, active_session=active_session)
    blocks_cmp = _aggregate_blocks(hourly_cmp, active_session=active_session)

    # Attach common metadata fields
    generated_bj = (_utcnow() + timedelta(hours=8)).strftime("%Y-%m-%d %H:%M:%S+0800")
    for frame in (hourly_main, blocks_main, daily_main, hourly_cmp, blocks_cmp):
        for row in frame:
            row["symbol"] = symbol
            row["exchange"] = "yahoo"
            row["exchange_actual"] = "yahoo"
            row["market_type"] = "index"
            row["mode"] = mode_actual
            row["mode_requested"] = mode
            row["horizon"] = f"{horizon_hours}h" if "hour_bj" in row else "1d"
            row["current_price"] = current_price
            row["forecast_generated_at_bj"] = generated_bj
            row["data_updated_at_bj"] = generated_bj
            row["model_version"] = f"index_{mode_actual}_v1"
            row["data_source_actual"] = "yahoo_chart"

    decision_row = _select_decision_row(hourly_main, active_session=active_session)
    decision_plan = _build_trade_plan_light(
        decision_row or {},
        risk_profile=risk_profile,
        p_bull=0.55,
        p_bear=0.45,
        conf_min=60.0,
        cost_bps=20.0,
    )

    sessions = [_enrich_session_row(r, cost_bps=cost_bps) for r in blocks_main]
    summary = _build_summary_from_blocks(sessions)
    compare_rows = _build_compare_rows(sessions, blocks_cmp, main_mode=mode_actual, compare_mode=compare_mode_actual)
    top_hourly = _build_top_tables([r for r in hourly_main if _safe_int(r.get("is_trading_hour"), 0) == 1], top_n=top_n, rank_key=rank_key, cost_bps=cost_bps)
    top_daily = _build_top_tables(daily_main, top_n=top_n, rank_key=rank_key, cost_bps=cost_bps)
    sim_path = _build_sim_path(
        daily_main,
        current_price,
        lookforward_days=lookforward_days,
        anchor_q10=_safe_float(decision_plan.get("q10")),
        anchor_q50=_safe_float(decision_plan.get("q50")),
        anchor_q90=_safe_float(decision_plan.get("q90")),
    )
    consensus = _session_consensus_light(decision_row, blocks_cmp)
    model_health_level, model_health_text = _model_health_text(hourly_main)

    data_version_seed = f"{key}|{symbol}|{mode_actual}|{horizon_hours}|{generated_bj}"
    data_version = hashlib.sha1(data_version_seed.encode("utf-8")).hexdigest()[:12]

    index_options = [
        {
            "key": ik,
            "label": f"{iv.get('name_zh')} ({iv.get('symbol')})",
            "name_zh": iv.get("name_zh"),
            "name_en": iv.get("name_en"),
            "symbol": iv.get("symbol"),
            "market": iv.get("market"),
        }
        for ik, iv in INDEX_SESSION_UNIVERSE.items()
    ]

    legacy_sessions: list[dict[str, Any]] = []
    for row in sessions:
        vol = _safe_float(row.get("volatility_score")) or 0.0
        legacy_sessions.append(
            {
                "name": str(row.get("session_name") or ""),
                "title": str(row.get("session_name_cn") or ""),
                "time": str(row.get("session_hours") or ""),
                "volatility": abs(vol) * 100.0 if abs(vol) <= 1 else vol,
                "direction": _direction_from_p_up(_as_prob(row.get("p_up"))),
                "prediction_text": f"预期{_direction_from_p_up(_as_prob(row.get('p_up')))}，建议分批执行并控制仓位。",
            }
        )

    payload = {
        "ok": True,
        "market": "cn" if market == "cn_equity" else "us",
        "index_key": key,
        "controls": {
            "index_options": index_options,
            "mode_options": ["forecast", "seasonality"],
            "horizon_options": [1, 2, 4, 6],
            "risk_profile_options": ["standard", "conservative", "aggressive"],
            "rank_options": RANK_OPTIONS,
            "selected": {
                "index_key": key,
                "mode": mode_actual,
                "horizon_hours": horizon_hours,
                "lookforward_days": lookforward_days,
                "risk_profile": risk_profile,
                "rank_key": rank_key,
                "top_n": top_n,
                "cost_bps": cost_bps,
            },
        },
        "meta": {
            "index_key": key,
            "index_name_zh": inst.get("name_zh"),
            "index_name_en": inst.get("name_en"),
            "symbol": symbol,
            "market": market,
            "mode_requested": mode,
            "mode_actual": mode_actual,
            "horizon_hours": horizon_hours,
            "lookforward_days": lookforward_days,
            "active_session": active_session,
            "current_price": current_price,
            "forecast_generated_at_bj": generated_bj,
            "data_updated_at_bj": generated_bj,
            "model_version": f"index_{mode_actual}_v1",
            "data_source_actual": "yahoo_chart",
            "data_version": data_version,
            "skipped_non_trading_days": [],
        },
        "sessions": sessions,
        "hourly": hourly_main,
        "daily": daily_main[:lookforward_days],
        "summary": summary,
        "decision": {
            "plan": decision_plan,
            "consensus": consensus,
            "model_health": model_health_level,
            "model_health_text": model_health_text,
            "threshold_text": "阈值 p_bull=0.55, p_bear=0.45, conf>=60, RR(TP1)>=1.0(不含TP2), cost=20.0bps",
        },
        "compare": {
            "main_mode": mode_actual,
            "compare_mode": compare_mode_actual,
            "rows": compare_rows,
        },
        "top": {
            "rank_key": rank_key,
            "top_n": top_n,
            "cost_bps": cost_bps,
            "hourly": top_hourly,
            "daily": top_daily,
        },
        "sim_path": sim_path,
        "notes": [
            f"指数时段范围：当前指数仅 `{_session_name_cn(active_session)}` 作为可交易时段，其它时段置空。",
            "小时级语义：从该小时开始的未来窗口回报，不代表该小时内必涨/必跌。"
        ],
        # Backward compatibility
        "sessions_legacy": legacy_sessions,
        "timestamp": _utcnow().isoformat(),
    }
    return payload, None


def create_app() -> Flask:
    app = Flask(__name__)
    CORS(app)
    app.config["SQLALCHEMY_DATABASE_URI"] = _notes_db_uri()
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    app.config["NOTES_AUTH_SECRET"] = os.getenv("NOTES_AUTH_SECRET", "notes-dev-secret-change-me")
    app.config["NOTES_TOKEN_TTL_SECONDS"] = int(os.getenv("NOTES_TOKEN_TTL_SECONDS", "604800"))
    db.init_app(app)

    with app.app_context():
        db.create_all()

    # ==================== Health Check ====================

    @app.get("/api/health")
    def health() -> Any:
        """Health check endpoint."""
        return jsonify({"ok": True, "service": "notes-api"})

    # ==================== Authentication ====================

    @app.post("/api/auth/register")
    def register() -> Any:
        """Register a new user."""
        payload = _json_payload()
        username = _normalize_username(payload.get("username"))
        email = _normalize_email(payload.get("email"))
        password = str(payload.get("password") or "")
        validation_error = _validate_register_input(username, email, password)
        if validation_error:
            return _err(validation_error, 400)
        duplicate = User.query.filter(
            or_(
                func.lower(User.username) == username.lower(),
                func.lower(User.email) == email,
            )
        ).first()
        if duplicate:
            return _err("username_or_email_already_exists", 409)
        user = User(
            username=username,
            email=email,
            password_hash=generate_password_hash(password),
            is_active=True,
        )
        db.session.add(user)
        try:
            db.session.commit()
        except IntegrityError:
            db.session.rollback()
            return _err("username_or_email_already_exists", 409)
        return jsonify({"ok": True, "user": user.to_public_dict()}), 201

    @app.post("/api/auth/login")
    def login() -> Any:
        """Login and get JWT token."""
        payload = _json_payload()
        username = _normalize_username(payload.get("username"))
        email = _normalize_email(payload.get("email"))
        password = str(payload.get("password") or "")
        if not password:
            return _err("password_required", 400)
        if not username and not email:
            return _err("username_or_email_required", 400)
        query = User.query
        if email:
            user = query.filter(func.lower(User.email) == email).first()
        else:
            user = query.filter(func.lower(User.username) == username.lower()).first()
        if user is None:
            return _err("invalid_credentials", 401)
        if not user.is_active:
            return _err("account_disabled", 403)
        if not check_password_hash(user.password_hash, password):
            return _err("invalid_credentials", 401)
        user.last_login_at = _utcnow()
        db.session.commit()
        token = _issue_token(app, user.id)
        return jsonify({
            "ok": True,
            "token": token,
            "token_type": "Bearer",
            "expires_in": int(app.config["NOTES_TOKEN_TTL_SECONDS"]),
            "user": user.to_public_dict(),
        })

    @app.get("/api/auth/me")
    def me() -> Any:
        """Get current user info (requires authentication)."""
        user, auth_err = _require_auth_user(app)
        if auth_err is not None:
            return auth_err
        return jsonify({"ok": True, "user": user.to_public_dict()})

    @app.post("/api/auth/logout")
    def logout() -> Any:
        """Client-side token mode: logout is a no-op on server."""
        return jsonify({"ok": True})

    @app.post("/api/auth/refresh")
    def refresh_token() -> Any:
        """Issue a new token if current token is valid."""
        user, auth_err = _require_auth_user(app)
        if auth_err is not None:
            return auth_err
        token = _issue_token(app, user.id)
        return jsonify({
            "ok": True,
            "token": token,
            "token_type": "Bearer",
            "expires_in": int(app.config["NOTES_TOKEN_TTL_SECONDS"]),
            "user": user.to_public_dict(),
        })

    # ==================== Notes API ====================

    @app.post("/api/notes")
    def create_note() -> Any:
        """Create a new note (requires authentication)."""
        user, auth_err = _require_auth_user(app)
        if auth_err is not None:
            return auth_err
        payload = _json_payload()
        title = str(payload.get("title") or "").strip()
        content = str(payload.get("content") or "").strip()
        note_type = str(payload.get("note_type") or "NOTE").strip().upper() or "NOTE"
        tags_list = _normalize_tags(payload.get("tags"))
        is_public = bool(payload.get("is_public", False))
        if not title and not content:
            return _err("title_or_content_required", 400)
        if note_type not in {"NOTE", "JOURNAL", "PLAN"}:
            note_type = "NOTE"
        note = Note(
            user_id=user.id,
            title=title,
            content=content,
            tags_csv=",".join(tags_list),
            note_type=note_type,
            is_public=is_public,
        )
        db.session.add(note)
        db.session.commit()
        return jsonify({"ok": True, "item": note.to_dict(include_author=True)}), 201

    @app.get("/api/notes")
    def list_notes() -> Any:
        """List user's notes (requires authentication)."""
        user, auth_err = _require_auth_user(app)
        if auth_err is not None:
            return auth_err
        q = str(request.args.get("q") or "").strip().lower()
        page_size = _parse_page_size(request.args.get("page_size"), default=20, max_value=100)
        query = Note.query.filter(Note.user_id == user.id)
        if q:
            query = query.filter(
                or_(
                    func.lower(Note.title).like(f"%{q}%"),
                    func.lower(Note.content).like(f"%{q}%"),
                    func.lower(Note.tags_csv).like(f"%{q}%"),
                )
            )
        items = query.order_by(Note.updated_at.desc(), Note.id.desc()).limit(page_size).all()
        return jsonify({
            "ok": True,
            "items": [item.to_dict(include_author=False) for item in items],
            "count": len(items),
        })

    @app.get("/api/notes/public")
    def list_public_notes() -> Any:
        """List public notes."""
        q = str(request.args.get("q") or "").strip().lower()
        page_size = _parse_page_size(request.args.get("page_size"), default=10, max_value=100)
        query = Note.query.filter(Note.is_public.is_(True))
        if q:
            query = query.filter(
                or_(
                    func.lower(Note.title).like(f"%{q}%"),
                    func.lower(Note.content).like(f"%{q}%"),
                    func.lower(Note.tags_csv).like(f"%{q}%"),
                )
            )
        items = query.order_by(Note.updated_at.desc(), Note.id.desc()).limit(page_size).all()
        return jsonify({
            "ok": True,
            "items": [item.to_dict(include_author=True) for item in items],
            "count": len(items),
        })

    # ==================== Tracking API ====================

    @app.get("/api/tracking/overview")
    def tracking_overview() -> Any:
        """
        Selection / Research / Tracking data for web page.
        Query params:
            market: all|crypto|cn|us|cn_equity|us_equity
            status: all|executable|watch|paused
            action: all|long|short|flat
            q: keyword
            limit: max rows for screener
            top_n: top rows for long/short/watch cards
            sort_by: edge_risk|edge_score|confidence_score|total_score|liquidity_score
            desc: true|false
            cost_bps: fallback cost basis points for edge calc
        """
        market = str(request.args.get("market", "all")).strip().lower()
        status_filter = str(request.args.get("status", "all")).strip().lower()
        action_filter = str(request.args.get("action", "all")).strip().lower()
        keyword = str(request.args.get("q", "")).strip().lower()
        limit = _parse_page_size(request.args.get("limit"), default=200, max_value=2000)
        top_n = _parse_page_size(request.args.get("top_n"), default=5, max_value=20)
        sort_by = str(request.args.get("sort_by", "edge_risk")).strip().lower()
        desc = _to_bool(request.args.get("desc"), default=True)
        cost_bps = _safe_float(request.args.get("cost_bps"))
        if cost_bps is None:
            cost_bps = 8.0
        cost_bps = min(200.0, max(0.0, cost_bps))

        market_filter = "" if market in {"", "all"} else _normalize_market_key(market)
        items = _tracking_items(cost_bps=cost_bps, market_filter=market_filter)
        if not items:
            return jsonify(
                {
                    "ok": True,
                    "items": [],
                    "top_long": [],
                    "top_short": [],
                    "top_watch": [],
                    "metrics": {
                        "total_candidates": 0,
                        "executable_count": 0,
                        "watch_count": 0,
                        "paused_count": 0,
                        "prediction_coverage": 0.0,
                        "hard_filter_pass_rate": 0.0,
                        "avg_missing_rate": 0.0,
                        "filtered_count": 0,
                    },
                    "count": 0,
                    "timestamp": _utcnow().isoformat(),
                }
            )

        base_view = items
        if keyword:
            base_view = [
                item
                for item in base_view
                if keyword in str(item.get("display_name", "")).lower()
                or keyword in str(item.get("name", "")).lower()
                or keyword in str(item.get("symbol", "")).lower()
                or keyword in str(item.get("reason", "")).lower()
            ]

        def _top_rows(rows: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
            return sorted(rows, key=lambda r: _safe_float(r.get("edge_risk")) or -999.0, reverse=True)[:n]

        top_long = _top_rows(
            [r for r in base_view if str(r.get("rule_status")) == "executable" and str(r.get("action")) == "Long"],
            top_n,
        )
        top_short = _top_rows(
            [r for r in base_view if str(r.get("rule_status")) == "executable" and str(r.get("action")) == "Short"],
            top_n,
        )
        top_watch = _top_rows([r for r in base_view if str(r.get("rule_status")) == "watch"], top_n)

        filtered = base_view
        if status_filter in {"executable", "watch", "paused"}:
            filtered = [r for r in filtered if str(r.get("rule_status")) == status_filter]
        if action_filter in {"long", "short", "flat"}:
            action_norm = action_filter.capitalize()
            filtered = [r for r in filtered if str(r.get("action")) == action_norm]

        sort_key_map = {
            "edge_risk": "edge_risk",
            "edge_score": "edge_score",
            "confidence_score": "confidence_score",
            "total_score": "total_score",
            "liquidity_score": "liquidity_score",
        }
        sort_key = sort_key_map.get(sort_by, "edge_risk")
        filtered = sorted(filtered, key=lambda r: _safe_float(r.get(sort_key)) or -999.0, reverse=bool(desc))
        filtered = filtered[:limit]

        total = len(base_view)
        executable_count = len([r for r in base_view if str(r.get("rule_status")) == "executable"])
        watch_count = len([r for r in base_view if str(r.get("rule_status")) == "watch"])
        paused_count = len([r for r in base_view if str(r.get("rule_status")) == "paused"])
        prediction_coverage = (
            len([r for r in base_view if _safe_float(r.get("predicted_change_pct")) is not None]) / total if total else 0.0
        )
        hard_filter_pass_rate = len([r for r in base_view if _to_bool(r.get("hard_filter_pass"), True)]) / total if total else 0.0
        missing_rates = [_safe_float(r.get("history_missing_rate")) for r in base_view]
        missing_rates = [x for x in missing_rates if x is not None]
        avg_missing_rate = (sum(missing_rates) / len(missing_rates)) if missing_rates else 0.0

        return jsonify(
            {
                "ok": True,
                "items": filtered,
                "top_long": top_long,
                "top_short": top_short,
                "top_watch": top_watch,
                "metrics": {
                    "total_candidates": total,
                    "executable_count": executable_count,
                    "watch_count": watch_count,
                    "paused_count": paused_count,
                    "prediction_coverage": prediction_coverage,
                    "hard_filter_pass_rate": hard_filter_pass_rate,
                    "avg_missing_rate": avg_missing_rate,
                    "filtered_count": len(filtered),
                },
                "count": len(filtered),
                "timestamp": _utcnow().isoformat(),
            }
        )

    @app.get("/api/tracking/detail/<track_key>")
    def tracking_detail(track_key: str) -> Any:
        market = str(request.args.get("market", "all")).strip().lower()
        cost_bps = _safe_float(request.args.get("cost_bps"))
        if cost_bps is None:
            cost_bps = 8.0
        market_filter = "" if market in {"", "all"} else _normalize_market_key(market)
        items = _tracking_items(cost_bps=cost_bps, market_filter=market_filter)
        wanted = str(track_key or "").strip().lower()
        if not wanted:
            return _err("track_key_required", 400)
        for item in items:
            item_key = str(item.get("track_key") or "").strip().lower()
            sym_key = _symbol_key(item.get("symbol"))
            if wanted == item_key or wanted == sym_key.lower() or wanted == str(item.get("symbol", "")).lower():
                return jsonify({"ok": True, "item": item, "timestamp": _utcnow().isoformat()})
        return _err("tracking_item_not_found", 404)

    # ==================== Execution API ====================

    @app.get("/api/execution/overview")
    def execution_overview() -> Any:
        """
        Paper trading / execution data for web page.
        Query params:
            limit: max rows for csv tables
            log_limit: max rows for jsonl logs
        """
        limit = _parse_page_size(request.args.get("limit"), default=200, max_value=2000)
        log_limit = _parse_page_size(request.args.get("log_limit"), default=200, max_value=1000)
        out_dir = _execution_output_dir()

        decision_logs = _get_latest_csv_rows(out_dir / "decision_packets_log.csv", count=limit)
        orders = _get_latest_csv_rows(out_dir / "paper_orders.csv", count=limit)
        fills = _get_latest_csv_rows(out_dir / "paper_fills.csv", count=limit)
        positions = _get_latest_csv_rows(out_dir / "paper_positions.csv", count=limit)
        daily_pnl = _get_latest_csv_rows(out_dir / "paper_daily_pnl.csv", count=limit)

        decision_logs = sorted(decision_logs, key=lambda r: str(r.get("generated_at_utc") or ""), reverse=True)
        orders = sorted(orders, key=lambda r: str(r.get("created_at_utc") or ""), reverse=True)
        fills = sorted(fills, key=lambda r: str(r.get("fill_time_utc") or ""), reverse=True)
        positions = sorted(
            positions,
            key=lambda r: str(_coalesce(r.get("entry_time_utc"), r.get("exit_time_utc"), "")),
            reverse=True,
        )
        daily_pnl_sorted_for_stats = sorted(daily_pnl, key=lambda r: str(r.get("date_utc") or ""))
        daily_pnl = sorted(daily_pnl, key=lambda r: str(r.get("date_utc") or ""), reverse=True)

        run_log = sorted(
            _read_jsonl_tail(out_dir / "paper_run_log.jsonl", limit=log_limit),
            key=lambda r: str(r.get("timestamp_utc") or ""),
            reverse=True,
        )
        gate_log = sorted(
            _read_jsonl_tail(out_dir / "gates_audit_log.jsonl", limit=log_limit),
            key=lambda r: str(r.get("timestamp_utc") or ""),
            reverse=True,
        )
        kill_switch_events = sorted(
            _read_jsonl_tail(out_dir / "kill_switch_events.jsonl", limit=log_limit),
            key=lambda r: str(r.get("timestamp_utc") or ""),
            reverse=True,
        )
        kill_switch_recovery = sorted(
            _read_jsonl_tail(out_dir / "kill_switch_recovery_log.jsonl", limit=log_limit),
            key=lambda r: str(r.get("timestamp_utc") or ""),
            reverse=True,
        )

        stats = _execution_stats(positions=positions, daily_pnl=daily_pnl_sorted_for_stats)
        open_positions = [r for r in positions if str(r.get("status") or "").strip().lower() == "open"]
        closed_positions = [r for r in positions if str(r.get("status") or "").strip().lower() == "closed"]

        return jsonify(
            {
                "ok": True,
                "stats": stats,
                "decision_logs": decision_logs,
                "orders": orders,
                "fills": fills,
                "positions": positions,
                "open_positions": open_positions,
                "closed_positions": closed_positions,
                "daily_pnl": daily_pnl,
                "run_log": run_log,
                "gate_log": gate_log,
                "kill_switch_events": kill_switch_events,
                "kill_switch_recovery": kill_switch_recovery,
                "counts": {
                    "decision_logs": len(decision_logs),
                    "orders": len(orders),
                    "fills": len(fills),
                    "positions": len(positions),
                    "open_positions": len(open_positions),
                    "closed_positions": len(closed_positions),
                    "daily_pnl": len(daily_pnl),
                    "run_log": len(run_log),
                    "gate_log": len(gate_log),
                },
                "timestamp": _utcnow().isoformat(),
            }
        )

    @app.post("/api/execution/clear-logs")
    def execution_clear_logs() -> Any:
        out_dir = _execution_output_dir()
        removed: list[str] = []
        for fn in [
            "decision_packet_latest.json",
            "decision_packets_log.csv",
            "paper_orders.csv",
            "paper_fills.csv",
            "paper_positions.csv",
            "paper_daily_pnl.csv",
            "paper_run_log.jsonl",
            "gates_audit_log.jsonl",
            "kill_switch_events.jsonl",
            "kill_switch_recovery_log.jsonl",
            ".kill_switch_last_seen.json",
            "kill_switch.state.json",
            "health_checks_streak.json",
        ]:
            p = out_dir / fn
            if p.exists():
                try:
                    p.unlink()
                    removed.append(fn)
                except Exception:
                    pass
        for p in out_dir.glob("decision_packet_*.json"):
            try:
                p.unlink()
                removed.append(p.name)
            except Exception:
                pass
        return jsonify({"ok": True, "removed": removed, "removed_count": len(removed), "timestamp": _utcnow().isoformat()})

    # ==================== Market Overview ====================

    @app.get("/api/market/overview")
    def market_overview() -> Any:
        """Get market overview for Crypto cards + generic asset list."""
        predictions = _get_market_predictions("crypto", limit=10)
        by_symbol = {str(item.get("symbol") or "").upper(): item for item in predictions}
        card_map: dict[str, dict[str, Any]] = {}
        for key, symbol in {"btc": "BTCUSDT", "eth": "ETHUSDT", "sol": "SOLUSDT"}.items():
            item = by_symbol.get(symbol, {})
            price = _safe_float(item.get("current_price"))
            q50 = _safe_float(item.get("predicted_change_pct"))
            support = _safe_float(item.get("support_level"))
            resistance = _safe_float(item.get("resistance_level"))
            if price is not None:
                if support is None and q50 is not None:
                    support = price * (1 + min(q50, 0))
                if resistance is None and q50 is not None:
                    resistance = price * (1 + max(q50, 0))
            card_map[key] = {
                "symbol": symbol,
                "name": item.get("name") or key.upper(),
                "price": price,
                "change": _safe_float(item.get("change_percent")) or (q50 * 100 if q50 is not None else 0.0),
                "high": resistance or price,
                "low": support or price,
                "volume": item.get("sample_size") or "--",
                "predicted": _safe_float(item.get("predicted_price")),
                "support": support,
                "resistance": resistance,
            }
        return jsonify({
            "ok": True,
            "assets": predictions,
            "btc": card_map.get("btc"),
            "eth": card_map.get("eth"),
            "sol": card_map.get("sol"),
            "timestamp": _utcnow().isoformat(),
        })

    @app.get("/api/market/indices")
    def market_indices() -> Any:
        """
        Get index cards for CN/US pages.
        Query params:
            market: cn | us | all
        """
        market = str(request.args.get("market", "all")).strip().lower()
        cn_payload = _live_index_payload("cn")
        us_payload = _live_index_payload("us")
        if market == "cn":
            return jsonify({"ok": True, "market": "cn", **cn_payload, "indices": cn_payload, "timestamp": _utcnow().isoformat()})
        if market == "us":
            return jsonify({"ok": True, "market": "us", **us_payload, "indices": us_payload, "timestamp": _utcnow().isoformat()})
        return jsonify({"ok": True, "market": "all", "cn": cn_payload, "us": us_payload, "timestamp": _utcnow().isoformat()})

    @app.get("/api/market/crypto")
    def market_crypto() -> Any:
        """Get crypto prices by symbols=BTC,ETH,SOL."""
        raw_symbols = str(request.args.get("symbols", "BTC,ETH,SOL"))
        requested = [s.strip().upper() for s in raw_symbols.split(",") if s.strip()]
        symbol_filter = ""
        items = _get_market_predictions("crypto", limit=100, symbol_filter=symbol_filter)
        out: list[dict[str, Any]] = []
        for item in items:
            sym = str(item.get("symbol") or "").upper()
            short = sym.replace("USDT", "").replace("USDC", "").replace("USD", "")
            if requested and short not in requested and sym not in requested:
                continue
            out.append(item)
        return jsonify({"ok": True, "items": out, "count": len(out), "timestamp": _utcnow().isoformat()})

    @app.get("/api/market/stocks")
    def market_stocks() -> Any:
        """Get stock prices by symbols list for CN/US."""
        raw_symbols = str(request.args.get("symbols", ""))
        requested = [s.strip().upper() for s in raw_symbols.split(",") if s.strip()]
        cn_items = _get_market_predictions("cn_equity", limit=500)
        us_items = _get_market_predictions("us_equity", limit=500)
        all_items = cn_items + us_items
        if requested:
            all_items = [item for item in all_items if _symbol_key(item.get("symbol")) in {_symbol_key(s) for s in requested}]
        return jsonify({"ok": True, "items": all_items, "count": len(all_items), "timestamp": _utcnow().isoformat()})

    @app.get("/api/market/history")
    def market_history() -> Any:
        """Unified market history endpoint forwarding to market-specific handlers."""
        symbol = str(request.args.get("symbol", "")).upper().strip()
        if not symbol:
            return _err("symbol_required", 400)
        if symbol.endswith("USDT") or symbol.endswith("USDC") or symbol.endswith("USD"):
            return crypto_history(symbol)
        if "." in symbol and symbol.split(".", 1)[0].isdigit():
            return cn_history(symbol)
        return us_history(symbol)

    # ==================== Crypto API ====================

    @app.get("/api/crypto/predictions")
    def crypto_predictions() -> Any:
        """Get crypto predictions for HTML/Streamlit pages."""
        symbol_filter = request.args.get("symbol", "").strip().upper()
        try:
            limit = min(int(request.args.get("limit", "100")), 500)
        except ValueError:
            limit = 100

        predictions = _get_market_predictions("crypto", limit=limit, symbol_filter=symbol_filter)
        if predictions:
            return jsonify({
                "ok": True,
                "predictions": predictions,
                "items": predictions,
                "count": len(predictions),
                "timestamp": _utcnow().isoformat(),
            })
        return jsonify({"ok": False, "error": "crypto_predictions_not_found"}), 503

    @app.get("/api/crypto/history/<symbol>")
    def crypto_history(symbol: str) -> Any:
        """
        Get historical price data for a crypto symbol.
        
        Args:
            symbol: Crypto symbol (e.g., BTCUSDT, ETHUSDT, SOLUSDT)
        
        Query params:
            limit: Number of bars (default 100, max 1000)
            interval: Time interval (hourly/daily, default hourly)
        
        Returns: { "ok": true, "symbol": "...", "bars": [...], "count": N }
        """
        symbol = symbol.upper().strip()
        try:
            limit = min(int(request.args.get("limit", "100")), 1000)
        except ValueError:
            limit = 100
        
        interval = request.args.get("interval", "hourly").lower()
        
        data_path = _get_data_path()

        # Prefer live exchange bars for crypto symbols.
        live_bars = _fetch_binance_history_bars(symbol=symbol, interval=interval, limit=limit)
        if live_bars:
            return jsonify({
                "ok": True,
                "symbol": symbol,
                "bars": live_bars,
                "count": len(live_bars),
                "interval": interval,
                "source": "binance_klines",
                "timestamp": _utcnow().isoformat()
            })

        yahoo_bars = _fetch_yahoo_history_bars(
            symbol=symbol,
            market="crypto",
            interval=interval,
            limit=limit,
        )
        if yahoo_bars:
            return jsonify({
                "ok": True,
                "symbol": symbol,
                "bars": yahoo_bars,
                "count": len(yahoo_bars),
                "interval": interval,
                "source": "yahoo_chart",
                "timestamp": _utcnow().isoformat()
            })
        
        # Try features file first (has OHLCV data)
        if interval == "daily":
            features_file = data_path / "features_daily.csv"
        else:
            features_file = data_path / "features_hourly.csv"
        
        if features_file.exists():
            rows = _read_csv_with_headers(features_file, limit=limit * 2)
            # Normalize symbol for comparison (e.g., BTCUSDT -> BTC, SOLUSDT -> SOL)
            base_symbol = symbol.replace("USDT", "").replace("USD", "").replace("USDC", "").upper()
            # Filter by symbol if available
            symbol_rows = [row for row in rows if row.get("symbol", "").upper() == base_symbol or row.get("symbol", "").upper() == symbol]
            
            if symbol_rows:
                bars = []
                for row in symbol_rows[-limit:]:
                    bar = {
                        "timestamp": row.get("timestamp_utc") or row.get("timestamp_market") or row.get("timestamp", ""),
                        "close": _safe_float(row.get("close")),
                        "volume": _safe_float(row.get("volume")),
                    }
                    # Add OHLC if available
                    if row.get("open"):
                        bar["open"] = _safe_float(row.get("open"))
                    if row.get("high"):
                        bar["high"] = _safe_float(row.get("high"))
                    if row.get("low"):
                        bar["low"] = _safe_float(row.get("low"))
                    bars.append(bar)
                return jsonify({
                    "ok": True,
                    "symbol": symbol,
                    "bars": bars,
                    "count": len(bars),
                    "interval": interval,
                    "timestamp": _utcnow().isoformat()
                })
        
        # Fallback to predictions file for price history
        if interval == "daily":
            pred_file = data_path / "predictions_daily.csv"
        else:
            pred_file = data_path / "predictions_hourly.csv"
        
        if pred_file.exists():
            rows = _get_latest_csv_rows(pred_file, limit)
            bars = []
            for row in rows:
                bars.append({
                    "timestamp": row.get("timestamp_utc") or row.get("timestamp_market", ""),
                    "close": _safe_float(row.get("close")),
                    "p_up": _safe_float(row.get("dir_h1_p_up")),
                    "p_down": _safe_float(row.get("dir_h1_p_down")),
                })
            return jsonify({
                "ok": True,
                "symbol": symbol,
                "bars": bars,
                "count": len(bars),
                "interval": interval,
                "timestamp": _utcnow().isoformat(),
                "source": "predictions"
            })
        
        return jsonify({"ok": False, "error": "history_data_not_found"}), 503

    @app.get("/api/crypto/symbols")
    def crypto_symbols() -> Any:
        """Get list of available crypto symbols."""
        predictions = _get_market_predictions("crypto", limit=100)
        symbols = []
        for item in predictions:
            sym = str(item.get("symbol") or "").upper()
            if not sym:
                continue
            symbols.append({
                "symbol": sym,
                "name": item.get("name") or sym.replace("USDT", ""),
                "instrument_id": sym.replace("USDT", "").lower(),
                "current_price": _safe_float(item.get("current_price")),
            })
        if not symbols:
            symbols = [
                {"symbol": "BTCUSDT", "name": "Bitcoin", "instrument_id": "btc"},
                {"symbol": "ETHUSDT", "name": "Ethereum", "instrument_id": "eth"},
                {"symbol": "SOLUSDT", "name": "Solana", "instrument_id": "sol"},
            ]
        return jsonify({"ok": True, "symbols": symbols, "count": len(symbols), "timestamp": _utcnow().isoformat()})

    @app.get("/api/crypto/signal/<symbol>")
    def crypto_signal(symbol: str) -> Any:
        """Get trading signal for a specific crypto symbol."""
        symbol = symbol.upper().strip()
        signal = _find_market_signal("crypto", symbol)
        if signal is None:
            signal = {
                "symbol": symbol,
                "action": "Flat",
                "policy_action": "Flat",
                "confidence": None,
                "confidence_score": None,
                "p_up": None,
                "p_down": None,
                "current_price": None,
                "target_price": None,
                "target_price_q50": None,
                "q50_change_pct": None,
                "volatility_score": None,
                "signal_strength": "Weak",
                "signal_strength_pp": 0.0,
                "trend_label": "neutral",
                "risk_level": "medium",
                "position_size": 0.0,
            }
        return jsonify({"ok": True, "symbol": symbol, "signal": signal, "timestamp": _utcnow().isoformat()})

    @app.get("/api/crypto/prediction/<symbol>")
    def crypto_prediction(symbol: str) -> Any:
        """Alias endpoint used by web frontend."""
        symbol = symbol.upper().strip()
        signal = _find_market_signal("crypto", symbol)
        if signal is None:
            return jsonify({"ok": False, "error": "signal_not_found", "symbol": symbol}), 404
        return jsonify({"ok": True, "symbol": symbol, "signal": signal, "prediction": signal, "timestamp": _utcnow().isoformat()})

    # ==================== CN Equity API ====================

    @app.get("/api/cn/predictions")
    def cn_predictions() -> Any:
        """Get A-share predictions."""
        symbol_filter = request.args.get("symbol", "").strip().upper()
        try:
            limit = min(int(request.args.get("limit", "100")), 500)
        except ValueError:
            limit = 100

        predictions = _get_market_predictions("cn_equity", limit=limit, symbol_filter=symbol_filter)
        if not predictions:
            # fallback sample set to keep page functional even when CN artifacts are missing
            sample = [
                {"symbol": "600519.SH", "name": "贵州茅台", "current_price": 1685.00, "change_percent": 1.25, "target_price": 1750.0},
                {"symbol": "300750.SZ", "name": "宁德时代", "current_price": 168.50, "change_percent": -2.35, "target_price": 160.0},
                {"symbol": "002594.SZ", "name": "比亚迪", "current_price": 245.80, "change_percent": 2.15, "target_price": 265.0},
                {"symbol": "601318.SH", "name": "中国平安", "current_price": 42.35, "change_percent": 0.25, "target_price": 45.0},
                {"symbol": "600036.SH", "name": "招商银行", "current_price": 32.85, "change_percent": 0.85, "target_price": 35.0},
            ]
            if symbol_filter:
                sample = [x for x in sample if _symbol_key(x.get("symbol")) == _symbol_key(symbol_filter)]
            predictions = sample[:limit]

        return jsonify({
            "ok": True,
            "predictions": predictions,
            "items": predictions,
            "count": len(predictions),
            "timestamp": _utcnow().isoformat(),
        })

    @app.get("/api/cn/history/<symbol>")
    def cn_history(symbol: str) -> Any:
        """Get historical price data for a CN stock symbol."""
        symbol = symbol.upper().strip()
        try:
            limit = min(int(request.args.get("limit", "100")), 1000)
        except ValueError:
            limit = 100
        
        interval = request.args.get("interval", "daily").lower()
        data_path = _get_data_path()
        
        if interval == "daily":
            features_file = data_path / "features_daily.csv"
        else:
            features_file = data_path / "features_hourly.csv"
        
        if features_file.exists():
            rows = _read_csv_with_headers(features_file, limit=limit * 20)
            symbol_key = _symbol_key(symbol)

            # Prefer strict symbol match when the file contains multi-symbol data.
            symbol_rows = [row for row in rows if _symbol_key(row.get("symbol")) == symbol_key]
            if symbol_rows:
                rows = symbol_rows
            else:
                # If symbol column does not exist in the file (single-instrument export),
                # keep original rows as fallback.
                has_symbol_values = any(str(row.get("symbol") or "").strip() for row in rows)
                if has_symbol_values:
                    rows = []

            bars = []
            for row in rows[-limit:]:
                close = _safe_float(row.get("close"))
                if close is None:
                    continue
                bar: dict[str, Any] = {
                    "timestamp": row.get("timestamp_utc") or row.get("timestamp_market") or row.get("timestamp", ""),
                    "close": close,
                }
                open_price = _safe_float(row.get("open"))
                high = _safe_float(row.get("high"))
                low = _safe_float(row.get("low"))
                volume = _safe_float(row.get("volume"))
                if open_price is not None:
                    bar["open"] = open_price
                if high is not None:
                    bar["high"] = high
                if low is not None:
                    bar["low"] = low
                if volume is not None:
                    bar["volume"] = volume
                bars.append(bar)
            if not bars:
                bars = _fetch_yahoo_history_bars(
                    symbol=symbol,
                    market="cn_equity",
                    interval=interval,
                    limit=limit,
                )
                if not bars:
                    return jsonify({"ok": False, "error": "cn_history_symbol_not_found", "symbol": symbol}), 404
                return jsonify({
                    "ok": True,
                    "symbol": symbol,
                    "bars": bars,
                    "count": len(bars),
                    "interval": interval,
                    "source": "yahoo_chart",
                    "timestamp": _utcnow().isoformat()
                })
            return jsonify({
                "ok": True,
                "symbol": symbol,
                "bars": bars,
                "count": len(bars),
                "interval": interval,
                "timestamp": _utcnow().isoformat()
            })
        
        bars = _fetch_yahoo_history_bars(
            symbol=symbol,
            market="cn_equity",
            interval=interval,
            limit=limit,
        )
        if bars:
            return jsonify({
                "ok": True,
                "symbol": symbol,
                "bars": bars,
                "count": len(bars),
                "interval": interval,
                "source": "yahoo_chart",
                "timestamp": _utcnow().isoformat()
            })
        return jsonify({"ok": False, "error": "cn_history_not_found"}), 503

    @app.get("/api/cn/symbols")
    def cn_symbols() -> Any:
        """Get list of available A-share symbols."""
        predictions = _get_market_predictions("cn_equity", limit=200)
        symbols: list[dict[str, Any]] = []
        for item in predictions:
            sym = str(item.get("symbol") or "").upper()
            if not sym:
                continue
            symbols.append({
                "symbol": sym,
                "name": item.get("name") or sym,
                "instrument_id": sym.lower().replace(".", "_"),
                "current_price": _safe_float(item.get("current_price")),
            })
        if not symbols:
            symbols = [
                {"symbol": "600519.SH", "name": "贵州茅台", "instrument_id": "moutai"},
                {"symbol": "300750.SZ", "name": "宁德时代", "instrument_id": "ningde"},
                {"symbol": "002594.SZ", "name": "比亚迪", "instrument_id": "byd"},
                {"symbol": "601318.SH", "name": "中国平安", "instrument_id": "pingan"},
                {"symbol": "600036.SH", "name": "招商银行", "instrument_id": "cmb"},
            ]
        return jsonify({"ok": True, "symbols": symbols, "count": len(symbols), "timestamp": _utcnow().isoformat()})

    @app.get("/api/cn/prediction/<symbol>")
    def cn_prediction(symbol: str) -> Any:
        """Alias endpoint used by web frontend."""
        symbol = symbol.upper().strip()
        signal = _find_market_signal("cn_equity", symbol)
        if signal is None:
            predictions = _get_market_predictions("cn_equity", limit=50, symbol_filter=symbol)
            if predictions:
                signal = _build_signal_payload(predictions[0])
            else:
                return jsonify({"ok": False, "error": "signal_not_found", "symbol": symbol}), 404
        return jsonify({"ok": True, "symbol": symbol, "signal": signal, "prediction": signal, "timestamp": _utcnow().isoformat()})


    # ==================== US Equity API ====================

    @app.get("/api/us/predictions")
    def us_predictions() -> Any:
        """Get US stock predictions."""
        symbol_filter = request.args.get("symbol", "").strip().upper()
        try:
            limit = min(int(request.args.get("limit", "100")), 500)
        except ValueError:
            limit = 100

        predictions = _get_market_predictions("us_equity", limit=limit, symbol_filter=symbol_filter)
        if not predictions:
            sample = [
                {"symbol": "AAPL", "name": "Apple Inc.", "current_price": 178.72, "change_percent": 1.25, "target_price": 185.0},
                {"symbol": "MSFT", "name": "Microsoft", "current_price": 415.50, "change_percent": 1.85, "target_price": 430.0},
                {"symbol": "TSLA", "name": "Tesla Inc.", "current_price": 245.30, "change_percent": -2.15, "target_price": 235.0},
                {"symbol": "GOOGL", "name": "Alphabet", "current_price": 156.80, "change_percent": 0.95, "target_price": 165.0},
                {"symbol": "NVDA", "name": "NVIDIA", "current_price": 875.20, "change_percent": 3.45, "target_price": 920.0},
            ]
            if symbol_filter:
                sample = [x for x in sample if _symbol_key(x.get("symbol")) == _symbol_key(symbol_filter)]
            predictions = sample[:limit]

        return jsonify({
            "ok": True,
            "predictions": predictions,
            "items": predictions,
            "count": len(predictions),
            "timestamp": _utcnow().isoformat(),
        })


    @app.get("/api/us/history/<symbol>")
    def us_history(symbol: str) -> Any:
        """Get historical price data for a US stock symbol."""
        symbol = symbol.upper().strip()
        try:
            limit = min(int(request.args.get("limit", "100")), 1000)
        except ValueError:
            limit = 100
        
        interval = request.args.get("interval", "daily").lower()
        data_path = _get_data_path()
        
        if interval == "daily":
            features_file = data_path / "features_daily.csv"
        else:
            features_file = data_path / "features_hourly.csv"
        
        if features_file.exists():
            rows = _read_csv_with_headers(features_file, limit=limit * 20)
            symbol_key = _symbol_key(symbol)

            # Prefer strict symbol match when available.
            symbol_rows = [row for row in rows if _symbol_key(row.get("symbol")) == symbol_key]
            if symbol_rows:
                rows = symbol_rows
            else:
                has_symbol_values = any(str(row.get("symbol") or "").strip() for row in rows)
                if has_symbol_values:
                    rows = []

            bars = []
            for row in rows[-limit:]:
                close = _safe_float(row.get("close"))
                if close is None:
                    continue
                bar: dict[str, Any] = {
                    "timestamp": row.get("timestamp_utc") or row.get("timestamp_market") or row.get("timestamp", ""),
                    "close": close,
                }
                open_price = _safe_float(row.get("open"))
                high = _safe_float(row.get("high"))
                low = _safe_float(row.get("low"))
                volume = _safe_float(row.get("volume"))
                if open_price is not None:
                    bar["open"] = open_price
                if high is not None:
                    bar["high"] = high
                if low is not None:
                    bar["low"] = low
                if volume is not None:
                    bar["volume"] = volume
                bars.append(bar)
            if not bars:
                bars = _fetch_yahoo_history_bars(
                    symbol=symbol,
                    market="us_equity",
                    interval=interval,
                    limit=limit,
                )
                if not bars:
                    return jsonify({"ok": False, "error": "us_history_symbol_not_found", "symbol": symbol}), 404
                return jsonify({
                    "ok": True,
                    "symbol": symbol,
                    "bars": bars,
                    "count": len(bars),
                    "interval": interval,
                    "source": "yahoo_chart",
                    "timestamp": _utcnow().isoformat()
                })
            return jsonify({
                "ok": True,
                "symbol": symbol,
                "bars": bars,
                "count": len(bars),
                "interval": interval,
                "timestamp": _utcnow().isoformat()
            })
        
        bars = _fetch_yahoo_history_bars(
            symbol=symbol,
            market="us_equity",
            interval=interval,
            limit=limit,
        )
        if bars:
            return jsonify({
                "ok": True,
                "symbol": symbol,
                "bars": bars,
                "count": len(bars),
                "interval": interval,
                "source": "yahoo_chart",
                "timestamp": _utcnow().isoformat()
            })
        return jsonify({"ok": False, "error": "us_history_not_found"}), 503

    @app.get("/api/us/symbols")
    def us_symbols() -> Any:
        """Get list of available US stock symbols."""
        predictions = _get_market_predictions("us_equity", limit=500)
        symbols: list[dict[str, Any]] = []
        for item in predictions:
            sym = str(item.get("symbol") or "").upper()
            if not sym:
                continue
            symbols.append({
                "symbol": sym,
                "name": item.get("name") or sym,
                "instrument_id": sym.lower(),
                "current_price": _safe_float(item.get("current_price")),
            })
        if not symbols:
            symbols = [
                {"symbol": "AAPL", "name": "Apple Inc.", "instrument_id": "aapl"},
                {"symbol": "MSFT", "name": "Microsoft", "instrument_id": "msft"},
                {"symbol": "TSLA", "name": "Tesla Inc.", "instrument_id": "tsla"},
                {"symbol": "GOOGL", "name": "Alphabet", "instrument_id": "googl"},
                {"symbol": "NVDA", "name": "NVIDIA", "instrument_id": "nvda"},
            ]
        return jsonify({"ok": True, "symbols": symbols, "count": len(symbols), "timestamp": _utcnow().isoformat()})

    @app.get("/api/us/prediction/<symbol>")
    def us_prediction(symbol: str) -> Any:
        """Alias endpoint used by web frontend."""
        symbol = symbol.upper().strip()
        signal = _find_market_signal("us_equity", symbol)
        if signal is None:
            predictions = _get_market_predictions("us_equity", limit=50, symbol_filter=symbol)
            if predictions:
                signal = _build_signal_payload(predictions[0])
            else:
                return jsonify({"ok": False, "error": "signal_not_found", "symbol": symbol}), 404
        return jsonify({"ok": True, "symbol": symbol, "signal": signal, "prediction": signal, "timestamp": _utcnow().isoformat()})


    # ==================== Session Forecast API ====================

    @app.get("/api/session/crypto")
    def session_crypto() -> Any:
        """Get crypto trading session forecast page payload (web-friendly, with legacy fields)."""
        symbol = str(request.args.get("symbol", "BTCUSDT")).strip().upper()
        exchange = str(request.args.get("exchange", "binance")).strip().lower()
        market_type = str(request.args.get("market_type", "perp")).strip().lower()
        mode = _normalize_mode(request.args.get("mode", "forecast"))
        horizon_hours = _safe_int(request.args.get("horizon_hours"), 4)
        lookforward_days = _safe_int(request.args.get("lookforward_days"), 14)
        risk_profile = str(request.args.get("risk_profile", "standard")).strip().lower()
        rank_key = str(request.args.get("rank_key", "edge_score")).strip().lower()
        cost_bps = _safe_float(request.args.get("cost_bps"))
        top_n = _safe_int(request.args.get("top_n"), 5)
        if cost_bps is None:
            cost_bps = 8.0

        payload, err = _build_crypto_session_payload(
            symbol=symbol,
            exchange=exchange,
            market_type=market_type,
            mode=mode,
            horizon_hours=horizon_hours,
            lookforward_days=lookforward_days,
            risk_profile=risk_profile,
            rank_key=rank_key,
            cost_bps=cost_bps,
            top_n=top_n,
        )
        if err:
            return jsonify({"ok": False, "error": err}), 503
        return jsonify(payload)

    @app.get("/api/session/index")
    def session_index() -> Any:
        """Get index trading session forecast page payload (web-friendly, with legacy fields)."""
        market = str(request.args.get("market", "cn")).strip().lower()
        index_key = str(request.args.get("index_key", "")).strip().lower()
        if not index_key:
            index_key = "sse" if market == "cn" else "nasdaq"
        mode = _normalize_mode(request.args.get("mode", "forecast"))
        horizon_hours = _safe_int(request.args.get("horizon_hours"), 4)
        lookforward_days = _safe_int(request.args.get("lookforward_days"), 14)
        risk_profile = str(request.args.get("risk_profile", "standard")).strip().lower()
        rank_key = str(request.args.get("rank_key", "edge_score")).strip().lower()
        cost_bps = _safe_float(request.args.get("cost_bps"))
        top_n = _safe_int(request.args.get("top_n"), 5)
        if cost_bps is None:
            cost_bps = 8.0

        payload, err = _build_index_session_payload(
            index_key=index_key,
            mode=mode,
            horizon_hours=horizon_hours,
            lookforward_days=lookforward_days,
            risk_profile=risk_profile,
            rank_key=rank_key,
            cost_bps=cost_bps,
            top_n=top_n,
        )
        if err:
            return jsonify({"ok": False, "error": err}), 503

        # Legacy shape for older web clients.
        summary = dict(payload.get("summary") or {})
        summary["trend"] = summary.get("overallTrend")
        summary["best_session"] = summary.get("bestSession")
        summary["risk_session"] = summary.get("riskSession")
        payload["summary"] = summary
        payload["session_cards"] = payload.get("sessions", [])
        payload["sessions"] = payload.get("sessions_legacy", [])
        return jsonify(payload)

    # ==================== Error Handlers ====================

    @app.errorhandler(404)
    def not_found(error):
        return jsonify({"ok": False, "error": "not_found"}), 404

    @app.errorhandler(500)
    def internal_error(error):
        return jsonify({"ok": False, "error": "internal_server_error"}), 500

    return app


if __name__ == "__main__":
    app = create_app()
    host = os.getenv("NOTES_HOST", "127.0.0.1")
    port = int(os.getenv("NOTES_PORT", "5001"))
    debug = os.getenv("NOTES_DEBUG", "0") == "1"
    app.run(host=host, port=port, debug=debug)



