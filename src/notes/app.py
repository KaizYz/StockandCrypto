from __future__ import annotations

import csv
import json
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer
from sqlalchemy import func, or_
from sqlalchemy.exc import IntegrityError
from werkzeug.security import check_password_hash, generate_password_hash

db = SQLAlchemy()

USERNAME_PATTERN = re.compile(r"^[A-Za-z0-9_]{3,32}$")


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
    
    # Build normalized result
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

    # ==================== Market Overview ====================

    @app.get("/api/market/overview")
    def market_overview() -> Any:
        """
        Get market overview with BTC/ETH/SOL prices and 24h changes.
        Returns: { "ok": true, "assets": [...], "generated_at_utc": "...", "timestamp": "..." }
        """
        data_path = _get_data_path() / "market_snapshot.json"
        if not data_path.exists():
            return jsonify({"ok": False, "error": "market_data_not_found"}), 503
        try:
            with open(data_path, "r", encoding="utf-8") as f:
                snapshot = json.load(f)
        except Exception as e:
            return jsonify({"ok": False, "error": f"failed_to_read_data: {e}"}), 500

        target_assets = {"btc", "eth", "sol"}
        assets = []
        for row in snapshot.get("rows", []):
            instrument_id = row.get("instrument_id", "").lower()
            if instrument_id in target_assets:
                assets.append({
                    "symbol": row.get("symbol", ""),
                    "name": row.get("name", ""),
                    "instrument_id": instrument_id,
                    "current_price": _safe_float(row.get("current_price")),
                    "predicted_change_pct": _safe_float(row.get("predicted_change_pct")),
                    "market": row.get("market", "crypto"),
                    "q10_change_pct": _safe_float(row.get("q10_change_pct")),
                    "q50_change_pct": _safe_float(row.get("q50_change_pct")),
                    "q90_change_pct": _safe_float(row.get("q90_change_pct")),
                    "error": row.get("error_message") if row.get("error_message") and not row.get("current_price") else None,
                })

        if not assets:
            for row in snapshot.get("rows", [])[:5]:
                assets.append({
                    "symbol": row.get("symbol", ""),
                    "name": row.get("name", ""),
                    "instrument_id": row.get("instrument_id", ""),
                    "current_price": _safe_float(row.get("current_price")),
                    "predicted_change_pct": _safe_float(row.get("predicted_change_pct")),
                    "market": row.get("market", ""),
                })

        return jsonify({
            "ok": True,
            "assets": assets,
            "generated_at_utc": snapshot.get("rows", [{}])[0].get("generated_at_utc", "") if snapshot.get("rows") else "",
            "timestamp": _utcnow().isoformat(),
        })

    # ==================== Crypto API ====================

    @app.get("/api/crypto/predictions")
    def crypto_predictions() -> Any:
        """
        Get crypto predictions for BTC/ETH/SOL.
        
        Query params:
            symbol: Filter by symbol (e.g., BTCUSDT)
            limit: Max number of results (default 100, max 500)
            normalized: Return normalized format (default true)
        
        Returns: { "ok": true, "predictions": [...], "count": N, "timestamp": "..." }
        """
        symbol_filter = request.args.get("symbol", "").strip().upper()
        try:
            limit = min(int(request.args.get("limit", "100")), 500)
        except ValueError:
            limit = 100
        
        normalized = request.args.get("normalized", "true").lower() != "false"
        
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
                })
        
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
                return jsonify({
                    "ok": True,
                    "predictions": predictions,
                    "count": len(predictions),
                    "timestamp": _utcnow().isoformat(),
                    "source": "policy_signals_hourly"
                })
        
        # Fallback to predictions_hourly
        hourly_file = data_path / "predictions_hourly.csv"
        if hourly_file.exists():
            rows = _get_latest_csv_rows(hourly_file, limit, symbol_filter if symbol_filter else None)
            if rows:
                return jsonify({
                    "ok": True,
                    "predictions": rows,
                    "count": len(rows),
                    "timestamp": _utcnow().isoformat(),
                    "source": "predictions_hourly"
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
        symbols = [
            {"symbol": "BTCUSDT", "name": "Bitcoin", "instrument_id": "btc"},
            {"symbol": "ETHUSDT", "name": "Ethereum", "instrument_id": "eth"},
            {"symbol": "SOLUSDT", "name": "Solana", "instrument_id": "sol"},
        ]
        data_path = _get_data_path()
        snapshot_file = data_path / "market_snapshot.json"
        if snapshot_file.exists():
            try:
                with open(snapshot_file, "r", encoding="utf-8") as f:
                    snapshot = json.load(f)
                crypto_assets = [row for row in snapshot.get("rows", []) if row.get("market") == "crypto"]
                if crypto_assets:
                    symbols = [{
                        "symbol": row.get("symbol", ""),
                        "name": row.get("name", ""),
                        "instrument_id": row.get("instrument_id", ""),
                        "current_price": _safe_float(row.get("current_price")),
                    } for row in crypto_assets]
            except Exception:
                pass
        return jsonify({"ok": True, "symbols": symbols, "count": len(symbols), "timestamp": _utcnow().isoformat()})

    @app.get("/api/crypto/signal/<symbol>")
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
        })

    # ==================== CN Equity API ====================

    @app.get("/api/cn/predictions")
    def cn_predictions() -> Any:
        """
        Get A-share (Chinese stock) predictions.
        
        Query params:
            symbol: Filter by symbol
            limit: Max results (default 100)
        
        Returns: { "ok": true, "predictions": [...], "count": N }
        """
        symbol_filter = request.args.get("symbol", "").strip().upper()
        try:
            limit = min(int(request.args.get("limit", "100")), 500)
        except ValueError:
            limit = 100
        
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
                    })
            except Exception:
                pass
        
        daily_file = data_path / "predictions_daily.csv"
        if daily_file.exists():
            rows = _get_latest_csv_rows(daily_file, limit, symbol_filter if symbol_filter else None)
            return jsonify({
                "ok": True,
                "predictions": rows,
                "count": len(rows),
                "timestamp": _utcnow().isoformat()
            })
        
        return jsonify({"ok": False, "error": "cn_predictions_not_found"}), 503

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
            rows = _read_csv_with_headers(features_file, limit=limit * 2)
            bars = []
            for row in rows[-limit:]:
                bars.append({
                    "timestamp": row.get("timestamp_utc") or row.get("timestamp_market", ""),
                    "close": _safe_float(row.get("close")),
                })
            return jsonify({
                "ok": True,
                "symbol": symbol,
                "bars": bars,
                "count": len(bars),
                "interval": interval,
                "timestamp": _utcnow().isoformat()
            })
        
        return jsonify({"ok": False, "error": "cn_history_not_found"}), 503

    @app.get("/api/cn/symbols")
    def cn_symbols() -> Any:
        """Get list of available A-share symbols."""
        symbols = [{"symbol": "600519.SS", "name": "贵州茅台", "instrument_id": "moutai"}]
        data_path = _get_data_path()
        snapshot_file = data_path / "market_snapshot.json"
        if snapshot_file.exists():
            try:
                with open(snapshot_file, "r", encoding="utf-8") as f:
                    snapshot = json.load(f)
                cn_assets = [row for row in snapshot.get("rows", []) if row.get("market") == "cn_equity"]
                if cn_assets:
                    symbols = [{
                        "symbol": row.get("symbol", ""),
                        "name": row.get("name", ""),
                        "instrument_id": row.get("instrument_id", ""),
                        "current_price": _safe_float(row.get("current_price")),
                    } for row in cn_assets]
            except Exception:
                pass
        return jsonify({"ok": True, "symbols": symbols, "count": len(symbols), "timestamp": _utcnow().isoformat()})

    # ==================== US Equity API ====================

    @app.get("/api/us/predictions")
    def us_predictions() -> Any:
        """
        Get US stock predictions.
        
        Query params:
            symbol: Filter by symbol
            limit: Max results (default 100)
        
        Returns: { "ok": true, "predictions": [...], "count": N }
        """
        symbol_filter = request.args.get("symbol", "").strip().upper()
        try:
            limit = min(int(request.args.get("limit", "100")), 500)
        except ValueError:
            limit = 100
        
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
                    })
            except Exception:
                pass
        
        daily_file = data_path / "predictions_daily.csv"
        if daily_file.exists():
            rows = _get_latest_csv_rows(daily_file, limit, symbol_filter if symbol_filter else None)
            return jsonify({
                "ok": True,
                "predictions": rows,
                "count": len(rows),
                "timestamp": _utcnow().isoformat()
            })
        
        return jsonify({"ok": False, "error": "us_predictions_not_found"}), 503

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
            rows = _read_csv_with_headers(features_file, limit=limit * 2)
            bars = []
            for row in rows[-limit:]:
                bars.append({
                    "timestamp": row.get("timestamp_utc") or row.get("timestamp_market", ""),
                    "close": _safe_float(row.get("close")),
                })
            return jsonify({
                "ok": True,
                "symbol": symbol,
                "bars": bars,
                "count": len(bars),
                "interval": interval,
                "timestamp": _utcnow().isoformat()
            })
        
        return jsonify({"ok": False, "error": "us_history_not_found"}), 503

    @app.get("/api/us/symbols")
    def us_symbols() -> Any:
        """Get list of available US stock symbols."""
        symbols = [{"symbol": "AAPL", "name": "Apple Inc.", "instrument_id": "aapl"}]
        data_path = _get_data_path()
        snapshot_file = data_path / "market_snapshot.json"
        if snapshot_file.exists():
            try:
                with open(snapshot_file, "r", encoding="utf-8") as f:
                    snapshot = json.load(f)
                us_assets = [row for row in snapshot.get("rows", []) if row.get("market") == "us_equity"]
                if us_assets:
                    symbols = [{
                        "symbol": row.get("symbol", ""),
                        "name": row.get("name", ""),
                        "instrument_id": row.get("instrument_id", ""),
                        "current_price": _safe_float(row.get("current_price")),
                    } for row in us_assets]
            except Exception:
                pass
        return jsonify({"ok": True, "symbols": symbols, "count": len(symbols), "timestamp": _utcnow().isoformat()})

    # ==================== Session Forecast API ====================

    @app.get("/api/session/crypto")
    def session_crypto() -> Any:
        """Get crypto trading session forecasts."""
        data_path = _get_data_path()
        session_file = data_path / "session_forecast_blocks.csv"
        if session_file.exists():
            rows = _read_csv_with_headers(session_file, limit=500)
            crypto_rows = [row for row in rows if row.get("symbol", "").upper() in ["BTCUSDT", "ETHUSDT", "SOLUSDT"]]
            if crypto_rows:
                return jsonify({"ok": True, "sessions": crypto_rows, "timestamp": _utcnow().isoformat()})
        hourly_file = data_path / "session_forecast_hourly.csv"
        if hourly_file.exists():
            rows = _get_latest_csv_rows(hourly_file, 50)
            return jsonify({"ok": True, "sessions": rows, "timestamp": _utcnow().isoformat()})
        return jsonify({"ok": False, "error": "crypto_session_data_not_found"}), 503

    @app.get("/api/session/index")
    def session_index() -> Any:
        """Get index trading session forecasts."""
        data_path = _get_data_path()
        daily_file = data_path / "session_forecast_daily.csv"
        if daily_file.exists():
            rows = _get_latest_csv_rows(daily_file, 50)
            return jsonify({"ok": True, "sessions": rows, "timestamp": _utcnow().isoformat()})
        hourly_file = data_path / "session_forecast_hourly.csv"
        if hourly_file.exists():
            rows = _get_latest_csv_rows(hourly_file, 50)
            return jsonify({"ok": True, "sessions": rows, "timestamp": _utcnow().isoformat()})
        return jsonify({"ok": False, "error": "index_session_data_not_found"}), 503

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
