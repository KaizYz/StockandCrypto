from __future__ import annotations

import json
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request
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
    updated_at = db.Column(db.DateTime(timezone=True), nullable=False, default=_utcnow, onupdate=_utcnow)
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
    updated_at = db.Column(db.DateTime(timezone=True), nullable=False, default=_utcnow, onupdate=_utcnow)
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


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["SQLALCHEMY_DATABASE_URI"] = _notes_db_uri()
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    app.config["NOTES_AUTH_SECRET"] = os.getenv("NOTES_AUTH_SECRET", "notes-dev-secret-change-me")
    app.config["NOTES_TOKEN_TTL_SECONDS"] = int(os.getenv("NOTES_TOKEN_TTL_SECONDS", "604800"))
    db.init_app(app)
    with app.app_context():
        db.create_all()

    @app.get("/api/health")
    def health() -> Any:
        return jsonify({"ok": True, "service": "notes-api"})

    @app.post("/api/auth/register")
    def register() -> Any:
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
        return jsonify(
            {
                "ok": True,
                "token": token,
                "token_type": "Bearer",
                "expires_in": int(app.config["NOTES_TOKEN_TTL_SECONDS"]),
                "user": user.to_public_dict(),
            }
        )

    @app.get("/api/auth/me")
    def me() -> Any:
        user, auth_err = _require_auth_user(app)
        if auth_err is not None:
            return auth_err
        return jsonify({"ok": True, "user": user.to_public_dict()})

    @app.post("/api/notes")
    def create_note() -> Any:
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
        items = (
            query.order_by(Note.updated_at.desc(), Note.id.desc())
            .limit(page_size)
            .all()
        )
        return jsonify(
            {
                "ok": True,
                "items": [item.to_dict(include_author=False) for item in items],
                "count": len(items),
            }
        )

    @app.get("/api/notes/public")
    def list_public_notes() -> Any:
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
        items = (
            query.order_by(Note.updated_at.desc(), Note.id.desc())
            .limit(page_size)
            .all()
        )
        return jsonify(
            {
                "ok": True,
                "items": [item.to_dict(include_author=True) for item in items],
                "count": len(items),
            }
        )

    # ==================== Market API for Homepage ====================

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

    @app.get("/api/market/overview")
    def market_overview() -> Any:
        """Return market overview with BTC/ETH/SOL prices and 24h changes."""
        data_path = _get_data_path() / "market_snapshot.json"
        if not data_path.exists():
            return jsonify({"ok": False, "error": "market_data_not_found"}), 503

        try:
            with open(data_path, "r", encoding="utf-8") as f:
                snapshot = json.load(f)
        except Exception as e:
            return jsonify({"ok": False, "error": f"failed_to_read_data: {e}"}), 500

        # Filter for main crypto assets and valid data
        target_assets = {"btc", "eth", "sol"}
        assets = []
        for row in snapshot.get("rows", []):
            instrument_id = row.get("instrument_id", "").lower()
            if instrument_id in target_assets:
                current_price = _safe_float(row.get("current_price"))
                predicted_change = _safe_float(row.get("predicted_change_pct"))
                assets.append({
                    "symbol": row.get("symbol", ""),
                    "name": row.get("name", ""),
                    "instrument_id": instrument_id,
                    "current_price": current_price,
                    "predicted_change_pct": predicted_change,
                    "market": row.get("market", "crypto"),
                    "error": row.get("error_message") if row.get("error_message") and not current_price else None,
                })

        # If no crypto found, include all assets as fallback
        if not assets:
            for row in snapshot.get("rows", [])[:5]:  # Limit to 5 assets
                current_price = _safe_float(row.get("current_price"))
                assets.append({
                    "symbol": row.get("symbol", ""),
                    "name": row.get("name", ""),
                    "instrument_id": row.get("instrument_id", ""),
                    "current_price": current_price,
                    "predicted_change_pct": _safe_float(row.get("predicted_change_pct")),
                    "market": row.get("market", ""),
                    "error": row.get("error_message") if row.get("error_message") and not current_price else None,
                })

        return jsonify({
            "ok": True,
            "assets": assets,
            "generated_at_utc": snapshot.get("rows", [{}])[0].get("generated_at_utc", "") if snapshot.get("rows") else "",
            "timestamp": _utcnow().isoformat(),
        })

    @app.get("/api/predictions/summary")
    def predictions_summary() -> Any:
        """Return latest prediction signals summary."""
        data_path = _get_data_path() / "predictions_latest_summary.json"
        if not data_path.exists():
            return jsonify({"ok": False, "error": "predictions_data_not_found"}), 503

        try:
            with open(data_path, "r", encoding="utf-8") as f:
                predictions = json.load(f)
        except Exception as e:
            return jsonify({"ok": False, "error": f"failed_to_read_data: {e}"}), 500

        # Extract hourly and daily summaries
        hourly = predictions.get("hourly", {}).get("latest", {})
        daily = predictions.get("daily", {}).get("latest", {})

        summary = {
            "hourly": {
                "timestamp_utc": hourly.get("timestamp_utc"),
                "close": _safe_float(hourly.get("close")),
                "direction_prediction": hourly.get("dir_h1_pred"),
                "prob_up": _safe_float(hourly.get("dir_h1_p_up")),
                "prob_down": _safe_float(hourly.get("dir_h1_p_down")),
                "return_q10": _safe_float(hourly.get("ret_h1_q0.1")),
                "return_q50": _safe_float(hourly.get("ret_h1_q0.5")),
                "return_q90": _safe_float(hourly.get("ret_h1_q0.9")),
            },
            "daily": {
                "timestamp_utc": daily.get("timestamp_utc"),
                "close": _safe_float(daily.get("close")),
                "direction_prediction": daily.get("dir_h1_pred"),
                "prob_up": _safe_float(daily.get("dir_h1_p_up")),
                "prob_down": _safe_float(daily.get("dir_h1_p_down")),
                "return_q10": _safe_float(daily.get("ret_h1_q0.1")),
                "return_q50": _safe_float(daily.get("ret_h1_q0.5")),
                "return_q90": _safe_float(daily.get("ret_h1_q0.9")),
            },
        }

        return jsonify({
            "ok": True,
            "summary": summary,
            "model_versions": {
                "hourly": predictions.get("hourly", {}).get("model_version"),
                "daily": predictions.get("daily", {}).get("model_version"),
            },
            "timestamp": _utcnow().isoformat(),
        })

    @app.get("/api/system/status")
    def system_status() -> Any:
        """Return system status and last update times."""
        data_path = _get_data_path()

        status = {
            "ok": True,
            "service": "StockandCrypto API",
            "version": "1.0.0",
            "timestamp": _utcnow().isoformat(),
        }

        # Check data files existence and modification times
        files_status = {}

        market_file = data_path / "market_snapshot.json"
        if market_file.exists():
            stat = market_file.stat()
            files_status["market_snapshot"] = {
                "exists": True,
                "size_bytes": stat.st_size,
                "modified_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
            }
        else:
            files_status["market_snapshot"] = {"exists": False}

        predictions_file = data_path / "predictions_latest_summary.json"
        if predictions_file.exists():
            stat = predictions_file.stat()
            files_status["predictions_summary"] = {
                "exists": True,
                "size_bytes": stat.st_size,
                "modified_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
            }
        else:
            files_status["predictions_summary"] = {"exists": False}

        status["data_files"] = files_status
        status["database"] = {"connected": True}

        return jsonify(status)

    return app


if __name__ == "__main__":
    app = create_app()
    host = os.getenv("NOTES_HOST", "127.0.0.1")
    port = int(os.getenv("NOTES_PORT", "5001"))
    debug = os.getenv("NOTES_DEBUG", "0") == "1"
    app.run(host=host, port=port, debug=debug)
