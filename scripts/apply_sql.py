from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.notes.app import create_app, db


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply a .sql file to the Notes database configured by Flask app."
    )
    parser.add_argument(
        "--sql",
        default="sql/schema.sql",
        help="Path to the SQL file to execute. Default: sql/schema.sql",
    )
    parser.add_argument(
        "--db-path",
        default="",
        help="Optional NOTES_DB_PATH override (relative or absolute path).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sql_path = Path(args.sql).resolve()

    if not sql_path.exists():
        print(f"[ERROR] SQL file not found: {sql_path}")
        return 1

    if args.db_path.strip():
        os.environ["NOTES_DB_PATH"] = args.db_path.strip()

    sql_text = sql_path.read_text(encoding="utf-8").strip()
    if not sql_text:
        print(f"[ERROR] SQL file is empty: {sql_path}")
        return 1

    app = create_app()
    with app.app_context():
        engine = db.engine
        if engine.dialect.name != "sqlite":
            print(
                f"[ERROR] Only sqlite is supported by this script right now. "
                f"Current dialect: {engine.dialect.name}"
            )
            return 2

        raw_conn = engine.raw_connection()
        try:
            raw_conn.executescript(sql_text)
            raw_conn.commit()
        except Exception as exc:
            raw_conn.rollback()
            print(f"[ERROR] Failed to execute SQL: {exc}")
            return 3
        finally:
            raw_conn.close()

        print("[OK] SQL applied successfully.")
        print(f"[INFO] SQL file: {sql_path}")
        print(f"[INFO] DB url: {engine.url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
