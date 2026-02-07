from __future__ import annotations

import hashlib
import importlib
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

from src.markets.session_forecast import build_session_forecast_bundle
from src.markets.snapshot import build_market_snapshot_from_instruments
from src.markets.universe import get_universe_catalog, load_universe
from src.models.policy import apply_policy_frame
from src.utils.config import load_config


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data(ttl=600, show_spinner=False)
def _load_main_config_cached(config_path: str = "configs/config.yaml") -> Dict[str, object]:
    return load_config(config_path)


@st.cache_data(ttl=180, show_spinner=False)
def _load_backtest_artifacts(processed_dir_str: str) -> Dict[str, pd.DataFrame]:
    root = Path(processed_dir_str) / "backtest"
    return {
        "metrics_summary": _load_csv(root / "metrics_summary.csv"),
        "metrics_by_fold": _load_csv(root / "metrics_by_fold.csv"),
        "compare": _load_csv(root / "compare_baselines.csv"),
        "equity": _load_csv(root / "equity.csv"),
        "trades": _load_csv(root / "trades.csv"),
        "latest_signals": _load_csv(root / "latest_signals.csv"),
    }


def _normalize_symbol_token(text: object) -> str:
    token = str(text or "").strip().lower()
    return token.replace("/", "").replace("-", "").replace("_", "")


@st.cache_data(ttl=300, show_spinner=False)
def _run_single_symbol_backtest_cached(
    market: str,
    symbol: str,
    provider: str,
    fallback_symbol: str = "",
    config_path: str = "configs/config.yaml",
) -> Dict[str, pd.DataFrame]:
    from src.evaluation.backtest_multi_market import run_single_symbol_backtest

    result = run_single_symbol_backtest(
        config_path=config_path,
        market=market,
        symbol=symbol,
        provider=provider,
        fallback_symbol=fallback_symbol,
    )
    return result


@st.cache_data(ttl=180, show_spinner=False)
def _load_symbol_signal_context_cached(
    market: str,
    symbol: str,
    provider: str,
    fallback_symbol: str = "",
    config_path: str = "configs/config.yaml",
) -> pd.DataFrame:
    from src.evaluation.backtest_multi_market import _build_model_like_signals, _fetch_daily_bars

    cfg = load_config(config_path)
    bt_cfg = cfg.get("backtest_multi_market", {})
    lookback_days_cfg = bt_cfg.get(
        "lookback_days",
        {"crypto": 540, "cn_equity": 1200, "us_equity": 1200},
    )
    mk = str(market)
    sym = str(symbol)
    prov = str(provider or ("binance" if mk == "crypto" else "yahoo"))
    lb_days = int(lookback_days_cfg.get(mk, 720))
    try:
        bars, _ = _fetch_daily_bars(
            market=mk,
            symbol=sym,
            provider=prov,
            lookback_days=lb_days,
            cfg=cfg,
        )
    except Exception:
        if mk == "crypto":
            fs = str(fallback_symbol or "").strip().upper()
            if not fs and prov == "coingecko":
                guess = str(sym).strip().upper()
                if guess.isalpha() and len(guess) <= 8:
                    fs = f"{guess}USDT"
            if fs.endswith("USDT"):
                bars, _ = _fetch_daily_bars(
                    market=mk,
                    symbol=fs,
                    provider="binance",
                    lookback_days=lb_days,
                    cfg=cfg,
                )
                sym = fs
            else:
                return pd.DataFrame()
        else:
            return pd.DataFrame()
    bars = bars.dropna(subset=["timestamp_utc", "open", "close"]).sort_values("timestamp_utc").reset_index(drop=True)
    if bars.empty:
        return pd.DataFrame()
    modeled = _build_model_like_signals(bars, cfg=cfg, market=mk)
    if modeled.empty:
        return pd.DataFrame()
    latest = modeled.tail(1).copy()
    latest["symbol"] = sym
    latest["market"] = mk
    latest["provider"] = prov
    return latest


def _reason_token_cn(token: object) -> str:
    key = str(token or "").strip().lower()
    mapping = {
        "ema_bull_cross": "EMA金叉",
        "ema_bear_cross": "EMA死叉",
        "macd_golden_cross": "MACD金叉",
        "macd_dead_cross": "MACD死叉",
        "supertrend_bullish": "SuperTrend看涨",
        "supertrend_bearish": "SuperTrend看跌",
        "volume_surge": "放量",
        "bos_up": "BOS向上突破",
        "bos_down": "BOS向下突破",
        "choch_bull": "CHOCH转多",
        "choch_bear": "CHOCH转空",
        "ema_trend_up": "EMA趋势上行",
        "ema_trend_down": "EMA趋势下行",
        "long_signal": "方向与幅度满足做多条件",
        "short_signal": "方向与幅度满足做空条件",
        "short_disallowed": "当前市场禁做空",
        "short_edge_below_threshold": "做空优势不足",
        "threshold_not_met": "阈值未触发",
        "probability_neutral": "方向概率中性",
        "position_below_minimum": "仓位低于最小阈值",
        "signal_neutral": "信号中性",
        "flat": "观望",
    }
    return mapping.get(key, str(token or "-"))


def _format_reason_tokens_cn(reason_text: object) -> str:
    raw = str(reason_text or "").strip()
    if not raw:
        return "-"
    return "；".join(_reason_token_cn(t) for t in raw.split(";") if str(t).strip())


@st.cache_data(ttl=15, show_spinner=False)
def _fetch_live_btc_price() -> float | None:
    try:
        r = requests.get(
            "https://api.binance.us/api/v3/ticker/price?symbol=BTCUSDT",
            timeout=8,
        )
        r.raise_for_status()
        data = r.json()
        return float(data["price"])
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def _load_universe_cached(market: str, pool_key: str) -> pd.DataFrame:
    return load_universe(market, pool_key)


@st.cache_data(ttl=30, show_spinner=False)
def _build_selected_snapshot_cached(
    instrument_id: str,
    name: str,
    market: str,
    symbol: str,
    provider: str,
    timezone: str,
    horizon_unit: str,
    horizon_steps: int,
    history_lookback_days: int,
    config_path: str = "configs/config.yaml",
    schema_version: str = "dashboard_pages_v1",
) -> pd.DataFrame:
    inst = [
        {
            "id": instrument_id,
            "name": name,
            "market": market,
            "symbol": symbol,
            "provider": provider,
            "timezone": timezone,
            "horizon_unit": horizon_unit,
            "horizon_steps": horizon_steps,
            "history_lookback_days": history_lookback_days,
        }
    ]
    return build_market_snapshot_from_instruments(inst, config_path=config_path)


@st.cache_data(ttl=300, show_spinner=False)
def _build_session_bundle_cached(
    symbol: str,
    exchange: str,
    market_type: str,
    mode: str,
    horizon_hours: int,
    lookforward_days: int,
    refresh_token: int = 0,
    config_path: str = "configs/config.yaml",
    schema_version: str = "session_page_v1",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    _ = refresh_token
    bundle = build_session_forecast_bundle(
        symbol=symbol,
        exchange=exchange,
        market_type=market_type,
        mode=mode,
        horizon_hours=horizon_hours,
        lookforward_days=lookforward_days,
        config_path=config_path,
    )
    return bundle.hourly, bundle.blocks, bundle.daily, bundle.metadata


def _is_finite_number(x: object) -> bool:
    try:
        return bool(np.isfinite(float(x)))
    except Exception:
        return False


def _safe_float(x: object) -> float:
    try:
        out = float(x)
        if np.isfinite(out):
            return out
        return float("nan")
    except Exception:
        return float("nan")


@st.cache_data(ttl=3600, show_spinner=False)
def _get_git_hash_short_cached() -> str:
    try:
        repo_root = Path(__file__).resolve().parents[1]
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(repo_root),
            stderr=subprocess.DEVNULL,
            timeout=2,
            text=True,
        )
        return out.strip() or "-"
    except Exception:
        return "-"


def _signal_strength_label(edge_pp: float, weak_pp: float, strong_pp: float) -> str:
    if not np.isfinite(edge_pp):
        return "-"
    if edge_pp < weak_pp:
        return "弱"
    if edge_pp < strong_pp:
        return "中"
    return "强"


def _append_signal_strength_columns(df: pd.DataFrame, weak_pp: float, strong_pp: float) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["p_up"] = pd.to_numeric(out.get("p_up"), errors="coerce")
    out["signal_strength"] = (out["p_up"] - 0.5).abs()
    out["signal_strength_pp"] = out["signal_strength"] * 100.0
    out["signal_strength_score"] = (out["signal_strength"] * 200.0).clip(0.0, 100.0)
    out["signal_strength_label"] = out["signal_strength_pp"].map(
        lambda x: _signal_strength_label(float(x), weak_pp=weak_pp, strong_pp=strong_pp)
        if _is_finite_number(x)
        else "-"
    )
    return out


def _append_edge_columns(df: pd.DataFrame, cost_bps: float) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    cost_pct = float(cost_bps) / 10000.0
    out["q10_change_pct"] = pd.to_numeric(out.get("q10_change_pct"), errors="coerce")
    out["q50_change_pct"] = pd.to_numeric(out.get("q50_change_pct"), errors="coerce")
    out["q90_change_pct"] = pd.to_numeric(out.get("q90_change_pct"), errors="coerce")
    width = (out["q90_change_pct"] - out["q10_change_pct"]).abs()
    out["edge_score"] = out["q50_change_pct"] - cost_pct
    out["edge_score_short"] = (-out["q50_change_pct"]) - cost_pct
    out["edge_risk"] = out["edge_score"] / width.where(width > 1e-12, np.nan)
    out["edge_risk_short"] = out["edge_score_short"] / width.where(width > 1e-12, np.nan)
    return out


def _format_signal_strength(label: object, edge_pp: object, score: object) -> str:
    if not (_is_finite_number(edge_pp) and _is_finite_number(score)):
        return "-"
    return f"{label} ({float(edge_pp):.2f}pp / {float(score):.0f})"


def _rank_metric_options() -> Dict[str, str]:
    return {
        "p_up": "按 P(up)",
        "q50_change_pct": "按 q50_change_pct",
        "volatility_score": "按 volatility_score",
        "edge_score": "按 edge_score（推荐）",
        "edge_risk": "按 edge_risk（高级）",
    }


def _sort_by_rank(df: pd.DataFrame, rank_key: str, side: str) -> pd.DataFrame:
    if df.empty:
        return df

    if side == "up":
        if rank_key == "p_up":
            return df.sort_values("p_up", ascending=False)
        if rank_key == "q50_change_pct":
            return df.sort_values("q50_change_pct", ascending=False)
        if rank_key == "volatility_score":
            return df.sort_values("volatility_score", ascending=False)
        if rank_key == "edge_risk":
            return df.sort_values("edge_risk", ascending=False)
        return df.sort_values("edge_score", ascending=False)

    if side == "down":
        if rank_key == "p_up":
            return df.sort_values("p_up", ascending=True)
        if rank_key == "q50_change_pct":
            return df.sort_values("q50_change_pct", ascending=True)
        if rank_key == "volatility_score":
            return df.sort_values("volatility_score", ascending=False)
        if rank_key == "edge_risk":
            return df.sort_values("edge_risk_short", ascending=False)
        return df.sort_values("edge_score_short", ascending=False)

    if rank_key == "p_up":
        return df.sort_values("signal_strength", ascending=False)
    if rank_key == "q50_change_pct":
        return df.sort_values("q50_change_pct", ascending=False, key=lambda s: s.abs())
    if rank_key == "edge_risk":
        return df.sort_values("edge_risk", ascending=False, key=lambda s: s.abs())
    if rank_key == "edge_score":
        return df.sort_values("edge_score", ascending=False, key=lambda s: s.abs())
    return df.sort_values("volatility_score", ascending=False)


def _style_signed_value(v: object) -> str:
    if not _is_finite_number(v):
        return ""
    value = float(v)
    if value > 0:
        return "color: #22c55e; font-weight: 600;"
    if value < 0:
        return "color: #ef4444; font-weight: 600;"
    return "color: #94a3b8;"


def _style_strength_label(v: object) -> str:
    text = str(v or "")
    if "强" in text:
        return "color: #22c55e; font-weight: 600;"
    if "中" in text:
        return "color: #f59e0b; font-weight: 600;"
    if "弱" in text:
        return "color: #94a3b8; font-weight: 600;"
    return ""


def _signal_strength_human_text(label: object) -> str:
    lv = str(label or "")
    if lv == "强":
        return "强信号（可执行）"
    if lv == "中":
        return "中等信号（需风控）"
    if lv == "弱":
        return "弱信号（≈随机）"
    return "-"


def _render_signal_badge(label: object) -> None:
    lv = str(label or "")
    if lv == "强":
        bg, fg = "#163a2f", "#86efac"
    elif lv == "中":
        bg, fg = "#3f3113", "#fcd34d"
    else:
        bg, fg = "#1f2937", "#cbd5e1"
    text = _signal_strength_human_text(lv)
    st.markdown(
        (
            "<span style='display:inline-block;padding:4px 10px;border-radius:999px;"
            f"background:{bg};color:{fg};font-size:12px;font-weight:700'>{text}</span>"
        ),
        unsafe_allow_html=True,
    )


def _auc_binary(y_true: pd.Series, y_score: pd.Series) -> float:
    y = pd.to_numeric(y_true, errors="coerce")
    s = pd.to_numeric(y_score, errors="coerce")
    mask = y.notna() & s.notna()
    if mask.sum() < 3:
        return float("nan")
    y = y[mask].astype(int)
    s = s[mask]
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = s.rank(method="average")
    sum_ranks_pos = float(ranks[y == 1].sum())
    return (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / float(n_pos * n_neg)


def _compute_recent_hourly_reliability(
    *,
    symbol: str,
    horizon_hours: int,
    window_days: int = 30,
    cfg: Dict[str, object] | None = None,
) -> Tuple[Dict[str, float], pd.DataFrame, str]:
    cfg_local = cfg or _load_main_config_cached("configs/config.yaml")
    model_symbol = str(((cfg_local.get("data", {}) if isinstance(cfg_local, dict) else {}).get("symbol", "BTCUSDT")))
    if str(symbol).upper() != model_symbol.upper():
        return {}, pd.DataFrame(), f"近期模型可信度面板当前仅支持主模型币种：{model_symbol}。"

    path = Path("data/processed/predictions_hourly.csv")
    if not path.exists():
        return {}, pd.DataFrame(), "缺少 predictions_hourly.csv，无法计算近期可信度。"

    df = pd.read_csv(path)
    p_col = f"dir_h{int(horizon_hours)}_p_up"
    q10_col = f"ret_h{int(horizon_hours)}_q0.1"
    q90_col = f"ret_h{int(horizon_hours)}_q0.9"
    required = {"close", p_col, q10_col, q90_col}
    if not required.issubset(df.columns):
        return {}, pd.DataFrame(), f"预测文件缺少列：{', '.join(sorted(required - set(df.columns)))}"

    ts_col = "timestamp_market" if "timestamp_market" in df.columns else "timestamp_utc"
    if ts_col not in df.columns:
        return {}, pd.DataFrame(), "预测文件缺少时间列。"

    work = df[[ts_col, "close", p_col, q10_col, q90_col]].copy()
    work[ts_col] = pd.to_datetime(work[ts_col], errors="coerce")
    work = work.dropna(subset=[ts_col]).sort_values(ts_col).reset_index(drop=True)
    work["close"] = pd.to_numeric(work["close"], errors="coerce")
    work["realized_ret"] = work["close"].shift(-int(horizon_hours)) / work["close"] - 1.0
    work["y"] = (work["realized_ret"] > 0).astype(float)
    work["p"] = pd.to_numeric(work[p_col], errors="coerce")
    work["q10"] = pd.to_numeric(work[q10_col], errors="coerce")
    work["q90"] = pd.to_numeric(work[q90_col], errors="coerce")
    work = work.dropna(subset=["realized_ret", "p", "q10", "q90"])
    if work.empty:
        return {}, pd.DataFrame(), "可用于评估的样本不足。"

    window_rows = max(24, int(window_days) * 24)
    w = work.tail(window_rows).copy()
    if w.empty:
        return {}, pd.DataFrame(), "滚动窗口样本为空。"

    pred_up = (w["p"] >= 0.5).astype(float)
    acc = float((pred_up == w["y"]).mean())
    auc = float(_auc_binary(w["y"], w["p"]))
    brier = float(((w["p"] - w["y"]) ** 2).mean())
    cov = float(((w["realized_ret"] >= w["q10"]) & (w["realized_ret"] <= w["q90"])).mean())
    width = float((w["q90"] - w["q10"]).mean())

    # Baselines on the same rolling window.
    naive_p = pd.Series(0.5, index=w.index, dtype=float)
    naive_pred = (naive_p >= 0.5).astype(float)
    prev_ret = w["close"] / w["close"].shift(1) - 1.0
    prev_p = (prev_ret > 0).astype(float).fillna(0.5)
    prev_pred = (prev_p >= 0.5).astype(float)

    rows = [
        {
            "模型": f"当前模型(h={int(horizon_hours)}h)",
            "样本范围": f"近{int(window_days)}天滚动",
            "Accuracy": acc,
            "AUC": auc,
            "Brier": brier,
        },
        {
            "模型": "Naive(0.5)",
            "样本范围": f"近{int(window_days)}天滚动",
            "Accuracy": float((naive_pred == w["y"]).mean()),
            "AUC": float("nan"),
            "Brier": float(((naive_p - w["y"]) ** 2).mean()),
        },
        {
            "模型": "Prev-bar",
            "样本范围": f"近{int(window_days)}天滚动",
            "Accuracy": float((prev_pred == w["y"]).mean()),
            "AUC": float(_auc_binary(w["y"], prev_p)),
            "Brier": float(((prev_p - w["y"]) ** 2).mean()),
        },
    ]

    wf_path = Path("data/processed/metrics_walk_forward_summary.csv")
    if wf_path.exists():
        wf = pd.read_csv(wf_path)
        req_cols = {"branch", "horizon", "task", "metric", "model", "mean"}
        if req_cols.issubset(wf.columns):
            wf = wf[
                (wf["branch"] == "hourly")
                & (pd.to_numeric(wf["horizon"], errors="coerce") == int(horizon_hours))
                & (wf["task"] == "direction")
                & (wf["metric"].isin(["accuracy", "roc_auc"]))
            ].copy()
        else:
            wf = pd.DataFrame()
        if not wf.empty:
            for model_name in ["baseline_logistic", "mvp_lightgbm", "baseline_prev_bar"]:
                sub = wf[wf["model"] == model_name]
                if sub.empty:
                    continue
                acc_m = pd.to_numeric(sub.loc[sub["metric"] == "accuracy", "mean"], errors="coerce")
                auc_m = pd.to_numeric(sub.loc[sub["metric"] == "roc_auc", "mean"], errors="coerce")
                rows.append(
                    {
                        "模型": model_name,
                        "样本范围": "walk_forward_summary(全样本)",
                        "Accuracy": float(acc_m.iloc[0]) if not acc_m.empty else float("nan"),
                        "AUC": float(auc_m.iloc[0]) if not auc_m.empty else float("nan"),
                        "Brier": float("nan"),
                    }
                )

    summary = {
        "accuracy": acc,
        "auc": auc,
        "brier": brier,
        "coverage": cov,
        "width": width,
        "samples": float(len(w)),
    }
    compare_df = pd.DataFrame(rows)
    return summary, compare_df, ""


@st.cache_data(ttl=300, show_spinner=False)
def _compute_recent_symbol_reliability_cached(
    *,
    market: str,
    symbol: str,
    provider: str,
    fallback_symbol: str = "",
    horizon_steps: int = 1,
    window_days: int = 30,
    config_path: str = "configs/config.yaml",
) -> Tuple[Dict[str, float], pd.DataFrame, str]:
    from src.evaluation.backtest_multi_market import _build_model_like_signals, _fetch_daily_bars

    cfg = load_config(config_path)
    bt_cfg = cfg.get("backtest_multi_market", {})
    lookback_days_cfg = bt_cfg.get(
        "lookback_days",
        {"crypto": 540, "cn_equity": 1200, "us_equity": 1200},
    )
    mk = str(market)
    sym = str(symbol)
    prov = str(provider or ("binance" if mk == "crypto" else "yahoo"))
    lb_days = int(lookback_days_cfg.get(mk, 720))

    try:
        bars, _ = _fetch_daily_bars(
            market=mk,
            symbol=sym,
            provider=prov,
            lookback_days=lb_days,
            cfg=cfg,
        )
    except Exception:
        if mk == "crypto":
            fs = str(fallback_symbol or "").strip().upper()
            if not fs and prov == "coingecko":
                guess = str(sym).strip().upper()
                if guess.isalpha() and len(guess) <= 8:
                    fs = f"{guess}USDT"
            if fs.endswith("USDT"):
                bars, _ = _fetch_daily_bars(
                    market=mk,
                    symbol=fs,
                    provider="binance",
                    lookback_days=lb_days,
                    cfg=cfg,
                )
                sym = fs
            else:
                return {}, pd.DataFrame(), "无法获取该标的历史数据，无法计算近期可信度。"
        else:
            return {}, pd.DataFrame(), "无法获取该标的历史数据，无法计算近期可信度。"

    bars = bars.dropna(subset=["timestamp_utc", "open", "close"]).sort_values("timestamp_utc").reset_index(drop=True)
    if bars.empty:
        return {}, pd.DataFrame(), "历史数据为空。"

    modeled = _build_model_like_signals(bars, cfg=cfg, market=mk)
    if modeled.empty:
        return {}, pd.DataFrame(), "信号建模结果为空。"

    h = max(1, int(horizon_steps))
    work = modeled.copy()
    work["close"] = pd.to_numeric(work["close"], errors="coerce")
    work["realized_ret_h"] = work["close"].shift(-h) / work["close"] - 1.0
    p_raw = pd.to_numeric(work.get("p_up"), errors="coerce")
    decay = float(np.exp(-0.12 * max(0, h - 1)))
    work["p_up_h"] = 0.5 + (p_raw - 0.5) * decay
    q10 = pd.to_numeric(work.get("q10_change_pct"), errors="coerce")
    q50 = pd.to_numeric(work.get("q50_change_pct"), errors="coerce")
    q90 = pd.to_numeric(work.get("q90_change_pct"), errors="coerce")
    work["q10_h"] = np.power(1.0 + q10, h) - 1.0
    work["q50_h"] = np.power(1.0 + q50, h) - 1.0
    work["q90_h"] = np.power(1.0 + q90, h) - 1.0
    work["y"] = (work["realized_ret_h"] > 0).astype(float)
    work = work.dropna(subset=["realized_ret_h", "p_up_h", "q10_h", "q90_h", "y"])
    if work.empty:
        return {}, pd.DataFrame(), "有效评估样本不足。"

    window_rows = max(20, int(window_days))
    w = work.tail(window_rows).copy()
    if w.empty:
        return {}, pd.DataFrame(), "滚动窗口为空。"

    pred_up = (w["p_up_h"] >= 0.5).astype(float)
    acc = float((pred_up == w["y"]).mean())
    auc = float(_auc_binary(w["y"], w["p_up_h"]))
    brier = float(((w["p_up_h"] - w["y"]) ** 2).mean())
    coverage = float(((w["realized_ret_h"] >= w["q10_h"]) & (w["realized_ret_h"] <= w["q90_h"])).mean())
    width = float((w["q90_h"] - w["q10_h"]).mean())

    naive_p = pd.Series(0.5, index=w.index, dtype=float)
    naive_pred = (naive_p >= 0.5).astype(float)
    prev_ret_h = w["close"] / w["close"].shift(h) - 1.0
    prev_p = (prev_ret_h > 0).astype(float).fillna(0.5)
    prev_pred = (prev_p >= 0.5).astype(float)
    rows = [
        {
            "模型": f"当前模型(h={h}d)",
            "样本范围": f"近{int(window_days)}天滚动",
            "Accuracy": acc,
            "AUC": auc,
            "Brier": brier,
            "Coverage": coverage,
        },
        {
            "模型": "Naive(0.5)",
            "样本范围": f"近{int(window_days)}天滚动",
            "Accuracy": float((naive_pred == w["y"]).mean()),
            "AUC": float("nan"),
            "Brier": float(((naive_p - w["y"]) ** 2).mean()),
            "Coverage": float("nan"),
        },
        {
            "模型": "Prev-h bar",
            "样本范围": f"近{int(window_days)}天滚动",
            "Accuracy": float((prev_pred == w["y"]).mean()),
            "AUC": float(_auc_binary(w["y"], prev_p)),
            "Brier": float(((prev_p - w["y"]) ** 2).mean()),
            "Coverage": float("nan"),
        },
    ]

    summary = {
        "accuracy": acc,
        "auc": auc,
        "brier": brier,
        "coverage": coverage,
        "width": width,
        "samples": float(len(w)),
    }
    return summary, pd.DataFrame(rows), ""


def _model_health_grade(summary: Dict[str, float] | None) -> str:
    if not summary:
        return "中"
    brier = _safe_float(summary.get("brier"))
    coverage = _safe_float(summary.get("coverage"))
    if not (np.isfinite(brier) and np.isfinite(coverage)):
        return "中"
    if brier <= 0.20 and 0.72 <= coverage <= 0.88:
        return "良"
    if brier <= 0.28 and 0.60 <= coverage <= 0.95:
        return "中"
    return "差"


def _parse_horizon_label(label: str) -> Tuple[str, int]:
    text = str(label or "").strip().lower()
    if text.endswith("h"):
        try:
            return "hour", max(1, int(text[:-1]))
        except Exception:
            return "hour", 1
    if text.endswith("d"):
        try:
            return "day", max(1, int(text[:-1]))
        except Exception:
            return "day", 1
    return "day", 1


def _default_lookback_days(market: str) -> int:
    if market == "crypto":
        return 365
    return 730


def _build_snapshot_fresh(inst: list[dict[str, object]]) -> pd.DataFrame:
    snapshot_mod = importlib.import_module("src.markets.snapshot")
    snapshot_mod = importlib.reload(snapshot_mod)
    return snapshot_mod.build_market_snapshot_from_instruments(inst, config_path="configs/config.yaml")


def _ensure_snapshot_factors(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    row = df.iloc[0]
    factor_cols = [
        "size_factor",
        "value_factor",
        "growth_factor",
        "momentum_factor",
        "reversal_factor",
        "low_vol_factor",
    ]
    has_any_factor = any(_is_finite_number(row.get(c)) for c in factor_cols)
    if has_any_factor:
        return df

    horizon_unit, horizon_steps = _parse_horizon_label(str(row.get("horizon_label", "1d")))
    fallback = [
        {
            "id": str(row.get("instrument_id", "unknown")),
            "name": str(row.get("name", "unknown")),
            "market": str(row.get("market", "unknown")),
            "symbol": str(row.get("symbol", "")),
            "provider": str(row.get("provider", "")),
            "timezone": str(row.get("timezone", "UTC")),
            "horizon_unit": horizon_unit,
            "horizon_steps": horizon_steps,
            "history_lookback_days": _default_lookback_days(str(row.get("market", ""))),
        }
    ]
    try:
        refreshed = _build_snapshot_fresh(fallback)
        if not refreshed.empty:
            return refreshed
    except Exception:
        pass
    return df


def _infer_horizons(columns: List[str]) -> List[int]:
    horizons = set()
    for c in columns:
        if c.startswith("dir_h") and c.endswith("_p_up"):
            mid = c.replace("dir_h", "").replace("_p_up", "")
            try:
                horizons.add(int(mid))
            except Exception:
                pass
    return sorted(horizons)


def _format_price(x: float | int | None) -> str:
    try:
        value = float(x)
        if not np.isfinite(value):
            return "-"
        return f"${value:,.2f}"
    except Exception:
        return "-"


def _format_change_pct(x: float | int | None) -> str:
    try:
        value = float(x)
        if not np.isfinite(value):
            return "-"
        return f"{value:+.2%}"
    except Exception:
        return "-"


def _format_float(x: float | int | None, digits: int = 4) -> str:
    try:
        value = float(x)
        if not np.isfinite(value):
            return "-"
        return f"{value:.{digits}f}"
    except Exception:
        return "-"


def _render_big_value(label: str, value: str, *, caption: str = "") -> None:
    st.markdown(f"**{label}**")
    st.markdown(
        (
            "<div style='font-size:3rem;font-weight:700;line-height:1.15;"
            "word-break:break-word;overflow-wrap:anywhere;'>"
            f"{value}</div>"
        ),
        unsafe_allow_html=True,
    )
    if caption:
        st.caption(caption)


def _trend_cn(label: str) -> str:
    mapping = {
        "bullish": "偏多",
        "bearish": "偏空",
        "sideways": "震荡",
    }
    return mapping.get(str(label), str(label))


def _risk_cn(label: str) -> str:
    mapping = {
        "low": "低",
        "medium": "中",
        "high": "高",
        "extreme": "极高",
    }
    return mapping.get(str(label), str(label))


def _policy_action_cn(label: str) -> str:
    mapping = {
        "Long": "做多",
        "Short": "做空",
        "Flat": "观望",
    }
    return mapping.get(str(label), str(label))


def _session_display_name(session_name: str) -> str:
    mapping = {"asia": "亚盘", "europe": "欧盘", "us": "美盘"}
    return mapping.get(str(session_name), str(session_name))


def _render_hourly_heatmap(
    hourly_df: pd.DataFrame, value_col: str, title: str, *, horizon_hours: int
) -> None:
    if hourly_df.empty or value_col not in hourly_df.columns:
        st.info("暂无热力图数据。")
        return
    work = hourly_df.copy()
    work = work.sort_values("hour_bj")
    x = [f"{int(h):02d}:00" for h in work["hour_bj"].tolist()]
    z_values = pd.to_numeric(work[value_col], errors="coerce").fillna(0.0).tolist()
    fig = go.Figure(
        data=go.Heatmap(
            z=[z_values],
            x=x,
            y=[title],
            colorscale="Blues",
            colorbar=dict(title=title),
        )
    )
    fig.update_layout(
        title=f"24小时热力图：从该小时开始的未来{int(horizon_hours)}h {title}（北京时间）",
        xaxis_title="小时",
        yaxis_title="指标",
        template="plotly_white",
        height=260,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_top_tables(
    hourly_df: pd.DataFrame,
    daily_df: pd.DataFrame,
    top_n: int,
    *,
    rank_key: str,
    cost_bps: float,
    weak_pp: float,
    strong_pp: float,
    horizon_hours: int,
) -> None:
    rank_name = _rank_metric_options().get(rank_key, rank_key)
    st.caption(
        f"榜单统一排序：{rank_name} | 成本估计：{float(cost_bps):.1f} bps | "
        f"小时级语义：从该小时起算未来 {int(horizon_hours)}h。"
    )

    if hourly_df.empty:
        st.info("暂无小时级榜单数据。")
    else:
        h = hourly_df.copy()
        h = _append_signal_strength_columns(h, weak_pp=weak_pp, strong_pp=strong_pp)
        h = _append_edge_columns(h, cost_bps=cost_bps)
        h["趋势"] = h["trend_label"].map(_trend_cn)
        h["风险"] = h["risk_level"].map(_risk_cn)
        h["上涨概率"] = h["p_up"].map(lambda x: _format_change_pct(x).replace("+", ""))
        h["下跌概率"] = h["p_down"].map(lambda x: _format_change_pct(x).replace("+", ""))
        h["预期涨跌幅"] = h["q50_change_pct"].map(_format_change_pct)
        h["目标价格(q50)"] = h["target_price_q50"].map(_format_price)
        h["置信度"] = h["confidence_score"].map(lambda x: _format_float(x, 1))
        h["信号强弱"] = h.apply(
            lambda r: _format_signal_strength(
                r.get("signal_strength_label", "-"),
                r.get("signal_strength_pp"),
                r.get("signal_strength_score"),
            ),
            axis=1,
        )
        h["机会值(edge)"] = h["edge_score"].map(_format_change_pct)
        h["风险调整机会"] = h["edge_risk"].map(lambda x: _format_float(x, 3))
        h["空头机会(edge)"] = h["edge_score_short"].map(_format_change_pct)
        h["空头风险调整机会"] = h["edge_risk_short"].map(lambda x: _format_float(x, 3))
        if "policy_action" in h.columns:
            h["策略动作"] = h["policy_action"].map(_policy_action_cn)
            h["建议仓位"] = h["policy_position_size"].map(
                lambda x: f"{float(x):.1%}" if _is_finite_number(x) else "-"
            )
            h["预期净优势"] = h["policy_expected_edge_pct"].map(_format_change_pct)

        up_rank = _sort_by_rank(h, rank_key=rank_key, side="up")
        down_rank = _sort_by_rank(h, rank_key=rank_key, side="down")
        vol_rank = _sort_by_rank(h, rank_key=rank_key, side="vol")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"**小时级：最可能上涨 Top N（{rank_name}）**")
            cols = [
                "hour_label",
                "上涨概率",
                "预期涨跌幅",
                "目标价格(q50)",
                "信号强弱",
                "机会值(edge)",
                "风险调整机会",
                "策略动作",
                "建议仓位",
                "预期净优势",
                "趋势",
                "风险",
                "置信度",
            ]
            cols = [c for c in cols if c in h.columns]
            st.dataframe(up_rank[cols].head(top_n), use_container_width=True, hide_index=True)
        with c2:
            st.markdown(f"**小时级：最可能下跌 Top N（{rank_name}）**")
            cols = [
                "hour_label",
                "下跌概率",
                "预期涨跌幅",
                "目标价格(q50)",
                "信号强弱",
                "空头机会(edge)",
                "空头风险调整机会",
                "策略动作",
                "建议仓位",
                "预期净优势",
                "趋势",
                "风险",
                "置信度",
            ]
            cols = [c for c in cols if c in h.columns]
            st.dataframe(down_rank[cols].head(top_n), use_container_width=True, hide_index=True)
        with c3:
            st.markdown(f"**小时级：最可能大波动 Top N（{rank_name}）**")
            cols = [
                "hour_label",
                "预期涨跌幅",
                "目标价格(q50)",
                "信号强弱",
                "机会值(edge)",
                "风险调整机会",
                "策略动作",
                "建议仓位",
                "预期净优势",
                "趋势",
                "风险",
                "置信度",
            ]
            cols = [c for c in cols if c in h.columns]
            st.dataframe(vol_rank[cols].head(top_n), use_container_width=True, hide_index=True)

    st.markdown("---")
    if daily_df.empty:
        st.info("暂无日线级榜单数据。")
    else:
        d = daily_df.copy()
        d = _append_signal_strength_columns(d, weak_pp=weak_pp, strong_pp=strong_pp)
        d = _append_edge_columns(d, cost_bps=cost_bps)
        d["趋势"] = d["trend_label"].map(_trend_cn)
        d["风险"] = d["risk_level"].map(_risk_cn)
        d["上涨概率"] = d["p_up"].map(lambda x: _format_change_pct(x).replace("+", ""))
        d["下跌概率"] = d["p_down"].map(lambda x: _format_change_pct(x).replace("+", ""))
        d["预期涨跌幅"] = d["q50_change_pct"].map(_format_change_pct)
        d["目标价格(q50)"] = d["target_price_q50"].map(_format_price)
        d["置信度"] = d["confidence_score"].map(lambda x: _format_float(x, 1))
        d["信号强弱"] = d.apply(
            lambda r: _format_signal_strength(
                r.get("signal_strength_label", "-"),
                r.get("signal_strength_pp"),
                r.get("signal_strength_score"),
            ),
            axis=1,
        )
        d["机会值(edge)"] = d["edge_score"].map(_format_change_pct)
        d["风险调整机会"] = d["edge_risk"].map(lambda x: _format_float(x, 3))
        d["空头机会(edge)"] = d["edge_score_short"].map(_format_change_pct)
        d["空头风险调整机会"] = d["edge_risk_short"].map(lambda x: _format_float(x, 3))
        if "policy_action" in d.columns:
            d["策略动作"] = d["policy_action"].map(_policy_action_cn)
            d["建议仓位"] = d["policy_position_size"].map(
                lambda x: f"{float(x):.1%}" if _is_finite_number(x) else "-"
            )
            d["预期净优势"] = d["policy_expected_edge_pct"].map(_format_change_pct)

        up_rank = _sort_by_rank(d, rank_key=rank_key, side="up")
        down_rank = _sort_by_rank(d, rank_key=rank_key, side="down")
        vol_rank = _sort_by_rank(d, rank_key=rank_key, side="vol")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"**日线级：最可能上涨 Top N（{rank_name}）**")
            cols = [
                "date_bj",
                "上涨概率",
                "预期涨跌幅",
                "目标价格(q50)",
                "信号强弱",
                "机会值(edge)",
                "风险调整机会",
                "策略动作",
                "建议仓位",
                "预期净优势",
                "趋势",
                "风险",
                "置信度",
            ]
            cols = [c for c in cols if c in d.columns]
            st.dataframe(up_rank[cols].head(top_n), use_container_width=True, hide_index=True)
        with c2:
            st.markdown(f"**日线级：最可能下跌 Top N（{rank_name}）**")
            cols = [
                "date_bj",
                "下跌概率",
                "预期涨跌幅",
                "目标价格(q50)",
                "信号强弱",
                "空头机会(edge)",
                "空头风险调整机会",
                "策略动作",
                "建议仓位",
                "预期净优势",
                "趋势",
                "风险",
                "置信度",
            ]
            cols = [c for c in cols if c in d.columns]
            st.dataframe(down_rank[cols].head(top_n), use_container_width=True, hide_index=True)
        with c3:
            st.markdown(f"**日线级：最可能大波动 Top N（{rank_name}）**")
            cols = [
                "date_bj",
                "预期涨跌幅",
                "目标价格(q50)",
                "信号强弱",
                "机会值(edge)",
                "风险调整机会",
                "策略动作",
                "建议仓位",
                "预期净优势",
                "趋势",
                "风险",
                "置信度",
            ]
            cols = [c for c in cols if c in d.columns]
            st.dataframe(vol_rank[cols].head(top_n), use_container_width=True, hide_index=True)


def _render_crypto_session_page() -> None:
    st.header("交易时间段预测（Crypto）")
    st.caption("北京时间24小时制；支持亚盘/欧盘/美盘、关键小时概率与未来N天日线预测。")

    if "session_refresh_token" not in st.session_state:
        st.session_state["session_refresh_token"] = 0

    cfg = {}
    try:
        import yaml

        cfg = yaml.safe_load(Path("configs/config.yaml").read_text(encoding="utf-8")) or {}
    except Exception:
        cfg = {}
    fc = cfg.get("forecast_config", {})
    source_cfg = fc.get("data_source", {})
    strength_cfg = fc.get("signal_strength", {})
    rank_cfg = fc.get("ranking", {})

    symbols = fc.get("symbols", {}).get("default", ["BTCUSDT", "ETHUSDT", "SOLUSDT"])
    exchanges = source_cfg.get("exchanges", ["binance", "bybit"])
    market_types = source_cfg.get("market_types", ["perp", "spot"])
    default_exchange = source_cfg.get("default_exchange", "binance")
    default_market_type = source_cfg.get("default_market_type", "perp")
    default_horizon = int(fc.get("hourly", {}).get("horizon_hours", 4))
    default_days = int(fc.get("daily", {}).get("lookforward_days", 14))
    weak_pp = float(strength_cfg.get("weak_threshold_pp", 2.0))
    strong_pp = float(strength_cfg.get("strong_threshold_pp", 5.0))
    default_cost_bps = float(rank_cfg.get("cost_bps", 8.0))

    f1, f2, f3, f4, f5 = st.columns([2, 1, 1, 1, 1])
    symbol = f1.selectbox("币种", options=symbols, index=0, key="session_symbol")
    exchange = f2.selectbox(
        "数据源",
        options=exchanges,
        index=exchanges.index(default_exchange) if default_exchange in exchanges else 0,
        key="session_exchange",
    )
    market_type = f3.selectbox(
        "市场类型",
        options=market_types,
        index=market_types.index(default_market_type) if default_market_type in market_types else 0,
        key="session_market_type",
    )
    mode = f4.selectbox("模式", options=["forecast", "seasonality"], index=0, key="session_mode")
    horizon_hours = int(
        f5.selectbox("小时周期", options=[4], index=0 if default_horizon == 4 else 0, key="session_horizon")
    )

    c1, c2 = st.columns([1, 3])
    lookforward_days = int(c1.slider("未来N天（日线）", 7, 30, default_days, 1, key="session_daily_n"))
    compare_view = bool(c2.checkbox("对照视图：Forecast vs Seasonality", value=False, key="session_compare_view"))
    if c2.button("刷新并重算", key="session_refresh_btn"):
        st.session_state["session_refresh_token"] += 1

    try:
        hourly_df, blocks_df, daily_df, meta = _build_session_bundle_cached(
            symbol=symbol,
            exchange=exchange,
            market_type=market_type,
            mode=mode,
            horizon_hours=horizon_hours,
            lookforward_days=lookforward_days,
            refresh_token=int(st.session_state["session_refresh_token"]),
        )
    except Exception as exc:
        st.error(f"时段预测计算失败：{exc}")
        return

    mode_actual = str(meta.get("mode_actual", mode))
    if mode_actual != mode:
        st.warning(f"请求模式是 `{mode}`，当前自动降级为 `{mode_actual}`。")
    st.caption(
        f"最新价格：{_format_price(meta.get('current_price'))} | "
        f"更新时间（北京时间）：{meta.get('data_updated_at_bj', '-')} | "
        f"模式/周期：{mode_actual} / {int(horizon_hours)}h"
    )

    data_version_seed = (
        f"{meta.get('symbol', '-')}"
        f"|{meta.get('exchange_actual', '-')}"
        f"|{meta.get('market_type', '-')}"
        f"|{meta.get('data_updated_at_bj', '-')}"
        f"|{meta.get('data_source_actual', '-')}"
    )
    data_version = hashlib.sha1(data_version_seed.encode("utf-8")).hexdigest()[:12]
    with st.expander("数据 & 模型信息", expanded=False):
        st.markdown(
            f"- 数据源：{meta.get('exchange_actual', '-')} / {meta.get('market_type', '-')} / {meta.get('symbol', '-')}\n"
            f"- 请求数据源：{meta.get('exchange', '-')}\n"
            f"- 数据更新时间（北京时间）：{meta.get('data_updated_at_bj', '-')}\n"
            f"- 预测生成时间（北京时间）：{meta.get('forecast_generated_at_bj', '-')}\n"
            f"- horizon={int(horizon_hours)}h / mode={mode_actual}\n"
            f"- model_version：{meta.get('model_version', '-')}\n"
            f"- data_version：{data_version}\n"
            f"- git_hash：{_get_git_hash_short_cached()}"
        )
        st.caption(f"Data Source Detail: {meta.get('data_source_actual', '-')}")

    # Session cards
    if blocks_df.empty:
        st.info("暂无时段汇总数据。")
    else:
        cards = _append_signal_strength_columns(blocks_df.copy(), weak_pp=weak_pp, strong_pp=strong_pp)
        cards["session_name_cn"] = cards.get("session_name_cn", cards["session_name"].map(_session_display_name))
        cards = cards.sort_values(
            "session_name", key=lambda s: s.map({"asia": 0, "europe": 1, "us": 2}).fillna(9)
        )
        cols = st.columns(3)
        for idx, (_, row) in enumerate(cards.iterrows()):
            col = cols[idx % 3]
            with col:
                st.markdown(f"**{row.get('session_name_cn', '-') }（{row.get('session_hours', '-') }）**")
                st.metric("上涨概率", _format_change_pct(row.get("p_up")).replace("+", ""))
                st.metric("下跌概率", _format_change_pct(row.get("p_down")).replace("+", ""))
                st.metric("预期涨跌幅(q50)", _format_change_pct(row.get("q50_change_pct")))
                st.metric("目标价格(q50)", _format_price(row.get("target_price_q50")))
                _render_signal_badge(row.get("signal_strength_label", "-"))
                st.caption(
                    "信号强度："
                    + _format_signal_strength(
                        row.get("signal_strength_label", "-"),
                        row.get("signal_strength_pp"),
                        row.get("signal_strength_score"),
                    )
                )
                st.caption(
                    f"趋势：{_trend_cn(str(row.get('trend_label', '-')))} | "
                    f"风险：{_risk_cn(str(row.get('risk_level', '-')))} | "
                    f"置信度：{_format_float(row.get('confidence_score'), 1)}"
                )
                if "policy_action" in row.index:
                    st.caption(
                        f"策略：{_policy_action_cn(str(row.get('policy_action', 'Flat')))} | "
                        f"仓位：{(float(row.get('policy_position_size')) if _is_finite_number(row.get('policy_position_size')) else 0.0):.1%} | "
                        f"净优势：{_format_change_pct(row.get('policy_expected_edge_pct'))}"
                    )

    st.caption(
        f"信号解释：`弱信号` (<{weak_pp:.1f}pp) 接近抛硬币，谨慎解读；"
        f"`中信号` ({weak_pp:.1f}-{strong_pp:.1f}pp)；`强信号` (>{strong_pp:.1f}pp)。"
    )

    if compare_view:
        compare_mode = "seasonality" if mode == "forecast" else "forecast"
        try:
            cmp_hourly, cmp_blocks, _, cmp_meta = _build_session_bundle_cached(
                symbol=symbol,
                exchange=exchange,
                market_type=market_type,
                mode=compare_mode,
                horizon_hours=horizon_hours,
                lookforward_days=lookforward_days,
                refresh_token=int(st.session_state["session_refresh_token"]),
            )
            st.markdown("### Forecast vs Seasonality 对照（同参数）")
            st.caption(
                f"当前模式：{mode_actual} | 对照模式：{cmp_meta.get('mode_actual', compare_mode)} | "
                "重点看 Δp_up（Forecast - Seasonality）。"
            )
            if not blocks_df.empty and not cmp_blocks.empty:
                left = blocks_df[["session_name", "session_name_cn", "p_up", "q50_change_pct"]].copy()
                right = cmp_blocks[["session_name", "p_up", "q50_change_pct"]].copy()
                merged = left.merge(right, on="session_name", how="inner", suffixes=("_main", "_cmp"))
                merged["Δp_up"] = merged["p_up_main"] - merged["p_up_cmp"]
                merged["Δq50"] = merged["q50_change_pct_main"] - merged["q50_change_pct_cmp"]
                show = merged.rename(
                    columns={
                        "session_name_cn": "时段",
                        "p_up_main": f"{mode_actual} p_up",
                        "p_up_cmp": f"{cmp_meta.get('mode_actual', compare_mode)} p_up",
                        "q50_change_pct_main": f"{mode_actual} q50",
                        "q50_change_pct_cmp": f"{cmp_meta.get('mode_actual', compare_mode)} q50",
                    }
                )
                fmt = {
                    c: "{:.2%}"
                    for c in [
                        f"{mode_actual} p_up",
                        f"{cmp_meta.get('mode_actual', compare_mode)} p_up",
                        f"{mode_actual} q50",
                        f"{cmp_meta.get('mode_actual', compare_mode)} q50",
                        "Δp_up",
                        "Δq50",
                    ]
                    if c in show.columns
                }
                styled_cmp = show.style.format(fmt, na_rep="-")
                delta_cols = [c for c in ["Δp_up", "Δq50"] if c in show.columns]
                if delta_cols:
                    styled_cmp = styled_cmp.applymap(_style_signed_value, subset=delta_cols)
                st.dataframe(styled_cmp, use_container_width=True, hide_index=True)
                st.caption("若 Δp_up 绝对值较大，说明模型观点与历史季节性节奏有明显偏离。")
            else:
                st.info("对照视图数据不足。")
        except Exception as exc:
            st.warning(f"对照视图构建失败：{exc}")

    st.markdown("---")
    tab1, tab2, tab3, tab4 = st.tabs(["上涨概率", "下跌概率", "波动强度", "置信度"])
    with tab1:
        _render_hourly_heatmap(hourly_df, "p_up", "上涨概率", horizon_hours=horizon_hours)
    with tab2:
        _render_hourly_heatmap(hourly_df, "p_down", "下跌概率", horizon_hours=horizon_hours)
    with tab3:
        _render_hourly_heatmap(hourly_df, "volatility_score", "波动强度", horizon_hours=horizon_hours)
    with tab4:
        _render_hourly_heatmap(hourly_df, "confidence_score", "置信度", horizon_hours=horizon_hours)

    st.markdown("---")
    st.subheader("未来N天日线预测")
    oneway_state = float(st.session_state.get("session_cost_bps_oneway", max(0.0, default_cost_bps / 2.0)))
    cost_side_state = str(st.session_state.get("session_cost_side", "双边(开+平)"))
    cost_bps_state = oneway_state * (2.0 if cost_side_state.startswith("双边") else 1.0)
    if daily_df.empty:
        st.info("暂无日线预测数据。")
    else:
        d = _append_signal_strength_columns(daily_df.copy(), weak_pp=weak_pp, strong_pp=strong_pp)
        d = _append_edge_columns(d, cost_bps=cost_bps_state)
        d["趋势"] = d["trend_label"].map(_trend_cn)
        d["风险"] = d["risk_level"].map(_risk_cn)
        d["信号强弱"] = d["signal_strength_label"]
        d["强度分(0-100)"] = d["signal_strength_score"]
        d["机会值(edge)"] = d["edge_score"]
        d["风险调整机会"] = d["edge_risk"]
        d["波动强度"] = pd.to_numeric(d.get("volatility_score"), errors="coerce")
        if "policy_action" in d.columns:
            d["策略动作"] = d["policy_action"].map(_policy_action_cn)
            d["建议仓位"] = d["policy_position_size"]
            d["预期净优势"] = pd.to_numeric(d["policy_expected_edge_pct"], errors="coerce")

        show_cols = [
            "date_bj",
            "day_of_week",
            "p_up",
            "p_down",
            "q50_change_pct",
            "target_price_q10",
            "target_price_q50",
            "target_price_q90",
            "start_window_top1",
            "波动强度",
            "信号强弱",
            "强度分(0-100)",
            "机会值(edge)",
            "风险调整机会",
            "策略动作",
            "建议仓位",
            "预期净优势",
            "趋势",
            "风险",
            "confidence_score",
        ]
        show_cols = [c for c in show_cols if c in d.columns]
        d_view = d[show_cols].rename(
            columns={
                "p_up": "上涨概率",
                "p_down": "下跌概率",
                "q50_change_pct": "预期涨跌幅(q50)",
                "target_price_q10": "目标价格(q10)",
                "target_price_q50": "目标价格(q50)",
                "target_price_q90": "目标价格(q90)",
                "start_window_top1": "start_window_top1",
                "confidence_score": "置信度",
                "建议仓位": "建议仓位",
                "预期净优势": "预期净优势",
            }
        )
        st.caption("关键列已置前（date_bj/day_of_week）并将趋势/风险/置信度放在表尾，便于快速扫读。")
        format_map_all = {
            "上涨概率": "{:.2%}",
            "下跌概率": "{:.2%}",
            "预期涨跌幅(q50)": "{:+.2%}",
            "目标价格(q10)": "${:,.2f}",
            "目标价格(q50)": "${:,.2f}",
            "目标价格(q90)": "${:,.2f}",
            "波动强度": "{:.2%}",
            "强度分(0-100)": "{:.0f}",
            "机会值(edge)": "{:+.2%}",
            "风险调整机会": "{:+.3f}",
            "建议仓位": "{:.1%}",
            "预期净优势": "{:+.2%}",
            "置信度": "{:.1f}",
        }
        format_map = {k: v for k, v in format_map_all.items() if k in d_view.columns}
        styled = d_view.style.format(format_map, na_rep="-")
        signed_cols = [c for c in ["预期涨跌幅(q50)", "机会值(edge)", "风险调整机会"] if c in d_view.columns]
        if signed_cols:
            styled = styled.applymap(_style_signed_value, subset=signed_cols)
        if "信号强弱" in d_view.columns:
            styled = styled.applymap(_style_strength_label, subset=["信号强弱"])
        if "波动强度" in d_view.columns:
            # pandas Styler.background_gradient depends on matplotlib.
            # Degrade gracefully when matplotlib is not installed.
            try:
                import matplotlib  # noqa: F401

                styled = styled.background_gradient(cmap="YlOrRd", subset=["波动强度"])
            except Exception:
                pass
        st.dataframe(styled, use_container_width=True, hide_index=True)

    st.markdown("---")
    t1, t2, t3 = st.columns([1, 1, 1.2])
    top_n = int(t1.slider("榜单显示 Top N", 3, 12, 5, 1, key="session_topn"))
    rank_key = t2.selectbox(
        "榜单排序标准",
        options=list(_rank_metric_options().keys()),
        index=list(_rank_metric_options().keys()).index("edge_score"),
        format_func=lambda k: _rank_metric_options().get(k, k),
        key="session_rank_key",
    )
    cost_mode = t3.selectbox(
        "成本口径",
        options=["双边(开+平)", "单边"],
        index=0,
        key="session_cost_side",
        help="默认使用双边成本，更保守。",
    )
    c31, c32 = st.columns([1, 1])
    one_way_cost_bps = float(
        c31.number_input(
            "单边成本（bps）",
            min_value=0.0,
            max_value=100.0,
            value=max(0.0, cost_bps_state / 2.0),
            step=1.0,
            key="session_cost_bps_oneway",
        )
    )
    cost_bps = one_way_cost_bps * (2.0 if cost_mode.startswith("双边") else 1.0)
    c32.metric("当前用于计算的成本", f"{cost_bps:.1f} bps")
    with st.expander("ⓘ edge 公式说明", expanded=False):
        st.markdown(
            "- `edge_score = q50_change_pct - cost_bps/10000`\n"
            "- `edge_risk = edge_score / (q90_change_pct - q10_change_pct)`\n"
            "- 当前 `cost_bps` 口径："
            + ("双边（开仓+平仓）" if cost_mode.startswith("双边") else "单边")
            + "。"
        )

    _render_top_tables(
        hourly_df=hourly_df,
        daily_df=daily_df,
        top_n=top_n,
        rank_key=rank_key,
        cost_bps=cost_bps,
        weak_pp=weak_pp,
        strong_pp=strong_pp,
        horizon_hours=horizon_hours,
    )

    with st.expander("近期模型可信度（滚动30天）", expanded=False):
        summary_30d, baseline_df, err_msg = _compute_recent_hourly_reliability(
            symbol=str(symbol),
            horizon_hours=int(horizon_hours),
            window_days=30,
            cfg=cfg,
        )
        if err_msg:
            st.info(err_msg)
        else:
            m1, m2, m3 = st.columns(3)
            m1.metric("Direction Accuracy", f"{summary_30d.get('accuracy', float('nan')):.2%}")
            m2.metric("ROC-AUC", f"{summary_30d.get('auc', float('nan')):.3f}")
            m3.metric("Brier", f"{summary_30d.get('brier', float('nan')):.4f}")
            m4, m5, m6 = st.columns(3)
            m4.metric("Coverage(80%目标)", f"{summary_30d.get('coverage', float('nan')):.2%}")
            m5.metric("平均区间宽度(q90-q10)", f"{summary_30d.get('width', float('nan')):.2%}")
            m6.metric("样本数", f"{int(summary_30d.get('samples', 0))}")
            st.caption("说明：Coverage 目标通常接近 80%；Brier 越低越好。")
            if not baseline_df.empty:
                st.markdown("**Baseline 对比（含 Naive / Prev-bar / 已有模型汇总）**")
                st.dataframe(
                    baseline_df.style.format(
                        {
                            "Accuracy": "{:.2%}",
                            "AUC": "{:.3f}",
                            "Brier": "{:.4f}",
                        },
                        na_rep="-",
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

    with st.expander("如何解读这个页面？", expanded=False):
        st.markdown(
            f"- 小时级语义：`从该小时开始的未来{int(horizon_hours)}h`，不是“该小时内必涨/必跌”。\n"
            "- 看 `P(up)/P(down)` 判断方向概率。\n"
            "- `信号强弱 = |P(up)-0.5|`：弱信号接近 coin flip，不要过度解读。\n"
            "- 看 `预期涨跌幅(q50)` 判断幅度。\n"
            "- 看 `波动强度` 和 `风险等级` 判断风险。\n"
            "- 看 `edge_score / edge_risk` 判断机会值与风险调整后性价比。\n"
            "- 看 `策略动作/建议仓位/预期净优势` 判断是否值得参与。\n"
            "- `Forecast` 与 `Seasonality` 是两种口径，分歧本身也是信息。"
        )


def _render_projection_chart(
    *,
    current_price: float,
    q10_change_pct: float,
    q50_change_pct: float,
    q90_change_pct: float,
    expected_date_label: str,
    title: str,
    entry_price: float | None = None,
    stop_loss_price: float | None = None,
    take_profit_price: float | None = None,
) -> None:
    if not pd.notna(current_price):
        return
    if not (pd.notna(q10_change_pct) and pd.notna(q50_change_pct) and pd.notna(q90_change_pct)):
        return

    q10_price = float(current_price * (1.0 + q10_change_pct))
    q50_price = float(current_price * (1.0 + q50_change_pct))
    q90_price = float(current_price * (1.0 + q90_change_pct))
    x = ["现在", expected_date_label if expected_date_label else "预期日期"]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=[current_price, q90_price],
            mode="lines+markers",
            name="q90",
            line=dict(width=1),
            marker=dict(size=6),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=[current_price, q10_price],
            mode="lines+markers",
            name="q10",
            line=dict(width=1),
            marker=dict(size=6),
            fill="tonexty",
            fillcolor="rgba(99, 110, 250, 0.20)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=[current_price, q50_price],
            mode="lines+markers",
            name="q50",
            line=dict(width=3),
            marker=dict(size=7),
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="时间",
        yaxis_title="价格",
        template="plotly_white",
        height=320,
        legend=dict(orientation="h"),
        margin=dict(l=20, r=20, t=50, b=20),
    )
    entry_v = _safe_float(entry_price if entry_price is not None else current_price)
    sl_v = _safe_float(stop_loss_price)
    tp_v = _safe_float(take_profit_price)
    if np.isfinite(entry_v):
        fig.add_hline(
            y=entry_v,
            line_dash="dot",
            line_color="#94a3b8",
            annotation_text="P0/Entry",
            annotation_position="top left",
        )
    if np.isfinite(sl_v):
        fig.add_hline(
            y=sl_v,
            line_dash="dash",
            line_color="#ef4444",
            annotation_text="SL",
            annotation_position="top left",
        )
    if np.isfinite(tp_v):
        fig.add_hline(
            y=tp_v,
            line_dash="dash",
            line_color="#22c55e",
            annotation_text="TP",
            annotation_position="top left",
        )
    st.plotly_chart(fig, use_container_width=True)


def _expected_date(latest_market: str, latest_utc: str, branch_name: str, horizon: int) -> str:
    ts_market = pd.to_datetime(latest_market, errors="coerce")
    if pd.notna(ts_market):
        if branch_name == "hourly":
            exp_ts = ts_market + pd.Timedelta(hours=int(horizon))
        else:
            exp_ts = ts_market + pd.Timedelta(days=int(horizon))
        return exp_ts.strftime("%Y-%m-%d %H:%M:%S %z")

    ts = pd.to_datetime(latest_utc, utc=True, errors="coerce")
    if pd.isna(ts):
        return "-"
    if branch_name == "hourly":
        exp_ts = ts + pd.Timedelta(hours=int(horizon))
    else:
        exp_ts = ts + pd.Timedelta(days=int(horizon))
    return exp_ts.strftime("%Y-%m-%d %H:%M:%S UTC")


def _render_factor_explain() -> None:
    with st.expander("指标解释（给非量化用户）", expanded=False):
        st.markdown(
            "- `市值因子`：规模相关，通常越大越稳。\n"
            "- `价值因子`：估值便宜程度（越高通常越便宜）。\n"
            "- `成长因子`：增长能力（盈利/营收或价格增长代理）。\n"
            "- `动能因子`：近期趋势强弱。\n"
            "- `反转因子`：短期是否有回撤后反弹特征。\n"
            "- `低波动因子`：波动越低，数值通常越好。"
        )


def _render_factor_top_contributions(row: pd.Series) -> None:
    factor_map = {
        "市值因子": _safe_float(row.get("size_factor")),
        "价值因子": _safe_float(row.get("value_factor")),
        "成长因子": _safe_float(row.get("growth_factor")),
        "动能因子": _safe_float(row.get("momentum_factor")),
        "反转因子": _safe_float(row.get("reversal_factor")),
        "低波动因子": _safe_float(row.get("low_vol_factor")),
    }
    rows = [{"因子": k, "贡献": v} for k, v in factor_map.items() if np.isfinite(v)]
    if not rows:
        return
    df = pd.DataFrame(rows)
    pos = df[df["贡献"] > 0].sort_values("贡献", ascending=False).head(3).copy()
    neg = df[df["贡献"] < 0].sort_values("贡献", ascending=True).head(3).copy()
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Top 正贡献因子**")
        if pos.empty:
            st.caption("暂无显著正贡献因子。")
        else:
            pos["贡献"] = pos["贡献"].map(_format_change_pct)
            st.dataframe(pos, use_container_width=True, hide_index=True)
    with c2:
        st.markdown("**Top 负贡献因子**")
        if neg.empty:
            st.caption("暂无显著负贡献因子。")
        else:
            neg["贡献"] = neg["贡献"].map(_format_change_pct)
            st.dataframe(neg, use_container_width=True, hide_index=True)
    st.caption("解释示例：成长/动能为正通常支持趋势延续；反转为负表示短线反弹支持偏弱。")


def _render_core_field_explain() -> None:
    with st.expander("这4个核心字段是什么意思？", expanded=False):
        st.markdown(
            "- `当前价格`：当前可交易市场的最新成交价。\n"
            "- `预测价格`：模型给出的目标价格（默认看中位数 q50）。\n"
            "- `预计涨跌幅`：从当前价格到预测价格的变化比例。\n"
            "- `预期日期`：这次预测对应的目标时间点（完整时间，含时区）。"
        )


def _task_label(task: str) -> str:
    mapping = {
        "direction": "方向预测（涨/跌）",
        "magnitude": "幅度预测（涨跌幅）",
        "magnitude_quantile": "区间预测（q10/q50/q90）",
    }
    return mapping.get(str(task), str(task))


def _branch_label(branch: str) -> str:
    mapping = {"hourly": "小时", "daily": "日线"}
    return mapping.get(str(branch), str(branch))


def _model_label(model: str) -> str:
    text = str(model or "")
    mapping = {
        "baseline_prev_bar": "基线模型（延续上一根K线方向）",
        "baseline_logistic": "基线逻辑回归",
        "mvp_lightgbm": "LightGBM（增强版）",
        "mvp_lightgbm_quantile_q50": "LightGBM 中位数预测（q50）",
        "mvp_lightgbm_quantile": "LightGBM 分位数预测（q10/q50/q90）",
    }
    if text in mapping:
        return mapping[text]
    if text.startswith("baseline_"):
        return f"基线模型（{text}）"
    return text


def _metric_meta(metric: str) -> tuple[str, str]:
    key = str(metric or "")
    mapping = {
        "accuracy": ("准确率", "越高越好（>50% 说明优于随机猜测）"),
        "f1": ("F1综合分", "越高越好（综合看精确率和召回率）"),
        "precision": ("精确率", "越高越好（模型说会涨时，真正上涨的比例）"),
        "recall": ("召回率", "越高越好（实际上涨里被识别出来的比例）"),
        "roc_auc": ("AUC", "越高越好（0.5 约等于随机）"),
        "mae": ("平均绝对误差", "越低越好（越接近真实值）"),
        "rmse": ("均方根误差", "越低越好（对大误差更敏感）"),
        "sign_accuracy": ("方向正确率", "越高越好（只看方向是否预测对）"),
    }
    if key in mapping:
        return mapping[key]
    if key.startswith("pinball_"):
        return ("分位数误差", "越低越好（区间预测误差）")
    return (key, "结合策略目标解读")


def _format_metric_value(metric: str, value: object) -> str:
    if not _is_finite_number(value):
        return "-"
    v = float(value)
    ratio_metrics = {"accuracy", "f1", "precision", "recall", "roc_auc", "sign_accuracy"}
    if str(metric) in ratio_metrics:
        return f"{v:.2%}"
    return f"{v:.4f}"


def _render_model_metrics_readable(metrics: pd.DataFrame) -> None:
    if metrics.empty:
        st.info("暂无可展示的模型评估结果。")
        return

    required_cols = {"branch", "task", "model", "horizon", "metric", "mean", "std"}
    missing = required_cols - set(metrics.columns)
    if missing:
        st.dataframe(metrics, use_container_width=True)
        return

    st.markdown("**Model Metrics（通俗版）**")
    with st.expander("怎么看这些指标？", expanded=False):
        st.markdown(
            "- `准确率/F1/AUC`：越高越好。\n"
            "- `MAE/RMSE/分位数误差`：越低越好。\n"
            "- `std(波动)`：越低代表越稳定。\n"
            "- 如果方向指标一般但幅度误差小，适合做区间预期，不适合单独做买卖信号。"
        )

    work = metrics.copy()
    work["branch"] = work["branch"].astype(str)
    work["task"] = work["task"].astype(str)
    work["_horizon_norm"] = (
        pd.to_numeric(work["horizon"], errors="coerce")
        .fillna(-1)
        .astype(int)
        .astype(str)
    )

    f1, f2, f3 = st.columns(3)
    branch_options = ["全部"] + sorted(work["branch"].dropna().unique().tolist())
    task_options = ["全部"] + sorted(work["task"].dropna().unique().tolist())
    horizon_values = sorted(work["_horizon_norm"].dropna().unique().tolist())
    horizon_options = ["全部"] + [str(h) for h in horizon_values]

    selected_branch = f1.selectbox("分支", branch_options, index=0, key="metrics_branch_filter")
    selected_task = f2.selectbox("任务", task_options, index=0, key="metrics_task_filter")
    selected_horizon = f3.selectbox("周期", horizon_options, index=0, key="metrics_horizon_filter")

    view = work.copy()
    if selected_branch != "全部":
        view = view[view["branch"] == selected_branch]
    if selected_task != "全部":
        view = view[view["task"] == selected_task]
    if selected_horizon != "全部":
        view = view[view["_horizon_norm"] == selected_horizon]

    if view.empty:
        st.info("当前筛选条件下没有指标记录。")
    else:
        out = pd.DataFrame(
            {
                "分支": view["branch"].map(_branch_label),
                "任务": view["task"].map(_task_label),
                "模型": view["model"].map(_model_label),
                "预测周期": view["_horizon_norm"],
                "指标": view["metric"].map(lambda x: _metric_meta(str(x))[0]),
                "平均表现": [
                    _format_metric_value(str(m), v) for m, v in zip(view["metric"], view["mean"])
                ],
                "稳定性(std)": view["std"].map(lambda x: _format_metric_value("std", x)),
                "解读": view["metric"].map(lambda x: _metric_meta(str(x))[1]),
            }
        )
        st.dataframe(out, use_container_width=True, hide_index=True)

    with st.expander("查看原始 Model Metrics 表", expanded=False):
        st.dataframe(metrics, use_container_width=True)


def _render_trade_signal_block(signal_row: pd.Series, *, header: str = "开单信号与理由") -> None:
    if signal_row is None or len(signal_row) == 0:
        return
    st.markdown(f"**{header}**")
    action_raw = str(signal_row.get("trade_signal", signal_row.get("policy_action", "Flat")))
    action_cn = _policy_action_cn(action_raw)
    trend = str(signal_row.get("trade_trend_context", "mixed"))
    trend_cn = {"bullish": "趋势偏多", "bearish": "趋势偏空", "mixed": "趋势混合"}.get(trend, trend)
    stop_price = _safe_float(signal_row.get("trade_stop_loss_price"))
    take_price = _safe_float(signal_row.get("trade_take_profit_price"))
    rr_ratio = _safe_float(signal_row.get("trade_rr_ratio"))
    support_score = _safe_float(signal_row.get("trade_support_score"))
    stop_text = _format_price(stop_price)
    take_text = _format_price(take_price)
    if action_raw == "Flat":
        stop_text = "不适用（观望）"
        take_text = "不适用（观望）"

    c1, c2, c3, c4, c5 = st.columns([1.6, 1.3, 1.2, 1.2, 1.0])
    with c1:
        _render_big_value("当前信号", action_cn, caption="观望 = 暂不下单")
    with c2:
        _render_big_value("趋势判断", trend_cn)
    with c3:
        _render_big_value("止损价", stop_text)
    with c4:
        _render_big_value("止盈价", take_text)
    with c5:
        _render_big_value("盈亏比(RR)", _format_float(rr_ratio, 2))
    if _is_finite_number(support_score):
        st.caption(f"技术共振分数: {_format_float(support_score, 0)}（>0 偏多，<0 偏空）")
    reason_text = _format_reason_tokens_cn(signal_row.get("trade_reason_tokens", signal_row.get("policy_reason", "-")))
    st.write(f"开单理由: {reason_text}")
    st.caption(
        "说明: 该理由由 EMA/MACD/SuperTrend/成交量/BOS/CHOCH 自动生成，"
        "用于解释信号，不构成投资建议。"
    )


def _ensure_policy_for_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    df = _ensure_snapshot_factors(df)
    if df.empty:
        return df
    work = df.copy()
    try:
        cfg = _load_main_config_cached()
        if "market_type" not in work.columns:
            work["market_type"] = np.where(work["market"].astype(str).eq("crypto"), "spot", "cash")
        if "p_up" not in work.columns:
            work["p_up"] = pd.to_numeric(work.get("policy_p_up_used"), errors="coerce")
        else:
            work["p_up"] = pd.to_numeric(work["p_up"], errors="coerce")
            fallback_p = pd.to_numeric(work.get("policy_p_up_used"), errors="coerce")
            work["p_up"] = work["p_up"].where(work["p_up"].notna(), fallback_p)
        if "volatility_score" not in work.columns and {"q90_change_pct", "q10_change_pct"}.issubset(work.columns):
            work["volatility_score"] = (
                pd.to_numeric(work["q90_change_pct"], errors="coerce")
                - pd.to_numeric(work["q10_change_pct"], errors="coerce")
            )
        if "confidence_score" not in work.columns:
            conf = (2.0 * (pd.to_numeric(work["p_up"], errors="coerce") - 0.5).abs()).clip(0.0, 1.0) * 100.0
            work["confidence_score"] = conf.fillna(50.0)
        if "risk_level" not in work.columns:
            width = pd.to_numeric(work.get("volatility_score"), errors="coerce").abs()
            work["risk_level"] = np.where(
                width < 0.02,
                "low",
                np.where(width < 0.05, "medium", np.where(width < 0.10, "high", "extreme")),
            )
        work = apply_policy_frame(
            work,
            cfg,
            market_col="market",
            market_type_col="market_type",
            p_up_col="p_up",
            q10_col="q10_change_pct",
            q50_col="q50_change_pct",
            q90_col="q90_change_pct",
            volatility_col="volatility_score",
            confidence_col="confidence_score",
            current_price_col="current_price",
            risk_level_col="risk_level",
        )
    except Exception:
        pass
    return work


def _find_policy_backtest_row(
    *,
    processed_dir: Path,
    market: str,
    symbol: str,
    aliases: List[str] | None = None,
) -> pd.Series | None:
    artifacts = _load_backtest_artifacts(str(processed_dir))
    summary = artifacts.get("metrics_summary", pd.DataFrame())
    if summary.empty:
        return None
    alias_tokens = {_normalize_symbol_token(symbol)}
    for a in aliases or []:
        alias_tokens.add(_normalize_symbol_token(a))
    sub = summary[(summary["market"].astype(str) == str(market))].copy()
    if sub.empty or "symbol" not in sub.columns:
        return None
    sub = sub[sub["symbol"].map(_normalize_symbol_token).isin(alias_tokens)]
    if sub.empty:
        return None
    policy = sub[sub["strategy"].astype(str) == "policy"].head(1)
    if policy.empty:
        return None
    return policy.iloc[0]


def _reliability_level_text(summary: Dict[str, float] | None) -> str:
    if not summary:
        return "暂无"
    brier = float(summary.get("brier", float("nan")))
    coverage = float(summary.get("coverage", float("nan")))
    if not np.isfinite(brier) or not np.isfinite(coverage):
        return "暂无"
    if brier <= 0.18 and 0.72 <= coverage <= 0.88:
        return f"高（Brier {brier:.3f}，Coverage {coverage:.1%}）"
    if brier <= 0.24 and 0.65 <= coverage <= 0.92:
        return f"中（Brier {brier:.3f}，Coverage {coverage:.1%}）"
    return f"低（Brier {brier:.3f}，Coverage {coverage:.1%}）"


def _build_trade_decision_plan(
    row: pd.Series,
    *,
    cfg: Dict[str, object] | None = None,
    risk_profile: str = "标准",
    model_health: str = "中",
    event_risk: bool = False,
) -> Dict[str, object]:
    cfg_local = cfg or _load_main_config_cached("configs/config.yaml")
    policy_cfg = (cfg_local.get("policy", {}) if isinstance(cfg_local, dict) else {})
    th_cfg = policy_cfg.get("thresholds", {})
    ex_cfg = policy_cfg.get("execution", {})
    p_bull = float(th_cfg.get("p_bull", 0.55))
    p_bear = float(th_cfg.get("p_bear", 0.45))
    conf_min = float((cfg_local.get("decision", {}) if isinstance(cfg_local, dict) else {}).get("confidence_min", 60.0))
    fee_bps = float(ex_cfg.get("fee_bps", 10.0))
    slippage_bps = float(ex_cfg.get("slippage_bps", 10.0))
    cost_bps = fee_bps + slippage_bps  # 双边成本：开+平
    cost_pct = cost_bps / 10000.0

    current_price = _safe_float(row.get("current_price"))
    p_up = _safe_float(row.get("p_up", row.get("policy_p_up_used")))
    q10 = _safe_float(row.get("q10_change_pct"))
    q50 = _safe_float(row.get("q50_change_pct", row.get("predicted_change_pct")))
    q90 = _safe_float(row.get("q90_change_pct"))
    conf = _safe_float(row.get("confidence_score"))
    risk_level = str(row.get("risk_level", "medium"))
    allow_short = bool(row.get("policy_allow_short", True))
    horizon_label = str(row.get("horizon_label", "4h"))
    width = _safe_float(q90 - q10)
    if not np.isfinite(width):
        width = _safe_float(row.get("volatility_score"))
    atr_proxy_pct = float(np.clip(max(0.003, (abs(width) / 2.0) if np.isfinite(width) else 0.003), 0.003, 0.03))

    edge_long = _safe_float(q50 - cost_pct)
    edge_short = _safe_float((-q50) - cost_pct)
    edge_risk_long = _safe_float(edge_long / width) if np.isfinite(width) and width > 1e-12 else float("nan")
    edge_risk_short = _safe_float(edge_short / width) if np.isfinite(width) and width > 1e-12 else float("nan")

    long_checks = [
        ("p_up >= 阈值", np.isfinite(p_up) and p_up >= p_bull),
        ("edge_score > 0（覆盖成本）", np.isfinite(edge_long) and edge_long > 0),
        ("confidence >= 最低阈值", np.isfinite(conf) and conf >= conf_min),
        ("风险非极高", str(risk_level) != "extreme"),
        ("模型健康非差", str(model_health) in {"良", "中"}),
        ("无重大事件风险", not bool(event_risk)),
    ]
    short_checks = [
        ("p_up <= 阈值", np.isfinite(p_up) and p_up <= p_bear),
        ("edge_score > 0（覆盖成本）", np.isfinite(edge_short) and edge_short > 0),
        ("confidence >= 最低阈值", np.isfinite(conf) and conf >= conf_min),
        ("允许做空", allow_short),
        ("风险非极高", str(risk_level) != "extreme"),
        ("模型健康非差", str(model_health) in {"良", "中"}),
        ("无重大事件风险", not bool(event_risk)),
    ]
    long_ok = all(ok for _, ok in long_checks)
    short_ok = all(ok for _, ok in short_checks)

    action = "WAIT"
    if long_ok and not short_ok:
        action = "LONG"
    elif short_ok and not long_ok:
        action = "SHORT"
    elif long_ok and short_ok:
        action = "LONG" if edge_risk_long >= edge_risk_short else "SHORT"

    entry = current_price
    sl = float("nan")
    tp = float("nan")
    tp2 = float("nan")
    rr = float("nan")
    profile = str(risk_profile or "标准")
    if profile not in {"保守", "标准", "激进"}:
        profile = "标准"
    action_reason = "规则未全部满足，建议观望。"
    if profile == "保守":
        atr_mult = 1.0
        tp_mode = "q50"
        sl_mode = "max_q10_atr"
    elif profile == "激进":
        atr_mult = 2.0
        tp_mode = "q90"
        sl_mode = "q10_pref"
    else:
        atr_mult = 1.5
        tp_mode = "mid"
        sl_mode = "atr_pref"
    if action == "LONG" and np.isfinite(current_price):
        atr_stop_ret = -atr_mult * atr_proxy_pct
        q10_ret = q10 if np.isfinite(q10) else float("nan")
        if sl_mode == "max_q10_atr":
            sl_ret = max(q10_ret if np.isfinite(q10_ret) else -999.0, atr_stop_ret)
        elif sl_mode == "q10_pref":
            sl_ret = q10_ret if np.isfinite(q10_ret) else atr_stop_ret
        else:
            sl_ret = atr_stop_ret if np.isfinite(atr_stop_ret) else q10_ret
        sl = current_price * (1.0 + sl_ret)
        if tp_mode == "q90":
            tp_ret = q90 if np.isfinite(q90) else q50
        elif tp_mode == "mid":
            if np.isfinite(q50) and np.isfinite(q90):
                tp_ret = 0.5 * (q50 + q90)
            else:
                tp_ret = q50 if np.isfinite(q50) else q90
        else:
            tp_ret = q50 if np.isfinite(q50) else q90
        tp = current_price * (1.0 + tp_ret) if np.isfinite(tp_ret) else float("nan")
        tp2 = current_price * (1.0 + q90) if np.isfinite(q90) else float("nan")
        risk = entry - sl
        reward = tp - entry
        rr = _safe_float(reward / risk) if np.isfinite(risk) and risk > 1e-12 else float("nan")
        action_reason = "做多条件满足：方向概率、edge、置信度与风险过滤均通过。"
    elif action == "SHORT" and np.isfinite(current_price):
        atr_stop_ret = atr_mult * atr_proxy_pct
        q90_ret = q90 if np.isfinite(q90) else float("nan")
        if sl_mode == "max_q10_atr":
            sl_ret = min(q90_ret if np.isfinite(q90_ret) else 999.0, atr_stop_ret)
        elif sl_mode == "q10_pref":
            sl_ret = q90_ret if np.isfinite(q90_ret) else atr_stop_ret
        else:
            sl_ret = atr_stop_ret if np.isfinite(atr_stop_ret) else q90_ret
        sl = current_price * (1.0 + sl_ret)
        if tp_mode == "q90":
            tp_ret = q10 if np.isfinite(q10) else q50
        elif tp_mode == "mid":
            if np.isfinite(q10) and np.isfinite(q50):
                tp_ret = 0.5 * (q10 + q50)
            else:
                tp_ret = q50 if np.isfinite(q50) else q10
        else:
            tp_ret = q50 if np.isfinite(q50) else q10
        tp = current_price * (1.0 + tp_ret) if np.isfinite(tp_ret) else float("nan")
        tp2 = current_price * (1.0 + q10) if np.isfinite(q10) else float("nan")
        risk = sl - entry
        reward = entry - tp
        rr = _safe_float(reward / risk) if np.isfinite(risk) and risk > 1e-12 else float("nan")
        action_reason = "做空条件满足：方向概率、edge、置信度与风险过滤均通过。"

    if action == "LONG":
        selected_checks = long_checks
    elif action == "SHORT":
        selected_checks = short_checks
    else:
        selected_checks = long_checks if sum(1 for _, ok in long_checks if ok) >= sum(1 for _, ok in short_checks if ok) else short_checks
    checks_passed = int(sum(1 for _, ok in selected_checks if ok))
    checks_total = int(len(selected_checks))

    strength = _signal_strength_label(abs(_safe_float(p_up - 0.5)) * 100.0 if np.isfinite(p_up) else float("nan"), 2.0, 5.0)
    return {
        "action": action,
        "action_cn": {"LONG": "做多", "SHORT": "做空", "WAIT": "观望"}.get(action, "观望"),
        "action_reason": action_reason,
        "entry": entry,
        "stop_loss": sl,
        "take_profit": tp,
        "take_profit_2": tp2,
        "rr": rr,
        "horizon_label": horizon_label,
        "risk_level": risk_level,
        "confidence_score": conf,
        "p_up": p_up,
        "p_down": 1.0 - p_up if np.isfinite(p_up) else float("nan"),
        "q10": q10,
        "q50": q50,
        "q90": q90,
        "edge_long": edge_long,
        "edge_short": edge_short,
        "edge_risk_long": edge_risk_long,
        "edge_risk_short": edge_risk_short,
        "cost_bps": cost_bps,
        "risk_profile": profile,
        "model_health": model_health,
        "event_risk": bool(event_risk),
        "long_checks": long_checks,
        "short_checks": short_checks,
        "selected_checks": selected_checks,
        "checks_passed": checks_passed,
        "checks_total": checks_total,
        "signal_strength": strength,
        "signal_strength_text": _signal_strength_human_text(strength),
        "policy_reason": str(row.get("policy_reason", "-")),
    }


def _render_trade_decision_summary(
    *,
    plan: Dict[str, object],
    reliability_summary: Dict[str, float] | None,
    backtest_policy_row: pd.Series | None = None,
) -> None:
    st.markdown("## 交易决策卡（3秒结论）")
    action = str(plan.get("action", "WAIT"))
    action_cn = str(plan.get("action_cn", "观望"))
    color = {"LONG": "#22c55e", "SHORT": "#ef4444", "WAIT": "#f59e0b"}.get(action, "#94a3b8")
    st.markdown(
        (
            "<div style='padding:12px 14px;border:1px solid rgba(148,163,184,.25);border-radius:12px;'>"
            "<div style='font-size:13px;color:#94a3b8'>最终建议</div>"
            f"<div style='font-size:42px;font-weight:800;color:{color};line-height:1.1'>{action} / {action_cn}</div>"
            f"<div style='margin-top:6px;color:#cbd5e1'>{plan.get('action_reason','-')}</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )
    st.caption(
        f"当前满足 {int(plan.get('checks_passed', 0))}/{int(plan.get('checks_total', 0))} 条开仓条件"
        f" -> 建议 {str(plan.get('action_cn', '观望'))}（{str(plan.get('risk_profile', '标准'))}）"
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("推荐入场", _format_price(plan.get("entry")))
    c2.metric("止损", _format_price(plan.get("stop_loss")))
    c3.metric("止盈(TP1)", _format_price(plan.get("take_profit")))
    c4.metric("盈亏比 R:R", _format_float(plan.get("rr"), 2))
    c5, c6, c7, c8 = st.columns(4)
    c5.metric("持仓周期", str(plan.get("horizon_label", "4h")))
    c6.metric("风险等级", _risk_cn(str(plan.get("risk_level", "-"))))
    c7.metric("信号强度", str(plan.get("signal_strength_text", "-")))
    c8.metric("模型可信度", _reliability_level_text(reliability_summary))
    c9, c10, c11, c12 = st.columns(4)
    c9.metric("止盈(TP2)", _format_price(plan.get("take_profit_2")))
    c10.metric("Long Edge", _format_change_pct(plan.get("edge_long")))
    c11.metric("Short Edge", _format_change_pct(plan.get("edge_short")))
    active_edge_risk = (
        plan.get("edge_risk_long")
        if str(plan.get("action", "WAIT")) == "LONG"
        else (plan.get("edge_risk_short") if str(plan.get("action", "WAIT")) == "SHORT" else float("nan"))
    )
    c12.metric("Edge/Risk", _format_float(active_edge_risk, 3))
    st.caption(
        f"P(up): {_format_change_pct(plan.get('p_up')).replace('+','')} | "
        f"P(down): {_format_change_pct(plan.get('p_down')).replace('+','')} | "
        f"成本口径: 双边 {float(plan.get('cost_bps', 0.0)):.1f} bps（开+平）"
    )
    if backtest_policy_row is not None:
        b1, b2, b3, b4, b5, b6 = st.columns(6)
        b1.metric("近回测胜率", _format_change_pct(backtest_policy_row.get("win_rate")).replace("+", ""))
        b2.metric("Profit Factor", _format_float(backtest_policy_row.get("profit_factor"), 2))
        b3.metric("Avg Win/Loss", _format_float(backtest_policy_row.get("avg_win_loss_ratio"), 2))
        b4.metric("最大回撤", _format_change_pct(backtest_policy_row.get("max_drawdown")))
        b5.metric("夏普", _format_float(backtest_policy_row.get("sharpe"), 2))
        b6.metric("交易次数", f"{int(_safe_float(backtest_policy_row.get('trades_count')))}")
        st.caption(
            f"Expectancy: {_format_float(backtest_policy_row.get('expectancy'), 4)} | "
            f"总收益: {_format_change_pct(backtest_policy_row.get('total_return'))} | "
            f"波动率: {_format_float(backtest_policy_row.get('volatility'), 4)}"
        )


def _render_rule_checklist(plan: Dict[str, object]) -> None:
    st.markdown("**信号触发规则（当前判定）**")
    lcol, scol = st.columns(2)
    with lcol:
        st.markdown("`Long` 触发条件")
        for label, ok in plan.get("long_checks", []):
            st.markdown(f"- {'✅' if ok else '❌'} {label}")
    with scol:
        st.markdown("`Short` 触发条件")
        for label, ok in plan.get("short_checks", []):
            st.markdown(f"- {'✅' if ok else '❌'} {label}")
    st.markdown("`当前建议` 对应条件")
    for label, ok in plan.get("selected_checks", []):
        st.markdown(f"- {'✅' if ok else '❌'} {label}")
    st.caption(
        "规则：Long 需满足 p_up、edge、置信度、风险过滤；"
        "Short 同理并要求允许做空。未满足则观望。"
    )


def _render_symbol_backtest_section(
    *,
    processed_dir: Path,
    market: str,
    symbol: str,
    symbol_aliases: List[str] | None = None,
    provider: str | None = None,
    fallback_symbol: str | None = None,
    title: str = "回测结果（开单效果）",
) -> None:
    artifacts = _load_backtest_artifacts(str(processed_dir))
    summary = artifacts.get("metrics_summary", pd.DataFrame())
    compare = artifacts.get("compare", pd.DataFrame())
    equity = artifacts.get("equity", pd.DataFrame())
    by_fold = artifacts.get("metrics_by_fold", pd.DataFrame())
    trades = artifacts.get("trades", pd.DataFrame())
    latest_signals = artifacts.get("latest_signals", pd.DataFrame())
    if summary.empty:
        summary = pd.DataFrame(columns=["market", "symbol", "strategy"])
    alias_tokens = {_normalize_symbol_token(symbol)}
    for a in symbol_aliases or []:
        alias_tokens.add(_normalize_symbol_token(a))

    def _match_symbol(df: pd.DataFrame, col: str = "symbol") -> pd.Series:
        return df[col].map(_normalize_symbol_token).isin(alias_tokens)

    sub = summary[
        (summary["market"].astype(str) == str(market))
        & _match_symbol(summary, col="symbol")
    ].copy()

    # On-demand single-symbol backtest fallback if precomputed table has no matching row.
    fallback_note = ""
    if sub.empty:
        try:
            prov = str(provider or ("binance" if str(market) == "crypto" and str(symbol).upper().endswith("USDT") else "yahoo"))
            with st.spinner("该标的不在预计算回测样本中，正在即时回测..."):
                realtime = _run_single_symbol_backtest_cached(
                    market=str(market),
                    symbol=str(symbol),
                    provider=prov,
                    fallback_symbol=str(fallback_symbol or ""),
                )
            summary_rt = realtime.get("metrics_summary", pd.DataFrame())
            compare_rt = realtime.get("compare_baselines", pd.DataFrame())
            equity_rt = realtime.get("equity", pd.DataFrame())
            by_fold_rt = realtime.get("metrics_by_fold", pd.DataFrame())
            trades_rt = realtime.get("trades", pd.DataFrame())
            latest_signal_rt = realtime.get("latest_signal", pd.DataFrame())
            if not summary_rt.empty:
                sub = summary_rt[
                    (summary_rt["market"].astype(str) == str(market))
                    & _match_symbol(summary_rt, col="symbol")
                ].copy()
                compare = compare_rt
                equity = equity_rt
                by_fold = by_fold_rt
                trades = trades_rt
                latest_signals = latest_signal_rt
                fallback_note = "已为当前标的即时补跑回测（未写入全量回测文件，仅用于本页展示）。"
        except Exception as exc:
            fallback_note = f"即时回测失败：{exc}"

    if sub.empty:
        st.info("该标的暂无回测记录。可能是历史数据不足或数据源不可用。")
        if fallback_note:
            st.caption(fallback_note)
        return

    st.markdown("---")
    st.subheader(title)
    if fallback_note:
        st.caption(fallback_note)

    sig_view = latest_signals[
        (latest_signals["market"].astype(str) == str(market))
        & _match_symbol(latest_signals, col="symbol")
    ].copy()
    if not sig_view.empty:
        _render_trade_signal_block(sig_view.iloc[-1], header="当前可执行信号（回测口径）")

    policy_row = sub[sub["strategy"] == "policy"].head(1)
    if not policy_row.empty:
        row = policy_row.iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("策略总收益", _format_change_pct(row.get("total_return")))
        c2.metric("策略夏普", _format_float(row.get("sharpe"), 2))
        c3.metric("最大回撤", _format_change_pct(row.get("max_drawdown")))
        c4.metric("胜率", _format_change_pct(row.get("win_rate")).replace("+", ""))
        st.caption(
            f"盈亏比: {_format_float(row.get('avg_win_loss_ratio'), 2)} | "
            f"Profit Factor: {_format_float(row.get('profit_factor'), 2)} | "
            f"交易次数: {int(_safe_float(row.get('trades_count')))}"
        )

    show = sub.copy()
    show["策略"] = show["strategy"].map(
        {
            "policy": "策略信号",
            "buy_hold": "买入并持有",
            "ma_crossover": "均线交叉",
            "naive_prev_bar": "前一日方向",
        }
    ).fillna(show["strategy"])
    show["总收益"] = show["total_return"].map(_format_change_pct)
    show["夏普"] = show["sharpe"].map(lambda x: _format_float(x, 2))
    show["最大回撤"] = show["max_drawdown"].map(_format_change_pct)
    show["胜率"] = show["win_rate"].map(lambda x: _format_change_pct(x).replace("+", ""))
    show["盈亏比"] = show["avg_win_loss_ratio"].map(lambda x: _format_float(x, 2))
    show["PF"] = show["profit_factor"].map(lambda x: _format_float(x, 2))
    show_cols = ["策略", "总收益", "夏普", "最大回撤", "胜率", "盈亏比", "PF"]
    st.dataframe(show[show_cols], use_container_width=True, hide_index=True)

    cmp = compare[
        (compare["market"].astype(str) == str(market))
        & _match_symbol(compare, col="symbol")
    ].copy()
    if not cmp.empty:
        cmp["相对总收益提升"] = cmp["delta_total_return"].map(_format_change_pct)
        cmp["相对夏普提升"] = cmp["delta_sharpe"].map(lambda x: _format_float(x, 2))
        cmp["相对回撤变化"] = cmp["delta_max_drawdown"].map(_format_change_pct)
        cmp["基准策略"] = cmp["baseline"].map(
            {
                "buy_hold": "买入并持有",
                "ma_crossover": "均线交叉",
                "naive_prev_bar": "前一日方向",
            }
        ).fillna(cmp["baseline"])
        st.markdown("**与基准对比（策略信号 - 基准）**")
        st.dataframe(
            cmp[["基准策略", "相对总收益提升", "相对夏普提升", "相对回撤变化"]],
            use_container_width=True,
            hide_index=True,
        )

    sub_eq = equity[
        (equity["market"].astype(str) == str(market))
        & _match_symbol(equity, col="symbol")
    ].copy()
    if not sub_eq.empty:
        sub_eq["timestamp_utc"] = pd.to_datetime(sub_eq["timestamp_utc"], utc=True, errors="coerce")
        sub_eq = sub_eq.dropna(subset=["timestamp_utc"]).sort_values(["strategy", "fold", "timestamp_utc"])
        rows: List[pd.DataFrame] = []
        for strategy, sdf in sub_eq.groupby("strategy", dropna=False):
            running = 1.0
            frames: List[pd.DataFrame] = []
            for _, fold_df in sdf.groupby("fold", dropna=False):
                part = fold_df.sort_values("timestamp_utc").copy()
                part["eq_chain"] = running * (1.0 + pd.to_numeric(part["strategy_ret"], errors="coerce").fillna(0.0)).cumprod()
                running = float(part["eq_chain"].iloc[-1]) if not part.empty else running
                frames.append(part)
            if frames:
                merged = pd.concat(frames, ignore_index=True)
                merged["strategy"] = strategy
                rows.append(merged[["timestamp_utc", "strategy", "eq_chain"]])
        if rows:
            eq_plot = pd.concat(rows, ignore_index=True)
            fig = go.Figure()
            for strategy, sdf in eq_plot.groupby("strategy", dropna=False):
                label = {
                    "policy": "策略信号",
                    "buy_hold": "买入并持有",
                    "ma_crossover": "均线交叉",
                    "naive_prev_bar": "前一日方向",
                }.get(str(strategy), str(strategy))
                fig.add_trace(
                    go.Scatter(
                        x=sdf["timestamp_utc"],
                        y=sdf["eq_chain"],
                        mode="lines",
                        name=label,
                    )
                )
            fig.update_layout(
                title=f"{symbol} 回测资金曲线（Walk-forward 串联）",
                xaxis_title="时间",
                yaxis_title="净值",
                template="plotly_white",
                height=340,
                margin=dict(l=20, r=20, t=50, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)

    fold_view = by_fold[
        (by_fold["market"].astype(str) == str(market))
        & _match_symbol(by_fold, col="symbol")
    ].copy()
    if not fold_view.empty:
        fold_view["总收益"] = fold_view["total_return"].map(_format_change_pct)
        fold_view["夏普"] = fold_view["sharpe"].map(lambda x: _format_float(x, 2))
        fold_view["最大回撤"] = fold_view["max_drawdown"].map(_format_change_pct)
        fold_view["策略"] = fold_view["strategy"].astype(str)
        st.markdown("**分折（fold）表现**")
        st.dataframe(
            fold_view[["fold", "策略", "总收益", "夏普", "最大回撤"]].sort_values(["策略", "fold"]),
            use_container_width=True,
            hide_index=True,
        )

    trade_view = trades[
        (trades["market"].astype(str) == str(market))
        & _match_symbol(trades, col="symbol")
    ].copy()
    if not trade_view.empty:
        trade_view = trade_view.sort_values("entry_time", ascending=False).head(30).copy()
        if "entry_signal_reason" in trade_view.columns:
            trade_view["开单理由"] = trade_view["entry_signal_reason"].map(_format_reason_tokens_cn)
        elif "reason" in trade_view.columns:
            trade_view["开单理由"] = trade_view["reason"].map(_format_reason_tokens_cn)
        trade_view["方向"] = trade_view.get("side", pd.Series(["-"] * len(trade_view))).map(
            {"long": "做多", "short": "做空"}
        ).fillna("-")
        trade_view["信号"] = trade_view.get("entry_signal", pd.Series(["-"] * len(trade_view))).map(
            _policy_action_cn
        )
        trade_view["进场价"] = trade_view["entry_price"].map(_format_price)
        trade_view["止损价"] = trade_view.get("stop_loss_price", pd.Series([np.nan] * len(trade_view))).map(
            _format_price
        )
        trade_view["止盈价"] = trade_view.get("take_profit_price", pd.Series([np.nan] * len(trade_view))).map(
            _format_price
        )
        trade_view["净收益"] = trade_view.get("net_pnl_pct", pd.Series([np.nan] * len(trade_view))).map(
            _format_change_pct
        )
        trade_view["退出原因"] = trade_view.get("exit_reason", pd.Series(["-"] * len(trade_view))).map(
            {"stop_loss": "止损", "take_profit": "止盈", "time_exit": "时间平仓", "flat": "无持仓（未开单）"}
        ).fillna("-")
        st.markdown("**最近开单记录（含止盈止损）**")
        show_cols = [
            "entry_time",
            "方向",
            "信号",
            "进场价",
            "止损价",
            "止盈价",
            "净收益",
            "退出原因",
            "开单理由",
        ]
        show_cols = [c for c in show_cols if c in trade_view.columns]
        st.dataframe(trade_view[show_cols], use_container_width=True, hide_index=True)


def _render_snapshot_result(
    df: pd.DataFrame,
    title_prefix: str,
    trade_plan: Dict[str, object] | None = None,
) -> None:
    work = _ensure_policy_for_snapshot(df)
    if work.empty:
        st.warning("当前选择未能生成预测快照。")
        return
    row = work.iloc[0]
    if pd.isna(row.get("current_price")):
        st.error(f"价格获取失败: {row.get('price_source', '-')}")
        if "error_message" in row and pd.notna(row.get("error_message")):
            st.caption(str(row.get("error_message")))
        return

    delta_abs = row.get("predicted_change_abs")
    delta_text = f"{float(delta_abs):+,.2f}" if _is_finite_number(delta_abs) else "-"
    action_text = _policy_action_cn(str(row.get("policy_action", "Flat")))
    s1, s2 = st.columns(2)
    with s1:
        _render_big_value("当前价格", _format_price(row.get("current_price")))
    with s2:
        _render_big_value("预测价格", _format_price(row.get("predicted_price")))
    s3, s4 = st.columns(2)
    with s3:
        _render_big_value("预计涨跌幅", _format_change_pct(row.get("predicted_change_pct")), caption=f"价差: {delta_text}")
    with s4:
        _render_big_value("策略动作", action_text, caption="观望 = 暂不下单")
    expected_date_full = str(row.get("expected_date_market", "-"))
    st.markdown("**预期日期（完整）**")
    st.code(expected_date_full)
    st.caption(
        f"价格源: {row.get('price_source', '-')} | 预测方法: {row.get('prediction_method', '-')}"
    )
    if "policy_position_size" in row.index:
        st.caption(
            f"建议仓位: {(float(row.get('policy_position_size')) if _is_finite_number(row.get('policy_position_size')) else 0.0):.1%} | "
            f"预期净优势: {_format_change_pct(row.get('policy_expected_edge_pct'))} | "
            f"策略理由: {row.get('policy_reason', '-')}"
        )
    signal_ctx = pd.DataFrame()
    try:
        market = str(row.get("market", ""))
        symbol = str(row.get("symbol", ""))
        provider = str(row.get("provider", "yahoo"))
        fallback_symbol = str(row.get("fallback_symbol", ""))
        if market == "crypto" and not fallback_symbol and provider == "coingecko":
            sym_guess = symbol.upper().strip()
            if sym_guess.isalpha() and len(sym_guess) <= 8:
                fallback_symbol = f"{sym_guess}USDT"
        signal_ctx = _load_symbol_signal_context_cached(
            market=market,
            symbol=symbol,
            provider=provider,
            fallback_symbol=fallback_symbol,
        )
    except Exception:
        signal_ctx = pd.DataFrame()
    if not signal_ctx.empty:
        _render_trade_signal_block(signal_ctx.iloc[-1], header="开单信号与风控计划")
    _render_core_field_explain()

    st.markdown("**量化因子（风险 + 市场行为）**")
    f1, f2, f3, f4, f5, f6 = st.columns(6)
    f1.metric("市值因子", _format_float(row.get("size_factor"), digits=3))
    f2.metric("价值因子", _format_float(row.get("value_factor"), digits=4))
    f3.metric("成长因子", _format_change_pct(row.get("growth_factor")))
    f4.metric("动能因子", _format_change_pct(row.get("momentum_factor")))
    f5.metric("反转因子", _format_change_pct(row.get("reversal_factor")))
    f6.metric("低波动因子", _format_change_pct(row.get("low_vol_factor")))

    market_cap = row.get("market_cap_usd")
    market_cap_text = _format_price(market_cap) if _is_finite_number(market_cap) else "-"
    st.caption(
        "风险因子来源: "
        f"size={row.get('size_factor_source', '-')}, "
        f"value={row.get('value_factor_source', '-')}, "
        f"growth={row.get('growth_factor_source', '-')} | "
        f"Market Cap(USD): {market_cap_text}"
    )
    _render_factor_explain()

    _render_projection_chart(
        current_price=float(row.get("current_price")),
        q10_change_pct=float(row.get("q10_change_pct")),
        q50_change_pct=float(row.get("q50_change_pct")),
        q90_change_pct=float(row.get("q90_change_pct")),
        expected_date_label=expected_date_full,
        title=f"{title_prefix} 预测可视化图（q10 / q50 / q90）",
        entry_price=(
            float(trade_plan.get("entry"))
            if isinstance(trade_plan, dict) and _is_finite_number(trade_plan.get("entry"))
            else None
        ),
        stop_loss_price=(
            float(trade_plan.get("stop_loss"))
            if isinstance(trade_plan, dict) and _is_finite_number(trade_plan.get("stop_loss"))
            else None
        ),
        take_profit_price=(
            float(trade_plan.get("take_profit"))
            if isinstance(trade_plan, dict) and _is_finite_number(trade_plan.get("take_profit"))
            else None
        ),
    )


def _render_branch(branch_name: str, df: pd.DataFrame, live_price: float | None = None) -> None:
    st.subheader(f"{branch_name.capitalize()} Branch")
    if df.empty:
        st.warning(f"No predictions found for {branch_name}.")
        return

    df = df.sort_values("timestamp_utc").reset_index(drop=True)
    latest = df.iloc[-1]
    st.caption(f"Latest UTC: {latest.get('timestamp_utc', '-')}")

    horizons = _infer_horizons(df.columns.tolist())
    if not horizons:
        st.warning("No horizon prediction columns found.")
        return

    selected_h = st.selectbox(
        f"{branch_name} horizon",
        options=horizons,
        index=0,
        key=f"{branch_name}_horizon",
    )

    c1, c2, c3 = st.columns(3)
    p_up = float(latest.get(f"dir_h{selected_h}_p_up", 0.0))
    p_down = float(latest.get(f"dir_h{selected_h}_p_down", 1.0 - p_up))
    c1.metric("P(up)", f"{p_up:.2%}")
    c2.metric("P(down)", f"{p_down:.2%}")
    c3.metric("Start Window", str(latest.get("start_window_name", "W?")))
    st.caption("说明：P(up)/P(down)是方向概率；预测价格是幅度模型结果，两者短期可能不完全一致。")

    q10_col = f"ret_h{selected_h}_q0.1"
    q50_col = f"ret_h{selected_h}_q0.5"
    q90_col = f"ret_h{selected_h}_q0.9"
    model_base_price = float(latest.get("close", float("nan")))
    current_price = float(live_price) if live_price is not None else model_base_price
    pred_ret_q50 = float(latest.get(q50_col, float("nan")))
    pred_price = (
        current_price * (1.0 + pred_ret_q50)
        if pd.notna(current_price) and pd.notna(pred_ret_q50)
        else float("nan")
    )
    delta_abs = (
        pred_price - current_price
        if pd.notna(pred_price) and pd.notna(current_price)
        else float("nan")
    )
    pred_price_base = (
        model_base_price * (1.0 + pred_ret_q50)
        if pd.notna(model_base_price) and pd.notna(pred_ret_q50)
        else float("nan")
    )
    expected_date = _expected_date(
        str(latest.get("timestamp_market", "")),
        str(latest.get("timestamp_utc", "")),
        branch_name,
        selected_h,
    )

    policy_row = {}
    try:
        cfg = _load_main_config_cached()
        branch_market_type = "perp" if branch_name == "hourly" else "spot"
        policy_input = pd.DataFrame(
            [
                {
                    "market": "crypto",
                    "market_type": branch_market_type,
                    "p_up": p_up,
                    "q10_change_pct": float(latest.get(q10_col, float("nan"))),
                    "q50_change_pct": pred_ret_q50,
                    "q90_change_pct": float(latest.get(q90_col, float("nan"))),
                    "volatility_score": float(latest.get(q90_col, float("nan")))
                    - float(latest.get(q10_col, float("nan"))),
                    "confidence_score": 50.0,
                    "current_price": current_price,
                    "risk_level": "medium",
                }
            ]
        )
        policy_eval = apply_policy_frame(policy_input, cfg)
        if not policy_eval.empty:
            policy_row = policy_eval.iloc[0].to_dict()
    except Exception:
        policy_row = {}

    delta_text = f"{delta_abs:+,.2f}" if _is_finite_number(delta_abs) else "-"
    action_text = _policy_action_cn(str(policy_row.get("policy_action", "Flat")))
    p1, p2 = st.columns(2)
    with p1:
        _render_big_value("当前价格", _format_price(current_price))
    with p2:
        _render_big_value("预测价格 (q50)", _format_price(pred_price))
    p3, p4 = st.columns(2)
    with p3:
        _render_big_value("预计涨跌幅", _format_change_pct(pred_ret_q50), caption=f"价差: {delta_text}")
    with p4:
        _render_big_value("策略动作", action_text, caption="观望 = 暂不下单")
    st.markdown("**预期日期（完整）**")
    st.code(expected_date)
    st.caption(
        f"模型基准价（最后收盘）: {_format_price(model_base_price)} | "
        f"按基准价口径预测价: {_format_price(pred_price_base)}"
    )
    if policy_row:
        st.caption(
            f"建议仓位: {(float(policy_row.get('policy_position_size')) if _is_finite_number(policy_row.get('policy_position_size')) else 0.0):.1%} | "
            f"预期净优势: {_format_change_pct(policy_row.get('policy_expected_edge_pct'))} | "
            f"策略理由: {policy_row.get('policy_reason', '-')}"
        )

    _render_projection_chart(
        current_price=float(current_price),
        q10_change_pct=float(latest.get(q10_col, float("nan"))),
        q50_change_pct=float(latest.get(q50_col, float("nan"))),
        q90_change_pct=float(latest.get(q90_col, float("nan"))),
        expected_date_label=expected_date,
        title=f"{branch_name.capitalize()} 预测可视化（q10 / q50 / q90）",
    )


def _render_btc_model_detail_section(
    btc_live: float | None,
    hourly_df: pd.DataFrame,
    daily_df: pd.DataFrame,
) -> None:
    st.markdown("---")
    st.subheader("BTC 模型详情（Hourly / Daily）")
    if btc_live is not None:
        st.info(f"BTC 实时价格 (Binance.US): **${btc_live:,.2f}**")
    else:
        st.warning("BTC 实时价格获取失败（不影响模型详情展示）。")

    btc_signal_ctx = _load_symbol_signal_context_cached(
        market="crypto",
        symbol="BTCUSDT",
        provider="binance",
        fallback_symbol="BTCUSDT",
    )
    if not btc_signal_ctx.empty:
        _render_trade_signal_block(btc_signal_ctx.iloc[-1], header="BTC 当前开单信号（技术触发 + 风控）")

    left, right = st.columns(2)
    with left:
        _render_branch("hourly", hourly_df, live_price=btc_live)
    with right:
        _render_branch("daily", daily_df, live_price=btc_live)


def _build_btc_model_snapshot_from_hourly(
    *,
    hourly_df: pd.DataFrame,
    btc_live: float | None,
    fallback_snapshot: pd.DataFrame,
) -> pd.DataFrame:
    if hourly_df.empty:
        return pd.DataFrame()

    required_cols = {
        "timestamp_utc",
        "timestamp_market",
        "close",
        "ret_h4_q0.1",
        "ret_h4_q0.5",
        "ret_h4_q0.9",
    }
    if not required_cols.issubset(set(hourly_df.columns)):
        return pd.DataFrame()

    df = hourly_df.sort_values("timestamp_utc").reset_index(drop=True)
    latest = df.iloc[-1]
    model_base_price = _safe_float(latest.get("close"))
    live_price = _safe_float(btc_live)
    current_price = live_price if np.isfinite(live_price) else model_base_price

    q10 = _safe_float(latest.get("ret_h4_q0.1"))
    q50 = _safe_float(latest.get("ret_h4_q0.5"))
    q90 = _safe_float(latest.get("ret_h4_q0.9"))
    p_up = _safe_float(latest.get("dir_h4_p_up"))
    p_down = _safe_float(latest.get("dir_h4_p_down"))
    if not (np.isfinite(current_price) and np.isfinite(q10) and np.isfinite(q50) and np.isfinite(q90)):
        return pd.DataFrame()

    predicted_price = current_price * (1.0 + q50)
    width = q90 - q10
    confidence_score = float(np.clip((2.0 * abs(p_up - 0.5)) * 100.0, 0.0, 100.0)) if np.isfinite(p_up) else 50.0
    risk_level = "medium"
    if np.isfinite(width):
        if width < 0.02:
            risk_level = "low"
        elif width < 0.05:
            risk_level = "medium"
        elif width < 0.10:
            risk_level = "high"
        else:
            risk_level = "extreme"
    expected_date_market = _expected_date(
        str(latest.get("timestamp_market", "")),
        str(latest.get("timestamp_utc", "")),
        "hourly",
        4,
    )

    ts_utc = pd.to_datetime(str(latest.get("timestamp_utc", "")), utc=True, errors="coerce")
    if pd.notna(ts_utc):
        expected_date_utc = (ts_utc + pd.Timedelta(hours=4)).strftime("%Y-%m-%d %H:%M:%S UTC")
    else:
        expected_date_utc = "-"

    base = {
        "instrument_id": "btc",
        "name": "Bitcoin",
        "market": "crypto",
        "symbol": "BTCUSDT",
        "provider": "binance",
        "timezone": "Asia/Shanghai",
        "horizon_label": "4h",
        "current_price": current_price,
        "predicted_price": predicted_price,
        "predicted_change_pct": q50,
        "predicted_change_abs": predicted_price - current_price,
        "expected_date_market": expected_date_market,
        "expected_date_utc": expected_date_utc,
        "p_up": p_up,
        "p_down": p_down if np.isfinite(p_down) else (1.0 - p_up if np.isfinite(p_up) else float("nan")),
        "volatility_score": width,
        "confidence_score": confidence_score,
        "risk_level": risk_level,
        "start_window_top1": str(latest.get("start_window_name", "W?")),
        "model_base_price": model_base_price,
        "q10_change_pct": q10,
        "q50_change_pct": q50,
        "q90_change_pct": q90,
        "price_source": "binance_ticker_live" if np.isfinite(live_price) else "model_last_close",
        "prediction_method": "mvp_lightgbm_quantile_q50 (hourly h=4)",
        "size_factor": float("nan"),
        "value_factor": float("nan"),
        "growth_factor": float("nan"),
        "momentum_factor": float("nan"),
        "reversal_factor": float("nan"),
        "low_vol_factor": float("nan"),
        "size_factor_source": "na",
        "value_factor_source": "na",
        "growth_factor_source": "na",
        "market_cap_usd": float("nan"),
        "generated_at_utc": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M UTC"),
    }

    # Reuse factor fields from fallback snapshot if available.
    if not fallback_snapshot.empty:
        ref = fallback_snapshot.iloc[0]
        factor_cols = [
            "size_factor",
            "value_factor",
            "growth_factor",
            "momentum_factor",
            "reversal_factor",
            "low_vol_factor",
            "size_factor_source",
            "value_factor_source",
            "growth_factor_source",
            "market_cap_usd",
        ]
        for col in factor_cols:
            if col in ref:
                base[col] = ref.get(col)

    return pd.DataFrame([base])


def _build_snapshot_for_selected(
    *,
    market: str,
    row: pd.Series,
) -> pd.DataFrame:
    provider = str(row.get("provider", "yahoo"))
    snapshot_symbol = str(row.get("snapshot_symbol", row.get("symbol", "")))
    if market == "crypto":
        return _build_selected_snapshot_cached(
            instrument_id=str(snapshot_symbol).lower(),
            name=str(row.get("name", snapshot_symbol)),
            market="crypto",
            symbol=snapshot_symbol,
            provider=provider,
            timezone="Asia/Shanghai",
            horizon_unit="hour",
            horizon_steps=4,
            history_lookback_days=365,
        )
    if market == "cn_equity":
        code = str(row.get("code", row.get("symbol", ""))).lower()
        return _build_selected_snapshot_cached(
            instrument_id=code,
            name=str(row.get("name", row.get("symbol", ""))),
            market="cn_equity",
            symbol=str(row.get("symbol", "")),
            provider="yahoo",
            timezone="Asia/Shanghai",
            horizon_unit="day",
            horizon_steps=3,
            history_lookback_days=730,
        )
    return _build_selected_snapshot_cached(
        instrument_id=str(row.get("symbol", "")).lower(),
        name=str(row.get("name", row.get("symbol", ""))),
        market="us_equity",
        symbol=str(row.get("symbol", "")),
        provider="yahoo",
        timezone="America/New_York",
        horizon_unit="day",
        horizon_steps=3,
        history_lookback_days=730,
    )


def _render_crypto_page(
    *,
    processed_dir: Path,
    btc_live: float | None,
    hourly_df: pd.DataFrame,
    daily_df: pd.DataFrame,
) -> None:
    st.header("Crypto 页面")
    cfg = _load_main_config_cached("configs/config.yaml")
    catalog = get_universe_catalog()["crypto"]
    pool_key = st.selectbox(
        "选择加密池",
        options=list(catalog.keys()),
        format_func=lambda k: catalog[k],
        key="crypto_pool_page",
    )
    uni = _load_universe_cached("crypto", pool_key)
    choice = st.selectbox("选择币种", uni["display"].tolist(), key="crypto_symbol_page")
    row = uni[uni["display"] == choice].iloc[0]
    snap = _build_snapshot_for_selected(market="crypto", row=row)

    snapshot_symbol = str(row.get("snapshot_symbol", "")).upper()
    symbol = str(row.get("symbol", "")).upper()
    is_btc = snapshot_symbol == "BTCUSDT" or symbol == "BTC"
    selected_snap = snap
    btc_main_snap_ready = False
    if is_btc:
        model_snap = _build_btc_model_snapshot_from_hourly(
            hourly_df=hourly_df,
            btc_live=btc_live,
            fallback_snapshot=snap,
        )
        if not model_snap.empty:
            selected_snap = model_snap
            btc_main_snap_ready = True

    # P0: 顶部决策卡（结论 -> 规则 -> 风险）
    selected_snap_for_plan = _ensure_policy_for_snapshot(selected_snap)
    bt_symbol = str(row.get("snapshot_symbol", row.get("symbol", "")))
    bt_policy_row = _find_policy_backtest_row(
        processed_dir=processed_dir,
        market="crypto",
        symbol=bt_symbol,
        aliases=[
            str(row.get("symbol", "")),
            str(row.get("snapshot_symbol", "")),
            str(row.get("name", "")),
        ],
    )
    if not selected_snap_for_plan.empty:
        plan_row = selected_snap_for_plan.iloc[0]
        plan = _build_trade_decision_plan(plan_row, cfg=cfg)
        horizon_unit, horizon_steps = _parse_horizon_label(str(plan_row.get("horizon_label", "4h")))
        horizon_hours = int(horizon_steps) if horizon_unit == "hour" else max(1, int(horizon_steps) * 24)
        rel_summary, _, rel_msg = _compute_recent_hourly_reliability(
            symbol=str(snapshot_symbol or symbol),
            horizon_hours=horizon_hours,
            window_days=30,
            cfg=cfg,
        )
        rel_7d, _, rel_7d_msg = _compute_recent_hourly_reliability(
            symbol=str(snapshot_symbol or symbol),
            horizon_hours=horizon_hours,
            window_days=7,
            cfg=cfg,
        )
        st.markdown("---")
        _render_trade_decision_summary(
            plan=plan,
            reliability_summary=rel_summary if not rel_msg else {},
            backtest_policy_row=bt_policy_row,
        )
        _render_rule_checklist(plan)
        if rel_msg:
            st.caption(f"模型可信度补充：{rel_msg}")
        elif not rel_7d_msg and rel_7d:
            acc_30 = _safe_float(rel_summary.get("accuracy"))
            acc_7 = _safe_float(rel_7d.get("accuracy"))
            brier_30 = _safe_float(rel_summary.get("brier"))
            brier_7 = _safe_float(rel_7d.get("brier"))
            if np.isfinite(acc_7) and np.isfinite(acc_30) and np.isfinite(brier_7) and np.isfinite(brier_30):
                if acc_7 + 0.03 < acc_30 or brier_7 > brier_30 + 0.03:
                    st.warning(
                        f"近期退化提示：7天表现弱于30天（Acc {acc_7:.1%} vs {acc_30:.1%}，"
                        f"Brier {brier_7:.3f} vs {brier_30:.3f}）。建议降低仓位。"
                    )
                else:
                    st.caption(
                        f"近期稳定：7天 vs 30天（Acc {acc_7:.1%}/{acc_30:.1%}，"
                        f"Brier {brier_7:.3f}/{brier_30:.3f}）。"
                    )
        edge_abs = _safe_float(plan.get("edge_long"))
        q50 = _safe_float(plan.get("q50"))
        if np.isfinite(q50) and np.isfinite(edge_abs) and abs(q50) < abs(edge_abs):
            st.warning("冲突提示：方向概率可能存在，但预期幅度不足以覆盖成本，建议观望或轻仓。")
        with st.expander("决策公式说明（可审计）", expanded=False):
            st.markdown(
                "- Long 触发：`p_up>=阈值` 且 `edge_score>0` 且 `confidence>=阈值` 且 `风险非极高`\n"
                "- Short 触发：`p_up<=阈值` 且 `edge_score>0` 且 `confidence>=阈值` 且 `允许做空`\n"
                "- `edge_score = q50 - cost_bps/10000`（双边成本）\n"
                "- Long 止损默认优先用 q10，下沿为正时使用 ATR 代理；止盈默认 q50，RR = (TP-entry)/(entry-SL)\n"
                "- Short 止损默认优先用 q90，上沿为负时使用 ATR 代理；止盈默认 q50，RR = (entry-TP)/(SL-entry)"
            )

    st.markdown("---")
    if btc_main_snap_ready:
        _render_snapshot_result(selected_snap, title_prefix="Crypto（BTC主模型）")
        st.caption("当前 BTC 顶部卡片与下方 BTC 模型详情已统一口径：Hourly h=4。")
    else:
        _render_snapshot_result(snap, title_prefix="Crypto")
        if not is_btc:
            st.caption("当前币种卡片使用快照基线口径（非 BTC 主模型）。")

    _render_btc_model_detail_section(btc_live=btc_live, hourly_df=hourly_df, daily_df=daily_df)

    st.markdown("---")
    st.subheader("模型效果解读")
    metrics_path = processed_dir / "metrics_walk_forward_summary.csv"
    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path)
        _render_model_metrics_readable(metrics)
    else:
        st.info("未找到模型评估结果（metrics_walk_forward_summary.csv）。")

    _render_symbol_backtest_section(
        processed_dir=processed_dir,
        market="crypto",
        symbol=bt_symbol,
        symbol_aliases=[
            str(row.get("symbol", "")),
            str(row.get("snapshot_symbol", "")),
            str(row.get("name", "")),
        ],
        provider=str(row.get("provider", "binance")),
        fallback_symbol=(
            f"{str(row.get('symbol', '')).upper()}USDT"
            if str(row.get("provider", "binance")) == "coingecko"
            else str(row.get("snapshot_symbol", ""))
        ),
        title="Crypto 回测结果（该币种）",
    )


def _render_cn_page() -> None:
    st.header("A股 页面")
    cfg = _load_main_config_cached("configs/config.yaml")
    catalog = get_universe_catalog()["cn_equity"]
    pool_key = st.selectbox(
        "选择A股股票池",
        options=list(catalog.keys()),
        format_func=lambda k: catalog[k],
        key="cn_pool_page",
    )
    uni = _load_universe_cached("cn_equity", pool_key)
    choice = st.selectbox("选择A股标的", uni["display"].tolist(), key="cn_symbol_page")
    row = uni[uni["display"] == choice].iloc[0]
    snap = _build_snapshot_for_selected(market="cn_equity", row=row)

    snap_for_plan = _ensure_policy_for_snapshot(snap)
    risk_col1, risk_col2 = st.columns(2)
    risk_profile = risk_col1.selectbox("风险偏好", ["保守", "标准", "激进"], index=1, key="cn_risk_profile")
    event_risk = bool(
        risk_col2.checkbox("未来3天有重大事件风险（政策/财报/宏观）", value=False, key="cn_event_risk")
    )

    bt_symbol = str(row.get("symbol", ""))
    bt_policy_row = _find_policy_backtest_row(
        processed_dir=Path("data/processed"),
        market="cn_equity",
        symbol=bt_symbol,
        aliases=[str(row.get("code", "")), str(row.get("name", ""))],
    )

    trade_plan: Dict[str, object] = {}
    rel_summary: Dict[str, float] = {}
    rel_compare = pd.DataFrame()
    rel_msg = ""
    if not snap_for_plan.empty:
        snap_row = snap_for_plan.iloc[0]
        h_unit, h_steps = _parse_horizon_label(str(snap_row.get("horizon_label", "3d")))
        h_days = max(1, int(h_steps) if h_unit == "day" else int(np.ceil(int(h_steps) / 24)))
        rel_summary, rel_compare, rel_msg = _compute_recent_symbol_reliability_cached(
            market="cn_equity",
            symbol=str(row.get("symbol", "")),
            provider="yahoo",
            fallback_symbol=str(row.get("symbol", "")),
            horizon_steps=h_days,
            window_days=30,
        )
        health_grade = _model_health_grade(rel_summary if not rel_msg else {})
        trade_plan = _build_trade_decision_plan(
            snap_row,
            cfg=cfg,
            risk_profile=risk_profile,
            model_health=health_grade,
            event_risk=event_risk,
        )
        st.markdown("---")
        _render_trade_decision_summary(
            plan=trade_plan,
            reliability_summary=rel_summary if not rel_msg else {},
            backtest_policy_row=bt_policy_row,
        )
        _render_rule_checklist(trade_plan)
        if rel_msg:
            st.caption(f"模型健康补充：{rel_msg}")
        with st.expander("模型健康与校准（近30天）", expanded=False):
            if rel_msg:
                st.info(rel_msg)
            else:
                c1, c2, c3 = st.columns(3)
                c1.metric("Brier", _format_float(rel_summary.get("brier"), 4))
                c2.metric("Coverage(80%目标)", _format_change_pct(rel_summary.get("coverage")).replace("+", ""))
                c3.metric("AUC", _format_float(rel_summary.get("auc"), 3))
                c4, c5, c6 = st.columns(3)
                c4.metric("Accuracy", _format_change_pct(rel_summary.get("accuracy")).replace("+", ""))
                c5.metric("区间宽度", _format_change_pct(rel_summary.get("width")))
                c6.metric("样本数", f"{int(rel_summary.get('samples', 0))}")
                st.caption(f"模型健康等级：{_model_health_grade(rel_summary)}")
                if not rel_compare.empty:
                    st.dataframe(
                        rel_compare.style.format(
                            {"Accuracy": "{:.2%}", "AUC": "{:.3f}", "Brier": "{:.4f}", "Coverage": "{:.2%}"},
                            na_rep="-",
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )

    st.markdown("---")
    _render_snapshot_result(snap, title_prefix="A股", trade_plan=trade_plan if trade_plan else None)
    if not snap_for_plan.empty:
        with st.expander("因子贡献摘要（Top 3 正/负）", expanded=False):
            _render_factor_top_contributions(snap_for_plan.iloc[0])

    _render_symbol_backtest_section(
        processed_dir=Path("data/processed"),
        market="cn_equity",
        symbol=bt_symbol,
        symbol_aliases=[str(row.get("code", "")), str(row.get("name", ""))],
        provider="yahoo",
        fallback_symbol=str(row.get("symbol", "")),
        title="A股 回测结果（该标的）",
    )


def _render_us_page() -> None:
    st.header("美股 页面")
    cfg = _load_main_config_cached("configs/config.yaml")
    catalog = get_universe_catalog()["us_equity"]
    pool_key = st.selectbox(
        "选择美股股票池",
        options=list(catalog.keys()),
        format_func=lambda k: catalog[k],
        key="us_pool_page",
    )
    uni = _load_universe_cached("us_equity", pool_key)
    choice = st.selectbox("选择美股标的", uni["display"].tolist(), key="us_symbol_page")
    row = uni[uni["display"] == choice].iloc[0]
    snap = _build_snapshot_for_selected(market="us_equity", row=row)
    snap_for_plan = _ensure_policy_for_snapshot(snap)

    risk_col1, risk_col2 = st.columns(2)
    risk_profile = risk_col1.selectbox("风险偏好", ["保守", "标准", "激进"], index=1, key="us_risk_profile")
    event_risk = bool(
        risk_col2.checkbox("未来3天有重大事件风险（财报/宏观）", value=False, key="us_event_risk")
    )

    bt_symbol = str(row.get("symbol", ""))
    bt_policy_row = _find_policy_backtest_row(
        processed_dir=Path("data/processed"),
        market="us_equity",
        symbol=bt_symbol,
        aliases=[str(row.get("name", ""))],
    )

    trade_plan: Dict[str, object] = {}
    rel_summary: Dict[str, float] = {}
    rel_compare = pd.DataFrame()
    rel_msg = ""
    if not snap_for_plan.empty:
        snap_row = snap_for_plan.iloc[0]
        h_unit, h_steps = _parse_horizon_label(str(snap_row.get("horizon_label", "3d")))
        h_days = max(1, int(h_steps) if h_unit == "day" else int(np.ceil(int(h_steps) / 24)))
        rel_summary, rel_compare, rel_msg = _compute_recent_symbol_reliability_cached(
            market="us_equity",
            symbol=str(row.get("symbol", "")),
            provider="yahoo",
            fallback_symbol=str(row.get("symbol", "")),
            horizon_steps=h_days,
            window_days=30,
        )
        health_grade = _model_health_grade(rel_summary if not rel_msg else {})
        trade_plan = _build_trade_decision_plan(
            snap_row,
            cfg=cfg,
            risk_profile=risk_profile,
            model_health=health_grade,
            event_risk=event_risk,
        )
        st.markdown("---")
        _render_trade_decision_summary(
            plan=trade_plan,
            reliability_summary=rel_summary if not rel_msg else {},
            backtest_policy_row=bt_policy_row,
        )
        _render_rule_checklist(trade_plan)
        if rel_msg:
            st.caption(f"模型健康补充：{rel_msg}")
        with st.expander("模型健康与校准（近30天）", expanded=False):
            if rel_msg:
                st.info(rel_msg)
            else:
                c1, c2, c3 = st.columns(3)
                c1.metric("Brier", _format_float(rel_summary.get("brier"), 4))
                c2.metric("Coverage(80%目标)", _format_change_pct(rel_summary.get("coverage")).replace("+", ""))
                c3.metric("AUC", _format_float(rel_summary.get("auc"), 3))
                c4, c5, c6 = st.columns(3)
                c4.metric("Accuracy", _format_change_pct(rel_summary.get("accuracy")).replace("+", ""))
                c5.metric("区间宽度", _format_change_pct(rel_summary.get("width")))
                c6.metric("样本数", f"{int(rel_summary.get('samples', 0))}")
                st.caption(f"模型健康等级：{_model_health_grade(rel_summary)}")
                if not rel_compare.empty:
                    st.dataframe(
                        rel_compare.style.format(
                            {"Accuracy": "{:.2%}", "AUC": "{:.3f}", "Brier": "{:.4f}", "Coverage": "{:.2%}"},
                            na_rep="-",
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )

    st.markdown("---")
    _render_snapshot_result(snap, title_prefix="美股", trade_plan=trade_plan if trade_plan else None)
    if not snap_for_plan.empty:
        with st.expander("因子贡献摘要（Top 3 正/负）", expanded=False):
            _render_factor_top_contributions(snap_for_plan.iloc[0])
    _render_symbol_backtest_section(
        processed_dir=Path("data/processed"),
        market="us_equity",
        symbol=bt_symbol,
        symbol_aliases=[str(row.get("name", ""))],
        provider="yahoo",
        fallback_symbol=str(row.get("symbol", "")),
        title="美股 回测结果（该标的）",
    )


def _status_cn(status: str) -> str:
    mapping = {"Active": "可执行", "Watch": "观察", "Retired": "暂停"}
    return mapping.get(status, status)


def _action_cn(action: str) -> str:
    mapping = {"Keep/Open": "持有或新开", "Monitor/Reduce": "观察或减仓", "Remove": "移除"}
    return mapping.get(action, action)


def _alert_cn(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return "无"
    mapping = {
        "prediction_unavailable": "预测不可用",
        "predicted_change_non_positive": "预测涨跌幅<=0",
        "history_too_short": "历史数据不足",
        "missing_rate_too_high": "缺失率过高",
        "liquidity_insufficient": "流动性不足",
        "risk_factor_missing": "风险因子缺失",
        "behavior_factor_missing": "行为因子缺失",
    }
    return "；".join(mapping.get(x, x) for x in text.split(";") if x.strip())


def _market_cn(market: str) -> str:
    mapping = {"crypto": "加密", "cn_equity": "A股", "us_equity": "美股"}
    return mapping.get(str(market), str(market))


def _split_alert_codes(text: object) -> List[str]:
    raw = str(text or "").strip()
    if not raw:
        return []
    return [x.strip() for x in raw.split(";") if x.strip()]


def _alert_tag_cn(code: str) -> str:
    mapping = {
        "prediction_unavailable": "无预测",
        "predicted_change_non_positive": "净优势不足",
        "history_too_short": "历史不足",
        "missing_rate_too_high": "缺失偏高",
        "liquidity_insufficient": "流动性不足",
        "risk_factor_missing": "风险因子缺失",
        "behavior_factor_missing": "行为因子缺失",
    }
    return mapping.get(str(code), str(code))


def _alert_fix_action_cn(code: str) -> str:
    mapping = {
        "prediction_unavailable": "补齐该标的预测流水后重跑 tracking。",
        "predicted_change_non_positive": "等待净优势转正或切换到反向/观望策略。",
        "history_too_short": "补足历史K线（建议>=365日线，或>=60小时级样本）。",
        "missing_rate_too_high": "修复数据缺失（更换源/补拉缺口）后再评估。",
        "liquidity_insufficient": "提高流动性后再纳入，或切换更高流动性标的。",
        "risk_factor_missing": "补齐市值/估值/成长因子快照。",
        "behavior_factor_missing": "补齐OHLCV后重算动量/反转/低波动因子。",
    }
    return mapping.get(str(code), "检查数据源与特征计算流程。")


def _confidence_bucket_cn(score: float) -> str:
    if not np.isfinite(score):
        return "未知"
    if score >= 80:
        return "高"
    if score >= 60:
        return "中"
    return "低"


def _risk_level_from_score(score: float) -> str:
    if not np.isfinite(score):
        return "high"
    if score <= 0.35:
        return "low"
    if score <= 0.55:
        return "medium"
    if score <= 0.75:
        return "high"
    return "extreme"


def _prepare_tracking_table(ranked: pd.DataFrame, coverage: pd.DataFrame, cost_bps: float) -> pd.DataFrame:
    if ranked.empty:
        return ranked.copy()

    work = ranked.copy()
    for c in ["market", "instrument_id", "name", "display", "symbol", "alerts", "policy_reason"]:
        if c not in work.columns:
            work[c] = ""
    work["market"] = work["market"].astype(str)
    work["instrument_id"] = work["instrument_id"].astype(str)

    if not coverage.empty:
        cov = coverage.copy()
        for c in ["market", "instrument_id"]:
            if c in cov.columns:
                cov[c] = cov[c].astype(str)
        cov_cols = [
            "market",
            "instrument_id",
            "prediction_available",
            "history_missing_rate",
            "hard_filter_pass",
            "alerts",
        ]
        cov_cols = [c for c in cov_cols if c in cov.columns]
        key_cols = [c for c in ["market", "instrument_id"] if c in cov_cols]
        cov = cov[cov_cols]
        if key_cols:
            cov = cov.drop_duplicates(subset=key_cols, keep="last")
            work = work.merge(cov, on=key_cols, how="left", suffixes=("", "_cov"))
        else:
            work["prediction_available"] = np.nan
            work["history_missing_rate"] = np.nan
            work["hard_filter_pass"] = np.nan
    else:
        work["prediction_available"] = np.nan
        work["history_missing_rate"] = np.nan
        work["hard_filter_pass"] = np.nan

    work["predicted_change_pct"] = pd.to_numeric(work.get("predicted_change_pct"), errors="coerce")
    work["policy_expected_edge_pct"] = pd.to_numeric(work.get("policy_expected_edge_pct"), errors="coerce")
    work["total_score"] = pd.to_numeric(work.get("total_score"), errors="coerce")
    work["liquidity_score"] = pd.to_numeric(work.get("liquidity_score"), errors="coerce")
    work["data_quality_score"] = pd.to_numeric(work.get("data_quality_score"), errors="coerce")
    work["history_score"] = pd.to_numeric(work.get("history_score"), errors="coerce")
    work["coverage_score"] = pd.to_numeric(work.get("coverage_score"), errors="coerce")
    work["factor_support_count"] = pd.to_numeric(work.get("factor_support_count"), errors="coerce").fillna(0.0)
    work["history_missing_rate"] = pd.to_numeric(work.get("history_missing_rate"), errors="coerce").fillna(0.0).clip(lower=0.0)

    pred_flag_raw = work.get("prediction_available", pd.Series([np.nan] * len(work), index=work.index))
    pred_flag = (
        pred_flag_raw.astype(str).str.lower().map({"true": True, "false": False, "1": True, "0": False})
    )
    pred_flag = pred_flag.where(pred_flag.notna(), work["predicted_change_pct"].notna())
    hard_flag_raw = work.get("hard_filter_pass", pd.Series([np.nan] * len(work), index=work.index))
    hard_flag = hard_flag_raw.astype(str).str.lower().map({"true": True, "false": False, "1": True, "0": False})
    hard_flag = hard_flag.fillna(True)
    work["prediction_available_flag"] = pred_flag.astype(bool)
    work["hard_filter_pass_flag"] = hard_flag.astype(bool)

    cost_pct = float(cost_bps) / 10000.0
    fallback_edge = work["predicted_change_pct"] - cost_pct
    work["edge_score"] = work["policy_expected_edge_pct"].where(work["policy_expected_edge_pct"].notna(), fallback_edge)
    work["edge_score_short"] = (-work["predicted_change_pct"]) - cost_pct

    total_norm = (work["total_score"] / 100.0).clip(lower=0.0, upper=1.0).fillna(0.0)
    quality_norm = (work["data_quality_score"] / 20.0).clip(lower=0.0, upper=1.0).fillna(0.0)
    factor_norm = (work["factor_support_count"] / 3.0).clip(lower=0.0, upper=1.0).fillna(0.0)
    conviction = (work["predicted_change_pct"].abs() * 20.0).clip(lower=0.0, upper=1.0).fillna(0.0)

    confidence = (
        100.0
        * (
            0.48 * total_norm
            + 0.20 * quality_norm
            + 0.15 * factor_norm
            + 0.17 * conviction
        )
        - (work["history_missing_rate"] * 30.0)
        - ((~work["hard_filter_pass_flag"]).astype(float) * 10.0)
    )
    work["confidence_score_est"] = confidence.clip(lower=0.0, upper=100.0)

    risk_score = (
        (1.0 - total_norm) * 0.45
        + (1.0 - quality_norm) * 0.20
        + (1.0 - factor_norm) * 0.15
        + (1.0 - conviction) * 0.08
        + work["history_missing_rate"].clip(upper=1.0) * 0.12
        + ((~work["hard_filter_pass_flag"]).astype(float) * 0.22)
    ).clip(lower=0.0, upper=1.0)
    work["risk_score"] = risk_score
    work["risk_level"] = work["risk_score"].map(_risk_level_from_score)
    work["risk_level_cn"] = work["risk_level"].map(_risk_cn)
    work["risk_rank"] = work["risk_level"].map({"low": 0, "medium": 1, "high": 2, "extreme": 3}).fillna(9)

    denom = work["risk_score"].clip(lower=0.08)
    work["edge_risk"] = work["edge_score"] / denom
    work["edge_risk_short"] = work["edge_score_short"] / denom

    work["市场"] = work["market"].map(_market_cn)
    work["原始状态"] = work["status"].map(_status_cn).fillna("-")
    work["建议动作"] = work["recommended_action"].map(_action_cn).fillna("-")
    work["策略动作"] = work["policy_action"].map(_policy_action_cn).fillna("观望")
    work["display_name"] = work["display"].fillna("").astype(str)
    work["display_name"] = work["display_name"].where(work["display_name"].str.strip() != "", work["name"].astype(str))
    work["display_name"] = work["display_name"].where(work["display_name"].str.strip() != "", work["symbol"].astype(str))
    work["track_key"] = work["market"].astype(str) + "|" + work["instrument_id"].astype(str)

    work["active_edge_score"] = np.where(work["策略动作"] == "做空", work["edge_score_short"], work["edge_score"])
    work["active_edge_risk"] = np.where(work["策略动作"] == "做空", work["edge_risk_short"], work["edge_risk"])
    work["inputs_ready"] = (
        work["prediction_available_flag"].astype(bool)
        & work["hard_filter_pass_flag"].astype(bool)
        & work["predicted_change_pct"].notna()
    )

    def _status_rule_row(r: pd.Series) -> Tuple[str, str]:
        conf_v = _safe_float(r.get("confidence_score_est"))
        edge_v = _safe_float(r.get("active_edge_score"))
        risk_v = str(r.get("risk_level", "high"))
        inputs_ok = bool(r.get("inputs_ready", False))
        hard_ok = bool(r.get("hard_filter_pass_flag", False))
        tags: List[str] = []
        tags.append("inputs_ready" if inputs_ok else "insufficient_inputs")
        tags.append("edge_ok" if np.isfinite(edge_v) and edge_v > 0 else "edge不足")
        tags.append("confidence_ok" if np.isfinite(conf_v) and conf_v >= 70 else "confidence偏低")
        tags.append("risk_ok" if risk_v in {"low", "medium"} else "risk偏高")
        if not hard_ok:
            tags.append("hard_filter_fail")

        if inputs_ok and hard_ok and np.isfinite(conf_v) and conf_v >= 70 and np.isfinite(edge_v) and edge_v > 0 and risk_v in {"low", "medium"}:
            return "可执行", " | ".join(tags)
        if (not inputs_ok) or (not hard_ok) or (np.isfinite(conf_v) and conf_v < 40) or risk_v == "extreme":
            return "暂停", " | ".join(tags)
        return "观察", " | ".join(tags)

    rule_tuple = work.apply(_status_rule_row, axis=1)
    work["状态(规则)"] = rule_tuple.map(lambda x: x[0] if isinstance(x, tuple) else "观察")
    work["状态触发标签"] = rule_tuple.map(lambda x: x[1] if isinstance(x, tuple) else "-")

    def _row_fix_actions(r: pd.Series) -> str:
        codes = _split_alert_codes(r.get("alerts"))
        if not codes and "alerts_cov" in r.index:
            codes = _split_alert_codes(r.get("alerts_cov"))
        if not codes:
            status_rule = str(r.get("状态(规则)", "观察"))
            if status_rule == "可执行":
                return "无需修复（当前可执行）。"
            if status_rule == "观察":
                return "继续跟踪，等待净优势/置信度进一步改善。"
            return "检查该标的数据源与预测流水是否齐全。"
        actions = []
        for c in codes:
            act = _alert_fix_action_cn(c)
            if act not in actions:
                actions.append(act)
        return "；".join(actions)

    def _row_alert_tags(r: pd.Series) -> str:
        codes = _split_alert_codes(r.get("alerts"))
        if not codes and "alerts_cov" in r.index:
            codes = _split_alert_codes(r.get("alerts_cov"))
        if not codes:
            return "无"
        tags = []
        for c in codes:
            t = _alert_tag_cn(c)
            if t not in tags:
                tags.append(t)
        return "、".join(tags)

    def _short_reason_line(r: pd.Series) -> str:
        action_tag = {"做多": "long_signal", "做空": "short_signal", "观望": "wait_signal"}.get(
            str(r.get("策略动作", "观望")),
            "wait_signal",
        )
        conf_txt = _confidence_bucket_cn(_safe_float(r.get("confidence_score_est")))
        risk_txt = _risk_cn(str(r.get("risk_level", "high")))
        reason_raw = _format_reason_tokens_cn(r.get("policy_reason", "-"))
        reason_core = "-"
        if isinstance(reason_raw, str) and reason_raw.strip() and reason_raw != "-":
            reason_core = reason_raw.split("；")[0]
        if reason_core == "-":
            return f"{action_tag} + {conf_txt}置信度 + {risk_txt}风险"
        return f"{action_tag} + {reason_core} + {conf_txt}置信度"

    work["告警标签"] = work.apply(_row_alert_tags, axis=1)
    work["修复动作"] = work.apply(_row_fix_actions, axis=1)
    work["短原因"] = work.apply(_short_reason_line, axis=1)
    return work


def _render_tracking_page(processed_dir: Path) -> None:
    st.header("Selection / Research / Tracking 页面")
    tracking_dir = processed_dir / "tracking"
    ranked = _load_csv(tracking_dir / "ranked_universe.csv")
    actions = _load_csv(tracking_dir / "tracking_actions.csv")
    coverage = _load_csv(tracking_dir / "coverage_matrix.csv")
    report_path = tracking_dir / "data_quality_report.md"

    if ranked.empty:
        st.info(
            "还没有 tracking 结果。请先运行 `python -m src.markets.tracking --config configs/config.yaml`。"
        )
        return

    ctrl1, ctrl2 = st.columns([1, 1])
    cost_bps = float(
        ctrl1.number_input(
            "成本估计（bps，双边：开+平）",
            min_value=0.0,
            max_value=200.0,
            value=8.0,
            step=1.0,
            key="track_cost_bps",
        )
    )
    top_k = int(ctrl2.slider("Top Opportunities 每组条数", 5, 10, 5, 1, key="track_top_opps_k"))

    prepared = _prepare_tracking_table(ranked=ranked, coverage=coverage, cost_bps=cost_bps)
    if prepared.empty:
        st.info("tracking 数据为空。")
        return

    total = len(prepared)
    executable_n = int((prepared["状态(规则)"] == "可执行").sum())
    watch_n = int((prepared["状态(规则)"] == "观察").sum())
    paused_n = int((prepared["状态(规则)"] == "暂停").sum())
    pred_cov = float(prepared["prediction_available_flag"].astype(bool).mean())
    hard_pass = float(prepared["hard_filter_pass_flag"].astype(bool).mean())
    avg_missing = float(pd.to_numeric(prepared["history_missing_rate"], errors="coerce").fillna(0.0).mean())

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("候选总数", f"{total}")
    k2.metric("可执行", f"{executable_n}")
    k3.metric("观察", f"{watch_n}")
    k4.metric("暂停", f"{paused_n}")
    k5.metric("预测可用率", f"{pred_cov:.1%}")
    k6.metric("硬门槛通过率", f"{hard_pass:.1%}")
    st.caption(
        f"平均缺失率：{avg_missing:.1%} | edge_score口径：优先 `policy_expected_edge_pct`，缺失时退化为 `predicted_change_pct - cost_bps/10000`。"
    )

    st.subheader("Top Opportunities（先看这里）")
    top_long = prepared[(prepared["状态(规则)"] == "可执行") & (prepared["策略动作"] == "做多")].sort_values(
        "active_edge_risk", ascending=False
    )
    top_short = prepared[(prepared["状态(规则)"] == "可执行") & (prepared["策略动作"] == "做空")].sort_values(
        "active_edge_risk", ascending=False
    )
    top_watch = prepared[prepared["状态(规则)"] == "观察"].sort_values("active_edge_risk", ascending=False)

    def _show_opps(df: pd.DataFrame) -> pd.DataFrame:
        show = df.head(top_k).copy()
        if show.empty:
            return show
        return pd.DataFrame(
            {
                "market": show["市场"],
                "symbol": show["display_name"],
                "final_action": show["策略动作"],
                "净优势(edge)": show["active_edge_score"].map(_format_change_pct),
                "风险": show["risk_level"].map(_risk_cn),
                "置信度": show["confidence_score_est"].map(lambda x: _format_float(x, 1)),
                "reason": show["短原因"],
            }
        )

    c_long, c_short, c_watch = st.columns(3)
    with c_long:
        st.markdown("**Top 做多机会（可执行）**")
        show = _show_opps(top_long)
        if show.empty:
            st.info("暂无满足条件的做多机会。")
        else:
            st.dataframe(show, use_container_width=True, hide_index=True)
    with c_short:
        st.markdown("**Top 做空机会（可执行）**")
        show = _show_opps(top_short)
        if show.empty:
            st.info("暂无满足条件的做空机会。")
        else:
            st.dataframe(show, use_container_width=True, hide_index=True)
    with c_watch:
        st.markdown("**Top 观察名单（潜力但未满足）**")
        show = _show_opps(top_watch)
        if show.empty:
            st.info("暂无观察名单。")
        else:
            st.dataframe(show, use_container_width=True, hide_index=True)

    with st.expander("状态判定规则（可执行/观察/暂停）", expanded=False):
        st.markdown(
            "- `可执行`：`confidence>=70` 且 `active_edge_score>0` 且 `risk_level<=中` 且 `inputs_ready=true`。\n"
            "- `观察`：方向存在但 edge/置信度/风险尚未满足执行门槛。\n"
            "- `暂停`：insufficient_inputs、hard_filter_fail、风险极高或置信度过低。\n"
            "- 触发标签示例：`inputs_ready | edge_ok | confidence_ok | risk_ok`。"
        )

    st.subheader("Screener（筛选 + 排序）")
    flt1, flt2, flt3, flt4 = st.columns([1, 1, 1, 1])
    market_options = ["全部"] + sorted(prepared["市场"].dropna().unique().tolist())
    status_options = ["全部", "可执行", "观察", "暂停"]
    action_options = ["全部", "做多", "做空", "观望"]
    preset_options = ["全部", "保守策略", "激进策略", "低风险", "高置信度"]
    market_sel = flt1.selectbox("市场", market_options, 0, key="track_market_v2")
    status_sel = flt2.selectbox("规则状态", status_options, 0, key="track_status_v2")
    action_sel = flt3.selectbox("策略动作", action_options, 0, key="track_action_v2")
    preset_sel = flt4.selectbox("快速预设", preset_options, 0, key="track_preset_v2")

    chip1, chip2, chip3, chip4, chip5 = st.columns(5)
    only_exec = chip1.toggle("只看可执行", value=False, key="track_chip_exec")
    only_long = chip2.toggle("只看做多", value=False, key="track_chip_long")
    only_short = chip3.toggle("只看做空", value=False, key="track_chip_short")
    exclude_paused = chip4.toggle("排除暂停", value=False, key="track_chip_no_pause")
    high_liq_only = chip5.toggle("高流动性", value=False, key="track_chip_liq")

    sort_col1, sort_col2, sort_col3 = st.columns([2, 1, 1])
    sort_map = {
        "按 edge_risk（默认）": "active_edge_risk",
        "按 edge_score": "active_edge_score",
        "按 confidence": "confidence_score_est",
        "按 risk_level": "risk_rank",
        "按 liquidity_score": "liquidity_score",
        "按 data_quality_score": "data_quality_score",
    }
    sort_label = sort_col1.selectbox("排序方式", list(sort_map.keys()), 0, key="track_sort_key")
    sort_key = sort_map[sort_label]
    sort_desc = bool(sort_col2.toggle("降序", value=True, key="track_sort_desc"))
    top_n = int(sort_col3.slider("主表展示前N", 20, 500, 120, 10, key="track_topn_v2"))

    view = prepared.copy()
    if market_sel != "全部":
        view = view[view["市场"] == market_sel]
    if status_sel != "全部":
        view = view[view["状态(规则)"] == status_sel]
    if action_sel != "全部":
        view = view[view["策略动作"] == action_sel]

    if preset_sel == "保守策略":
        view = view[
            (view["状态(规则)"] == "可执行")
            & (view["risk_level"].isin(["low", "medium"]))
            & (view["confidence_score_est"] >= 75)
            & (view["active_edge_score"] > 0)
        ]
    elif preset_sel == "激进策略":
        view = view[
            (view["状态(规则)"] != "暂停")
            & (view["active_edge_score"].abs() > 0)
            & (view["confidence_score_est"] >= 55)
        ]
    elif preset_sel == "低风险":
        view = view[view["risk_level"].isin(["low", "medium"])]
    elif preset_sel == "高置信度":
        view = view[view["confidence_score_est"] >= 80]

    if only_exec:
        view = view[view["状态(规则)"] == "可执行"]
    if only_long and not only_short:
        view = view[view["策略动作"] == "做多"]
    if only_short and not only_long:
        view = view[view["策略动作"] == "做空"]
    if exclude_paused:
        view = view[view["状态(规则)"] != "暂停"]
    if high_liq_only and "liquidity_score" in view.columns:
        liq_threshold = float(pd.to_numeric(prepared["liquidity_score"], errors="coerce").quantile(0.7))
        view = view[pd.to_numeric(view["liquidity_score"], errors="coerce") >= liq_threshold]

    if sort_key in view.columns:
        if sort_key == "risk_rank":
            view = view.sort_values([sort_key, "active_edge_risk"], ascending=[True, False])
        else:
            view = view.sort_values(sort_key, ascending=not sort_desc)

    show = view.head(top_n).copy()
    show["置信度"] = show["confidence_score_est"].map(lambda x: _format_float(x, 1))
    show["风险等级"] = show["risk_level"].map(_risk_cn)
    show["机会值(edge)"] = show["active_edge_score"].map(_format_change_pct)
    show["风险调整(edge_risk)"] = show["active_edge_risk"].map(lambda x: _format_float(x, 3))
    show["预计涨跌幅"] = show["predicted_change_pct"].map(_format_change_pct)
    show["总分(0-100)"] = show["total_score"].map(lambda x: _format_float(x, 1))
    show["数据质量"] = show["data_quality_score"].map(lambda x: _format_float(x, 1))
    show["流动性"] = show["liquidity_score"].map(lambda x: _format_float(x, 1))
    show["因子支持数"] = show["factor_support_count"].map(lambda x: _format_float(x, 0))
    show = show.rename(columns={"display_name": "标的", "symbol": "代码"})

    main_cols = [
        "市场",
        "标的",
        "代码",
        "状态(规则)",
        "策略动作",
        "原始状态",
        "置信度",
        "风险等级",
        "机会值(edge)",
        "风险调整(edge_risk)",
        "预计涨跌幅",
        "总分(0-100)",
        "数据质量",
        "流动性",
        "因子支持数",
        "告警标签",
        "状态触发标签",
        "修复动作",
        "短原因",
    ]
    main_cols = [c for c in main_cols if c in show.columns]
    st.dataframe(show[main_cols], use_container_width=True, hide_index=True)
    export_cols = [c for c in show.columns if c not in {"track_key"}]
    st.download_button(
        "Download CSV（当前筛选）",
        data=show[export_cols].to_csv(index=False).encode("utf-8-sig"),
        file_name="tracking_screener_filtered.csv",
        mime="text/csv",
        use_container_width=False,
    )

    st.subheader("单标的展开详情（Drill-down）")
    detail_source = view if not view.empty else prepared
    detail_options = detail_source["track_key"].dropna().astype(str).tolist()
    detail_name_col = "标的" if "标的" in detail_source.columns else "display_name"
    detail_map = {
        k: f"{detail_source.loc[detail_source['track_key'] == k, '市场'].iloc[0]} | {detail_source.loc[detail_source['track_key'] == k, detail_name_col].iloc[0]}"
        for k in detail_options
    }
    if detail_options:
        selected_key = st.selectbox(
            "选择标的",
            detail_options,
            index=0,
            format_func=lambda x: detail_map.get(x, x),
            key="track_drill_symbol",
        )
        row = detail_source.loc[detail_source["track_key"] == selected_key].iloc[0]
        d1, d2, d3, d4, d5, d6 = st.columns(6)
        d1.metric("规则状态", str(row.get("状态(规则)", "-")))
        d2.metric("策略动作", str(row.get("策略动作", "-")))
        d3.metric("机会值(edge)", _format_change_pct(row.get("active_edge_score")))
        d4.metric("风险调整(edge_risk)", _format_float(row.get("active_edge_risk"), 3))
        d5.metric("置信度", _format_float(row.get("confidence_score_est"), 1))
        d6.metric("风险", _risk_cn(str(row.get("risk_level", "-"))))
        e1, e2, e3, e4 = st.columns(4)
        e1.metric("当前价格", _format_price(row.get("current_price")))
        e2.metric("预测价格", _format_price(row.get("predicted_price")))
        e3.metric("预测涨跌幅", _format_change_pct(row.get("predicted_change_pct")))
        e4.metric("因子支持数", _format_float(row.get("factor_support_count"), 0))
        st.caption(
            f"短原因：{row.get('短原因', '-')} | 触发标签：{row.get('状态触发标签', '-')} | "
            f"告警：{row.get('告警标签', '无')}"
        )
        st.caption(f"修复建议：{row.get('修复动作', '-')}")
        if _is_finite_number(row.get("policy_position_size")):
            st.caption(f"建议仓位：{float(row.get('policy_position_size')):.1%}")
    else:
        st.info("当前筛选条件下没有标的可展开。")

    st.subheader("动作建议明细（含暂停原因 / 修复建议）")
    action_view = prepared.copy()
    action_view["建议仓位"] = action_view["policy_position_size"].map(
        lambda x: f"{float(x):.1%}" if _is_finite_number(x) else "-"
    )
    action_view["预期净优势"] = action_view["active_edge_score"].map(_format_change_pct)
    action_view["预计涨跌幅"] = action_view["predicted_change_pct"].map(_format_change_pct)
    action_cols = [
        "市场",
        "display_name",
        "状态(规则)",
        "策略动作",
        "建议仓位",
        "预期净优势",
        "预计涨跌幅",
        "短原因",
        "状态触发标签",
        "告警标签",
        "修复动作",
    ]
    action_cols = [c for c in action_cols if c in action_view.columns]
    action_view = action_view.rename(columns={"display_name": "标的"})
    action_cols = ["标的" if c == "display_name" else c for c in action_cols]
    st.dataframe(action_view[action_cols].head(200), use_container_width=True, hide_index=True)

    st.subheader("Tracking：Watchlist + 信号变化提醒")
    watch_path = tracking_dir / "watchlist.csv"
    changes_path = tracking_dir / "signal_changes.csv"
    watch_cols = [
        "track_key",
        "market",
        "instrument_id",
        "name",
        "added_time_bj",
        "last_signal_change_bj",
        "last_status",
        "last_action",
        "last_review_note",
        "signal_change_7d",
    ]
    watch = _load_csv(watch_path)
    if watch.empty:
        watch = pd.DataFrame(columns=watch_cols)
    else:
        for c in watch_cols:
            if c not in watch.columns:
                watch[c] = ""
        watch = watch[watch_cols].copy()

    now_bj_text = pd.Timestamp.now(tz="Asia/Shanghai").strftime("%Y-%m-%d %H:%M:%S %z")
    if not watch.empty:
        idx_map = prepared.set_index("track_key")
        change_records: List[Dict[str, object]] = []
        for i in watch.index:
            key = str(watch.at[i, "track_key"])
            if key not in idx_map.index:
                continue
            cur = idx_map.loc[key]
            new_status = str(cur.get("状态(规则)", "-"))
            new_action = str(cur.get("策略动作", "-"))
            prev_status = str(watch.at[i, "last_status"] or "").strip()
            prev_action = str(watch.at[i, "last_action"] or "").strip()
            watch.at[i, "market"] = str(cur.get("market", ""))
            watch.at[i, "instrument_id"] = str(cur.get("instrument_id", ""))
            watch.at[i, "name"] = str(cur.get("display_name", ""))
            watch.at[i, "last_status"] = new_status
            watch.at[i, "last_action"] = new_action
            if prev_status and ((prev_status != new_status) or (prev_action != new_action)):
                watch.at[i, "last_signal_change_bj"] = now_bj_text
                change_records.append(
                    {
                        "track_key": key,
                        "market": str(cur.get("market", "")),
                        "name": str(cur.get("display_name", "")),
                        "change_time_bj": now_bj_text,
                        "from_status": prev_status,
                        "to_status": new_status,
                        "from_action": prev_action,
                        "to_action": new_action,
                    }
                )
        if change_records:
            old_changes = _load_csv(changes_path)
            new_changes = pd.DataFrame(change_records)
            all_changes = new_changes if old_changes.empty else pd.concat([old_changes, new_changes], ignore_index=True)
            all_changes.to_csv(changes_path, index=False, encoding="utf-8-sig")
        watch.to_csv(watch_path, index=False, encoding="utf-8-sig")

    label_map = {
        str(r["track_key"]): f"{r['市场']} | {r['display_name']}"
        for _, r in prepared[["track_key", "市场", "display_name"]].drop_duplicates().iterrows()
    }
    default_keys = watch["track_key"].dropna().astype(str).tolist() if not watch.empty else []
    selected_keys = st.multiselect(
        "选择需要持续跟踪的标的",
        options=list(label_map.keys()),
        default=[k for k in default_keys if k in label_map],
        format_func=lambda x: label_map.get(x, x),
        key="track_watch_keys_v2",
    )

    edit_base = pd.DataFrame({"track_key": selected_keys})
    if not edit_base.empty:
        edit_base = edit_base.merge(
            prepared[["track_key", "市场", "display_name", "状态(规则)", "策略动作"]].drop_duplicates("track_key"),
            on="track_key",
            how="left",
        )
        if not watch.empty:
            edit_base = edit_base.merge(watch[["track_key", "last_review_note"]], on="track_key", how="left")
        else:
            edit_base["last_review_note"] = ""
        edit_base = edit_base.rename(
            columns={
                "display_name": "标的",
                "状态(规则)": "当前状态",
                "策略动作": "当前动作",
                "last_review_note": "研究备注",
            }
        )
        edited = st.data_editor(
            edit_base[["track_key", "市场", "标的", "当前状态", "当前动作", "研究备注"]],
            hide_index=True,
            use_container_width=True,
            key="track_watch_editor_v2",
        )
    else:
        edited = pd.DataFrame(columns=["track_key", "研究备注"])
        st.info("还未选择跟踪标的。")

    if st.button("保存 Watchlist", key="track_watch_save_v2"):
        prev_watch = watch.set_index("track_key") if not watch.empty else pd.DataFrame().set_index(pd.Index([]))
        keep_rows: List[Dict[str, object]] = []
        change_records: List[Dict[str, object]] = []
        idx_map = prepared.set_index("track_key")
        now_text = pd.Timestamp.now(tz="Asia/Shanghai").strftime("%Y-%m-%d %H:%M:%S %z")
        note_map = {}
        if not edited.empty and "track_key" in edited.columns:
            note_map = dict(zip(edited["track_key"].astype(str), edited["研究备注"].fillna("").astype(str)))

        for key in selected_keys:
            if key not in idx_map.index:
                continue
            cur = idx_map.loc[key]
            prev = prev_watch.loc[key] if key in prev_watch.index else None
            prev_status = str(prev["last_status"]).strip() if prev is not None and "last_status" in prev else ""
            prev_action = str(prev["last_action"]).strip() if prev is not None and "last_action" in prev else ""
            new_status = str(cur.get("状态(规则)", "-"))
            new_action = str(cur.get("策略动作", "-"))
            added_time = str(prev["added_time_bj"]) if prev is not None and str(prev.get("added_time_bj", "")).strip() else now_text
            last_signal_change = (
                str(prev["last_signal_change_bj"]) if prev is not None and str(prev.get("last_signal_change_bj", "")).strip() else "-"
            )
            if prev_status and ((prev_status != new_status) or (prev_action != new_action)):
                last_signal_change = now_text
                change_records.append(
                    {
                        "track_key": key,
                        "market": str(cur.get("market", "")),
                        "name": str(cur.get("display_name", "")),
                        "change_time_bj": now_text,
                        "from_status": prev_status,
                        "to_status": new_status,
                        "from_action": prev_action,
                        "to_action": new_action,
                    }
                )
            keep_rows.append(
                {
                    "track_key": key,
                    "market": str(cur.get("market", "")),
                    "instrument_id": str(cur.get("instrument_id", "")),
                    "name": str(cur.get("display_name", "")),
                    "added_time_bj": added_time,
                    "last_signal_change_bj": last_signal_change,
                    "last_status": new_status,
                    "last_action": new_action,
                    "last_review_note": note_map.get(str(key), ""),
                    "signal_change_7d": 0,
                }
            )

        new_watch = pd.DataFrame(keep_rows, columns=watch_cols)
        if change_records:
            old_changes = _load_csv(changes_path)
            new_changes = pd.DataFrame(change_records)
            all_changes = new_changes if old_changes.empty else pd.concat([old_changes, new_changes], ignore_index=True)
            all_changes.to_csv(changes_path, index=False, encoding="utf-8-sig")

        changes_all = _load_csv(changes_path)
        if not changes_all.empty and "change_time_bj" in changes_all.columns:
            changes_all["change_time_bj"] = pd.to_datetime(changes_all["change_time_bj"], errors="coerce", utc=True).dt.tz_convert(
                "Asia/Shanghai"
            )
            window_start = pd.Timestamp.now(tz="Asia/Shanghai") - pd.Timedelta(days=7)
            recent7 = changes_all[changes_all["change_time_bj"] >= window_start]
            cnt7 = recent7.groupby("track_key").size().to_dict()
            new_watch["signal_change_7d"] = new_watch["track_key"].map(cnt7).fillna(0).astype(int)

        new_watch.to_csv(watch_path, index=False, encoding="utf-8-sig")
        st.success(f"已保存 Watchlist：{len(new_watch)} 个标的。")
        st.rerun()

    watch_latest = _load_csv(watch_path)
    changes_latest = _load_csv(changes_path)
    if not watch_latest.empty:
        for c in watch_cols:
            if c not in watch_latest.columns:
                watch_latest[c] = ""
        if not changes_latest.empty and "change_time_bj" in changes_latest.columns:
            changes_latest["change_time_bj"] = pd.to_datetime(changes_latest["change_time_bj"], errors="coerce", utc=True).dt.tz_convert(
                "Asia/Shanghai"
            )
            recent7 = changes_latest[
                changes_latest["change_time_bj"] >= (pd.Timestamp.now(tz="Asia/Shanghai") - pd.Timedelta(days=7))
            ]
            cnt7 = recent7.groupby("track_key").size().to_dict()
            watch_latest["signal_change_7d"] = watch_latest["track_key"].astype(str).map(cnt7).fillna(0).astype(int)
        watch_show = watch_latest.copy()
        watch_show["市场"] = watch_show["market"].map(_market_cn)
        watch_show = watch_show.rename(
            columns={
                "name": "标的",
                "added_time_bj": "加入时间",
                "last_signal_change_bj": "最近信号变化",
                "last_status": "当前状态",
                "last_action": "当前动作",
                "signal_change_7d": "近7天变化次数",
                "last_review_note": "研究备注",
            }
        )
        watch_cols_show = ["市场", "标的", "加入时间", "最近信号变化", "当前状态", "当前动作", "近7天变化次数", "研究备注"]
        watch_cols_show = [c for c in watch_cols_show if c in watch_show.columns]
        st.dataframe(watch_show[watch_cols_show], use_container_width=True, hide_index=True)

    st.markdown("**信号变化提醒（最近24小时）**")
    if not changes_latest.empty and "change_time_bj" in changes_latest.columns:
        changes_latest["change_time_bj"] = pd.to_datetime(changes_latest["change_time_bj"], errors="coerce", utc=True).dt.tz_convert(
            "Asia/Shanghai"
        )
        recent24 = changes_latest[
            changes_latest["change_time_bj"] >= (pd.Timestamp.now(tz="Asia/Shanghai") - pd.Timedelta(hours=24))
        ].copy()
        if recent24.empty:
            st.info("最近24小时没有状态/动作变化。")
        else:
            recent24["市场"] = recent24["market"].map(_market_cn)
            recent24["变化时间"] = recent24["change_time_bj"].dt.strftime("%Y-%m-%d %H:%M:%S %z")
            recent24 = recent24.rename(
                columns={
                    "name": "标的",
                    "from_status": "原状态",
                    "to_status": "新状态",
                    "from_action": "原动作",
                    "to_action": "新动作",
                }
            )
            st.dataframe(
                recent24[["变化时间", "市场", "标的", "原状态", "新状态", "原动作", "新动作"]],
                use_container_width=True,
                hide_index=True,
            )
            promote_n = int(((recent24["原状态"] == "观察") & (recent24["新状态"] == "可执行")).sum())
            degrade_n = int(((recent24["原状态"] == "可执行") & (recent24["新状态"] != "可执行")).sum())
            risk_up_n = int((recent24["新状态"] == "暂停").sum())
            a1, a2, a3 = st.columns(3)
            a1.metric("观察 -> 可执行", f"{promote_n}")
            a2.metric("可执行 -> 降级", f"{degrade_n}")
            a3.metric("变为暂停", f"{risk_up_n}")
    else:
        st.info("暂无变化日志。保存 watchlist 后将开始记录。")

    st.subheader("数据覆盖率 / 缺口分析")
    code_list: List[str] = []
    for raw in prepared.get("alerts", pd.Series([""] * len(prepared), index=prepared.index)).tolist():
        code_list.extend(_split_alert_codes(raw))
    if not code_list and "alerts_cov" in prepared.columns:
        for raw in prepared["alerts_cov"].tolist():
            code_list.extend(_split_alert_codes(raw))
    if code_list:
        gap_df = pd.Series(code_list).value_counts().rename_axis("code").reset_index(name="count")
        gap_df["问题"] = gap_df["code"].map(_alert_tag_cn)
        gap_df["修复建议"] = gap_df["code"].map(_alert_fix_action_cn)
        gap_df = gap_df[["问题", "count", "修复建议"]].rename(columns={"count": "数量"})
        st.dataframe(gap_df.head(12), use_container_width=True, hide_index=True)
    else:
        st.success("当前没有明显的数据覆盖告警。")

    if not actions.empty:
        with st.expander("原始 tracking_actions 快照", expanded=False):
            st.dataframe(actions.head(200), use_container_width=True, hide_index=True)

    if report_path.exists():
        with st.expander("数据质量报告", expanded=False):
            st.markdown(report_path.read_text(encoding="utf-8"))


def main() -> None:
    st.set_page_config(page_title="Multi-Market Forecast Dashboard", layout="wide")
    st.title("Multi-Market Forecast Dashboard")
    st.caption("页面导航：Crypto / A股 / 美股 / 交易时间段预测 / Selection-Research-Tracking")
    st.markdown(
        """
<style>
div[data-testid="stMetricValue"] {
  white-space: normal !important;
  overflow: visible !important;
  text-overflow: clip !important;
}
</style>
""",
        unsafe_allow_html=True,
    )

    if st.button("Clear cache and reload", use_container_width=False):
        st.cache_data.clear()
        st.rerun()

    processed_dir = Path("data/processed")
    hourly = _load_csv(processed_dir / "predictions_hourly.csv")
    daily = _load_csv(processed_dir / "predictions_daily.csv")
    btc_live = _fetch_live_btc_price()

    page = st.sidebar.radio(
        "页面",
        options=[
            "Crypto 页面",
            "A股 页面",
            "美股 页面",
            "交易时间段预测（Crypto）",
            "Selection / Research / Tracking 页面",
        ],
        index=0,
    )

    if page == "Crypto 页面":
        _render_crypto_page(
            processed_dir=processed_dir,
            btc_live=btc_live,
            hourly_df=hourly,
            daily_df=daily,
        )
    elif page == "A股 页面":
        _render_cn_page()
    elif page == "美股 页面":
        _render_us_page()
    elif page == "交易时间段预测（Crypto）":
        _render_crypto_session_page()
    else:
        _render_tracking_page(processed_dir)


if __name__ == "__main__":
    main()
