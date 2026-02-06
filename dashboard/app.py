from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


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


def _render_branch(branch_name: str, df: pd.DataFrame) -> None:
    st.subheader(f"{branch_name.capitalize()} Branch")
    if df.empty:
        st.warning(f"No predictions found for {branch_name}. Run predict pipeline first.")
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

    # Start-window probability bars
    start_cols = [c for c in df.columns if c.startswith("start_p_w")]
    if start_cols:
        start_map = {f"start_p_w{i}": f"W{i}" for i in range(4)}
        y = [float(latest.get(c, 0.0)) for c in start_cols]
        x = [start_map.get(c, c) for c in start_cols]
        fig_start = go.Figure(go.Bar(x=x, y=y))
        fig_start.update_layout(
            title=f"{branch_name.capitalize()} Start Window Probabilities",
            xaxis_title="Window",
            yaxis_title="Probability",
            template="plotly_white",
            height=280,
        )
        st.plotly_chart(fig_start, use_container_width=True)

    # Price + interval band
    q10_col = f"ret_h{selected_h}_q0.1"
    q50_col = f"ret_h{selected_h}_q0.5"
    q90_col = f"ret_h{selected_h}_q0.9"
    if all(c in df.columns for c in [q10_col, q50_col, q90_col, "close"]):
        chart_df = df.tail(200).copy()
        base_close = chart_df["close"]
        chart_df["pred_low"] = base_close * (1.0 + chart_df[q10_col])
        chart_df["pred_mid"] = base_close * (1.0 + chart_df[q50_col])
        chart_df["pred_high"] = base_close * (1.0 + chart_df[q90_col])

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=chart_df["timestamp_utc"],
                y=chart_df["close"],
                mode="lines",
                name="Close",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=chart_df["timestamp_utc"],
                y=chart_df["pred_mid"],
                mode="lines",
                name="Pred q50",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=chart_df["timestamp_utc"],
                y=chart_df["pred_high"],
                mode="lines",
                line=dict(width=0),
                name="q90",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=chart_df["timestamp_utc"],
                y=chart_df["pred_low"],
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                name="q10-q90 band",
            )
        )
        fig.update_layout(
            title=f"{branch_name.capitalize()} Price + Predicted Interval (h={selected_h})",
            xaxis_title="Time (UTC)",
            yaxis_title="Price",
            template="plotly_white",
            height=380,
        )
        st.plotly_chart(fig, use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="Crypto Forecast Dashboard", layout="wide")
    st.title("BTC Forecast Dashboard")
    st.caption("Direction + Start Window + Quantile Interval (Hourly / Daily)")

    processed_dir = Path("data/processed")
    hourly = _load_csv(processed_dir / "predictions_hourly.csv")
    daily = _load_csv(processed_dir / "predictions_daily.csv")

    left, right = st.columns(2)
    with left:
        _render_branch("hourly", hourly)
    with right:
        _render_branch("daily", daily)

    st.markdown("---")
    st.subheader("Model Metrics")
    metrics_path = processed_dir / "metrics_walk_forward_summary.csv"
    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path)
        st.dataframe(metrics, use_container_width=True)
    else:
        st.info("No walk-forward summary found. Run `python -m src.evaluation.walk_forward --config configs/config.yaml`.")


if __name__ == "__main__":
    main()

