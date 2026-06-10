"""
Streamlit dashboard for the Trading Lab.

Launch with:
    streamlit run dashboard.py

Features:
  - Four top-level metric cards (P&L, Win Rate, Sharpe, Max Drawdown)
  - Equity curve (cumulative PnL) — Plotly line chart
  - P&L per trade — Plotly bar chart
  - 20 most recent trades table
  - ML predictions vs actual close prices — Plotly line chart
  - Sidebar: symbol filter + date range
  - Auto-refresh every 60 seconds
"""
from __future__ import annotations

import time

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from backtester import compute_metrics
from database import get_all_predictions, get_all_trades, init_db

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    layout="wide",
    page_title="Trading Lab Dashboard",
    page_icon="📈",
)

# ---------------------------------------------------------------------------
# Ensure DB tables exist
# ---------------------------------------------------------------------------
init_db()

# ---------------------------------------------------------------------------
# Custom CSS — subtle dark-mode styling
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Main background */
    .main { background-color: #0f1117; }
    /* Metric card accent */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #1e2130 0%, #252b40 100%);
        border: 1px solid #2e3555;
        border-radius: 12px;
        padding: 16px 20px;
    }
    /* Section headers */
    h2, h3 { color: #7eb3ff; }
    /* Sidebar */
    [data-testid="stSidebar"] { background-color: #13161f; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Filters")
    st.markdown("---")

    trades_raw = get_all_trades()
    all_symbols = (
        sorted(trades_raw["symbol"].dropna().unique().tolist())
        if not trades_raw.empty and "symbol" in trades_raw.columns
        else []
    )
    symbol_filter = st.multiselect("Symbol", options=all_symbols, default=all_symbols)

    date_min = (
        trades_raw["timestamp"].min().date()
        if not trades_raw.empty and "timestamp" in trades_raw.columns
        else pd.Timestamp("2020-01-01").date()
    )
    date_max = (
        trades_raw["timestamp"].max().date()
        if not trades_raw.empty and "timestamp" in trades_raw.columns
        else pd.Timestamp.today().date()
    )
    date_range = st.date_input("Date Range", value=(date_min, date_max))

    st.markdown("---")
    st.caption("Auto-refresh: every 60 seconds")

# ---------------------------------------------------------------------------
# Data loading & filtering
# ---------------------------------------------------------------------------
def load_and_filter_trades() -> pd.DataFrame:
    df = get_all_trades()
    if df.empty:
        return df

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    if symbol_filter:
        df = df[df["symbol"].isin(symbol_filter)]

    if len(date_range) == 2:
        start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
        if "timestamp" in df.columns:
            df = df[(df["timestamp"] >= start) & (df["timestamp"] <= end)]

    return df


trades = load_and_filter_trades()
predictions_df = get_all_predictions()

metrics = compute_metrics(trades)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("📈 Trading Lab Dashboard")
st.markdown(f"*Last updated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*")
st.markdown("---")

# ---------------------------------------------------------------------------
# Metric cards
# ---------------------------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)

with c1:
    pnl_val = metrics["total_pnl"]
    st.metric(
        "💰 Total P&L",
        f"${pnl_val:,.2f}",
        delta=f"${pnl_val:+,.2f}",
    )
with c2:
    st.metric("🏆 Win Rate", f"{metrics['win_rate']:.1f}%")
with c3:
    st.metric("📊 Sharpe Ratio", f"{metrics['sharpe_ratio']:.3f}")
with c4:
    st.metric("📉 Max Drawdown", f"{metrics['max_drawdown']:.2f}%")

st.markdown("---")

# ---------------------------------------------------------------------------
# Equity Curve
# ---------------------------------------------------------------------------
st.subheader("Equity Curve")

if not metrics["equity_curve"].empty:
    eq = metrics["equity_curve"].reset_index()
    eq.columns = ["Trade #", "Cumulative P&L ($)"]
    fig_eq = px.line(
        eq,
        x="Trade #",
        y="Cumulative P&L ($)",
        title="Cumulative P&L Over Trades",
        template="plotly_dark",
        color_discrete_sequence=["#7eb3ff"],
    )
    fig_eq.update_layout(
        plot_bgcolor="#151823",
        paper_bgcolor="#151823",
        margin=dict(t=40, b=20),
    )
    fig_eq.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.4)
    st.plotly_chart(fig_eq, use_container_width=True)
else:
    st.info("No trade data yet — equity curve will appear once trades with P&L are logged.")

# ---------------------------------------------------------------------------
# P&L per Trade Bar Chart
# ---------------------------------------------------------------------------
st.subheader("P&L Per Trade")

if not trades.empty and "pnl" in trades.columns:
    pnl_df = trades.dropna(subset=["pnl"]).reset_index(drop=True)
    pnl_df["color"] = pnl_df["pnl"].apply(lambda v: "profit" if v >= 0 else "loss")
    fig_bar = px.bar(
        pnl_df,
        x=pnl_df.index,
        y="pnl",
        color="color",
        color_discrete_map={"profit": "#4caf87", "loss": "#e05c5c"},
        labels={"x": "Trade #", "pnl": "P&L ($)", "color": ""},
        title="Individual Trade P&L",
        template="plotly_dark",
    )
    fig_bar.update_layout(
        plot_bgcolor="#151823",
        paper_bgcolor="#151823",
        margin=dict(t=40, b=20),
        showlegend=False,
    )
    fig_bar.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    st.plotly_chart(fig_bar, use_container_width=True)
else:
    st.info("No P&L data available yet.")

# ---------------------------------------------------------------------------
# ML Predictions vs Actual Close Prices
# ---------------------------------------------------------------------------
st.subheader("ML Predictions vs Actual Close")

if predictions_df is not None and not predictions_df.empty:
    pred_plot = predictions_df.copy()
    if symbol_filter:
        pred_plot = pred_plot[pred_plot["symbol"].isin(symbol_filter)]

    if "timestamp" in pred_plot.columns:
        pred_plot["timestamp"] = pd.to_datetime(pred_plot["timestamp"])
        pred_plot = pred_plot.sort_values("timestamp")

    if not pred_plot.empty:
        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(
            x=pred_plot["timestamp"],
            y=pred_plot["close_price"],
            mode="lines",
            name="Actual Close",
            line=dict(color="#7eb3ff"),
        ))
        fig_pred.add_trace(go.Scatter(
            x=pred_plot["timestamp"],
            y=pred_plot["predicted_price"],
            mode="lines",
            name="ML Prediction",
            line=dict(color="#f4b942", dash="dot"),
        ))
        fig_pred.update_layout(
            title="Model Predictions vs Actual Price",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            template="plotly_dark",
            plot_bgcolor="#151823",
            paper_bgcolor="#151823",
            margin=dict(t=40, b=20),
        )
        st.plotly_chart(fig_pred, use_container_width=True)
    else:
        st.info("No predictions found for the selected symbol / date range.")
else:
    st.info("No prediction data available yet.")

st.markdown("---")

# ---------------------------------------------------------------------------
# Recent Trades Table
# ---------------------------------------------------------------------------
st.subheader("20 Most Recent Trades")

if not trades.empty:
    cols_to_show = [
        c for c in ["timestamp", "symbol", "signal", "side", "entry_price",
                     "exit_price", "qty", "pnl", "ml_prediction"]
        if c in trades.columns
    ]
    recent = trades.sort_values("timestamp", ascending=False).head(20)[cols_to_show]
    st.dataframe(recent, use_container_width=True)
else:
    st.info("No trades logged yet.")

# ---------------------------------------------------------------------------
# Auto-refresh countdown (60 seconds)
# ---------------------------------------------------------------------------
st.markdown("---")
placeholder = st.empty()
for remaining in range(60, 0, -1):
    placeholder.caption(f"⏱ Auto-refreshing in {remaining}s…")
    time.sleep(1)

st.rerun()
