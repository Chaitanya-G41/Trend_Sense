"""
TrendSense Dashboard — Page 1: Overview
=========================================
KPI cards, sales trend, and recent alerts.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os, sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, project_root)
import config


def render():
    """Render the Overview page."""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>Overview</h1>
        <p>AI-Powered Social Media Trend Prediction for Business Decision Making</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    walmart_df, trends_df = load_data()
    
    if walmart_df is None:
        st.warning("No data available. Please run the pipeline first: `python run_pipeline.py`")
        st.code("python run_pipeline.py --synthetic-trends", language="bash")
        return
    
    # ─── KPI Cards ───
    col1, col2, col3, col4 = st.columns(4)
    
    total_stores = walmart_df["Store"].nunique() if "Store" in walmart_df.columns else 1
    avg_sales = walmart_df["Weekly_Sales"].mean()
    total_sales = walmart_df["Weekly_Sales"].sum()
    holiday_impact = walmart_df[walmart_df["Holiday_Flag"] == 1]["Weekly_Sales"].mean() - avg_sales if "Holiday_Flag" in walmart_df.columns else 0
    
    with col1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-header">
                Total Stores
            </div>
            <div class="kpi-value">{total_stores}</div>
            <div class="kpi-subtext">Active locations</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-header">
                Avg Weekly Sales
            </div>
            <div class="kpi-value">${avg_sales:,.0f}</div>
            <div class="kpi-subtext">Per Store</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-header">
                Total Revenue
            </div>
            <div class="kpi-value">${total_sales/1e6:,.1f}M</div>
            <div class="kpi-subtext">All Stores</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        delta_sign = "+" if holiday_impact > 0 else ""
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-header">
                Holiday Impact
            </div>
            <div class="kpi-value">{delta_sign}${holiday_impact:,.0f}</div>
            <div class="kpi-subtext">Avg Boost</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    
    # ─── Sales Trend Chart ───
    col_left, col_right = st.columns([2, 1], gap="large")
    
    with col_left:
        st.subheader("Weekly Sales Trend")
        
        # Aggregate by date
        if "Store" in walmart_df.columns:
            agg_df = walmart_df.groupby("Date").agg(
                Weekly_Sales=("Weekly_Sales", "mean"),
                Holiday_Flag=("Holiday_Flag", "max")
            ).reset_index()
        else:
            agg_df = walmart_df.copy()
        
        fig = go.Figure()
        
        # Main sales line
        fig.add_trace(go.Scatter(
            x=agg_df["Date"], y=agg_df["Weekly_Sales"],
            mode="lines",
            name="Avg Weekly Sales",
            line=dict(color="#7C3AED", width=2.5, shape="spline", smoothing=0.5),
            fill="tonexty" if len(agg_df) < 500 else None,
            fillcolor="rgba(124, 58, 237, 0.08)",
        ))
        
        # Holiday markers
        if "Holiday_Flag" in agg_df.columns:
            holidays = agg_df[agg_df["Holiday_Flag"] == 1]
            fig.add_trace(go.Scatter(
                x=holidays["Date"], y=holidays["Weekly_Sales"],
                mode="markers",
                name="Holiday Weeks",
                marker=dict(color="#A78BFA", size=8, symbol="circle", line=dict(color="#FFFFFF", width=1.5)),
            ))
        
        fig.update_layout(
            template="plotly_white",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            height=400,
            margin=dict(l=0, r=0, t=10, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, font=dict(color="#6B7280")),
            xaxis=dict(gridcolor="#F3F4F6", title=""),
            yaxis=dict(gridcolor="#F3F4F6", title="Sales ($)", color="#6B7280"),
            hovermode="x unified",
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col_right:
        st.subheader("Sales Distribution")
        
        fig2 = go.Figure()
        fig2.add_trace(go.Histogram(
            x=walmart_df["Weekly_Sales"],
            nbinsx=40,
            marker_color="#A78BFA",
            opacity=0.8,
        ))
        fig2.update_layout(
            template="plotly_white",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            height=400,
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis=dict(title="Weekly Sales ($)", gridcolor="#F3F4F6", color="#6B7280"),
            yaxis=dict(title="Frequency", gridcolor="#F3F4F6", color="#6B7280"),
            bargap=0.1,
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    # ─── Store Performance ───
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    st.subheader("Store Performance Comparison")
    
    if "Store" in walmart_df.columns:
        store_perf = walmart_df.groupby("Store")["Weekly_Sales"].agg(["mean", "std", "min", "max"]).reset_index()
        store_perf.columns = ["Store", "Avg Sales", "Std Dev", "Min Sales", "Max Sales"]
        store_perf = store_perf.sort_values("Avg Sales", ascending=False)
        
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(
            x=store_perf["Store"].astype(str),
            y=store_perf["Avg Sales"],
            marker=dict(
                color=store_perf["Avg Sales"],
                colorscale=[[0, "#C4B5FD"], [1, "#7C3AED"]],
                showscale=False,
            ),
            text=store_perf["Avg Sales"].apply(lambda x: f"${x/1000:,.0f}k"),
            textposition="outside",
            textfont=dict(color="#6B7280", size=10)
        ))
        
        fig3.update_layout(
            template="plotly_white",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            height=300,
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis=dict(title="Store ID", gridcolor="#F3F4F6", color="#6B7280"),
            yaxis=dict(title="Avg Weekly Sales ($)", gridcolor="#F3F4F6", color="#6B7280"),
            bargap=0.2,
        )
        st.plotly_chart(fig3, use_container_width=True)
    
    # ─── Recent TVI Alerts ───
    if trends_df is not None and len(trends_df) > 0:
        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
        st.subheader("Recent Trend Alerts")
        
        from src.tvi import compute_tvi_features
        
        keywords = [k for k in config.TREND_KEYWORDS.keys() if k in trends_df.columns]
        
        alert_cols = st.columns(min(len(keywords), 4))
        for idx, keyword in enumerate(keywords[:4]):
            with alert_cols[idx]:
                tvi_feat = compute_tvi_features(trends_df, keyword)
                recent_spikes = tvi_feat[tvi_feat["is_spike"]].tail(3)
                
                spike_count = len(tvi_feat[tvi_feat["is_spike"]])
                latest_tvi = tvi_feat["tvi"].dropna().iloc[-1] if len(tvi_feat["tvi"].dropna()) > 0 else 0
                
                status_icon = "[OK]" if abs(latest_tvi) < 10 else ("[WARN]" if abs(latest_tvi) < 30 else "[ALERT]")
                
                st.metric(
                    label=f"{status_icon} {keyword}",
                    value=f"{latest_tvi:.1f}%",
                    delta=f"{spike_count} spikes detected",
                )


@st.cache_data(ttl=300)
def load_data():
    """Load Walmart and Trends data from processed or raw directories."""
    walmart_df = None
    trends_df = None
    
    # Try processed data first
    processed_path = os.path.join(config.PROCESSED_DATA_DIR, "walmart_featured.csv")
    if os.path.exists(processed_path):
        walmart_df = pd.read_csv(processed_path, parse_dates=["Date"])
    else:
        # Try raw data
        raw_path = os.path.join(config.RAW_DATA_DIR, "Walmart.csv")
        if os.path.exists(raw_path):
            walmart_df = pd.read_csv(raw_path)
            try:
                walmart_df["Date"] = pd.to_datetime(walmart_df["Date"], format="%d-%m-%Y")
            except (ValueError, TypeError):
                walmart_df["Date"] = pd.to_datetime(walmart_df["Date"], dayfirst=True)
    
    # Load trends
    trends_path = os.path.join(config.TRENDS_DATA_DIR, "google_trends_india.csv")
    if os.path.exists(trends_path):
        trends_df = pd.read_csv(trends_path, parse_dates=["date"])
    
    return walmart_df, trends_df
