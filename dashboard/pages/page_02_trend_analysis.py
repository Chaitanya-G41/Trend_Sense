"""
TrendSense Dashboard — Page 2: Trend Analysis
================================================
Google Trends visualization, TVI computation, and spike detection.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os, sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, project_root)
import config
from src.tvi import compute_tvi_features, get_tvi_summary


def render():
    """Render the Trend Analysis page."""
    
    st.markdown("""
    <div class="main-header">
        <h1>📊 Trend Analysis</h1>
        <p>Google Trends interest, Trend Velocity Index (TVI), and spike detection</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load trends data
    trends_df = load_trends_data()
    
    if trends_df is None:
        st.warning("⚠️ No trends data found. Run the pipeline first:")
        st.code("python run_pipeline.py --synthetic-trends", language="bash")
        return
    
    # Keyword selector
    keywords = [col for col in trends_df.columns if col != "date"]
    
    if not keywords:
        st.error("No keyword data found in trends dataset.")
        return
    
    selected_keyword = st.selectbox(
        "🔍 Select Trend Keyword",
        keywords,
        index=0,
    )
    
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    
    # Compute TVI
    tvi_feat = compute_tvi_features(trends_df, selected_keyword)
    summary = get_tvi_summary(tvi_feat)
    
    # ─── TVI Summary KPIs ───
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("📈 Mean TVI", f"{summary['mean_tvi']:.1f}%")
    with col2:
        st.metric("📊 Std TVI", f"{summary['std_tvi']:.1f}%")
    with col3:
        st.metric("🔥 Total Spikes", summary['total_spikes'])
    with col4:
        st.metric("🔴 Severe Spikes", summary['severe_spikes'])
    with col5:
        st.metric("📉 Spike Rate", f"{summary['spike_rate_pct']}%")
    
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    
    # ─── Main Visualization: 3-panel chart ───
    st.subheader(f"📈 Trend Analysis — '{selected_keyword}'")
    
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=[
            "Google Trends Search Interest (0-100)",
            "Trend Velocity Index (TVI) — Rate of Change",
            "Spike Detection & Severity",
        ],
        row_heights=[0.3, 0.35, 0.35],
    )
    
    dates = tvi_feat["date"]
    
    # Panel 1: Raw trend
    fig.add_trace(go.Scatter(
        x=dates, y=tvi_feat["raw_trend"],
        mode="lines",
        name="Search Interest",
        line=dict(color="#3B82F6", width=2),
        fill="tozeroy",
        fillcolor="rgba(59, 130, 246, 0.1)",
    ), row=1, col=1)
    
    # Add smoothed trend
    smoothed = tvi_feat["raw_trend"].rolling(window=4, min_periods=1).mean()
    fig.add_trace(go.Scatter(
        x=dates, y=smoothed,
        mode="lines",
        name="4-Week Moving Avg",
        line=dict(color="#F59E0B", width=2, dash="dash"),
    ), row=1, col=1)
    
    # Panel 2: TVI bars
    tvi_vals = tvi_feat["tvi"].fillna(0)
    colors_tvi = ["#10B981" if v >= 0 else "#EF4444" for v in tvi_vals]
    
    fig.add_trace(go.Bar(
        x=dates, y=tvi_vals,
        name="TVI",
        marker_color=colors_tvi,
        opacity=0.75,
    ), row=2, col=1)
    
    # Add TVI smoothed line
    fig.add_trace(go.Scatter(
        x=dates, y=tvi_feat["tvi_smoothed"].fillna(0),
        mode="lines",
        name="Smoothed TVI",
        line=dict(color="#F59E0B", width=2),
    ), row=2, col=1)
    
    # Zero line
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=2, col=1)
    
    # Panel 3: Spike scatter
    severity_colors = {
        "NONE": "rgba(148, 163, 184, 0.3)",
        "MILD": "#F59E0B",
        "MODERATE": "#F97316",
        "SEVERE": "#EF4444",
    }
    severity_sizes = {"NONE": 4, "MILD": 10, "MODERATE": 15, "SEVERE": 22}
    
    for severity in ["NONE", "MILD", "MODERATE", "SEVERE"]:
        mask = tvi_feat["spike_severity"] == severity
        if mask.any():
            fig.add_trace(go.Scatter(
                x=dates[mask],
                y=tvi_vals[mask],
                mode="markers",
                name=f"{severity} Spike",
                marker=dict(
                    color=severity_colors[severity],
                    size=severity_sizes[severity],
                    line=dict(width=1, color="white") if severity != "NONE" else dict(width=0),
                ),
            ), row=3, col=1)
    
    # Threshold lines
    mean_tvi = summary["mean_tvi"]
    std_tvi = summary["std_tvi"]
    for sigma, label, color in [(2, "+2σ (Moderate)", "#F97316"), (3, "+3σ (Severe)", "#EF4444")]:
        threshold = mean_tvi + sigma * std_tvi
        fig.add_hline(y=threshold, line_dash="dot", line_color=color, opacity=0.6,
                      annotation_text=label, row=3, col=1)
    
    fig.update_layout(
        template="plotly_white",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=900,
        margin=dict(l=20, r=20, t=40, b=20),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    
    for i in range(1, 4):
        fig.update_xaxes(gridcolor="rgba(0,0,0,0.05)", row=i, col=1)
        fig.update_yaxes(gridcolor="rgba(0,0,0,0.05)", row=i, col=1)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # ─── Spike Timeline ───
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    st.subheader("🔥 Spike Timeline")
    
    spikes = tvi_feat[tvi_feat["is_spike"]].copy()
    
    if len(spikes) > 0:
        spikes_display = spikes[["date", "raw_trend", "tvi", "spike_severity"]].copy()
        spikes_display.columns = ["Date", "Search Interest", "TVI (%)", "Severity"]
        spikes_display = spikes_display.sort_values("Date", ascending=False)
        
        # Color-code severity
        def severity_badge(severity):
            colors = {"MILD": "🟡", "MODERATE": "🟠", "SEVERE": "🔴"}
            return f"{colors.get(severity, '⚪')} {severity}"
        
        spikes_display["Severity"] = spikes_display["Severity"].apply(severity_badge)
        spikes_display["TVI (%)"] = spikes_display["TVI (%)"].round(1)
        
        st.dataframe(
            spikes_display.head(20),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No spikes detected for this keyword with current thresholds.")
    
    # ─── Multi-keyword Comparison ───
    if len(keywords) > 1:
        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
        st.subheader("📊 Multi-Keyword Comparison")
        
        compare_keywords = st.multiselect(
            "Select keywords to compare",
            keywords,
            default=keywords[:min(4, len(keywords))],
        )
        
        if compare_keywords:
            fig_compare = go.Figure()
            
            colors_list = ["#6366F1", "#F59E0B", "#10B981", "#EF4444", "#8B5CF6",
                          "#3B82F6", "#EC4899", "#14B8A6"]
            
            for idx, kw in enumerate(compare_keywords):
                color = colors_list[idx % len(colors_list)]
                fig_compare.add_trace(go.Scatter(
                    x=trends_df["date"],
                    y=trends_df[kw],
                    mode="lines",
                    name=kw,
                    line=dict(color=color, width=2),
                ))
            
            fig_compare.update_layout(
                template="plotly_white",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                height=400,
                title="Search Interest Comparison",
                xaxis=dict(gridcolor="rgba(0,0,0,0.05)"),
                yaxis=dict(title="Interest (0-100)", gridcolor="rgba(0,0,0,0.05)"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            
            st.plotly_chart(fig_compare, use_container_width=True)


@st.cache_data(ttl=300)
def load_trends_data():
    """Load Google Trends data."""
    trends_path = os.path.join(config.TRENDS_DATA_DIR, "google_trends_india.csv")
    if os.path.exists(trends_path):
        return pd.read_csv(trends_path, parse_dates=["date"])
    return None
