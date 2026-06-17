"""
TrendSense Dashboard — Page 4: Decision Support
===================================================
The core DSS interface: HOLD / INCREASE STOCK / URGENT RESTOCK
with confidence scores, TVI status, and historical decisions.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os, sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, project_root)
import config
from src.decision_engine import generate_decision, generate_batch_decisions, get_decision_summary
from src.tvi import compute_tvi_features, get_tvi_summary


def render():
    """Render the Decision Support page."""
    
    st.markdown("""
    <div class="main-header">
        <h1>🎯 Decision Support System</h1>
        <p>AI-powered inventory recommendations — HOLD / INCREASE STOCK / URGENT RESTOCK</p>
    </div>
    """, unsafe_allow_html=True)
    
    category = st.session_state.get("selected_category", "General")
    
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    
    # ─── Interactive Decision Simulator ───
    st.subheader("⚙️ Decision Simulator")
    st.markdown("*Adjust parameters to see how the decision engine responds in real-time*")
    
    sim_col1, sim_col2, sim_col3 = st.columns(3)
    
    with sim_col1:
        current_stock = st.number_input(
            "📦 Current Stock (units)",
            min_value=100, max_value=100000, value=10000, step=500,
            help="Current inventory level in units"
        )
        predicted_demand = st.number_input(
            "📈 Predicted Demand (units)",
            min_value=100, max_value=100000, value=12000, step=500,
            help="ML model's predicted weekly demand"
        )
    
    with sim_col2:
        model_confidence = st.slider(
            "🎯 Model Confidence",
            min_value=0.5, max_value=0.99, value=0.85, step=0.01,
            help="Confidence score from the ML model"
        )
        sim_category = st.selectbox(
            "📂 Product Category",
            list(config.DEFAULT_CATEGORY_LAGS.keys()),
            index=list(config.DEFAULT_CATEGORY_LAGS.keys()).index(category)
            if category in config.DEFAULT_CATEGORY_LAGS else 0,
        )
    
    with sim_col3:
        tvi_spike = st.checkbox("⚡ TVI Spike Active", value=False)
        if tvi_spike:
            tvi_severity = st.selectbox("Spike Severity", ["MILD", "MODERATE", "SEVERE"], index=1)
            tvi_value = st.slider("TVI Value", -50.0, 100.0, 45.0)
        else:
            tvi_severity = "NONE"
            tvi_value = 0.0
    
    # Generate decision
    tvi_status = {
        "is_spike": tvi_spike,
        "severity": tvi_severity,
        "tvi_value": tvi_value,
    }
    
    decision = generate_decision(
        predicted_demand=predicted_demand,
        current_stock=current_stock,
        tvi_status=tvi_status,
        model_confidence=model_confidence,
        category=sim_category,
    )
    
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    
    # ─── Decision Output Card ───
    action = decision["action"]
    confidence = decision["confidence_pct"]
    change_pct = decision["predicted_change_pct"]
    
    card_classes = {
        "HOLD": ("decision-hold", "🟢", "#10B981"),
        "INCREASE STOCK": ("decision-increase", "🟡", "#F59E0B"),
        "URGENT RESTOCK": ("decision-urgent", "🔴", "#EF4444"),
    }
    
    card_class, icon, accent_color = card_classes.get(action, ("decision-hold", "⚪", "#94A3B8"))
    
    st.markdown(f"""
    <div class="{card_class}" style="text-align: center;">
        <div style="font-size: 3.5rem; margin-bottom: 8px;">{icon}</div>
        <div style="font-size: 2.2rem; font-weight: 800; letter-spacing: 2px;">{action}</div>
        <div style="font-size: 1.3rem; margin-top: 8px; opacity: 0.9;">
            Confidence: <span style="font-weight: 800; font-size: 1.5rem;">{confidence}</span>
        </div>
        <hr style="border-color: rgba(255,255,255,0.2); margin: 16px 40px;">
        <div style="display: flex; justify-content: center; gap: 48px; flex-wrap: wrap;">
            <div>
                <div style="font-size: 0.8rem; opacity: 0.7; text-transform: uppercase;">Predicted Demand</div>
                <div style="font-size: 1.4rem; font-weight: 700;">{decision['predicted_demand']:,.0f}</div>
            </div>
            <div>
                <div style="font-size: 0.8rem; opacity: 0.7; text-transform: uppercase;">Current Stock</div>
                <div style="font-size: 1.4rem; font-weight: 700;">{decision['current_stock']:,.0f}</div>
            </div>
            <div>
                <div style="font-size: 0.8rem; opacity: 0.7; text-transform: uppercase;">Change</div>
                <div style="font-size: 1.4rem; font-weight: 700;">{change_pct:+.1f}%</div>
            </div>
            <div>
                <div style="font-size: 0.8rem; opacity: 0.7; text-transform: uppercase;">Category</div>
                <div style="font-size: 1.4rem; font-weight: 700;">{sim_category}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Rationale expander
    with st.expander("📋 Decision Rationale", expanded=True):
        st.markdown(f"**{decision['rationale']}**")
        
        if decision["spike_boost_applied"]:
            st.warning(f"⚡ Urgency was boosted by a **{tvi_severity}** TVI spike (TVI = {tvi_value:.1f})")
    
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    
    # ─── Decision Threshold Visualization ───
    st.subheader("📊 Decision Thresholds")
    
    col_gauge, col_explain = st.columns([2, 1])
    
    with col_gauge:
        # Gauge chart showing where current prediction falls
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=change_pct,
            title={"text": f"Predicted Demand Change", "font": {"size": 16, "color": "#F8FAFC"}},
            number={"suffix": "%", "font": {"color": "#F8FAFC"}},
            delta={"reference": 0, "increasing": {"color": "#EF4444"}, "decreasing": {"color": "#10B981"}},
            gauge={
                "axis": {"range": [-20, 80], "tickcolor": "#94A3B8"},
                "bar": {"color": accent_color},
                "bgcolor": "#1E293B",
                "bordercolor": "#334155",
                "steps": [
                    {"range": [-20, 10], "color": "rgba(16, 185, 129, 0.2)"},
                    {"range": [10, 40], "color": "rgba(245, 158, 11, 0.2)"},
                    {"range": [40, 80], "color": "rgba(239, 68, 68, 0.2)"},
                ],
                "threshold": {
                    "line": {"color": "#F8FAFC", "width": 3},
                    "thickness": 0.8,
                    "value": change_pct,
                },
            },
        ))
        
        fig_gauge.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font={"color": "#F8FAFC"},
            height=300,
            margin=dict(l=30, r=30, t=60, b=20),
        )
        
        st.plotly_chart(fig_gauge, use_container_width=True)
    
    with col_explain:
        st.markdown("""
        **Decision Zones:**
        
        🟢 **HOLD** — Change < 10%  
        Demand is stable. Maintain current inventory levels.
        
        🟡 **INCREASE STOCK** — 10% ≤ Change < 40%  
        Moderate growth expected. Gradually increase inventory.
        
        🔴 **URGENT RESTOCK** — Change ≥ 40%  
        Significant surge detected. Immediate restocking needed.
        
        ⚡ **TVI Spike Boost**  
        When a MODERATE or SEVERE TVI spike is detected, urgency is boosted by one level.
        """)
    
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    
    # ─── Batch Decisions (Simulation) ───
    st.subheader("📋 Batch Decision Simulation")
    st.markdown("*Simulating decisions across multiple scenarios*")
    
    np.random.seed(42)
    n_scenarios = 12
    
    scenarios_stock = np.random.uniform(5000, 20000, n_scenarios)
    scenarios_demand = scenarios_stock * (1 + np.random.uniform(-0.15, 0.60, n_scenarios))
    
    tvi_statuses = []
    for _ in range(n_scenarios):
        is_spike = np.random.random() < 0.25
        sev = np.random.choice(["MILD", "MODERATE", "SEVERE"], p=[0.5, 0.35, 0.15]) if is_spike else "NONE"
        tvi_statuses.append({"is_spike": is_spike, "severity": sev, "tvi_value": np.random.normal(0, 30) if is_spike else 0})
    
    categories = np.random.choice(list(config.DEFAULT_CATEGORY_LAGS.keys()), n_scenarios).tolist()
    
    batch_df = generate_batch_decisions(
        predictions=scenarios_demand,
        current_stocks=scenarios_stock,
        tvi_statuses=tvi_statuses,
        model_confidence=model_confidence,
        categories=categories,
    )
    
    summary = get_decision_summary(batch_df)
    
    # Summary cards
    sum_c1, sum_c2, sum_c3, sum_c4 = st.columns(4)
    with sum_c1:
        st.metric("🟢 HOLD", f"{summary['hold_count']} ({summary['hold_pct']}%)")
    with sum_c2:
        st.metric("🟡 INCREASE", f"{summary['increase_count']} ({summary['increase_pct']}%)")
    with sum_c3:
        st.metric("🔴 URGENT", f"{summary['urgent_count']} ({summary['urgent_pct']}%)")
    with sum_c4:
        st.metric("📊 Avg Confidence", f"{summary['avg_confidence']}%")
    
    # Batch results table
    display_df = batch_df[["category", "action", "confidence_pct", "predicted_demand",
                            "current_stock", "predicted_change_pct", "tvi_severity",
                            "spike_boost_applied"]].copy()
    display_df.columns = ["Category", "Decision", "Confidence", "Pred. Demand",
                          "Current Stock", "Change %", "TVI Status", "Spike Boost"]
    display_df["Pred. Demand"] = display_df["Pred. Demand"].apply(lambda x: f"{x:,.0f}")
    display_df["Current Stock"] = display_df["Current Stock"].apply(lambda x: f"{x:,.0f}")
    display_df["Change %"] = display_df["Change %"].apply(lambda x: f"{x:+.1f}%")
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    # Decision distribution donut chart
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    
    col_donut, col_lag = st.columns(2)
    
    with col_donut:
        st.subheader("📊 Decision Distribution")
        
        labels = ["HOLD", "INCREASE STOCK", "URGENT RESTOCK"]
        values = [summary["hold_count"], summary["increase_count"], summary["urgent_count"]]
        colors_pie = ["#10B981", "#F59E0B", "#EF4444"]
        
        fig_donut = go.Figure(data=[go.Pie(
            labels=labels, values=values,
            hole=0.55,
            marker=dict(colors=colors_pie, line=dict(color="#0F172A", width=3)),
            textinfo="label+percent",
            textfont=dict(size=13, color="#F8FAFC"),
        )])
        
        fig_donut.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#F8FAFC"),
            height=350,
            showlegend=False,
            margin=dict(l=20, r=20, t=20, b=20),
        )
        
        st.plotly_chart(fig_donut, use_container_width=True)
    
    with col_lag:
        st.subheader("⏱️ Category Lag Periods")
        st.markdown("*Optimal delay between social signal and purchase event*")
        
        lag_data = config.DEFAULT_CATEGORY_LAGS
        
        fig_lag = go.Figure()
        categories_list = list(lag_data.keys())
        lags_list = list(lag_data.values())
        
        colors_lag = ["#6366F1", "#8B5CF6", "#A78BFA", "#C4B5FD", "#DDD6FE"]
        
        fig_lag.add_trace(go.Bar(
            x=categories_list,
            y=lags_list,
            marker=dict(color=colors_lag[:len(categories_list)],
                       line=dict(color="#F8FAFC", width=1)),
            text=[f"{l}d" for l in lags_list],
            textposition="outside",
            textfont=dict(color="#F8FAFC", size=14, family="Inter"),
        ))
        
        fig_lag.update_layout(
            template="plotly_dark",
            plot_bgcolor="rgba(15, 23, 42, 0.8)",
            paper_bgcolor="rgba(0,0,0,0)",
            height=350,
            xaxis=dict(gridcolor="rgba(148,163,184,0.1)", title="Category"),
            yaxis=dict(gridcolor="rgba(148,163,184,0.1)", title="Lag (days)"),
            margin=dict(l=20, r=20, t=20, b=20),
        )
        
        st.plotly_chart(fig_lag, use_container_width=True)
    
    # ─── Confidence Distribution ───
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    st.subheader("🎯 Confidence Score Analysis")
    
    fig_conf = go.Figure()
    
    for action, color in [("HOLD", "#10B981"), ("INCREASE STOCK", "#F59E0B"), ("URGENT RESTOCK", "#EF4444")]:
        mask = batch_df["action"] == action
        if mask.any():
            fig_conf.add_trace(go.Box(
                y=batch_df.loc[mask, "confidence"] * 100,
                name=action,
                marker_color=color,
                boxmean=True,
            ))
    
    fig_conf.update_layout(
        template="plotly_dark",
        plot_bgcolor="rgba(15, 23, 42, 0.8)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=300,
        yaxis=dict(title="Confidence (%)", gridcolor="rgba(148,163,184,0.1)"),
        xaxis=dict(gridcolor="rgba(148,163,184,0.1)"),
        margin=dict(l=20, r=20, t=20, b=20),
    )
    
    st.plotly_chart(fig_conf, use_container_width=True)
