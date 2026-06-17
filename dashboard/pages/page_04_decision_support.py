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
        <h1><span class="material-symbols-outlined" style="margin-right: 12px; color: #7C3AED;">lightbulb</span> Decision Support System</h1>
        <p>AI-powered inventory recommendations — HOLD / INCREASE STOCK / URGENT RESTOCK</p>
    </div>
    """, unsafe_allow_html=True)
    
    category = st.session_state.get("selected_category", "General")
    
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    
    # ─── Interactive Decision Simulator ───
    st.subheader(":material/settings: Decision Simulator")
    st.markdown("*Adjust parameters to see how the decision engine responds in real-time*")
    
    sim_col1, sim_col2, sim_col3 = st.columns(3)
    
    with sim_col1:
        current_stock = st.number_input(
            " Current Stock (units)",
            min_value=100, max_value=100000, value=10000, step=500,
            help="Current inventory level in units"
        )
        predicted_demand = st.number_input(
            " Predicted Demand (units)",
            min_value=100, max_value=100000, value=12000, step=500,
            help="ML model's predicted weekly demand"
        )
    
    with sim_col2:
        model_confidence = st.slider(
            " Model Confidence",
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
        tvi_spike = st.checkbox(" TVI Spike Active", value=False)
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
        "HOLD": ("decision-hold", "check_circle", "#10B981"),
        "INCREASE STOCK": ("decision-increase", "trending_up", "#F59E0B"),
        "URGENT RESTOCK": ("decision-urgent", "warning", "#EF4444"),
    }
    
    card_class, icon, accent_color = card_classes.get(action, ("decision-hold", "help", "#9CA3AF"))
    
    st.markdown(f"""
    <div class="{card_class}" style="text-align: center; margin-bottom: 24px;">
        <div style="margin-bottom: 12px;"><span class="material-symbols-outlined" style="font-size: 3.5rem; color: {accent_color};">{icon}</span></div>
        <div style="font-size: 2rem; font-weight: 800; letter-spacing: 0.05em; color: #111827;">{action}</div>
        <div style="font-size: 1.1rem; margin-top: 8px; color: #6B7280;">
            Confidence: <span style="font-weight: 700; color: #111827;">{confidence}</span>
        </div>
        <hr style="border: none; height: 1px; background-color: rgba(0,0,0,0.05); margin: 24px 40px;">
        <div style="display: flex; justify-content: center; gap: 48px; flex-wrap: wrap;">
            <div>
                <div style="font-size: 0.75rem; color: #6B7280; text-transform: uppercase;">Predicted Demand</div>
                <div style="font-size: 1.25rem; font-weight: 600; color: #111827;">{decision['predicted_demand']:,.0f}</div>
            </div>
            <div>
                <div style="font-size: 0.75rem; color: #6B7280; text-transform: uppercase;">Current Stock</div>
                <div style="font-size: 1.25rem; font-weight: 600; color: #111827;">{decision['current_stock']:,.0f}</div>
            </div>
            <div>
                <div style="font-size: 0.75rem; color: #6B7280; text-transform: uppercase;">Change</div>
                <div style="font-size: 1.25rem; font-weight: 600; color: #111827;">{change_pct:+.1f}%</div>
            </div>
            <div>
                <div style="font-size: 0.75rem; color: #6B7280; text-transform: uppercase;">Category</div>
                <div style="font-size: 1.25rem; font-weight: 600; color: #111827;">{sim_category}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Rationale expander
    with st.expander("Decision Rationale", expanded=True):
        st.markdown(f"**{decision['rationale']}**")
        
        if decision["spike_boost_applied"]:
            st.warning(f":material/warning: Urgency was boosted by a **{tvi_severity}** TVI spike (TVI = {tvi_value:.1f})")
    
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    
    # ─── Decision Threshold Visualization ───
    st.subheader(":material/rule: Decision Thresholds")
    
    col_gauge, col_explain = st.columns([2, 1])
    
    with col_gauge:
        # Gauge chart showing where current prediction falls
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=change_pct,
            title={"text": f"Predicted Demand Change", "font": {"size": 16, "color": "#1E293B"}},
            number={"suffix": "%", "font": {"color": "#1E293B"}},
            delta={"reference": 0, "increasing": {"color": "#EF4444"}, "decreasing": {"color": "#10B981"}},
            gauge={
                "axis": {"range": [-20, 80], "tickcolor": "#475569"},
                "bar": {"color": accent_color},
                "bgcolor": "#F1F5F9",
                "bordercolor": "#E2E8F0",
                "steps": [
                    {"range": [-20, 10], "color": "rgba(16, 185, 129, 0.15)"},
                    {"range": [10, 40], "color": "rgba(245, 158, 11, 0.15)"},
                    {"range": [40, 80], "color": "rgba(239, 68, 68, 0.15)"},
                ],
                "threshold": {
                    "line": {"color": "#1E293B", "width": 3},
                    "thickness": 0.8,
                    "value": change_pct,
                },
            },
        ))
        
        fig_gauge.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font={"color": "#111827", "family": "Inter"},
            height=300,
            margin=dict(l=30, r=30, t=60, b=20),
        )
        
        st.plotly_chart(fig_gauge, use_container_width=True)
    
    with col_explain:
        st.markdown("""
        **Decision Zones:**
        
        <span style="color:#10B981; font-weight:bold;">HOLD</span> — Change < 10%  
        Demand is stable. Maintain current inventory levels.
        
        <span style="color:#F59E0B; font-weight:bold;">INCREASE STOCK</span> — 10% ≤ Change < 40%  
        Moderate growth expected. Gradually increase inventory.
        
        <span style="color:#EF4444; font-weight:bold;">URGENT RESTOCK</span> — Change ≥ 40%  
        Significant surge detected. Immediate restocking needed.
        
        **TVI Spike Boost**  
        When a MODERATE or SEVERE TVI spike is detected, urgency is boosted by one level.
        """, unsafe_allow_html=True)
    
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    
    # ─── Batch Decisions (Simulation) ───
    st.subheader(":material/fact_check: Batch Decision Simulation")
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
        st.metric(":material/check_circle: HOLD", f"{summary['hold_count']} ({summary['hold_pct']}%)")
    with sum_c2:
        st.metric(":material/trending_up: INCREASE", f"{summary['increase_count']} ({summary['increase_pct']}%)")
    with sum_c3:
        st.metric(":material/warning: URGENT", f"{summary['urgent_count']} ({summary['urgent_pct']}%)")
    with sum_c4:
        st.metric(":material/psychology: Avg Confidence", f"{summary['avg_confidence']}%")
    
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
    
    col_donut, col_lag = st.columns(2, gap="large")
    
    with col_donut:
        st.subheader(":material/pie_chart: Decision Distribution")
        
        labels = ["HOLD", "INCREASE STOCK", "URGENT RESTOCK"]
        values = [summary["hold_count"], summary["increase_count"], summary["urgent_count"]]
        colors_pie = ["#10B981", "#F59E0B", "#EF4444"]
        
        fig_donut = go.Figure(data=[go.Pie(
            labels=labels, values=values,
            hole=0.55,
            marker=dict(colors=colors_pie, line=dict(color="#FFFFFF", width=3)),
            textinfo="label+percent",
            textfont=dict(size=13, color="#1E293B"),
        )])
        
        fig_donut.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#111827", family="Inter"),
            height=350,
            showlegend=False,
            margin=dict(l=0, r=0, t=20, b=0),
        )
        
        st.plotly_chart(fig_donut, use_container_width=True)
    
    with col_lag:
        st.subheader(":material/timer: Category Lag Periods")
        st.markdown("*Optimal delay between social signal and purchase event*")
        
        lag_data = config.DEFAULT_CATEGORY_LAGS
        
        fig_lag = go.Figure()
        categories_list = list(lag_data.keys())
        lags_list = list(lag_data.values())
        
        colors_lag = ["#7C3AED", "#8B5CF6", "#A78BFA", "#C4B5FD", "#E5E7EB"]
        
        fig_lag.add_trace(go.Bar(
            x=categories_list,
            y=lags_list,
            marker=dict(color=colors_lag[:len(categories_list)],
                       line=dict(color="#FFFFFF", width=1)),
            text=[f"{l}d" for l in lags_list],
            textposition="outside",
            textfont=dict(color="#6B7280", size=14, family="Inter"),
        ))
        
        fig_lag.update_layout(
            template="plotly_white",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            height=350,
            xaxis=dict(gridcolor="#F3F4F6", title="", color="#6B7280"),
            yaxis=dict(gridcolor="#F3F4F6", title="Lag (days)", color="#6B7280"),
            margin=dict(l=0, r=0, t=20, b=0),
        )
        
        st.plotly_chart(fig_lag, use_container_width=True)
    
    # ─── Confidence Distribution ───
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    st.subheader(":material/bar_chart: Confidence Score Analysis")
    
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
        template="plotly_white",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=300,
        yaxis=dict(title="Confidence (%)", gridcolor="#F3F4F6", color="#6B7280"),
        xaxis=dict(gridcolor="#F3F4F6", color="#6B7280"),
        margin=dict(l=0, r=0, t=20, b=0),
    )
    
    st.plotly_chart(fig_conf, use_container_width=True)
