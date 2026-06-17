"""
TrendSense — Streamlit Dashboard (Main App)
=============================================
Multi-page dashboard for visualizing trend analysis, model predictions,
and inventory decision support.

Launch: streamlit run dashboard/app.py
"""

import streamlit as st
import sys
import os

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

import config

# ──────────────────────────────────
# Page Configuration
# ──────────────────────────────────
st.set_page_config(
    page_title="TrendSense | AI-Powered Demand Forecasting",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────
# Custom CSS
# ──────────────────────────────────
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* Global styles */
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    
    /* Dark theme override */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0F172A 0%, #1E293B 100%);
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: #F8FAFC;
    }
    
    /* KPI Card styling */
    .kpi-card {
        background: linear-gradient(135deg, #1E293B 0%, #334155 100%);
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(99, 102, 241, 0.2);
    }
    
    .kpi-value {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #6366F1, #8B5CF6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 8px 0;
    }
    
    .kpi-label {
        font-size: 0.85rem;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
    }
    
    /* Decision cards */
    .decision-hold {
        background: linear-gradient(135deg, #064E3B, #065F46);
        border: 2px solid #10B981;
        border-radius: 16px;
        padding: 20px;
        color: #F8FAFC;
    }
    
    .decision-increase {
        background: linear-gradient(135deg, #78350F, #92400E);
        border: 2px solid #F59E0B;
        border-radius: 16px;
        padding: 20px;
        color: #F8FAFC;
    }
    
    .decision-urgent {
        background: linear-gradient(135deg, #7F1D1D, #991B1B);
        border: 2px solid #EF4444;
        border-radius: 16px;
        padding: 20px;
        color: #F8FAFC;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.4); }
        50% { box-shadow: 0 0 0 10px rgba(239, 68, 68, 0); }
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #0F172A 0%, #1E1B4B 50%, #312E81 100%);
        border-radius: 20px;
        padding: 32px;
        margin-bottom: 24px;
        border: 1px solid rgba(99, 102, 241, 0.2);
    }
    
    .main-header h1 {
        background: linear-gradient(135deg, #818CF8, #C084FC);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 800;
        margin: 0;
    }
    
    .main-header p {
        color: #CBD5E1;
        font-size: 1.1rem;
        margin-top: 8px;
    }
    
    /* Sidebar logo area */
    .sidebar-brand {
        text-align: center;
        padding: 20px 0 30px 0;
        border-bottom: 1px solid rgba(99, 102, 241, 0.2);
        margin-bottom: 20px;
    }
    
    .sidebar-brand h2 {
        background: linear-gradient(135deg, #818CF8, #C084FC);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 1.5rem;
        font-weight: 800;
    }
    
    /* Metric delta */
    [data-testid="stMetricDelta"] {
        font-weight: 600;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    [data-testid="stHeader"] {display: none;}
    
    /* Custom separator */
    .custom-divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, #6366F1, transparent);
        margin: 24px 0;
        border: none;
    }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────
# Sidebar
# ──────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <h2>📈 TrendSense</h2>
        <p style="color: #94A3B8; font-size: 0.8rem;">AI-Powered Demand Forecasting</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Navigation")
    page = st.radio(
        "Go to",
        ["Overview", "Trend Analysis", "Predictions", "Decision Support"],
        label_visibility="collapsed",
    )

# ──────────────────────────────────
# Page Router
# ──────────────────────────────────
# Default selected category to General since we removed the sidebar dropdown
st.session_state["selected_category"] = "General"

if page == "Overview":
    from pages import page_01_overview
    page_01_overview.render()
elif page == "Trend Analysis":
    from pages import page_02_trend_analysis
    page_02_trend_analysis.render()
elif page == "Predictions":
    from pages import page_03_predictions
    page_03_predictions.render()
elif page == "Decision Support":
    from pages import page_04_decision_support
    page_04_decision_support.render()

