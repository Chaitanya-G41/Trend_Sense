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
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');
    
    /* Global styles */
    .stApp {
        background: linear-gradient(135deg, #F8FAFC 0%, #EFF6FF 100%);
        font-family: 'Outfit', sans-serif;
        color: #1E293B;
    }
    
    /* Sidebar background */
    [data-testid="stSidebar"] {
        background: #FFFFFF !important;
        border-right: 1px solid #E2E8F0;
    }
    
    [data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] p, [data-testid="stSidebar"] h3 {
        color: #334155 !important;
    }
    
    /* Style radio navigation items */
    .stRadio div[role="radiogroup"] {
        gap: 8px;
    }
    
    .stRadio div[role="radiogroup"] > label {
        background: #F8FAFC;
        color: #475569;
        border-radius: 12px;
        padding: 10px 16px;
        margin-bottom: 6px;
        border: 1px solid #E2E8F0;
        transition: all 0.2s ease;
        font-weight: 500;
        cursor: pointer;
    }
    
    .stRadio div[role="radiogroup"] > label:hover {
        background: #F1F5F9;
        border-color: #CBD5E1;
        transform: translateX(3px);
    }
    
    .stRadio div[role="radiogroup"] > label[data-checked="true"] {
        background: linear-gradient(135deg, #EEF2FF 0%, #E0E7FF 100%) !important;
        border-color: #6366F1 !important;
        color: #4F46E5 !important;
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.08);
        font-weight: 600;
    }
    
    /* KPI Card styling */
    .kpi-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 18px;
        padding: 24px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.02);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .kpi-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 25px rgba(99, 102, 241, 0.08);
    }
    
    .kpi-value {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #4F46E5, #EC4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 8px 0;
    }
    
    .kpi-label {
        font-size: 0.85rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
    }
    
    /* Decision cards */
    .decision-hold {
        background: linear-gradient(135deg, #ECFDF5, #D1FAE5);
        border: 1px solid #10B981;
        border-radius: 18px;
        padding: 24px;
        color: #065F46;
    }
    
    .decision-increase {
        background: linear-gradient(135deg, #FFFBEB, #FEF3C7);
        border: 1px solid #F59E0B;
        border-radius: 18px;
        padding: 24px;
        color: #78350F;
    }
    
    .decision-urgent {
        background: linear-gradient(135deg, #FEF2F2, #FEE2E2);
        border: 1px solid #EF4444;
        border-radius: 18px;
        padding: 24px;
        color: #991B1B;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.2); }
        50% { box-shadow: 0 0 0 8px rgba(239, 68, 68, 0); }
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #EEF2FF 0%, #F5F3FF 50%, #FDF2F8 100%);
        border-radius: 20px;
        padding: 32px;
        margin-bottom: 24px;
        border: 1px solid #E0E7FF;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.01);
    }
    
    .main-header h1 {
        background: linear-gradient(135deg, #4F46E5, #9333EA);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 800;
        margin: 0;
    }
    
    .main-header p {
        color: #475569;
        font-size: 1.1rem;
        margin-top: 8px;
    }
    
    /* Sidebar logo area */
    .sidebar-brand {
        text-align: center;
        padding: 20px 0;
        border-bottom: 1px solid #F1F5F9;
        margin-bottom: 20px;
    }
    
    .sidebar-brand h2 {
        background: linear-gradient(135deg, #4F46E5, #EC4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 1.6rem;
        font-weight: 800;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    [data-testid="stHeader"] {display: none;}
    
    /* Custom separator */
    .custom-divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, #E2E8F0, transparent);
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

