"""
TrendSense — Streamlit Dashboard (Main App)
=============================================
Multi-page dashboard for visualizing trend analysis, model predictions,
and inventory decision support.
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
    page_title="TrendSense",
    page_icon=":material/analytics:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────
# Custom CSS (SaaS Redesign)
# ──────────────────────────────────
st.markdown("""
<style>
    /* Import Inter Font and Material Symbols */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@24,300,0,0');
    
    /* Global styles */
    .stApp {
        background: #F8FAFC;
        font-family: 'Inter', sans-serif;
        color: #111827;
    }
    
    /* Typography Overrides */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Inter', sans-serif !important;
        color: #111827 !important;
        font-weight: 600 !important;
        letter-spacing: -0.01em;
    }
    
    /* Sidebar background */
    [data-testid="stSidebar"] {
        background: #FFFFFF !important;
        border-right: 1px solid #E5E7EB;
    }
    
    /* Minimalist Header Area */
    .main-header {
        margin-bottom: 40px;
        padding-bottom: 20px;
        border-bottom: 1px solid #E5E7EB;
    }
    
    .main-header h1 {
        font-size: 2.25rem !important;
        font-weight: 700 !important;
        color: #111827 !important;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
    }
    
    .main-header p {
        color: #6B7280;
        font-size: 1rem;
        margin-top: 0;
        font-weight: 400;
    }
    
    /* KPI Card styling (Glassmorphism & SaaS) */
    .kpi-card {
        background: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05), 0 1px 2px rgba(0,0,0,0.03);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        height: 100%;
        display: flex;
        flex-direction: column;
    }
    
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05), 0 4px 6px -2px rgba(0, 0, 0, 0.03);
        border-color: #C4B5FD;
    }
    
    .kpi-header {
        display: flex;
        align-items: center;
        margin-bottom: 12px;
        color: #6B7280;
        font-size: 0.875rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .kpi-header span.material-symbols-outlined {
        font-size: 1.25rem;
        margin-right: 8px;
        color: #A78BFA;
    }
    
    .kpi-value {
        font-size: 2rem;
        font-weight: 700;
        color: #111827;
        margin: 4px 0;
        line-height: 1.2;
    }
    
    .kpi-subtext {
        font-size: 0.875rem;
        color: #9CA3AF;
        margin-top: 8px;
    }
    
    /* Decision Cards */
    .decision-hold {
        background: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-left: 4px solid #10B981;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    
    .decision-increase {
        background: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-left: 4px solid #F59E0B;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    
    .decision-urgent {
        background: #FEF2F2;
        border: 1px solid #FECACA;
        border-left: 4px solid #EF4444;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    
    /* Sidebar Brand */
    .sidebar-brand {
        display: flex;
        align-items: center;
        padding: 16px 0 24px 0;
        margin-bottom: 8px;
    }
    
    .sidebar-brand .material-symbols-outlined {
        color: #7C3AED;
        font-size: 1.75rem;
        margin-right: 12px;
    }
    
    .sidebar-brand h2 {
        font-size: 1.25rem;
        font-weight: 700;
        color: #111827;
        margin: 0;
        letter-spacing: -0.02em;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* Custom separator */
    .custom-divider {
        height: 1px;
        background: #E5E7EB;
        margin: 32px 0;
        border: none;
    }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────
# Sidebar Brand Override
# ──────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <span class="material-symbols-outlined">analytics</span>
        <h2>TrendSense</h2>
    </div>
    """, unsafe_allow_html=True)
    
# ──────────────────────────────────
# Streamlit Native Navigation (SaaS Router)
# ──────────────────────────────────
st.session_state["selected_category"] = "General"

from pages import page_01_overview, page_02_trend_analysis, page_03_predictions, page_04_decision_support

pg = st.navigation([
    st.Page(page_01_overview.render, title="Overview", icon=":material/dashboard:", url_path="overview"),
    st.Page(page_02_trend_analysis.render, title="Trend Analysis", icon=":material/trending_up:", url_path="trend_analysis"),
    st.Page(page_03_predictions.render, title="Predictions", icon=":material/online_prediction:", url_path="predictions"),
    st.Page(page_04_decision_support.render, title="Decision Support", icon=":material/lightbulb:", url_path="decision_support"),
])

pg.run()
