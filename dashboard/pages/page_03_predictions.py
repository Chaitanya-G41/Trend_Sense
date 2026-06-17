"""
TrendSense Dashboard — Page 3: Predictions
============================================
Model comparison, feature importance, and prediction vs actual charts.
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


def render():
    """Render the Predictions page."""
    
    st.markdown("""
    <div class="main-header">
        <h1><span class="material-symbols-outlined" style="margin-right: 12px; color: #7C3AED;">online_prediction</span> Model Predictions</h1>
        <p>XGBoost vs Random Forest vs ARIMA — performance comparison and analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load saved models and data
    models_info = load_models()
    featured_df = load_featured_data()
    
    if not models_info or featured_df is None:
        st.warning("No trained models or processed data found. Please run the pipeline first:")
        st.code("python run_pipeline.py", language="bash")
        return
    
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    
    # ─── Model Comparison ───
    st.subheader(" Model Performance Comparison")
    
    # Build comparison from saved models or defaults
    comparison_data = get_comparison_data(models_info)
    
    if comparison_data:
        # Create comparison cards
        cols = st.columns(len(comparison_data))
        
        for idx, (model_name, metrics) in enumerate(comparison_data.items()):
            with cols[idx]:
                is_best = metrics.get("MAPE (%)", 100) == min(
                    m.get("MAPE (%)", 100) for m in comparison_data.values()
                )
                border_color = "#7C3AED" if is_best else "#E5E7EB"
                badge = " (BEST)" if is_best else ""
                text_color = "#7C3AED" if is_best else "#111827"
                
                st.markdown(f"""
                <div style="background: #FFFFFF;
                            border: 2px solid {border_color}; border-radius: 16px;
                            padding: 24px; text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
                    <div style="font-size: 0.875rem; color: #6B7280; text-transform: uppercase;
                                letter-spacing: 0.05em; font-weight: 500;">{model_name}<span style="color:#A78BFA; font-weight:700;">{badge}</span></div>
                    <div style="font-size: 2rem; font-weight: 700; color: {text_color};
                                margin: 8px 0;">{metrics.get('MAPE (%)', 'N/A')}%</div>
                    <div style="font-size: 0.75rem; color: #9CA3AF;">MAPE</div>
                    <hr style="border: none; height: 1px; background-color: #E5E7EB; margin: 16px 0;">
                    <div style="display: flex; justify-content: space-around;">
                        <div>
                            <div style="color: #6B7280; font-size: 0.75rem;">RMSE</div>
                            <div style="color: #111827; font-weight: 600;">{metrics.get('RMSE', 'N/A')}</div>
                        </div>
                        <div>
                            <div style="color: #6B7280; font-size: 0.75rem;">R²</div>
                            <div style="color: #111827; font-weight: 600;">{metrics.get('R²', 'N/A')}</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
        
        # Bar chart comparison
        fig = make_subplots(rows=1, cols=3, subplot_titles=["MAPE (%) ↓ Lower is Better",
                                                              "RMSE ↓ Lower is Better",
                                                              "R² ↑ Higher is Better"])
        
        models = list(comparison_data.keys())
        colors = ["#E5E7EB", "#C4B5FD", "#7C3AED"][:len(models)]
        
        mape_vals = [comparison_data[m].get("MAPE (%)", 0) for m in models]
        rmse_vals = [comparison_data[m].get("RMSE", 0) for m in models]
        r2_vals = [comparison_data[m].get("R²", 0) for m in models]
        
        fig.add_trace(go.Bar(x=models, y=mape_vals, marker_color=colors,
                             text=[f"{v:.1f}%" for v in mape_vals], textposition="outside",
                             textfont=dict(color="#6B7280")), row=1, col=1)
        fig.add_trace(go.Bar(x=models, y=rmse_vals, marker_color=colors,
                             text=[f"{v:.0f}" for v in rmse_vals], textposition="outside",
                             textfont=dict(color="#6B7280")), row=1, col=2)
        fig.add_trace(go.Bar(x=models, y=r2_vals, marker_color=colors,
                             text=[f"{v:.3f}" for v in r2_vals], textposition="outside",
                             textfont=dict(color="#6B7280")), row=1, col=3)
        
        fig.update_layout(
            template="plotly_white",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            height=400,
            showlegend=False,
            margin=dict(l=0, r=0, t=40, b=0),
        )
        for i in range(1, 4):
            fig.update_xaxes(gridcolor="#F3F4F6", color="#6B7280", row=1, col=i)
            fig.update_yaxes(gridcolor="#F3F4F6", color="#6B7280", row=1, col=i)
        
        st.plotly_chart(fig, use_container_width=True)
    
    # ─── Δ-MAPE Improvement ───
    if comparison_data and len(comparison_data) >= 2:
        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
        st.subheader(" Δ-MAPE: Improvement Over Baseline")
        
        model_names = list(comparison_data.keys())
        baseline_mape = comparison_data[model_names[0]].get("MAPE (%)", 0)
        
        for name in model_names[1:]:
            delta = baseline_mape - comparison_data[name].get("MAPE (%)", 0)
            color = "#10B981" if delta > 0 else "#EF4444"
            st.markdown(f"""
            <div style="background: #FFFFFF; border: 1px solid #E2E8F0;
                        border-radius: 12px; padding: 16px; margin: 8px 0;
                        display: flex; align-items: center; justify-content: space-between;
                        box-shadow: 0 4px 10px rgba(0,0,0,0.01);">
                <span style="color: #334155; font-weight: 600;">{name} vs {model_names[0]}</span>
                <span style="color: {color}; font-weight: 800; font-size: 1.3rem;">
                    {'↓' if delta > 0 else '↑'} {abs(delta):.2f}% MAPE
                </span>
            </div>
            """, unsafe_allow_html=True)
    
    # ─── Prediction vs Actual ───
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    st.subheader(":material/monitoring: Prediction vs Actual (Test Set)")
    
    # Generate predictions using loaded model or simulate
    predictions, actuals = generate_predictions(featured_df, models_info)
    
    if predictions is not None and actuals is not None:
        fig2 = go.Figure()
        
        x_range = list(range(len(actuals)))
        
        fig2.add_trace(go.Scatter(
            x=x_range, y=actuals,
            mode="lines",
            name="Actual Sales",
            line=dict(color="#111827", width=2.5),
        ))
        
        fig2.add_trace(go.Scatter(
            x=x_range, y=predictions,
            mode="lines",
            name="Predicted Sales",
            line=dict(color="#7C3AED", width=2.5, dash="dash"),
        ))
        
        # Confidence band
        error = np.abs(np.array(actuals) - np.array(predictions))
        std_error = np.std(error)
        upper = np.array(predictions) + 1.96 * std_error
        lower = np.array(predictions) - 1.96 * std_error
        
        fig2.add_trace(go.Scatter(
            x=x_range + x_range[::-1],
            y=list(upper) + list(lower[::-1]),
            fill="toself",
            fillcolor="rgba(124, 58, 237, 0.08)",
            line=dict(color="rgba(0,0,0,0)"),
            name="95% Confidence",
        ))
        
        fig2.update_layout(
            template="plotly_white",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            height=400,
            xaxis=dict(title="Week (Test Period)", gridcolor="#F3F4F6", color="#6B7280"),
            yaxis=dict(title="Weekly Sales ($)", gridcolor="#F3F4F6", color="#6B7280"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, font=dict(color="#6B7280")),
            margin=dict(l=0, r=0, t=30, b=0),
            hovermode="x unified"
        )
        
        st.plotly_chart(fig2, use_container_width=True)
    
    # ─── Feature Importance ───
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    st.subheader(":material/tune: Feature Importance (XGBoost)")
    
    importance_df = get_feature_importance(models_info, featured_df)
    
    if importance_df is not None and len(importance_df) > 0:
        top_n = st.slider("Show top N features", 5, min(30, len(importance_df)), 15)
        top = importance_df.head(top_n).sort_values("importance")
        
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(
            y=top["feature"],
            x=top["importance"],
            orientation="h",
            marker=dict(
                color=top["importance"],
                colorscale="Viridis",
                showscale=True,
            ),
        ))
        
        fig3.update_layout(
            template="plotly_white",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            height=max(400, top_n * 28),
            xaxis=dict(title="Feature Importance", gridcolor="rgba(0,0,0,0.05)"),
            yaxis=dict(gridcolor="rgba(0,0,0,0.05)"),
            margin=dict(l=200),
        )
        
        st.plotly_chart(fig3, use_container_width=True)
        
        # Highlight TVI features
        tvi_features = top[top["feature"].str.contains("tvi|spike|trend", case=False, na=False)]
        if len(tvi_features) > 0:
            st.success(f" **TVI features in top {top_n}**: {len(tvi_features)} features "
                      f"({len(tvi_features)/top_n*100:.0f}% of top features are trend-derived)")


def load_models():
    """Load saved model artifacts."""
    models = {}
    for name in ["xgboost_tvi", "random_forest", "best_model"]:
        path = os.path.join(config.MODELS_DIR, f"{name}.pkl")
        if os.path.exists(path):
            try:
                models[name] = joblib.load(path)
            except Exception:
                pass
    return models


@st.cache_data(ttl=300)
def load_featured_data():
    """Load feature-engineered dataset."""
    path = os.path.join(config.PROCESSED_DATA_DIR, "walmart_featured.csv")
    if os.path.exists(path):
        return pd.read_csv(path, parse_dates=["Date"])
    return None


def get_comparison_data(models_info):
    """Get model comparison metrics."""
    comparison = {}
    
    # From saved models
    if models_info:
        for key, data in models_info.items():
            if isinstance(data, dict) and "metrics" in data and data["metrics"]:
                name_map = {
                    "xgboost_tvi": "XGBoost + TVI",
                    "random_forest": "Random Forest",
                    "best_model": "Best Model",
                }
                name = name_map.get(key, key)
                if name != "Best Model":
                    comparison[name] = data["metrics"]
    
    return comparison


def generate_predictions(featured_df, models_info):
    """Generate or load predictions for the test set."""
    if featured_df is None:
        return None, None
    
    target_col = "Weekly_Sales"
    if target_col not in featured_df.columns:
        return None, None
    
    # Try using saved XGBoost model
    if "xgboost_tvi" in models_info:
        try:
            model = models_info["xgboost_tvi"]["model"]
            from src.feature_engineering import get_feature_columns
            feature_cols = get_feature_columns(featured_df, target_col)
            
            test_weeks = config.TEST_SIZE_WEEKS
            test_df = featured_df.tail(test_weeks)
            
            X_test = test_df[feature_cols].fillna(0)
            predictions = model.predict(X_test)
            actuals = test_df[target_col].values
            
            return predictions, actuals
        except Exception:
            pass
    
    return None, None


def get_feature_importance(models_info, featured_df):
    """Get feature importance from saved model."""
    if "xgboost_tvi" in models_info:
        try:
            model = models_info["xgboost_tvi"]["model"]
            from src.feature_engineering import get_feature_columns
            feature_cols = get_feature_columns(featured_df, "Weekly_Sales")
            
            importances = model.feature_importances_
            if len(importances) == len(feature_cols):
                return pd.DataFrame({
                    "feature": feature_cols,
                    "importance": importances,
                }).sort_values("importance", ascending=False)
        except Exception:
            pass
    
    return None
