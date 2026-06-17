"""
TrendSense Configuration
========================
Global paths, category mappings, model hyperparameters, and decision thresholds.
"""

import os

# ──────────────────────────────────────────────
# Project Paths
# ──────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
TRENDS_DATA_DIR = os.path.join(DATA_DIR, "trends")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")

# Create directories if they don't exist
for d in [RAW_DATA_DIR, TRENDS_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR]:
    os.makedirs(d, exist_ok=True)

# ──────────────────────────────────────────────
# Kaggle Dataset References
# ──────────────────────────────────────────────
KAGGLE_DATASETS = {
    "walmart": "yasserh/walmart-dataset",
    "rossmann": "pratyushakar/rossmann-store-sales",
}

# ──────────────────────────────────────────────
# Google Trends Configuration
# ──────────────────────────────────────────────
TRENDS_GEO = "IN"  # India
TRENDS_TIMEFRAME = "2010-01-01 2024-12-31"

# Keyword → Category Mapping
TREND_KEYWORDS = {
    "smartphones": "Electronics",
    "Diwali offers": "Festival",
    "fashion sale": "Fashion",
    "electronics sale": "Electronics",
    "laptop deals": "Electronics",
    "online shopping": "General",
    "festival sale": "Festival",
    "clothing sale": "Fashion",
}

# ──────────────────────────────────────────────
# Category-Specific Lag Defaults (days)
# Optimised via lag_optimizer.py at runtime
# ──────────────────────────────────────────────
DEFAULT_CATEGORY_LAGS = {
    "Fashion": 3,       # Impulse buying
    "Electronics": 14,  # Research-driven
    "Festival": 7,      # Moderate decision time
    "General": 7,       # Moderate decision time
    "Groceries": 2,     # Immediate need
}

# ──────────────────────────────────────────────
# Indian Holiday / Festive Calendar
# (approximate dates — varies by year)
# ──────────────────────────────────────────────
INDIAN_HOLIDAYS = {
    "Republic Day": {"month": 1, "day": 26},
    "Holi": {"month": 3, "day": 14},
    "Independence Day": {"month": 8, "day": 15},
    "Raksha Bandhan": {"month": 8, "day": 19},
    "Ganesh Chaturthi": {"month": 9, "day": 7},
    "Navratri Start": {"month": 10, "day": 3},
    "Dussehra": {"month": 10, "day": 12},
    "Dhanteras": {"month": 10, "day": 29},
    "Diwali": {"month": 11, "day": 1},
    "Christmas": {"month": 12, "day": 25},
}

# E-commerce sale events (approximate)
ECOMMERCE_EVENTS = {
    "Big Billion Day": {"month": 10, "day": 8, "duration_days": 6},
    "Great Indian Festival": {"month": 10, "day": 5, "duration_days": 7},
    "Republic Day Sale": {"month": 1, "day": 20, "duration_days": 6},
    "End of Season Sale": {"month": 6, "day": 20, "duration_days": 14},
}

# ──────────────────────────────────────────────
# TVI (Trend Velocity Index) Parameters
# ──────────────────────────────────────────────
TVI_SPIKE_THRESHOLDS = {
    "MILD": 1.0,      # > 1 sigma
    "MODERATE": 2.0,   # > 2 sigma
    "SEVERE": 3.0,     # > 3 sigma
}

# ──────────────────────────────────────────────
# Model Hyperparameters
# ──────────────────────────────────────────────
ARIMA_PARAMS = {
    "order": (1, 1, 1),
    "seasonal_order": (1, 1, 1, 52),  # weekly seasonality
}

RANDOM_FOREST_PARAMS = {
    "n_estimators": 200,
    "max_depth": 10,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "random_state": 42,
    "n_jobs": -1,
}

XGBOOST_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,
}

# Time-series cross-validation
CV_SPLITS = 5
TEST_SIZE_WEEKS = 12  # 12 weeks holdout for final test

# ──────────────────────────────────────────────
# Decision Engine Thresholds
# ──────────────────────────────────────────────
DECISION_THRESHOLDS = {
    "HOLD": 0.03,            # predicted change < 3.0%
    "INCREASE_STOCK": 0.40,  # 3.0% ≤ predicted change < 40%
    # URGENT RESTOCK: predicted change ≥ 40%
}

# TVI spike boosts urgency by one level
TVI_SPIKE_URGENCY_BOOST = True

# Confidence score bounds
CONFIDENCE_MIN = 0.50
CONFIDENCE_MAX = 0.99

# ──────────────────────────────────────────────
# Visualization Defaults
# ──────────────────────────────────────────────
PLOT_STYLE = "seaborn-v0_8-darkgrid"
PLOT_FIGSIZE = (14, 6)
PLOT_DPI = 150
COLOR_PALETTE = {
    "primary": "#6366F1",      # Indigo
    "secondary": "#8B5CF6",    # Violet
    "success": "#10B981",      # Emerald
    "warning": "#F59E0B",      # Amber
    "danger": "#EF4444",       # Red
    "info": "#3B82F6",         # Blue
    "hold": "#10B981",         # Green
    "increase": "#F59E0B",     # Amber
    "urgent": "#EF4444",       # Red
    "background": "#0F172A",   # Dark slate
    "surface": "#1E293B",      # Slate
    "text": "#F8FAFC",         # Light
}
