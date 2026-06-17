"""
TrendSense — Utility Functions
================================
Helper functions for plotting, metrics formatting, and general utilities.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


# ──────────────────────────────────────────────
# Plotting Helpers
# ──────────────────────────────────────────────

def setup_plot_style():
    """Set up consistent plotting style for all figures."""
    try:
        plt.style.use(config.PLOT_STYLE)
    except (OSError, ValueError):
        plt.style.use("seaborn-v0_8")

    plt.rcParams.update({
        "figure.figsize": config.PLOT_FIGSIZE,
        "figure.dpi": config.PLOT_DPI,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "figure.facecolor": "white",
        "axes.facecolor": "#f8f9fa",
        "axes.grid": True,
        "grid.alpha": 0.3,
    })


def plot_sales_trend(df, date_col="Date", sales_col="Weekly_Sales", title="Weekly Sales Trend",
                     save_path=None):
    """Plot weekly sales over time."""
    setup_plot_style()
    fig, ax = plt.subplots(figsize=config.PLOT_FIGSIZE)

    if "Store" in df.columns:
        # Aggregate across stores
        agg = df.groupby(date_col)[sales_col].mean()
        ax.plot(agg.index, agg.values, color=config.COLOR_PALETTE["primary"],
                linewidth=1.5, alpha=0.9)
        ax.fill_between(agg.index, agg.values, alpha=0.15,
                        color=config.COLOR_PALETTE["primary"])
    else:
        ax.plot(df[date_col], df[sales_col], color=config.COLOR_PALETTE["primary"],
                linewidth=1.5, alpha=0.9)

    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Weekly Sales ($)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.xticks(rotation=45)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=config.PLOT_DPI, bbox_inches="tight")
        print(f"💾 Saved: {save_path}")

    return fig


def plot_tvi_analysis(tvi_features, keyword="", save_path=None):
    """Plot TVI analysis with raw trend, TVI, and spikes."""
    setup_plot_style()
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    dates = tvi_features["date"]
    colors = config.COLOR_PALETTE

    # 1. Raw trend
    axes[0].plot(dates, tvi_features["raw_trend"], color=colors["info"],
                 linewidth=1.5, label="Search Interest")
    axes[0].fill_between(dates, tvi_features["raw_trend"], alpha=0.15,
                         color=colors["info"])
    axes[0].set_title(f"Google Trends Interest — '{keyword}'", fontweight="bold")
    axes[0].set_ylabel("Interest (0-100)")
    axes[0].legend()

    # 2. TVI (rate of change)
    tvi_vals = tvi_features["tvi"]
    pos_mask = tvi_vals >= 0
    axes[1].bar(dates[pos_mask], tvi_vals[pos_mask], color=colors["success"],
                alpha=0.7, width=5, label="Positive TVI")
    axes[1].bar(dates[~pos_mask], tvi_vals[~pos_mask], color=colors["danger"],
                alpha=0.7, width=5, label="Negative TVI")
    axes[1].axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    axes[1].set_title("Trend Velocity Index (TVI)", fontweight="bold")
    axes[1].set_ylabel("TVI (%)")
    axes[1].legend()

    # 3. Spike detection
    spike_colors = {"NONE": "gray", "MILD": colors["warning"],
                    "MODERATE": "#FF6B00", "SEVERE": colors["danger"]}

    for severity, color in spike_colors.items():
        mask = tvi_features["spike_severity"] == severity
        if mask.any():
            axes[2].scatter(dates[mask], tvi_vals[mask], c=color,
                           s=30 if severity == "NONE" else 80,
                           alpha=0.5 if severity == "NONE" else 0.9,
                           label=severity, zorder=5 if severity != "NONE" else 1)

    axes[2].axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    axes[2].set_title("Spike Detection", fontweight="bold")
    axes[2].set_ylabel("TVI (%)")
    axes[2].set_xlabel("Date")
    axes[2].legend()

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=config.PLOT_DPI, bbox_inches="tight")
        print(f"💾 Saved: {save_path}")

    return fig


def plot_model_comparison(comparison_df, save_path=None):
    """Plot model comparison as bar chart."""
    setup_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    models = comparison_df.index.tolist()
    colors = [config.COLOR_PALETTE["danger"], config.COLOR_PALETTE["warning"],
              config.COLOR_PALETTE["success"]]

    # MAPE comparison
    mape_vals = comparison_df["MAPE (%)"].values
    bars = axes[0].bar(models, mape_vals, color=colors[:len(models)], alpha=0.85)
    axes[0].set_title("MAPE (%) — Lower is Better", fontweight="bold")
    axes[0].set_ylabel("MAPE (%)")
    for bar, val in zip(bars, mape_vals):
        axes[0].text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.3,
                     f"{val:.1f}%", ha="center", va="bottom", fontweight="bold")

    # RMSE comparison
    rmse_vals = comparison_df["RMSE"].values
    bars = axes[1].bar(models, rmse_vals, color=colors[:len(models)], alpha=0.85)
    axes[1].set_title("RMSE — Lower is Better", fontweight="bold")
    axes[1].set_ylabel("RMSE")
    for bar, val in zip(bars, rmse_vals):
        axes[1].text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.3,
                     f"{val:.0f}", ha="center", va="bottom", fontweight="bold")

    # R² comparison
    r2_vals = comparison_df["R²"].values
    bars = axes[2].bar(models, r2_vals, color=colors[:len(models)], alpha=0.85)
    axes[2].set_title("R² Score — Higher is Better", fontweight="bold")
    axes[2].set_ylabel("R²")
    for bar, val in zip(bars, r2_vals):
        axes[2].text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.005,
                     f"{val:.3f}", ha="center", va="bottom", fontweight="bold")

    for ax in axes:
        ax.tick_params(axis="x", rotation=15)

    plt.suptitle("TrendSense — Model Comparison", fontweight="bold", fontsize=15, y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=config.PLOT_DPI, bbox_inches="tight")
        print(f"💾 Saved: {save_path}")

    return fig


def plot_prediction_vs_actual(y_true, y_pred, dates=None, model_name="XGBoost",
                               save_path=None):
    """Plot predicted vs actual sales values."""
    setup_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    colors = config.COLOR_PALETTE

    # Line plot
    x = dates if dates is not None else range(len(y_true))
    axes[0].plot(x, y_true, color=colors["info"], linewidth=2, label="Actual", alpha=0.9)
    axes[0].plot(x, y_pred, color=colors["danger"], linewidth=2, label="Predicted",
                 linestyle="--", alpha=0.9)
    axes[0].fill_between(x, y_true, y_pred, alpha=0.1, color=colors["warning"])
    axes[0].set_title(f"{model_name}: Predicted vs Actual", fontweight="bold")
    axes[0].set_ylabel("Weekly Sales")
    axes[0].legend()

    # Scatter plot
    axes[1].scatter(y_true, y_pred, alpha=0.5, c=colors["primary"], s=40)
    max_val = max(np.max(y_true), np.max(y_pred))
    min_val = min(np.min(y_true), np.min(y_pred))
    axes[1].plot([min_val, max_val], [min_val, max_val], "r--", alpha=0.7,
                 label="Perfect Prediction")
    axes[1].set_title("Prediction Scatter", fontweight="bold")
    axes[1].set_xlabel("Actual Sales")
    axes[1].set_ylabel("Predicted Sales")
    axes[1].legend()

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=config.PLOT_DPI, bbox_inches="tight")
        print(f"💾 Saved: {save_path}")

    return fig


def plot_feature_importance(feature_importance_df, top_n=15, model_name="XGBoost",
                             save_path=None):
    """Plot top feature importances as horizontal bar chart."""
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(10, 8))

    top = feature_importance_df.head(top_n).sort_values("importance")
    colors_gradient = plt.cm.viridis(np.linspace(0.3, 0.9, len(top)))

    ax.barh(top["feature"], top["importance"], color=colors_gradient, alpha=0.85)
    ax.set_title(f"{model_name} — Top {top_n} Feature Importances", fontweight="bold")
    ax.set_xlabel("Importance")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=config.PLOT_DPI, bbox_inches="tight")
        print(f"💾 Saved: {save_path}")

    return fig


def plot_lag_correlation(corr_df, category="", save_path=None):
    """Plot cross-correlation at different lags."""
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = [config.COLOR_PALETTE["success"] if row["is_significant"] else config.COLOR_PALETTE["info"]
              for _, row in corr_df.iterrows()]

    bars = ax.bar(corr_df["lag_weeks"], corr_df["correlation"], color=colors, alpha=0.8)

    # Highlight optimal lag
    best_idx = corr_df["correlation"].abs().idxmax()
    best_lag = corr_df.loc[best_idx, "lag_weeks"]
    ax.axvline(x=best_lag, color=config.COLOR_PALETTE["danger"], linestyle="--",
               linewidth=2, label=f"Optimal Lag: {best_lag} weeks")

    ax.set_title(f"TVI-Sales Cross-Correlation — {category}", fontweight="bold")
    ax.set_xlabel("Lag (weeks)")
    ax.set_ylabel("Pearson Correlation")
    ax.axhline(y=0, color="gray", linestyle="-", alpha=0.3)
    ax.legend()

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=config.PLOT_DPI, bbox_inches="tight")
        print(f"💾 Saved: {save_path}")

    return fig


# ──────────────────────────────────────────────
# Data Utility Functions
# ──────────────────────────────────────────────

def print_section(title: str, char: str = "=", width: int = 60):
    """Print a formatted section header."""
    print(f"\n{char * width}")
    print(f"  {title}")
    print(f"{char * width}")


def format_number(n: float) -> str:
    """Format number with Indian-style commas and abbreviations."""
    if abs(n) >= 1e7:
        return f"₹{n / 1e7:.1f} Cr"
    elif abs(n) >= 1e5:
        return f"₹{n / 1e5:.1f} L"
    elif abs(n) >= 1e3:
        return f"₹{n / 1e3:.1f} K"
    else:
        return f"₹{n:,.0f}"


def save_dataframe(df: pd.DataFrame, filename: str, directory: str = None):
    """Save DataFrame to CSV in the processed data directory."""
    directory = directory or config.PROCESSED_DATA_DIR
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, filename)
    df.to_csv(path, index=False)
    print(f"💾 Saved: {path} ({len(df)} rows)")
    return path
