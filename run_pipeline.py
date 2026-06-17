"""
TrendSense — End-to-End Pipeline Runner
=========================================
Orchestrates the complete workflow:
  1. Data Ingestion (Walmart + Google Trends)
  2. TVI Computation
  3. Feature Engineering
  4. Model Training & Evaluation
  5. Lag Optimization
  6. Decision Generation

Usage:
    python run_pipeline.py                # Full pipeline
    python run_pipeline.py --mode test    # Quick test with synthetic data
    python run_pipeline.py --skip-download  # Skip data download (use cached)
"""

import argparse
import sys
import os
import warnings

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from src.data_ingestion import download_walmart_data, fetch_google_trends, _generate_synthetic_trends
from src.tvi import compute_tvi_features, get_tvi_summary
from src.feature_engineering import merge_all_features, get_feature_columns
from src.models import run_full_training, save_model
from src.lag_optimizer import optimize_lags_by_category
from src.decision_engine import generate_decision, generate_batch_decisions, get_decision_summary, format_decision_output
from src.utils import (
    print_section, plot_sales_trend, plot_tvi_analysis,
    plot_model_comparison, plot_prediction_vs_actual,
    plot_feature_importance, save_dataframe,
)


def run_pipeline(skip_download=False, use_synthetic_trends=False, save_plots=True):
    """
    Execute the complete TrendSense pipeline.
    
    Parameters
    ----------
    skip_download : bool
        Skip Kaggle downloads (use cached data)
    use_synthetic_trends : bool
        Use synthetic Google Trends data instead of live API
    save_plots : bool
        Save plots to reports/figures/
    """
    print_section("🚀 TrendSense — Full Pipeline Execution", "═")
    
    # ────────────────────────────────
    # PHASE 1: Data Ingestion
    # ────────────────────────────────
    print_section("PHASE 1: Data Ingestion")
    
    # Walmart data
    try:
        walmart_df = download_walmart_data()
    except FileNotFoundError:
        print("⚠️ Walmart data not found. Generating synthetic sales data...")
        walmart_df = _generate_synthetic_sales()
    
    # Google Trends data
    if use_synthetic_trends:
        trends_df = _generate_synthetic_trends(
            list(config.TREND_KEYWORDS.keys()),
            config.TRENDS_TIMEFRAME
        )
    else:
        try:
            trends_df = fetch_google_trends()
        except Exception as e:
            print(f"⚠️ Google Trends fetch failed: {e}")
            print("   Using synthetic trends data...")
            trends_df = _generate_synthetic_trends(
                list(config.TREND_KEYWORDS.keys()),
                config.TRENDS_TIMEFRAME
            )
    
    print(f"\n📊 Data Summary:")
    print(f"   Walmart: {walmart_df.shape[0]} rows × {walmart_df.shape[1]} cols")
    print(f"   Trends:  {trends_df.shape[0]} rows × {trends_df.shape[1]} cols")
    
    # ────────────────────────────────
    # PHASE 2: TVI Analysis
    # ────────────────────────────────
    print_section("PHASE 2: TVI Computation & Spike Detection")
    
    tvi_results = {}
    keywords = [k for k in config.TREND_KEYWORDS.keys() if k in trends_df.columns]
    
    for keyword in keywords:
        print(f"\n🔍 Analyzing keyword: '{keyword}'")
        tvi_feat = compute_tvi_features(trends_df, keyword)
        tvi_summary = get_tvi_summary(tvi_feat)
        tvi_results[keyword] = {"features": tvi_feat, "summary": tvi_summary}
        
        print(f"   Mean TVI: {tvi_summary['mean_tvi']:.2f}")
        print(f"   Spikes detected: {tvi_summary['total_spikes']} "
              f"(Severe: {tvi_summary['severe_spikes']}, "
              f"Moderate: {tvi_summary['moderate_spikes']}, "
              f"Mild: {tvi_summary['mild_spikes']})")
        
        # Save TVI plot
        if save_plots:
            plot_path = os.path.join(config.FIGURES_DIR, f"tvi_analysis_{keyword.replace(' ', '_')}.png")
            try:
                plot_tvi_analysis(tvi_feat, keyword, save_path=plot_path)
            except Exception:
                pass
    
    # ────────────────────────────────
    # PHASE 3: Feature Engineering
    # ────────────────────────────────
    print_section("PHASE 3: Feature Engineering")
    
    # Use "General" category with first available keyword
    featured_df = merge_all_features(
        walmart_df, trends_df,
        category="General",
        target_col="Weekly_Sales",
    )
    
    # Save processed dataset
    save_dataframe(featured_df, "walmart_featured.csv")
    
    # ────────────────────────────────
    # PHASE 4: Model Training
    # ────────────────────────────────
    print_section("PHASE 4: Model Training & Evaluation")
    
    feature_cols = get_feature_columns(featured_df, "Weekly_Sales")
    training_results = run_full_training(
        featured_df,
        target_col="Weekly_Sales",
        feature_cols=feature_cols,
        save=True,
    )
    
    # Save comparison plot
    if save_plots and training_results.get("comparison") is not None:
        try:
            plot_model_comparison(
                training_results["comparison"],
                save_path=os.path.join(config.FIGURES_DIR, "model_comparison.png")
            )
        except Exception:
            pass
    
    # Save prediction plot for best model
    if save_plots:
        best_model_name = "XGBoost + TVI"
        if best_model_name in training_results.get("results", {}):
            result = training_results["results"][best_model_name]
            try:
                plot_prediction_vs_actual(
                    training_results["y_test"].values,
                    result["predictions"],
                    model_name=best_model_name,
                    save_path=os.path.join(config.FIGURES_DIR, "prediction_vs_actual.png")
                )
            except Exception:
                pass
    
    # Feature importance plot
    if save_plots:
        for name, result in training_results.get("results", {}).items():
            if "feature_importance" in result:
                try:
                    plot_feature_importance(
                        result["feature_importance"],
                        model_name=name,
                        save_path=os.path.join(config.FIGURES_DIR,
                                              f"feature_importance_{name.replace(' ', '_').replace('+', '')}.png")
                    )
                except Exception:
                    pass
    
    # ────────────────────────────────
    # PHASE 5: Lag Optimization
    # ────────────────────────────────
    print_section("PHASE 5: Category-Specific Lag Optimization")
    
    lag_results = optimize_lags_by_category(walmart_df, trends_df)
    
    # ────────────────────────────────
    # PHASE 6: Decision Generation
    # ────────────────────────────────
    print_section("PHASE 6: Decision Generation")
    
    # Generate sample decisions using model predictions
    if "XGBoost + TVI" in training_results.get("results", {}):
        xgb_result = training_results["results"]["XGBoost + TVI"]
        predictions = xgb_result["predictions"]
        actuals = training_results["y_test"].values
        
        # Create TVI statuses for test period from actual X_test data
        n_test = len(predictions)
        tvi_statuses = []
        X_test = training_results["X_test"]
        
        # Find the TVI column names
        tvi_col = next((c for c in X_test.columns if c.startswith("tvi_") and not c.startswith("tvi_accel") and not c.startswith("tvi_smooth")), None)
        severity_col = next((c for c in X_test.columns if c.startswith("spike_severity_") and c.endswith("_encoded")), None)
        
        severity_map_reverse = {0: "NONE", 1: "MILD", 2: "MODERATE", 3: "SEVERE"}
        
        for i in range(n_test):
            tvi_val = X_test[tvi_col].iloc[i] if tvi_col else 0.0
            sev_encoded = int(X_test[severity_col].iloc[i]) if severity_col else 0
            severity = severity_map_reverse.get(sev_encoded, "NONE")
            is_spike = severity != "NONE"
            
            tvi_statuses.append({
                "is_spike": is_spike,
                "severity": severity,
                "tvi_value": tvi_val,
            })
        
        # Use previous week's actual sales as "current stock" for realistic simulation
        y_train = training_results["y_train"].values
        current_stocks = np.concatenate(([y_train[-1]], actuals[:-1]))
        
        decisions_df = generate_batch_decisions(
            predictions=predictions,
            current_stocks=actuals,
            tvi_statuses=tvi_statuses,
            model_confidence=0.85,
        )
        
        # Print sample decisions
        print("\n📋 Sample Decisions (first 5):")
        print(decisions_df[["action", "confidence_pct", "predicted_change_pct",
                           "tvi_severity", "spike_boost_applied"]].head().to_string())
        
        # Summary
        summary = get_decision_summary(decisions_df)
        print(f"\n📊 Decision Summary:")
        print(f"   Total: {summary['total_decisions']}")
        print(f"   🟢 HOLD: {summary['hold_count']} ({summary['hold_pct']}%)")
        print(f"   🟡 INCREASE STOCK: {summary['increase_count']} ({summary['increase_pct']}%)")
        print(f"   🔴 URGENT RESTOCK: {summary['urgent_count']} ({summary['urgent_pct']}%)")
        print(f"   Avg Confidence: {summary['avg_confidence']}%")
        print(f"   TVI-Influenced: {summary['spike_influenced']}")
        
        # Save decisions
        save_dataframe(decisions_df, "decisions.csv")
    
    # ────────────────────────────────
    # DONE
    # ────────────────────────────────
    print_section("✅ Pipeline Complete!", "═")
    print(f"\n📁 Outputs saved to:")
    print(f"   Data:    {config.PROCESSED_DATA_DIR}")
    print(f"   Models:  {config.MODELS_DIR}")
    print(f"   Figures: {config.FIGURES_DIR}")
    print(f"\n🌐 Launch dashboard with: streamlit run dashboard/app.py")
    
    return {
        "walmart_df": walmart_df,
        "trends_df": trends_df,
        "tvi_results": tvi_results,
        "featured_df": featured_df,
        "training_results": training_results,
        "lag_results": lag_results,
    }


def _generate_synthetic_sales(n_stores=5, n_weeks=143):
    """Generate synthetic Walmart-like sales data for testing."""
    np.random.seed(42)
    rows = []
    start_date = pd.Timestamp("2010-02-05")
    
    for store in range(1, n_stores + 1):
        base_sales = np.random.uniform(800000, 2500000)
        for week in range(n_weeks):
            date = start_date + pd.Timedelta(weeks=week)
            
            # Seasonality
            seasonal = 100000 * np.sin(2 * np.pi * week / 52)
            
            # Holiday spikes
            is_holiday = 1 if date.month in [11, 12] and date.day > 20 else 0
            holiday_boost = is_holiday * np.random.uniform(200000, 500000)
            
            # Random noise
            noise = np.random.normal(0, 50000)
            
            weekly_sales = base_sales + seasonal + holiday_boost + noise
            
            rows.append({
                "Store": store,
                "Date": date,
                "Weekly_Sales": max(0, weekly_sales),
                "Holiday_Flag": is_holiday,
                "Temperature": np.random.uniform(30, 100),
                "Fuel_Price": np.random.uniform(2.5, 4.5),
                "CPI": np.random.uniform(120, 230),
                "Unemployment": np.random.uniform(4, 14),
            })
    
    df = pd.DataFrame(rows)
    print(f"📊 Synthetic sales data: {len(df)} rows, {n_stores} stores, {n_weeks} weeks")
    return df


def run_test_pipeline():
    """Quick pipeline test with small synthetic data."""
    print_section("🧪 TrendSense — Test Pipeline", "═")
    return run_pipeline(skip_download=True, use_synthetic_trends=True, save_plots=False)


# ──────────────────────────────────────────────
# CLI Entry Point
# ──────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TrendSense Pipeline Runner")
    parser.add_argument("--mode", choices=["full", "test"], default="full",
                        help="Run mode: 'full' for complete pipeline, 'test' for quick test")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip data download, use cached files")
    parser.add_argument("--synthetic-trends", action="store_true",
                        help="Use synthetic Google Trends data")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip saving plots")
    
    args = parser.parse_args()
    
    if args.mode == "test":
        run_test_pipeline()
    else:
        run_pipeline(
            skip_download=args.skip_download,
            use_synthetic_trends=args.synthetic_trends,
            save_plots=not args.no_plots,
        )
