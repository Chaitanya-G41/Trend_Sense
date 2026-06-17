"""
TrendSense — Lag Optimizer Module
===================================
Discovers the optimal delay between a social media trend spike
and its sales impact per product category.

Key innovation: Fashion (impulse) → 3 days, Electronics (research) → 14 days.
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Tuple

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


def compute_cross_correlation(
    tvi_series: pd.Series,
    sales_series: pd.Series,
    max_lag: int = 21,
) -> pd.DataFrame:
    """
    Compute cross-correlation between TVI and sales at various lag periods.
    
    Positive lag means TVI leads sales (expected: trends predict sales).
    
    Parameters
    ----------
    tvi_series : pd.Series
        Trend Velocity Index values
    sales_series : pd.Series
        Weekly sales values
    max_lag : int
        Maximum lag to test (in weeks, since data is weekly)
        
    Returns
    -------
    pd.DataFrame
        Cross-correlation results with columns: [lag_weeks, correlation, p_value]
    """
    # Align series and drop NaN
    combined = pd.DataFrame({"tvi": tvi_series.values, "sales": sales_series.values}).dropna()
    
    if len(combined) < max_lag + 10:
        print(f"   ⚠️ Insufficient data ({len(combined)} points) for max_lag={max_lag}")
        max_lag = max(1, len(combined) // 3)
    
    results = []
    
    for lag in range(0, max_lag + 1):
        if lag >= len(combined):
            break
            
        # TVI leads sales by 'lag' weeks
        tvi_shifted = combined["tvi"].iloc[:-lag] if lag > 0 else combined["tvi"]
        sales_target = combined["sales"].iloc[lag:] if lag > 0 else combined["sales"]
        
        # Ensure equal length
        min_len = min(len(tvi_shifted), len(sales_target))
        if min_len < 5:
            continue
            
        tvi_aligned = tvi_shifted.iloc[:min_len].values
        sales_aligned = sales_target.iloc[:min_len].values
        
        # Pearson correlation
        corr, p_value = stats.pearsonr(tvi_aligned, sales_aligned)
        
        results.append({
            "lag_weeks": lag,
            "correlation": round(corr, 4),
            "p_value": round(p_value, 4),
            "is_significant": p_value < 0.05,
            "n_samples": min_len,
        })
    
    return pd.DataFrame(results)


def find_optimal_lag(
    tvi_series: pd.Series,
    sales_series: pd.Series,
    max_lag: int = 21,
) -> dict:
    """
    Find the optimal lag period that maximizes correlation between TVI and sales.
    
    Parameters
    ----------
    tvi_series : pd.Series
        Trend Velocity Index values
    sales_series : pd.Series
        Weekly sales values
    max_lag : int
        Maximum lag to test (in weeks)
        
    Returns
    -------
    dict
        Optimal lag info: {lag_weeks, correlation, p_value, all_correlations}
    """
    corr_df = compute_cross_correlation(tvi_series, sales_series, max_lag)
    
    if corr_df.empty:
        return {
            "lag_weeks": config.DEFAULT_CATEGORY_LAGS.get("General", 7) // 7,
            "correlation": 0.0,
            "p_value": 1.0,
            "all_correlations": corr_df,
        }
    
    # Find lag with maximum absolute correlation
    corr_df["abs_correlation"] = corr_df["correlation"].abs()
    best_idx = corr_df["abs_correlation"].idxmax()
    best_row = corr_df.loc[best_idx]
    
    return {
        "lag_weeks": int(best_row["lag_weeks"]),
        "lag_days": int(best_row["lag_weeks"]) * 7,
        "correlation": best_row["correlation"],
        "p_value": best_row["p_value"],
        "is_significant": best_row["is_significant"],
        "all_correlations": corr_df,
    }


def optimize_lags_by_category(
    sales_df: pd.DataFrame,
    trends_df: pd.DataFrame,
    category_keywords: Dict[str, list] = None,
    max_lag: int = 21,
    sales_col: str = "Weekly_Sales",
) -> Dict[str, dict]:
    """
    Find optimal lag for each product category.
    
    Parameters
    ----------
    sales_df : pd.DataFrame
        Sales data (aggregated weekly)
    trends_df : pd.DataFrame
        Google Trends data with keyword columns
    category_keywords : dict, optional
        Mapping of category → list of keywords.
        Defaults to config.TREND_KEYWORDS (inverted)
    max_lag : int
        Maximum lag to test
    sales_col : str
        Sales column name
        
    Returns
    -------
    dict
        Category → optimal lag info
    """
    # Build category → keywords mapping from config
    if category_keywords is None:
        category_keywords = {}
        for keyword, category in config.TREND_KEYWORDS.items():
            if category not in category_keywords:
                category_keywords[category] = []
            category_keywords[category].append(keyword)
    
    print("\n" + "=" * 60)
    print("🔍 TrendSense — Category Lag Optimization")
    print("=" * 60)
    
    # Aggregate sales to a single series (mean across all stores)
    if "Store" in sales_df.columns:
        weekly_sales = sales_df.groupby("Date")[sales_col].mean().sort_index()
    else:
        weekly_sales = sales_df.set_index("Date")[sales_col].sort_index()
    
    results = {}
    
    for category, keywords in category_keywords.items():
        print(f"\n📂 Category: {category}")
        print(f"   Keywords: {keywords}")
        
        category_results = []
        
        for keyword in keywords:
            if keyword not in trends_df.columns:
                print(f"   ⚠️ Keyword '{keyword}' not found in trends data, skipping")
                continue
            
            # Compute TVI for this keyword
            from src.tvi import compute_tvi
            tvi = compute_tvi(trends_df[keyword])
            
            # Align lengths
            min_len = min(len(tvi), len(weekly_sales))
            if min_len < 10:
                print(f"   ⚠️ Insufficient data overlap for '{keyword}', skipping")
                continue
            
            tvi_aligned = tvi.iloc[:min_len].reset_index(drop=True)
            sales_aligned = weekly_sales.iloc[:min_len].reset_index(drop=True)
            
            # Find optimal lag
            lag_result = find_optimal_lag(tvi_aligned, sales_aligned, max_lag)
            lag_result["keyword"] = keyword
            category_results.append(lag_result)
            
            sig = "✓" if lag_result["is_significant"] else "✗"
            print(f"   📊 '{keyword}': optimal lag = {lag_result['lag_weeks']}w "
                  f"({lag_result['lag_days']}d), corr = {lag_result['correlation']:.3f} "
                  f"(p={lag_result['p_value']:.3f}) [{sig}]")
        
        if category_results:
            # Use the keyword with strongest correlation for this category
            best = max(category_results, key=lambda x: abs(x["correlation"]))
            results[category] = {
                "optimal_lag_weeks": best["lag_weeks"],
                "optimal_lag_days": best["lag_days"],
                "best_keyword": best["keyword"],
                "correlation": best["correlation"],
                "p_value": best["p_value"],
                "is_significant": best["is_significant"],
                "default_lag_days": config.DEFAULT_CATEGORY_LAGS.get(category, 7),
                "all_keyword_results": category_results,
            }
        else:
            # Use default
            default_lag = config.DEFAULT_CATEGORY_LAGS.get(category, 7)
            results[category] = {
                "optimal_lag_weeks": default_lag // 7,
                "optimal_lag_days": default_lag,
                "best_keyword": None,
                "correlation": 0.0,
                "p_value": 1.0,
                "is_significant": False,
                "default_lag_days": default_lag,
                "all_keyword_results": [],
            }
    
    # Print summary
    print("\n" + "-" * 60)
    print("📋 LAG OPTIMIZATION SUMMARY")
    print("-" * 60)
    print(f"{'Category':<15} {'Optimal Lag':<12} {'Default Lag':<12} {'Corr':<8} {'Sig?':<5}")
    print("-" * 60)
    for cat, info in results.items():
        sig = "✓" if info["is_significant"] else "✗"
        print(f"{cat:<15} {info['optimal_lag_days']}d ({info['optimal_lag_weeks']}w)"
              f"   {info['default_lag_days']}d          "
              f"{info['correlation']:.3f}   {sig}")
    print("-" * 60)
    
    return results


def validate_lag_improvement(
    sales_df: pd.DataFrame,
    trends_df: pd.DataFrame,
    category: str,
    keyword: str,
    optimal_lag: int,
    fixed_lag: int = 7,
    sales_col: str = "Weekly_Sales",
) -> dict:
    """
    Validate that using optimal lag improves MAPE over fixed lag.
    
    Parameters
    ----------
    sales_df : pd.DataFrame
        Sales data
    trends_df : pd.DataFrame
        Trends data
    category : str
        Product category
    keyword : str
        Trend keyword
    optimal_lag : int
        Optimal lag in weeks (from find_optimal_lag)
    fixed_lag : int
        Fixed lag in days for comparison
    sales_col : str
        Sales column name
        
    Returns
    -------
    dict
        Validation results with MAPE comparison
    """
    from src.feature_engineering import merge_all_features, get_feature_columns
    from src.models import temporal_train_test_split, train_xgboost
    
    # Save original config and test with both lags
    results = {}
    
    for lag_label, lag_value in [("fixed", fixed_lag // 7), ("optimal", optimal_lag)]:
        # Override lag for this category temporarily
        original_lag = config.DEFAULT_CATEGORY_LAGS.get(category, 7)
        config.DEFAULT_CATEGORY_LAGS[category] = lag_value * 7
        
        try:
            # Run feature engineering with this lag
            featured_df = merge_all_features(sales_df, trends_df, category, sales_col)
            
            # Train model
            feature_cols = get_feature_columns(featured_df, sales_col)
            X_train, X_test, y_train, y_test = temporal_train_test_split(
                featured_df, sales_col, feature_cols
            )
            
            if len(X_train) > 0 and len(X_test) > 0:
                xgb_result = train_xgboost(X_train, y_train, X_test, y_test)
                results[lag_label] = {
                    "lag_weeks": lag_value,
                    "mape": xgb_result["metrics"]["MAPE (%)"],
                }
            
        finally:
            config.DEFAULT_CATEGORY_LAGS[category] = original_lag
    
    improvement = None
    if "fixed" in results and "optimal" in results:
        improvement = results["fixed"]["mape"] - results["optimal"]["mape"]
    
    return {
        "category": category,
        "keyword": keyword,
        "fixed_lag_mape": results.get("fixed", {}).get("mape"),
        "optimal_lag_mape": results.get("optimal", {}).get("mape"),
        "mape_improvement": improvement,
    }


# ──────────────────────────────────────────────
# Main execution for testing
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("TrendSense — Lag Optimizer Test")
    print("=" * 60)
    
    # Create synthetic data with a known lag relationship
    np.random.seed(42)
    n = 104  # 2 years of weekly data
    
    # Simulate TVI signal
    tvi = np.random.normal(0, 15, n)
    tvi[40:44] = [30, 60, 80, 50]  # Festive spike
    
    # Sales follow TVI with a 3-week lag
    sales = np.zeros(n)
    for i in range(3, n):
        sales[i] = 20000 + 200 * tvi[i - 3] + np.random.normal(0, 1000)
    
    tvi_series = pd.Series(tvi)
    sales_series = pd.Series(sales)
    
    # Test cross-correlation
    corr_df = compute_cross_correlation(tvi_series, sales_series, max_lag=10)
    print("\nCross-correlation results:")
    print(corr_df.to_string())
    
    # Find optimal lag
    result = find_optimal_lag(tvi_series, sales_series, max_lag=10)
    print(f"\n🎯 Optimal lag: {result['lag_weeks']} weeks ({result['lag_weeks'] * 7} days)")
    print(f"   Correlation: {result['correlation']:.4f}")
    print(f"   P-value: {result['p_value']:.4f}")
    print(f"   Significant: {result['is_significant']}")
    
    print("\n✅ Lag optimizer test complete!")
