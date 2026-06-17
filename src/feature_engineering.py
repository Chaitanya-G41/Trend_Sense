"""
TrendSense — Feature Engineering Module
=========================================
Creates ML-ready features from raw sales and trend data.
Includes temporal, holiday, lag, rolling, and TVI-based features.
"""

import pandas as pd
import numpy as np
from typing import Optional, List

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.tvi import compute_tvi_features


def create_temporal_features(df: pd.DataFrame, date_col: str = "Date") -> pd.DataFrame:
    """
    Create calendar-based temporal features.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input data with a date column
    date_col : str
        Name of the date column
        
    Returns
    -------
    pd.DataFrame
        DataFrame with added temporal features
    """
    df = df.copy()
    dt = pd.to_datetime(df[date_col])
    
    df["week_of_year"] = dt.dt.isocalendar().week.astype(int)
    df["month"] = dt.dt.month
    df["quarter"] = dt.dt.quarter
    df["year"] = dt.dt.year
    df["day_of_year"] = dt.dt.dayofyear
    df["is_month_start"] = dt.dt.is_month_start.astype(int)
    df["is_month_end"] = dt.dt.is_month_end.astype(int)
    df["is_year_end"] = (dt.dt.month == 12).astype(int)
    
    # Cyclical encoding for week and month (captures circular nature)
    df["week_sin"] = np.sin(2 * np.pi * df["week_of_year"] / 52)
    df["week_cos"] = np.cos(2 * np.pi * df["week_of_year"] / 52)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    
    return df


def create_holiday_features(df: pd.DataFrame, date_col: str = "Date") -> pd.DataFrame:
    """
    Create Indian holiday and festive season features.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input data with a date column
    date_col : str
        Name of the date column
        
    Returns
    -------
    pd.DataFrame
        DataFrame with added holiday features
    """
    df = df.copy()
    dt = pd.to_datetime(df[date_col])
    
    # Festive season flags (October-November)
    df["is_festive_season"] = dt.dt.month.isin([10, 11]).astype(int)
    
    # Diwali week proximity (approximate: last week of Oct / first week of Nov)
    df["is_diwali_week"] = ((dt.dt.month == 10) & (dt.dt.day >= 25) | 
                            (dt.dt.month == 11) & (dt.dt.day <= 7)).astype(int)
    
    # Big Billion Day / Great Indian Festival (early October)
    df["is_ecommerce_sale"] = ((dt.dt.month == 10) & (dt.dt.day <= 15)).astype(int)
    
    # Republic Day sale (late January)
    df["is_republic_day_sale"] = ((dt.dt.month == 1) & (dt.dt.day >= 20)).astype(int)
    
    # End of season sale (June-July)
    df["is_eos_sale"] = dt.dt.month.isin([6, 7]).astype(int)
    
    # Days to nearest holiday (approximate)
    holiday_dates = []
    for year in dt.dt.year.unique():
        for name, info in config.INDIAN_HOLIDAYS.items():
            try:
                hdate = pd.Timestamp(year=year, month=info["month"], day=info["day"])
                holiday_dates.append(hdate)
            except ValueError:
                pass
    
    if holiday_dates:
        holiday_dates = sorted(holiday_dates)
        
        def days_to_nearest(date):
            diffs = [abs((h - date).days) for h in holiday_dates]
            return min(diffs) if diffs else 999
        
        df["days_to_nearest_holiday"] = dt.apply(days_to_nearest)
        df["is_near_holiday"] = (df["days_to_nearest_holiday"] <= 7).astype(int)
    else:
        df["days_to_nearest_holiday"] = 999
        df["is_near_holiday"] = 0
    
    return df


def create_lag_features(
    df: pd.DataFrame,
    target_col: str = "Weekly_Sales",
    lags: list = None,
    group_col: str = "Store",
) -> pd.DataFrame:
    """
    Create autoregressive lag features.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input data sorted by date
    target_col : str
        Column to create lags for
    lags : list
        List of lag periods (in weeks)
    group_col : str
        Column to group by (e.g., Store)
        
    Returns
    -------
    pd.DataFrame
        DataFrame with added lag features
    """
    df = df.copy()
    lags = lags or [1, 2, 4, 8, 12]
    
    if group_col and group_col in df.columns:
        for lag in lags:
            df[f"{target_col}_lag_{lag}w"] = df.groupby(group_col)[target_col].shift(lag)
    else:
        for lag in lags:
            df[f"{target_col}_lag_{lag}w"] = df[target_col].shift(lag)
    
    return df


def create_rolling_features(
    df: pd.DataFrame,
    target_col: str = "Weekly_Sales",
    windows: list = None,
    group_col: str = "Store",
) -> pd.DataFrame:
    """
    Create rolling window statistics features.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input data sorted by date
    target_col : str
        Column to compute rolling stats for
    windows : list
        List of rolling window sizes (in weeks)
    group_col : str
        Column to group by (e.g., Store)
        
    Returns
    -------
    pd.DataFrame
        DataFrame with added rolling features
    """
    df = df.copy()
    windows = windows or [4, 8, 12]
    
    for window in windows:
        if group_col and group_col in df.columns:
            grouped = df.groupby(group_col)[target_col]
            df[f"{target_col}_rolling_mean_{window}w"] = grouped.transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
            df[f"{target_col}_rolling_std_{window}w"] = grouped.transform(
                lambda x: x.rolling(window=window, min_periods=1).std()
            )
            df[f"{target_col}_rolling_min_{window}w"] = grouped.transform(
                lambda x: x.rolling(window=window, min_periods=1).min()
            )
            df[f"{target_col}_rolling_max_{window}w"] = grouped.transform(
                lambda x: x.rolling(window=window, min_periods=1).max()
            )
        else:
            df[f"{target_col}_rolling_mean_{window}w"] = df[target_col].rolling(window=window, min_periods=1).mean()
            df[f"{target_col}_rolling_std_{window}w"] = df[target_col].rolling(window=window, min_periods=1).std()
    
    return df


def create_tvi_features(
    df: pd.DataFrame,
    trends_df: pd.DataFrame,
    keyword: str,
    lag_weeks: int = 1,
    date_col: str = "Date",
) -> pd.DataFrame:
    """
    Merge TVI features from Google Trends into the sales DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        Sales data with a date column
    trends_df : pd.DataFrame
        Google Trends data with 'date' column and keyword columns
    keyword : str
        The keyword to compute TVI for
    lag_weeks : int
        How many weeks to lag the TVI features (to prevent leakage)
    date_col : str
        Date column in sales DataFrame
        
    Returns
    -------
    pd.DataFrame
        Sales data with TVI features merged in
    """
    df = df.copy()
    
    # Compute TVI features for the keyword
    tvi_df = compute_tvi_features(trends_df, keyword)
    
    # Rename columns to include keyword prefix
    tvi_cols = {
        "tvi": f"tvi_{keyword}",
        "tvi_acceleration": f"tvi_accel_{keyword}",
        "tvi_smoothed": f"tvi_smooth_{keyword}",
        "is_spike": f"is_spike_{keyword}",
        "spike_severity": f"spike_severity_{keyword}",
        "raw_trend": f"trend_raw_{keyword}",
    }
    tvi_df = tvi_df.rename(columns=tvi_cols)
    
    # Apply lag to prevent data leakage
    for col in tvi_cols.values():
        if col in tvi_df.columns:
            tvi_df[col] = tvi_df[col].shift(lag_weeks)
    
    # Merge on date (week-level)
    df[date_col] = pd.to_datetime(df[date_col])
    tvi_df["date"] = pd.to_datetime(tvi_df["date"])
    
    # Use week-start for merge compatibility
    df["_merge_week"] = df[date_col].dt.to_period("W").apply(lambda x: x.start_time)
    tvi_df["_merge_week"] = tvi_df["date"].dt.to_period("W").apply(lambda x: x.start_time)
    
    # Drop raw date from TVI to avoid conflicts
    tvi_merge = tvi_df.drop(columns=["date"])
    
    df = df.merge(tvi_merge, on="_merge_week", how="left")
    df.drop(columns=["_merge_week"], inplace=True)
    
    return df


def encode_spike_severity(df: pd.DataFrame, severity_col: str) -> pd.DataFrame:
    """
    Encode spike severity as ordinal numeric.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data with a spike severity column
    severity_col : str
        Name of the severity column
        
    Returns
    -------
    pd.DataFrame
        Data with encoded severity
    """
    df = df.copy()
    severity_map = {"NONE": 0, "MILD": 1, "MODERATE": 2, "SEVERE": 3}
    encoded_col = f"{severity_col}_encoded"
    df[encoded_col] = df[severity_col].map(severity_map).fillna(0).astype(int)
    return df


def merge_all_features(
    sales_df: pd.DataFrame,
    trends_df: pd.DataFrame,
    category: str = "General",
    target_col: str = "Weekly_Sales",
    date_col: str = "Date",
) -> pd.DataFrame:
    """
    End-to-end feature engineering pipeline.
    Combines all feature types into a single ML-ready DataFrame.
    
    Parameters
    ----------
    sales_df : pd.DataFrame
        Raw sales data
    trends_df : pd.DataFrame
        Google Trends data
    category : str
        Product category for lag optimization
    target_col : str
        Target column name
    date_col : str
        Date column name
        
    Returns
    -------
    pd.DataFrame
        ML-ready feature DataFrame
    """
    print(f"🔧 Engineering features for category: {category}")
    
    df = sales_df.copy()
    
    # 1. Temporal features
    df = create_temporal_features(df, date_col)
    print(f"   ✅ Temporal features: {df.shape[1]} columns")
    
    # 2. Holiday features
    df = create_holiday_features(df, date_col)
    print(f"   ✅ Holiday features: {df.shape[1]} columns")
    
    # 3. Lag features
    df = create_lag_features(df, target_col)
    print(f"   ✅ Lag features: {df.shape[1]} columns")
    
    # 4. Rolling statistics
    df = create_rolling_features(df, target_col)
    print(f"   ✅ Rolling features: {df.shape[1]} columns")
    
    # 5. TVI features (for each available keyword in the category)
    category_keywords = [k for k, v in config.TREND_KEYWORDS.items() if v == category]
    if not category_keywords:
        # Use first available keyword as fallback
        category_keywords = list(config.TREND_KEYWORDS.keys())[:1]
    
    lag_weeks = config.DEFAULT_CATEGORY_LAGS.get(category, 7) // 7  # Convert days to weeks
    lag_weeks = max(lag_weeks, 1)  # At least 1 week lag
    
    for keyword in category_keywords:
        if keyword in trends_df.columns:
            df = create_tvi_features(df, trends_df, keyword, lag_weeks, date_col)
            # Encode severity
            sev_col = f"spike_severity_{keyword}"
            if sev_col in df.columns:
                df = encode_spike_severity(df, sev_col)
            print(f"   ✅ TVI features for '{keyword}': {df.shape[1]} columns")
    
    # 6. Drop rows with NaN from lagging (first few weeks)
    initial_rows = len(df)
    df = df.dropna(subset=[f"{target_col}_lag_1w"]).reset_index(drop=True)
    dropped = initial_rows - len(df)
    print(f"   ℹ️ Dropped {dropped} rows with NaN from lagging")
    
    print(f"📊 Final feature set: {df.shape[0]} rows × {df.shape[1]} columns")
    
    return df


def get_feature_columns(df: pd.DataFrame, target_col: str = "Weekly_Sales", exclude_cols: list = None) -> list:
    """
    Get list of feature columns (exclude target, date, and non-numeric columns).
    
    Parameters
    ----------
    df : pd.DataFrame
        Feature-engineered DataFrame
    target_col : str
        Target column to exclude
    exclude_cols : list
        Additional columns to exclude
        
    Returns
    -------
    list
        List of feature column names
    """
    exclude = {target_col, "Date", "date", "Store", "_merge_week"}
    if exclude_cols:
        exclude.update(exclude_cols)
    
    # Only include numeric columns
    feature_cols = [
        col for col in df.select_dtypes(include=[np.number]).columns
        if col not in exclude
    ]
    
    return feature_cols


# ──────────────────────────────────────────────
# Main execution for testing
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("TrendSense — Feature Engineering Test")
    print("=" * 60)
    
    # Create sample sales data
    np.random.seed(42)
    n_weeks = 104  # 2 years
    dates = pd.date_range("2023-01-01", periods=n_weeks, freq="W-SUN")
    
    sales_df = pd.DataFrame({
        "Store": 1,
        "Date": dates,
        "Weekly_Sales": np.random.uniform(10000, 50000, n_weeks),
        "Holiday_Flag": np.random.choice([0, 1], n_weeks, p=[0.85, 0.15]),
    })
    
    # Create sample trends data
    trends_df = pd.DataFrame({
        "date": dates,
        "smartphones": np.random.randint(20, 80, n_weeks),
        "Diwali offers": np.random.randint(10, 90, n_weeks),
    })
    
    # Run full pipeline
    featured_df = merge_all_features(sales_df, trends_df, category="Electronics")
    
    print(f"\nFeature columns:")
    feature_cols = get_feature_columns(featured_df)
    for col in feature_cols:
        print(f"  • {col}")
    
    print(f"\nSample (first 3 rows):")
    print(featured_df.head(3).to_string())
    
    print("\n✅ Feature engineering test complete!")
