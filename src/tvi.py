"""
TrendSense — Trend Velocity Index (TVI) Module
================================================
Original metric: Computes the rate of change (acceleration) of social media
search interest, not raw volume.

A search volume growing 100→400 in three days signals a viral spike
needing urgent restocking — entirely different from gradual growth.
"""

import pandas as pd
import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


def compute_tvi(trend_series: pd.Series) -> pd.Series:
    """
    Compute Trend Velocity Index — percentage rate of change.
    
    Formula: TVI(t) = [Trend(t) - Trend(t-1)] / Trend(t-1) × 100
    
    Parameters
    ----------
    trend_series : pd.Series
        Raw Google Trends interest values (0-100)
        
    Returns
    -------
    pd.Series
        TVI values (percentage change)
    """
    # Replace 0 with NaN, forward-fill, compute pct_change, and cap at +/- 500%
    safe_series = trend_series.replace(0, np.nan).ffill()
    tvi = safe_series.pct_change() * 100
    tvi = tvi.clip(-500.0, 500.0)
    return tvi


def compute_tvi_acceleration(tvi_series: pd.Series) -> pd.Series:
    """
    Compute TVI acceleration — rate of change of TVI (2nd derivative).
    
    Formula: TVI_accel(t) = TVI(t) - TVI(t-1)
    
    Parameters
    ----------
    tvi_series : pd.Series
        TVI values from compute_tvi()
        
    Returns
    -------
    pd.Series
        TVI acceleration values
    """
    return tvi_series.diff()


def compute_tvi_rolling(trend_series: pd.Series, window: int = 4) -> pd.Series:
    """
    Compute smoothed TVI using a rolling window to reduce noise.
    
    Parameters
    ----------
    trend_series : pd.Series
        Raw Google Trends interest values
    window : int
        Rolling window size (in weeks)
        
    Returns
    -------
    pd.Series
        Smoothed TVI values
    """
    smoothed = trend_series.rolling(window=window, min_periods=1).mean()
    return compute_tvi(smoothed)


def detect_spikes(
    tvi_series: pd.Series,
    threshold_sigma: float = 2.0,
    dates: pd.Series = None,
) -> pd.DataFrame:
    """
    Detect viral spikes in TVI data using statistical thresholds.
    
    A spike is flagged when TVI(t) > μ + threshold_sigma × σ
    
    Parameters
    ----------
    tvi_series : pd.Series
        TVI values
    threshold_sigma : float
        Number of standard deviations above mean to flag as spike
    dates : pd.Series, optional
        Corresponding dates for the TVI values
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: [date, tvi_value, severity, is_spike]
    """
    # Remove NaN values for statistics
    clean_tvi = tvi_series.dropna()
    
    if len(clean_tvi) == 0:
        return pd.DataFrame(columns=["date", "tvi_value", "severity", "is_spike"])
    
    mean_tvi = clean_tvi.mean()
    std_tvi = clean_tvi.std()
    
    # Classify each point
    results = []
    for i, tvi_val in enumerate(tvi_series):
        if pd.isna(tvi_val):
            severity = "NONE"
            is_spike = False
        else:
            severity = classify_spike_severity(tvi_val, mean_tvi, std_tvi)
            is_spike = severity != "NONE"
        
        date_val = dates.iloc[i] if dates is not None else i
        results.append({
            "date": date_val,
            "tvi_value": tvi_val,
            "severity": severity,
            "is_spike": is_spike,
        })
    
    return pd.DataFrame(results)


def classify_spike_severity(tvi_value: float, mean: float, std: float) -> str:
    """
    Classify spike severity based on standard deviation thresholds.
    
    Parameters
    ----------
    tvi_value : float
        Current TVI value
    mean : float
        Mean TVI over the period
    std : float
        Standard deviation of TVI
        
    Returns
    -------
    str
        One of: 'NONE', 'MILD', 'MODERATE', 'SEVERE'
    """
    if std == 0:
        return "NONE"
    
    z_score = (tvi_value - mean) / std
    
    if z_score >= config.TVI_SPIKE_THRESHOLDS["SEVERE"]:
        return "SEVERE"
    elif z_score >= config.TVI_SPIKE_THRESHOLDS["MODERATE"]:
        return "MODERATE"
    elif z_score >= config.TVI_SPIKE_THRESHOLDS["MILD"]:
        return "MILD"
    else:
        return "NONE"


def compute_tvi_features(trend_df: pd.DataFrame, keyword: str) -> pd.DataFrame:
    """
    Compute full TVI feature set for a given keyword.
    
    Returns DataFrame with columns:
    - date, raw_trend, tvi, tvi_acceleration, tvi_smoothed,
      is_spike, spike_severity
    
    Parameters
    ----------
    trend_df : pd.DataFrame
        Google Trends data with 'date' column and keyword columns
    keyword : str
        The keyword column to compute TVI for
        
    Returns
    -------
    pd.DataFrame
        Full TVI feature set
    """
    if keyword not in trend_df.columns:
        raise ValueError(f"Keyword '{keyword}' not found in trends data. Available: {list(trend_df.columns)}")
    
    result = pd.DataFrame()
    result["date"] = trend_df["date"]
    result["raw_trend"] = trend_df[keyword].values
    
    # TVI (rate of change)
    result["tvi"] = compute_tvi(result["raw_trend"])
    
    # TVI acceleration (2nd derivative)
    result["tvi_acceleration"] = compute_tvi_acceleration(result["tvi"])
    
    # Smoothed TVI (rolling window)
    result["tvi_smoothed"] = compute_tvi_rolling(result["raw_trend"], window=4)
    
    # Spike detection
    spike_df = detect_spikes(result["tvi"], threshold_sigma=2.0, dates=result["date"])
    result["is_spike"] = spike_df["is_spike"].values
    result["spike_severity"] = spike_df["severity"].values
    
    return result


def get_tvi_summary(tvi_features: pd.DataFrame) -> dict:
    """
    Generate a summary of TVI analysis.
    
    Parameters
    ----------
    tvi_features : pd.DataFrame
        Output from compute_tvi_features()
        
    Returns
    -------
    dict
        Summary statistics
    """
    clean_tvi = tvi_features["tvi"].dropna()
    
    total_spikes = tvi_features["is_spike"].sum()
    severe_spikes = (tvi_features["spike_severity"] == "SEVERE").sum()
    moderate_spikes = (tvi_features["spike_severity"] == "MODERATE").sum()
    mild_spikes = (tvi_features["spike_severity"] == "MILD").sum()
    
    return {
        "total_weeks": len(tvi_features),
        "mean_tvi": clean_tvi.mean() if len(clean_tvi) > 0 else 0,
        "std_tvi": clean_tvi.std() if len(clean_tvi) > 0 else 0,
        "max_tvi": clean_tvi.max() if len(clean_tvi) > 0 else 0,
        "min_tvi": clean_tvi.min() if len(clean_tvi) > 0 else 0,
        "total_spikes": int(total_spikes),
        "severe_spikes": int(severe_spikes),
        "moderate_spikes": int(moderate_spikes),
        "mild_spikes": int(mild_spikes),
        "spike_rate_pct": round(total_spikes / max(len(tvi_features), 1) * 100, 2),
    }


# ──────────────────────────────────────────────
# Main execution for testing
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("TrendSense — TVI Module Test")
    print("=" * 60)
    
    # Create sample trend data
    np.random.seed(42)
    n = 52  # 1 year of weekly data
    dates = pd.date_range("2024-01-01", periods=n, freq="W-SUN")
    
    # Simulate trend with a festive spike
    base = 30 + 5 * np.sin(2 * np.pi * np.arange(n) / 52) + np.random.normal(0, 3, n)
    base[40:44] = [60, 85, 95, 70]  # Diwali spike in Oct-Nov
    base = np.clip(base, 0, 100)
    
    sample_df = pd.DataFrame({
        "date": dates,
        "smartphones": base.astype(int)
    })
    
    # Compute TVI features
    tvi_features = compute_tvi_features(sample_df, "smartphones")
    print("\nTVI Features (first 10 rows):")
    print(tvi_features.head(10).to_string())
    
    # Summary
    summary = get_tvi_summary(tvi_features)
    print(f"\nTVI Summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    
    # Check spike detection
    spikes = tvi_features[tvi_features["is_spike"]]
    print(f"\n🔥 Detected {len(spikes)} spikes:")
    if len(spikes) > 0:
        print(spikes[["date", "tvi_value" if "tvi_value" in spikes.columns else "tvi", "spike_severity"]].to_string())
    
    print("\n✅ TVI module test complete!")
