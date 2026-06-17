"""
TrendSense — Data Ingestion Module
====================================
Handles downloading datasets from Kaggle and fetching Google Trends data.
"""

import os
import time
import pandas as pd
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


def download_walmart_data(dest_dir: str = None) -> pd.DataFrame:
    """
    Download the Walmart Store Sales dataset from Kaggle.
    
    Dataset: yasserh/walmart-dataset
    Columns: Store, Date, Weekly_Sales, Holiday_Flag, Temperature, 
             Fuel_Price, CPI, Unemployment
    ~6,400 rows across 45 stores (2010-2012)
    
    Parameters
    ----------
    dest_dir : str, optional
        Destination directory. Defaults to config.RAW_DATA_DIR
        
    Returns
    -------
    pd.DataFrame
        Walmart sales data with parsed dates
    """
    dest_dir = dest_dir or config.RAW_DATA_DIR
    csv_path = os.path.join(dest_dir, "Walmart.csv")
    
    # Check if already downloaded
    if not os.path.exists(csv_path):
        print("📥 Downloading Walmart dataset from Kaggle...")
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
            api = KaggleApi()
            api.authenticate()
            api.dataset_download_files(
                config.KAGGLE_DATASETS["walmart"],
                path=dest_dir,
                unzip=True
            )
            print(f"✅ Walmart data downloaded to {dest_dir}")
        except Exception as e:
            print(f"⚠️ Kaggle download failed: {e}")
            print("Please manually download from: https://www.kaggle.com/datasets/yasserh/walmart-dataset")
            print(f"Place the CSV file in: {dest_dir}")
            raise FileNotFoundError(f"Walmart.csv not found in {dest_dir}")
    else:
        print(f"✅ Walmart data already exists at {csv_path}")
    
    # Load and parse
    df = pd.read_csv(csv_path)
    
    # Handle different date formats
    try:
        df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y")
    except (ValueError, TypeError):
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
    
    df = df.sort_values(["Store", "Date"]).reset_index(drop=True)
    
    print(f"📊 Walmart data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"   Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
    print(f"   Stores: {df['Store'].nunique()}")
    
    return df


def download_rossmann_data(dest_dir: str = None) -> pd.DataFrame:
    """
    Download the Rossmann Store Sales dataset from Kaggle.
    
    Dataset: pratyushakar/rossmann-store-sales
    Columns: Store, DayOfWeek, Date, Sales, Customers, Open, 
             Promo, StateHoliday, SchoolHoliday
    ~1M rows across 1,115 stores (2013-2015)
    
    Parameters
    ----------
    dest_dir : str, optional
        Destination directory. Defaults to config.RAW_DATA_DIR
        
    Returns
    -------
    pd.DataFrame
        Rossmann sales data with parsed dates
    """
    dest_dir = dest_dir or config.RAW_DATA_DIR
    csv_path = os.path.join(dest_dir, "train.csv")
    
    # Also check for "rossmann_train.csv" as alternate name
    alt_csv_path = os.path.join(dest_dir, "rossmann_train.csv")
    
    if os.path.exists(alt_csv_path):
        csv_path = alt_csv_path
    
    if not os.path.exists(csv_path):
        print("📥 Downloading Rossmann dataset from Kaggle...")
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
            api = KaggleApi()
            api.authenticate()
            api.dataset_download_files(
                config.KAGGLE_DATASETS["rossmann"],
                path=dest_dir,
                unzip=True
            )
            print(f"✅ Rossmann data downloaded to {dest_dir}")
        except Exception as e:
            print(f"⚠️ Kaggle download failed: {e}")
            print("Please manually download from: https://www.kaggle.com/datasets/pratyushakar/rossmann-store-sales")
            print(f"Place the CSV file in: {dest_dir}")
            raise FileNotFoundError(f"Rossmann train.csv not found in {dest_dir}")
    else:
        print(f"✅ Rossmann data already exists at {csv_path}")
    
    # Load and parse
    df = pd.read_csv(csv_path, low_memory=False)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Store", "Date"]).reset_index(drop=True)
    
    print(f"📊 Rossmann data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"   Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
    print(f"   Stores: {df['Store'].nunique()}")
    
    return df


def aggregate_rossmann_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate daily Rossmann data to weekly granularity to match Walmart format.
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw Rossmann daily data
        
    Returns
    -------
    pd.DataFrame
        Weekly aggregated data with columns similar to Walmart
    """
    df = df[df["Open"] == 1].copy()  # Only open stores
    
    # Create week start column (ISO week)
    df["Week_Start"] = df["Date"].dt.to_period("W").apply(lambda x: x.start_time)
    
    weekly = df.groupby(["Store", "Week_Start"]).agg(
        Weekly_Sales=("Sales", "sum"),
        Weekly_Customers=("Customers", "sum"),
        Promo_Days=("Promo", "sum"),
        Holiday_Flag=("StateHoliday", lambda x: int((x != "0").any())),
        Open_Days=("Open", "sum"),
    ).reset_index()
    
    weekly.rename(columns={"Week_Start": "Date"}, inplace=True)
    weekly = weekly.sort_values(["Store", "Date"]).reset_index(drop=True)
    
    print(f"📊 Rossmann weekly aggregation: {weekly.shape[0]} rows")
    return weekly


def fetch_google_trends(
    keywords: list = None,
    timeframe: str = None,
    geo: str = None,
    cache: bool = True,
) -> pd.DataFrame:
    """
    Fetch Google Trends data for given keywords (India region).
    
    Uses pytrends library. Results are cached to avoid rate limiting.
    Google Trends returns weekly interest scores (0-100) for each keyword.
    
    Parameters
    ----------
    keywords : list, optional
        Search keywords. Defaults to keys of config.TREND_KEYWORDS
    timeframe : str, optional
        Time range. Defaults to config.TRENDS_TIMEFRAME
    geo : str, optional
        Geographic region. Defaults to config.TRENDS_GEO ('IN')
    cache : bool
        Whether to cache results as CSV
        
    Returns
    -------
    pd.DataFrame
        Weekly Google Trends interest data with 'date' index
    """
    keywords = keywords or list(config.TREND_KEYWORDS.keys())
    timeframe = timeframe or config.TRENDS_TIMEFRAME
    geo = geo or config.TRENDS_GEO
    
    cache_path = os.path.join(config.TRENDS_DATA_DIR, "google_trends_india.csv")
    
    # Check cache first
    if cache and os.path.exists(cache_path):
        print(f"✅ Loading cached Google Trends data from {cache_path}")
        df = pd.read_csv(cache_path, parse_dates=["date"])
        return df
    
    print(f"📥 Fetching Google Trends for {len(keywords)} keywords...")
    print(f"   Region: {geo}, Timeframe: {timeframe}")
    
    try:
        from pytrends.request import TrendReq
        
        pytrends = TrendReq(hl="en-US", tz=330)  # IST offset
        
        all_trends = []
        
        # pytrends allows max 5 keywords per request
        for i in range(0, len(keywords), 5):
            batch = keywords[i:i + 5]
            print(f"   Fetching batch: {batch}")
            
            pytrends.build_payload(batch, timeframe=timeframe, geo=geo)
            trend_data = pytrends.interest_over_time()
            
            if not trend_data.empty:
                trend_data = trend_data.drop(columns=["isPartial"], errors="ignore")
                all_trends.append(trend_data)
            
            # Respect rate limits
            if i + 5 < len(keywords):
                print("   ⏳ Waiting 15s to respect rate limits...")
                time.sleep(15)
        
        if all_trends:
            df = pd.concat(all_trends, axis=1)
            df = df.loc[:, ~df.columns.duplicated()]  # Remove duplicate columns
            df.index.name = "date"
            df = df.reset_index()
            
            # Cache results
            if cache:
                df.to_csv(cache_path, index=False)
                print(f"💾 Cached trends data to {cache_path}")
            
            print(f"📊 Google Trends data: {df.shape[0]} weeks, {df.shape[1] - 1} keywords")
            return df
        else:
            print("⚠️ No Google Trends data returned")
            return pd.DataFrame()
            
    except ImportError:
        print("⚠️ pytrends not installed. Install with: pip install pytrends")
        raise
    except Exception as e:
        print(f"⚠️ Google Trends fetch failed: {e}")
        print("Generating synthetic trends data as fallback...")
        return _generate_synthetic_trends(keywords, timeframe)


def _generate_synthetic_trends(keywords: list, timeframe: str) -> pd.DataFrame:
    """
    Generate synthetic Google Trends data for testing and development.
    Simulates realistic search interest patterns with seasonality and spikes.
    
    Parameters
    ----------
    keywords : list
        Keywords to generate data for
    timeframe : str
        Time range string like '2010-01-01 2024-12-31'
        
    Returns
    -------
    pd.DataFrame
        Synthetic weekly trend data
    """
    dates = timeframe.split(" ")
    start_date = pd.to_datetime(dates[0])
    end_date = pd.to_datetime(dates[1]) if len(dates) > 1 else pd.to_datetime("2024-12-31")
    
    date_range = pd.date_range(start=start_date, end=end_date, freq="W-SUN")
    
    np.random.seed(42)
    data = {"date": date_range}
    
    for keyword in keywords:
        n = len(date_range)
        
        # Base trend (gradual increase over years)
        base = np.linspace(20, 50, n) + np.random.normal(0, 3, n)
        
        # Weekly seasonality
        weekly_pattern = 5 * np.sin(2 * np.pi * np.arange(n) / 52)
        
        # Festive spikes (Oct-Nov for Diwali/festive season)
        festive_spike = np.zeros(n)
        for i, d in enumerate(date_range):
            if d.month in [10, 11]:  # Diwali season
                festive_spike[i] = np.random.uniform(20, 50)
            elif d.month == 1 and d.day < 31:  # Republic Day sales
                festive_spike[i] = np.random.uniform(10, 25)
        
        # Random viral spikes
        spike_indices = np.random.choice(n, size=max(1, n // 50), replace=False)
        viral_spikes = np.zeros(n)
        viral_spikes[spike_indices] = np.random.uniform(30, 60, size=len(spike_indices))
        
        # Combine and clip to 0-100
        trend = base + weekly_pattern + festive_spike + viral_spikes
        trend = np.clip(trend, 0, 100).astype(int)
        
        data[keyword] = trend
    
    df = pd.DataFrame(data)
    
    # Cache synthetic data
    cache_path = os.path.join(config.TRENDS_DATA_DIR, "google_trends_india.csv")
    df.to_csv(cache_path, index=False)
    print(f"📊 Synthetic trends generated: {df.shape[0]} weeks, {len(keywords)} keywords")
    print(f"💾 Cached to {cache_path}")
    
    return df


def load_or_create_dataset(dataset_name: str = "walmart") -> pd.DataFrame:
    """
    Convenience function to load a dataset, downloading if needed.
    
    Parameters
    ----------
    dataset_name : str
        One of 'walmart', 'rossmann', 'trends'
        
    Returns
    -------
    pd.DataFrame
    """
    if dataset_name == "walmart":
        return download_walmart_data()
    elif dataset_name == "rossmann":
        return download_rossmann_data()
    elif dataset_name == "trends":
        return fetch_google_trends()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Use 'walmart', 'rossmann', or 'trends'")


# ──────────────────────────────────────────────
# Main execution for testing
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("TrendSense — Data Ingestion Test")
    print("=" * 60)
    
    # Test Walmart
    try:
        walmart_df = download_walmart_data()
        print(f"\nWalmart sample:\n{walmart_df.head()}\n")
    except FileNotFoundError:
        print("⏭️ Skipping Walmart (not downloaded yet)")
    
    # Test Google Trends (synthetic fallback)
    trends_df = fetch_google_trends()
    print(f"\nTrends sample:\n{trends_df.head()}\n")
    
    print("✅ Data ingestion module test complete!")
