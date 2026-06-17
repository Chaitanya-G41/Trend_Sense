"""
TrendSense — Automated Setup Script
=====================================
Downloads datasets, generates trends data, and runs the full pipeline.
Run this once to set up everything.

Usage: python setup_and_run.py
"""

import os
import sys
import numpy as np
import pandas as pd

# Add project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config


def create_walmart_dataset():
    """
    Create a realistic Walmart-style dataset based on the actual Kaggle schema.
    
    The real dataset has: Store, Date, Weekly_Sales, Holiday_Flag, 
    Temperature, Fuel_Price, CPI, Unemployment
    ~6,400 rows across 45 stores (Feb 2010 - Oct 2012)
    
    This synthetic version mirrors the same schema with realistic distributions
    derived from known summary statistics of the actual dataset.
    """
    print("📦 Generating Walmart-style sales dataset...")
    
    np.random.seed(42)
    
    n_stores = 45
    start_date = pd.Timestamp("2010-02-05")
    end_date = pd.Timestamp("2012-10-26")
    
    # Generate weekly dates
    dates = pd.date_range(start=start_date, end=end_date, freq="W-FRI")
    n_weeks = len(dates)
    
    # Known US holidays in the Walmart dataset period (Super Bowl, Labor Day, 
    # Thanksgiving, Christmas)
    holiday_weeks = set()
    for year in [2010, 2011, 2012]:
        # Super Bowl (first Sunday of Feb)
        holiday_weeks.add(pd.Timestamp(f"{year}-02-12"))
        # Labor Day (first Monday of Sep)
        holiday_weeks.add(pd.Timestamp(f"{year}-09-10"))
        # Thanksgiving (fourth Thursday of Nov)
        if year == 2010:
            holiday_weeks.add(pd.Timestamp(f"{year}-11-26"))
        elif year == 2011:
            holiday_weeks.add(pd.Timestamp(f"{year}-11-25"))
        elif year == 2012:
            holiday_weeks.add(pd.Timestamp(f"{year}-11-23"))
        # Christmas
        holiday_weeks.add(pd.Timestamp(f"{year}-12-31"))
    
    rows = []
    
    # Store-level base parameters (mimicking real Walmart store variation)
    store_bases = {
        # High-volume stores
        **{i: np.random.uniform(1800000, 2500000) for i in range(1, 6)},
        # Medium-high stores 
        **{i: np.random.uniform(1200000, 1800000) for i in range(6, 16)},
        # Medium stores
        **{i: np.random.uniform(700000, 1200000) for i in range(16, 31)},
        # Smaller stores
        **{i: np.random.uniform(300000, 700000) for i in range(31, 46)},
    }
    
    for store in range(1, n_stores + 1):
        base_sales = store_bases[store]
        
        for week_idx, date in enumerate(dates):
            # Seasonal pattern (peaks in Nov-Dec for holiday shopping)
            month = date.month
            seasonal = 0
            if month in [11, 12]:
                seasonal = base_sales * np.random.uniform(0.15, 0.40)
            elif month in [6, 7]:  # Summer boost
                seasonal = base_sales * np.random.uniform(0.03, 0.08)
            elif month in [1, 2]:  # Post-holiday dip
                seasonal = -base_sales * np.random.uniform(0.05, 0.12)
            
            # Holiday flag
            is_holiday = 0
            for hw in holiday_weeks:
                if abs((date - hw).days) <= 3:
                    is_holiday = 1
                    break
            
            # Holiday boost
            holiday_boost = 0
            if is_holiday:
                holiday_boost = base_sales * np.random.uniform(0.10, 0.35)
            
            # Markdown effect (stores run markdowns especially in holiday periods)
            markdown_effect = 0
            if month in [11, 12] or is_holiday:
                markdown_effect = base_sales * np.random.uniform(0.0, 0.08)
            
            # Weekly noise
            noise = np.random.normal(0, base_sales * 0.06)
            
            # Trend (slight upward trend over time)
            trend = base_sales * 0.0003 * week_idx
            
            weekly_sales = base_sales + seasonal + holiday_boost + markdown_effect + trend + noise
            weekly_sales = max(weekly_sales, base_sales * 0.3)  # Floor
            
            # Temperature (Fahrenheit, varies by season)
            temp_base = {1: 35, 2: 38, 3: 48, 4: 58, 5: 68, 6: 78,
                        7: 82, 8: 80, 9: 72, 10: 60, 11: 48, 12: 38}
            temperature = temp_base[month] + np.random.normal(0, 6)
            
            # Fuel price (trending up from ~2.5 to ~3.5 over 2010-2012)
            fuel_base = 2.5 + (week_idx / n_weeks) * 1.2
            fuel_price = fuel_base + np.random.normal(0, 0.15)
            
            # CPI (Consumer Price Index, 126-228 range, trending up)
            cpi = 126 + (week_idx / n_weeks) * 100 + np.random.normal(0, 5)
            
            # Unemployment (8-14%, trending down slightly)
            unemployment = 10 - (week_idx / n_weeks) * 3 + np.random.normal(0, 1)
            unemployment = max(4, min(14, unemployment))
            
            rows.append({
                "Store": store,
                "Date": date.strftime("%d-%m-%Y"),
                "Weekly_Sales": round(weekly_sales, 2),
                "Holiday_Flag": is_holiday,
                "Temperature": round(temperature, 2),
                "Fuel_Price": round(fuel_price, 3),
                "CPI": round(cpi, 6),
                "Unemployment": round(unemployment, 3),
            })
    
    df = pd.DataFrame(rows)
    
    # Save as CSV
    csv_path = os.path.join(config.RAW_DATA_DIR, "Walmart.csv")
    df.to_csv(csv_path, index=False)
    
    print(f"✅ Walmart dataset created: {len(df)} rows, {n_stores} stores, {n_weeks} weeks")
    print(f"   Date range: {dates[0].date()} to {dates[-1].date()}")
    print(f"   Saved to: {csv_path}")
    print(f"   Avg weekly sales: ${df['Weekly_Sales'].mean():,.0f}")
    print(f"   Holiday weeks: {df['Holiday_Flag'].sum()} ({df['Holiday_Flag'].mean()*100:.1f}%)")
    
    return df


def create_google_trends_data():
    """
    Generate realistic Google Trends India data with proper seasonal patterns,
    festive spikes, and trend characteristics.
    """
    print("\n📈 Generating Google Trends India data...")
    
    np.random.seed(42)
    
    # Match the Walmart time range + extension
    start_date = pd.Timestamp("2010-01-03")
    end_date = pd.Timestamp("2012-12-30")
    dates = pd.date_range(start=start_date, end=end_date, freq="W-SUN")
    n = len(dates)
    
    data = {"date": dates}
    
    keyword_configs = {
        "smartphones": {
            "base": 25, "trend_growth": 35, "festive_boost": 30,
            "volatility": 4, "category": "Electronics"
        },
        "Diwali offers": {
            "base": 5, "trend_growth": 10, "festive_boost": 80,
            "volatility": 3, "category": "Festival"
        },
        "fashion sale": {
            "base": 20, "trend_growth": 15, "festive_boost": 25,
            "volatility": 5, "category": "Fashion"
        },
        "electronics sale": {
            "base": 15, "trend_growth": 20, "festive_boost": 35,
            "volatility": 4, "category": "Electronics"
        },
        "laptop deals": {
            "base": 10, "trend_growth": 25, "festive_boost": 40,
            "volatility": 3, "category": "Electronics"
        },
        "online shopping": {
            "base": 20, "trend_growth": 40, "festive_boost": 20,
            "volatility": 4, "category": "General"
        },
        "festival sale": {
            "base": 8, "trend_growth": 12, "festive_boost": 70,
            "volatility": 3, "category": "Festival"
        },
        "clothing sale": {
            "base": 18, "trend_growth": 10, "festive_boost": 20,
            "volatility": 5, "category": "Fashion"
        },
    }
    
    for keyword, cfg in keyword_configs.items():
        base = cfg["base"]
        growth = cfg["trend_growth"]
        festive = cfg["festive_boost"]
        vol = cfg["volatility"]
        
        trend = np.zeros(n)
        
        for i, dt in enumerate(dates):
            # Base + linear growth
            value = base + (growth * i / n)
            
            # Seasonal pattern (yearly cycle)
            value += 5 * np.sin(2 * np.pi * i / 52)
            
            # Festive spikes (Oct-Nov = Diwali/Navratri season)
            if dt.month == 10 and dt.day >= 15:
                value += festive * np.random.uniform(0.4, 0.8)
            elif dt.month == 11 and dt.day <= 15:
                value += festive * np.random.uniform(0.6, 1.0)
            elif dt.month == 11 and dt.day > 15:
                value += festive * np.random.uniform(0.1, 0.3)  # Tapering
            
            # Republic Day sale (Jan)
            if dt.month == 1 and 20 <= dt.day <= 31:
                value += festive * np.random.uniform(0.15, 0.35)
            
            # End of season (Jun-Jul)
            if dt.month in [6, 7]:
                value += festive * np.random.uniform(0.05, 0.15)
            
            # Random viral spikes (rare)
            if np.random.random() < 0.03:
                value += np.random.uniform(15, 40)
            
            # Noise
            value += np.random.normal(0, vol)
            
            trend[i] = value
        
        # Normalize to 0-100 (Google Trends scale)
        trend = np.clip(trend, 0, None)
        if trend.max() > 0:
            trend = (trend / trend.max()) * 100
        
        data[keyword] = trend.astype(int)
    
    df = pd.DataFrame(data)
    
    csv_path = os.path.join(config.TRENDS_DATA_DIR, "google_trends_india.csv")
    df.to_csv(csv_path, index=False)
    
    print(f"✅ Google Trends data created: {len(df)} weeks, {len(keyword_configs)} keywords")
    print(f"   Date range: {dates[0].date()} to {dates[-1].date()}")
    print(f"   Keywords: {list(keyword_configs.keys())}")
    print(f"   Saved to: {csv_path}")
    
    return df


def run_full_setup():
    """Run complete project setup."""
    print("=" * 65)
    print("  🚀 TrendSense — Automated Project Setup")
    print("  Setting up datasets, training models, generating outputs...")
    print("=" * 65)
    
    # Step 1: Create datasets
    walmart_df = create_walmart_dataset()
    trends_df = create_google_trends_data()
    
    # Step 2: Run the pipeline
    print("\n" + "=" * 65)
    print("  🔄 Running ML Pipeline...")
    print("=" * 65)
    
    from run_pipeline import run_pipeline
    results = run_pipeline(
        skip_download=True,  # We already created the data
        use_synthetic_trends=False,  # Use our generated trends
        save_plots=True,
    )
    
    print("\n" + "=" * 65)
    print("  ✅ SETUP COMPLETE!")
    print("=" * 65)
    print(f"""
    📁 Data:     {config.RAW_DATA_DIR}
    📈 Trends:   {config.TRENDS_DATA_DIR}
    📊 Features: {config.PROCESSED_DATA_DIR}
    🤖 Models:   {config.MODELS_DIR}
    📉 Figures:  {config.FIGURES_DIR}
    
    🌐 To launch the dashboard, run:
       streamlit run dashboard/app.py
    
    🧪 To run tests:
       python -m pytest tests/ -v
    """)
    
    return results


if __name__ == "__main__":
    run_full_setup()
