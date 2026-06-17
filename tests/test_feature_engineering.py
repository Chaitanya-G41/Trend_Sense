"""
TrendSense — Unit Tests: Feature Engineering
===============================================
"""

import sys
import os
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.feature_engineering import (
    create_temporal_features,
    create_holiday_features,
    create_lag_features,
    create_rolling_features,
    encode_spike_severity,
    get_feature_columns,
)


@pytest.fixture
def sample_sales_df():
    """Create sample sales DataFrame for testing."""
    np.random.seed(42)
    n = 52
    dates = pd.date_range("2024-01-01", periods=n, freq="W-SUN")
    return pd.DataFrame({
        "Store": 1,
        "Date": dates,
        "Weekly_Sales": np.random.uniform(10000, 50000, n),
        "Holiday_Flag": np.random.choice([0, 1], n, p=[0.85, 0.15]),
    })


class TestTemporalFeatures:
    """Tests for create_temporal_features."""

    def test_adds_expected_columns(self, sample_sales_df):
        result = create_temporal_features(sample_sales_df)
        expected = ["week_of_year", "month", "quarter", "year",
                    "week_sin", "week_cos", "month_sin", "month_cos"]
        for col in expected:
            assert col in result.columns, f"Missing: {col}"

    def test_week_of_year_range(self, sample_sales_df):
        result = create_temporal_features(sample_sales_df)
        assert result["week_of_year"].between(1, 53).all()

    def test_month_range(self, sample_sales_df):
        result = create_temporal_features(sample_sales_df)
        assert result["month"].between(1, 12).all()

    def test_cyclical_encoding_bounds(self, sample_sales_df):
        result = create_temporal_features(sample_sales_df)
        assert result["week_sin"].between(-1, 1).all()
        assert result["week_cos"].between(-1, 1).all()

    def test_does_not_modify_original(self, sample_sales_df):
        original_cols = set(sample_sales_df.columns)
        create_temporal_features(sample_sales_df)
        assert set(sample_sales_df.columns) == original_cols


class TestHolidayFeatures:
    """Tests for create_holiday_features."""

    def test_adds_festive_season(self, sample_sales_df):
        result = create_holiday_features(sample_sales_df)
        assert "is_festive_season" in result.columns
        assert result["is_festive_season"].isin([0, 1]).all()

    def test_festive_season_oct_nov(self, sample_sales_df):
        result = create_holiday_features(sample_sales_df)
        oct_nov = result[result["Date"].dt.month.isin([10, 11])]
        if len(oct_nov) > 0:
            assert (oct_nov["is_festive_season"] == 1).all()

    def test_days_to_holiday_non_negative(self, sample_sales_df):
        result = create_holiday_features(sample_sales_df)
        if "days_to_nearest_holiday" in result.columns:
            assert (result["days_to_nearest_holiday"] >= 0).all()


class TestLagFeatures:
    """Tests for create_lag_features."""

    def test_creates_expected_lags(self, sample_sales_df):
        result = create_lag_features(sample_sales_df, lags=[1, 2, 4])
        assert "Weekly_Sales_lag_1w" in result.columns
        assert "Weekly_Sales_lag_2w" in result.columns
        assert "Weekly_Sales_lag_4w" in result.columns

    def test_lag_1_shifts_correctly(self, sample_sales_df):
        result = create_lag_features(sample_sales_df, lags=[1])
        # Lag-1 of row i should be value of row i-1
        assert pd.isna(result["Weekly_Sales_lag_1w"].iloc[0])
        assert result["Weekly_Sales_lag_1w"].iloc[1] == sample_sales_df["Weekly_Sales"].iloc[0]

    def test_lag_produces_nans(self, sample_sales_df):
        result = create_lag_features(sample_sales_df, lags=[4])
        # First 4 rows should have NaN for lag-4
        assert result["Weekly_Sales_lag_4w"].iloc[:4].isna().all()


class TestRollingFeatures:
    """Tests for create_rolling_features."""

    def test_creates_rolling_columns(self, sample_sales_df):
        result = create_rolling_features(sample_sales_df, windows=[4])
        assert "Weekly_Sales_rolling_mean_4w" in result.columns
        assert "Weekly_Sales_rolling_std_4w" in result.columns

    def test_rolling_mean_reasonable(self, sample_sales_df):
        result = create_rolling_features(sample_sales_df, windows=[4])
        # Rolling mean should be between min and max of original
        rm = result["Weekly_Sales_rolling_mean_4w"].dropna()
        assert rm.min() >= sample_sales_df["Weekly_Sales"].min() * 0.9
        assert rm.max() <= sample_sales_df["Weekly_Sales"].max() * 1.1


class TestEncodeSeverity:
    """Tests for encode_spike_severity."""

    def test_encodes_correctly(self):
        df = pd.DataFrame({"sev": ["NONE", "MILD", "MODERATE", "SEVERE"]})
        result = encode_spike_severity(df, "sev")
        assert list(result["sev_encoded"]) == [0, 1, 2, 3]

    def test_handles_unknown(self):
        df = pd.DataFrame({"sev": ["UNKNOWN"]})
        result = encode_spike_severity(df, "sev")
        assert result["sev_encoded"].iloc[0] == 0  # Maps to 0 (fillna)


class TestFeatureColumns:
    """Tests for get_feature_columns."""

    def test_excludes_target(self, sample_sales_df):
        df = create_temporal_features(sample_sales_df)
        cols = get_feature_columns(df, "Weekly_Sales")
        assert "Weekly_Sales" not in cols

    def test_excludes_date(self, sample_sales_df):
        df = create_temporal_features(sample_sales_df)
        cols = get_feature_columns(df, "Weekly_Sales")
        assert "Date" not in cols

    def test_only_numeric(self, sample_sales_df):
        df = create_temporal_features(sample_sales_df)
        df["string_col"] = "test"
        cols = get_feature_columns(df, "Weekly_Sales")
        assert "string_col" not in cols


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
