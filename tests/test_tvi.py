"""
TrendSense — Unit Tests: TVI Module
=====================================
"""

import sys
import os
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.tvi import (
    compute_tvi,
    compute_tvi_acceleration,
    compute_tvi_rolling,
    detect_spikes,
    classify_spike_severity,
    compute_tvi_features,
    get_tvi_summary,
)


class TestComputeTVI:
    """Tests for compute_tvi function."""

    def test_basic_increase(self):
        """TVI should be positive for increasing trend."""
        series = pd.Series([10, 20, 30, 40, 50])
        tvi = compute_tvi(series)
        assert tvi.iloc[1] == 100.0  # 10→20 is 100% increase
        assert all(tvi.dropna() > 0)

    def test_basic_decrease(self):
        """TVI should be negative for decreasing trend."""
        series = pd.Series([50, 40, 30, 20, 10])
        tvi = compute_tvi(series)
        assert all(tvi.dropna() < 0)

    def test_constant_trend(self):
        """TVI should be 0 for constant trend."""
        series = pd.Series([50, 50, 50, 50])
        tvi = compute_tvi(series)
        assert all(tvi.dropna() == 0)

    def test_first_value_is_nan(self):
        """First TVI value should be NaN (no previous value)."""
        series = pd.Series([10, 20, 30])
        tvi = compute_tvi(series)
        assert pd.isna(tvi.iloc[0])

    def test_handles_zeros(self):
        """Should handle zero values without division errors."""
        series = pd.Series([0, 10, 0, 10])
        tvi = compute_tvi(series)
        assert not tvi.isna().all()  # Not all NaN


class TestTVIAcceleration:
    """Tests for compute_tvi_acceleration function."""

    def test_acceleration_positive(self):
        """Acceleration should be positive when TVI is increasing."""
        tvi = pd.Series([0, 10, 30, 60])
        accel = compute_tvi_acceleration(tvi)
        assert accel.dropna().iloc[-1] > 0

    def test_acceleration_negative(self):
        """Acceleration should be negative when TVI is decreasing."""
        tvi = pd.Series([60, 30, 10, 0])
        accel = compute_tvi_acceleration(tvi)
        assert accel.dropna().iloc[-1] < 0


class TestSpikeDetection:
    """Tests for spike detection functions."""

    def test_classify_none(self):
        """Normal values should be classified as NONE."""
        assert classify_spike_severity(5.0, 5.0, 10.0) == "NONE"

    def test_classify_mild(self):
        """Values > 1σ should be MILD."""
        assert classify_spike_severity(16.0, 5.0, 10.0) == "MILD"

    def test_classify_moderate(self):
        """Values > 2σ should be MODERATE."""
        assert classify_spike_severity(26.0, 5.0, 10.0) == "MODERATE"

    def test_classify_severe(self):
        """Values > 3σ should be SEVERE."""
        assert classify_spike_severity(36.0, 5.0, 10.0) == "SEVERE"

    def test_zero_std(self):
        """Zero standard deviation should return NONE."""
        assert classify_spike_severity(10.0, 5.0, 0.0) == "NONE"

    def test_detect_spikes_returns_dataframe(self):
        """detect_spikes should return a DataFrame with expected columns."""
        tvi = pd.Series([1, 2, 50, 3, 1])
        dates = pd.Series(pd.date_range("2024-01-01", periods=5, freq="W"))
        result = detect_spikes(tvi, threshold_sigma=2.0, dates=dates)
        assert isinstance(result, pd.DataFrame)
        assert "is_spike" in result.columns
        assert "severity" in result.columns

    def test_spike_detected_for_outlier(self):
        """Should detect spike for extreme outlier."""
        tvi = pd.Series([1, 2, 1, 2, 1, 100, 1, 2])
        result = detect_spikes(tvi, threshold_sigma=2.0)
        assert result["is_spike"].sum() >= 1


class TestComputeTVIFeatures:
    """Tests for full TVI feature computation."""

    def test_returns_all_columns(self):
        """Should return DataFrame with all expected columns."""
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=10, freq="W"),
            "smartphones": [10, 20, 30, 25, 35, 28, 40, 90, 45, 30],
        })
        result = compute_tvi_features(df, "smartphones")
        expected_cols = ["date", "raw_trend", "tvi", "tvi_acceleration",
                        "tvi_smoothed", "is_spike", "spike_severity"]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_invalid_keyword_raises(self):
        """Should raise ValueError for missing keyword."""
        df = pd.DataFrame({"date": [1, 2, 3], "smartphones": [10, 20, 30]})
        with pytest.raises(ValueError):
            compute_tvi_features(df, "nonexistent_keyword")


class TestTVISummary:
    """Tests for get_tvi_summary function."""

    def test_summary_keys(self):
        """Summary should contain all expected keys."""
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=10, freq="W"),
            "test": range(10, 20),
        })
        features = compute_tvi_features(df, "test")
        summary = get_tvi_summary(features)
        
        expected_keys = ["total_weeks", "mean_tvi", "std_tvi", "total_spikes",
                        "severe_spikes", "moderate_spikes", "mild_spikes", "spike_rate_pct"]
        for key in expected_keys:
            assert key in summary, f"Missing key: {key}"

    def test_summary_total_weeks(self):
        """total_weeks should match input length."""
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=20, freq="W"),
            "test": range(20),
        })
        features = compute_tvi_features(df, "test")
        summary = get_tvi_summary(features)
        assert summary["total_weeks"] == 20


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
