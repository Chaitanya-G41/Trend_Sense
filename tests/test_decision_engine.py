"""
TrendSense — Unit Tests: Decision Engine
==========================================
"""

import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.decision_engine import (
    generate_decision,
    compute_confidence_score,
    format_decision_output,
    generate_batch_decisions,
    get_decision_summary,
)


class TestGenerateDecision:
    """Tests for generate_decision function."""

    def test_hold_for_stable_demand(self):
        """Should return HOLD when change < 10%."""
        decision = generate_decision(
            predicted_demand=10500,
            current_stock=10000,
            model_confidence=0.85,
        )
        assert decision["action"] == "HOLD"

    def test_increase_for_moderate_growth(self):
        """Should return INCREASE STOCK for 10-40% change."""
        decision = generate_decision(
            predicted_demand=12500,
            current_stock=10000,
            model_confidence=0.85,
        )
        assert decision["action"] == "INCREASE STOCK"

    def test_urgent_for_high_demand(self):
        """Should return URGENT RESTOCK for ≥40% change."""
        decision = generate_decision(
            predicted_demand=15000,
            current_stock=10000,
            model_confidence=0.85,
        )
        assert decision["action"] == "URGENT RESTOCK"

    def test_tvi_spike_boosts_urgency(self):
        """MODERATE spike should boost HOLD to INCREASE STOCK."""
        decision = generate_decision(
            predicted_demand=10800,  # 8% change → normally HOLD
            current_stock=10000,
            tvi_status={"is_spike": True, "severity": "MODERATE", "tvi_value": 45.0},
            model_confidence=0.75,
        )
        assert decision["action"] == "INCREASE STOCK"
        assert decision["spike_boost_applied"] is True

    def test_severe_spike_boosts_to_urgent(self):
        """SEVERE spike should boost INCREASE to URGENT."""
        decision = generate_decision(
            predicted_demand=12000,  # 20% → normally INCREASE
            current_stock=10000,
            tvi_status={"is_spike": True, "severity": "SEVERE", "tvi_value": 80.0},
            model_confidence=0.85,
        )
        assert decision["action"] == "URGENT RESTOCK"

    def test_no_boost_for_mild_spike(self):
        """MILD spike should NOT boost urgency."""
        decision = generate_decision(
            predicted_demand=10500,
            current_stock=10000,
            tvi_status={"is_spike": True, "severity": "MILD", "tvi_value": 15.0},
            model_confidence=0.80,
        )
        assert decision["spike_boost_applied"] is False

    def test_zero_stock_handled(self):
        """Should handle zero current stock without errors."""
        decision = generate_decision(
            predicted_demand=5000,
            current_stock=0,
            model_confidence=0.80,
        )
        assert decision["action"] in ["HOLD", "INCREASE STOCK", "URGENT RESTOCK"]

    def test_decision_has_required_fields(self):
        """Decision dict should contain all required fields."""
        decision = generate_decision(10000, 10000)
        required = ["action", "urgency", "confidence", "confidence_pct",
                    "predicted_demand", "current_stock", "predicted_change_pct",
                    "rationale", "tvi_spike", "spike_boost_applied"]
        for field in required:
            assert field in decision, f"Missing field: {field}"

    def test_confidence_in_range(self):
        """Confidence should be between 0.5 and 0.99."""
        decision = generate_decision(15000, 10000, model_confidence=0.90)
        assert 0.50 <= decision["confidence"] <= 0.99


class TestConfidenceScore:
    """Tests for compute_confidence_score."""

    def test_tvi_agreement_boosts_confidence(self):
        """Spike + high demand prediction should increase confidence."""
        base = 0.80
        with_spike = compute_confidence_score(
            base, 0.30,
            {"is_spike": True, "severity": "SEVERE", "tvi_value": 50},
            "Electronics"
        )
        without_spike = compute_confidence_score(
            base, 0.30,
            {"is_spike": False, "severity": "NONE", "tvi_value": 0},
            "Electronics"
        )
        assert with_spike > without_spike

    def test_disagreement_lowers_confidence(self):
        """Spike + negative prediction should decrease confidence."""
        base = 0.80
        conf = compute_confidence_score(
            base, -0.10,
            {"is_spike": True, "severity": "MODERATE", "tvi_value": 30},
            "Fashion"
        )
        assert conf < base

    def test_extreme_prediction_penalty(self):
        """Very large predicted changes should lower confidence."""
        base = 0.80
        normal = compute_confidence_score(
            base, 0.20,
            {"is_spike": False, "severity": "NONE", "tvi_value": 0},
            "General"
        )
        extreme = compute_confidence_score(
            base, 0.80,
            {"is_spike": False, "severity": "NONE", "tvi_value": 0},
            "General"
        )
        assert extreme <= normal


class TestFormatDecision:
    """Tests for format_decision_output."""

    def test_format_contains_action(self):
        decision = generate_decision(15000, 10000, model_confidence=0.85)
        output = format_decision_output(decision)
        assert decision["action"] in output

    def test_format_contains_confidence(self):
        decision = generate_decision(15000, 10000, model_confidence=0.85)
        output = format_decision_output(decision)
        assert decision["confidence_pct"] in output


class TestBatchDecisions:
    """Tests for generate_batch_decisions."""

    def test_returns_correct_length(self):
        predictions = np.array([10000, 12000, 18000])
        stocks = np.array([10000, 10000, 10000])
        result = generate_batch_decisions(predictions, stocks)
        assert len(result) == 3

    def test_returns_dataframe(self):
        predictions = np.array([10000, 15000])
        stocks = np.array([10000, 10000])
        result = generate_batch_decisions(predictions, stocks)
        import pandas as pd
        assert isinstance(result, pd.DataFrame)


class TestDecisionSummary:
    """Tests for get_decision_summary."""

    def test_summary_counts(self):
        predictions = np.array([10500, 13000, 18000, 9500, 15000])
        stocks = np.array([10000, 10000, 10000, 10000, 10000])
        batch_df = generate_batch_decisions(predictions, stocks)
        summary = get_decision_summary(batch_df)
        
        total = summary["hold_count"] + summary["increase_count"] + summary["urgent_count"]
        assert total == summary["total_decisions"]

    def test_percentages_sum_to_100(self):
        predictions = np.array([10500, 13000, 18000])
        stocks = np.array([10000, 10000, 10000])
        batch_df = generate_batch_decisions(predictions, stocks)
        summary = get_decision_summary(batch_df)
        
        total_pct = summary["hold_pct"] + summary["increase_pct"] + summary["urgent_pct"]
        assert abs(total_pct - 100.0) < 0.5  # Allow small rounding error


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
