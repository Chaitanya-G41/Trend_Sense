"""
TrendSense — Decision Engine Module
======================================
Converts ML model predictions into actionable business decisions:
  • HOLD — demand stable, maintain current stock
  • INCREASE STOCK — moderate demand increase expected
  • URGENT RESTOCK — significant demand surge detected

Each decision includes a confidence score (e.g., 87%).
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


# ──────────────────────────────────────────────
# Decision Logic
# ──────────────────────────────────────────────

def generate_decision(
    predicted_demand: float,
    current_stock: float,
    tvi_status: dict = None,
    model_confidence: float = 0.80,
    category: str = "General",
) -> dict:
    """
    Generate inventory decision based on predicted demand vs current stock.
    
    Decision Logic:
        predicted_change < 10%           → HOLD
        10% ≤ predicted_change < 40%     → INCREASE STOCK
        predicted_change ≥ 40%           → URGENT RESTOCK
        TVI spike detected (>2σ)         → Boost urgency by one level
    
    Parameters
    ----------
    predicted_demand : float
        Model's predicted weekly demand
    current_stock : float
        Current inventory level
    tvi_status : dict, optional
        TVI spike information: {is_spike: bool, severity: str, tvi_value: float}
    model_confidence : float
        Model confidence score (0-1)
    category : str
        Product category for context
        
    Returns
    -------
    dict
        Decision with action, confidence, rationale, and metadata
    """
    if current_stock <= 0:
        current_stock = 1  # Prevent division by zero
    
    # Calculate predicted change
    predicted_change = (predicted_demand - current_stock) / current_stock
    predicted_change_pct = predicted_change * 100
    
    # Default TVI status
    if tvi_status is None:
        tvi_status = {"is_spike": False, "severity": "NONE", "tvi_value": 0.0}
    
    # Base decision from thresholds
    if predicted_change < config.DECISION_THRESHOLDS["HOLD"]:
        action = "HOLD"
        urgency = 1
        color = "green"
        rationale = (f"Predicted demand change is {predicted_change_pct:+.1f}%, "
                     f"below the {config.DECISION_THRESHOLDS['HOLD']*100:.0f}% threshold. "
                     f"Current stock levels are adequate.")
    elif predicted_change < config.DECISION_THRESHOLDS["INCREASE_STOCK"]:
        action = "INCREASE STOCK"
        urgency = 2
        color = "amber"
        rationale = (f"Predicted demand increase of {predicted_change_pct:+.1f}% "
                     f"suggests moderate growth. Consider increasing inventory by "
                     f"{predicted_change_pct:.0f}% for {category} category.")
    else:
        action = "URGENT RESTOCK"
        urgency = 3
        color = "red"
        rationale = (f"Significant demand surge predicted: {predicted_change_pct:+.1f}%. "
                     f"Immediate restocking recommended for {category} category "
                     f"to prevent stockouts.")
    
    # TVI spike urgency boost
    spike_boost_applied = False
    if config.TVI_SPIKE_URGENCY_BOOST and tvi_status.get("is_spike", False):
        severity = tvi_status.get("severity", "NONE")
        if severity in ("MODERATE", "SEVERE") and urgency < 3:
            urgency += 1
            spike_boost_applied = True
            
            # Upgrade action
            if urgency == 2:
                action = "INCREASE STOCK"
                color = "amber"
            elif urgency == 3:
                action = "URGENT RESTOCK"
                color = "red"
            
            rationale += (f"\n⚡ Urgency boosted due to {severity} TVI spike detected "
                         f"(TVI = {tvi_status.get('tvi_value', 0):.1f}).")
    
    # Adjust confidence based on factors
    confidence = compute_confidence_score(
        model_confidence, predicted_change, tvi_status, category
    )
    
    return {
        "action": action,
        "urgency": urgency,
        "color": color,
        "confidence": confidence,
        "confidence_pct": f"{confidence * 100:.0f}%",
        "predicted_demand": round(predicted_demand, 2),
        "current_stock": round(current_stock, 2),
        "predicted_change_pct": round(predicted_change_pct, 1),
        "category": category,
        "rationale": rationale,
        "tvi_spike": tvi_status.get("is_spike", False),
        "tvi_severity": tvi_status.get("severity", "NONE"),
        "spike_boost_applied": spike_boost_applied,
    }


def compute_confidence_score(
    model_confidence: float,
    predicted_change: float,
    tvi_status: dict,
    category: str,
) -> float:
    """
    Compute overall decision confidence score.
    
    Factors:
    - Model prediction confidence (base)
    - TVI spike agreement (boost if spike aligns with high demand prediction)
    - Prediction magnitude (lower confidence for extreme predictions)
    
    Parameters
    ----------
    model_confidence : float
        Base model confidence (0-1)
    predicted_change : float
        Predicted demand change ratio
    tvi_status : dict
        TVI spike information
    category : str
        Product category
        
    Returns
    -------
    float
        Confidence score between CONFIDENCE_MIN and CONFIDENCE_MAX
    """
    base = model_confidence
    
    # TVI agreement bonus: if both TVI and model predict increase → higher confidence
    tvi_bonus = 0.0
    if tvi_status.get("is_spike", False):
        severity = tvi_status.get("severity", "NONE")
        if predicted_change > 0.1:  # Model also predicts increase
            if severity == "SEVERE":
                tvi_bonus = 0.10
            elif severity == "MODERATE":
                tvi_bonus = 0.07
            elif severity == "MILD":
                tvi_bonus = 0.03
        elif predicted_change < 0:  # Disagreement → lower confidence
            tvi_bonus = -0.10
    
    # Extreme prediction penalty (very high changes are less certain)
    magnitude_penalty = 0.0
    if abs(predicted_change) > 0.6:
        magnitude_penalty = -0.05
    elif abs(predicted_change) > 1.0:
        magnitude_penalty = -0.10
    
    confidence = base + tvi_bonus + magnitude_penalty
    
    # Clip to configured bounds
    confidence = np.clip(confidence, config.CONFIDENCE_MIN, config.CONFIDENCE_MAX)
    
    return round(float(confidence), 2)


def format_decision_output(decision: dict) -> str:
    """
    Format decision as a human-readable string for display.
    
    Parameters
    ----------
    decision : dict
        Decision from generate_decision()
        
    Returns
    -------
    str
        Formatted decision string
    """
    action = decision["action"]
    confidence = decision["confidence_pct"]
    category = decision["category"]
    change = decision["predicted_change_pct"]
    
    # Action symbols
    symbols = {
        "HOLD": "🟢",
        "INCREASE STOCK": "🟡",
        "URGENT RESTOCK": "🔴",
    }
    symbol = symbols.get(action, "⚪")
    
    lines = [
        "━" * 50,
        f"  {symbol}  {action}  —  Confidence: {confidence}",
        "━" * 50,
        f"  Category:          {category}",
        f"  Predicted Demand:  {decision['predicted_demand']:,.0f} units",
        f"  Current Stock:     {decision['current_stock']:,.0f} units",
        f"  Predicted Change:  {change:+.1f}%",
    ]
    
    if decision["tvi_spike"]:
        lines.append(f"  TVI Alert:         {decision['tvi_severity']} spike detected")
    
    if decision["spike_boost_applied"]:
        lines.append(f"  ⚡ Urgency boosted by TVI spike")
    
    lines.extend([
        "",
        f"  📋 Rationale:",
        f"  {decision['rationale']}",
        "━" * 50,
    ])
    
    return "\n".join(lines)


# ──────────────────────────────────────────────
# Batch Decision Generation
# ──────────────────────────────────────────────

def generate_batch_decisions(
    predictions: np.ndarray,
    current_stocks: np.ndarray,
    tvi_statuses: list = None,
    model_confidence: float = 0.80,
    categories: list = None,
) -> pd.DataFrame:
    """
    Generate decisions for multiple stores/products at once.
    
    Parameters
    ----------
    predictions : array-like
        Predicted demand values
    current_stocks : array-like
        Current stock levels
    tvi_statuses : list of dict, optional
        TVI status for each item
    model_confidence : float
        Model confidence score
    categories : list, optional
        Product categories
        
    Returns
    -------
    pd.DataFrame
        DataFrame with all decisions
    """
    n = len(predictions)
    
    if tvi_statuses is None:
        tvi_statuses = [{"is_spike": False, "severity": "NONE", "tvi_value": 0.0}] * n
    
    if categories is None:
        categories = ["General"] * n
    
    decisions = []
    for i in range(n):
        decision = generate_decision(
            predicted_demand=predictions[i],
            current_stock=current_stocks[i],
            tvi_status=tvi_statuses[i],
            model_confidence=model_confidence,
            category=categories[i],
        )
        decision["index"] = i
        decisions.append(decision)
    
    return pd.DataFrame(decisions)


def get_decision_summary(decisions_df: pd.DataFrame) -> dict:
    """
    Summarize batch decisions.
    
    Parameters
    ----------
    decisions_df : pd.DataFrame
        Output from generate_batch_decisions()
        
    Returns
    -------
    dict
        Summary statistics
    """
    total = len(decisions_df)
    
    summary = {
        "total_decisions": total,
        "hold_count": int((decisions_df["action"] == "HOLD").sum()),
        "increase_count": int((decisions_df["action"] == "INCREASE STOCK").sum()),
        "urgent_count": int((decisions_df["action"] == "URGENT RESTOCK").sum()),
        "avg_confidence": round(decisions_df["confidence"].mean() * 100, 1),
        "spike_influenced": int(decisions_df["spike_boost_applied"].sum()),
    }
    
    summary["hold_pct"] = round(summary["hold_count"] / max(total, 1) * 100, 1)
    summary["increase_pct"] = round(summary["increase_count"] / max(total, 1) * 100, 1)
    summary["urgent_pct"] = round(summary["urgent_count"] / max(total, 1) * 100, 1)
    
    return summary


# ──────────────────────────────────────────────
# Main execution for testing
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("TrendSense — Decision Engine Test")
    print("=" * 60)
    
    # Test Case 1: HOLD — demand stable
    print("\n📌 Test 1: Stable demand (HOLD expected)")
    decision1 = generate_decision(
        predicted_demand=10500,
        current_stock=10000,
        model_confidence=0.85,
        category="Electronics"
    )
    print(format_decision_output(decision1))
    
    # Test Case 2: INCREASE STOCK — moderate growth
    print("\n📌 Test 2: Moderate growth (INCREASE STOCK expected)")
    decision2 = generate_decision(
        predicted_demand=13000,
        current_stock=10000,
        model_confidence=0.82,
        category="Fashion"
    )
    print(format_decision_output(decision2))
    
    # Test Case 3: URGENT RESTOCK — demand surge
    print("\n📌 Test 3: Demand surge (URGENT RESTOCK expected)")
    decision3 = generate_decision(
        predicted_demand=18000,
        current_stock=10000,
        tvi_status={"is_spike": True, "severity": "SEVERE", "tvi_value": 85.5},
        model_confidence=0.90,
        category="Festival"
    )
    print(format_decision_output(decision3))
    
    # Test Case 4: TVI spike boost
    print("\n📌 Test 4: TVI spike boosts HOLD → INCREASE STOCK")
    decision4 = generate_decision(
        predicted_demand=10800,
        current_stock=10000,
        tvi_status={"is_spike": True, "severity": "MODERATE", "tvi_value": 45.0},
        model_confidence=0.75,
        category="General"
    )
    print(format_decision_output(decision4))
    
    # Test batch decisions
    print("\n📌 Test 5: Batch decisions")
    np.random.seed(42)
    batch_df = generate_batch_decisions(
        predictions=np.array([10500, 13000, 18000, 9500, 15000]),
        current_stocks=np.array([10000, 10000, 10000, 10000, 10000]),
        tvi_statuses=[
            {"is_spike": False, "severity": "NONE", "tvi_value": 0},
            {"is_spike": True, "severity": "MILD", "tvi_value": 20},
            {"is_spike": True, "severity": "SEVERE", "tvi_value": 85},
            {"is_spike": False, "severity": "NONE", "tvi_value": -5},
            {"is_spike": True, "severity": "MODERATE", "tvi_value": 45},
        ],
        categories=["Electronics", "Fashion", "Festival", "General", "Electronics"],
    )
    
    print(batch_df[["action", "confidence_pct", "predicted_change_pct", "category", "spike_boost_applied"]].to_string())
    
    summary = get_decision_summary(batch_df)
    print(f"\n📊 Summary:")
    for k, v in summary.items():
        print(f"   {k}: {v}")
    
    print("\n✅ Decision engine test complete!")
