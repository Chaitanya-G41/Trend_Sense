"""
TrendSense — ML Models Module
===============================
Training and evaluation of forecasting models:
- ARIMA (baseline, no TVI)
- Random Forest (intermediate)
- XGBoost (primary, with TVI features)

Includes time-series cross-validation and model comparison.
"""

import pandas as pd
import numpy as np
import warnings
import joblib
import os
from typing import Tuple, Dict, Optional

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

# Suppress convergence warnings during ARIMA fitting
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# ──────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute regression evaluation metrics.

    Parameters
    ----------
    y_true : array-like
        Actual values
    y_pred : array-like
        Predicted values

    Returns
    -------
    dict
        Dictionary with MAPE, RMSE, MAE, R² scores
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    # Remove zeros from MAPE calculation to avoid division issues
    mask = y_true != 0
    if mask.sum() == 0:
        mape = np.nan
    else:
        mape = mean_absolute_percentage_error(y_true[mask], y_pred[mask]) * 100

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = np.mean(np.abs(y_true - y_pred))
    r2 = r2_score(y_true, y_pred)

    return {
        "MAPE (%)": round(mape, 2),
        "RMSE": round(rmse, 2),
        "MAE": round(mae, 2),
        "R²": round(r2, 4),
    }


# ──────────────────────────────────────────────
# ARIMA Baseline
# ──────────────────────────────────────────────

def train_arima_baseline(
    train_series: pd.Series,
    test_series: pd.Series,
    order: tuple = None,
    seasonal_order: tuple = None,
) -> dict:
    """
    Train ARIMA/SARIMAX model as baseline (no external features).

    Parameters
    ----------
    train_series : pd.Series
        Training time series (target values)
    test_series : pd.Series
        Test time series for evaluation
    order : tuple, optional
        ARIMA order (p, d, q). Defaults from config
    seasonal_order : tuple, optional
        Seasonal ARIMA order. Defaults from config

    Returns
    -------
    dict
        Model results with predictions and metrics
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    order = order or config.ARIMA_PARAMS["order"]
    seasonal_order = seasonal_order or config.ARIMA_PARAMS.get("seasonal_order", (0, 0, 0, 0))

    print(f"📈 Training ARIMA{order} baseline...")

    try:
        model = SARIMAX(
            train_series,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        fitted = model.fit(disp=False, maxiter=200)

        # Forecast
        predictions = fitted.forecast(steps=len(test_series))
        predictions = np.maximum(predictions, 0)  # Sales can't be negative

        metrics = compute_metrics(test_series.values, predictions.values)

        print(f"   MAPE: {metrics['MAPE (%)']}%  |  RMSE: {metrics['RMSE']}  |  R²: {metrics['R²']}")

        return {
            "model_name": "ARIMA",
            "model": fitted,
            "predictions": predictions,
            "metrics": metrics,
            "order": order,
            "seasonal_order": seasonal_order,
        }

    except Exception as e:
        print(f"   ⚠️ ARIMA fitting failed: {e}")
        print("   Falling back to naive forecast (last known value)...")
        predictions = pd.Series([train_series.iloc[-1]] * len(test_series), index=test_series.index)
        metrics = compute_metrics(test_series.values, predictions.values)

        return {
            "model_name": "ARIMA (naive fallback)",
            "model": None,
            "predictions": predictions,
            "metrics": metrics,
        }


# ──────────────────────────────────────────────
# Random Forest
# ──────────────────────────────────────────────

def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    params: dict = None,
) -> dict:
    """
    Train Random Forest regressor.

    Parameters
    ----------
    X_train, X_test : pd.DataFrame
        Feature matrices
    y_train, y_test : pd.Series
        Target vectors
    params : dict, optional
        Model hyperparameters. Defaults from config

    Returns
    -------
    dict
        Model results with predictions, metrics, and feature importance
    """
    params = params or config.RANDOM_FOREST_PARAMS

    print(f"🌲 Training Random Forest (n_estimators={params['n_estimators']})...")

    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    predictions = np.maximum(predictions, 0)

    metrics = compute_metrics(y_test.values, predictions)

    # Feature importance
    feature_importance = pd.DataFrame({
        "feature": X_train.columns,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    print(f"   MAPE: {metrics['MAPE (%)']}%  |  RMSE: {metrics['RMSE']}  |  R²: {metrics['R²']}")

    return {
        "model_name": "Random Forest",
        "model": model,
        "predictions": predictions,
        "metrics": metrics,
        "feature_importance": feature_importance,
        "params": params,
    }


# ──────────────────────────────────────────────
# XGBoost
# ──────────────────────────────────────────────

def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    params: dict = None,
) -> dict:
    """
    Train XGBoost regressor (primary model).

    Parameters
    ----------
    X_train, X_test : pd.DataFrame
        Feature matrices
    y_train, y_test : pd.Series
        Target vectors
    params : dict, optional
        Model hyperparameters. Defaults from config

    Returns
    -------
    dict
        Model results with predictions, metrics, and feature importance
    """
    from xgboost import XGBRegressor

    params = params or config.XGBOOST_PARAMS

    print(f"🚀 Training XGBoost (n_estimators={params['n_estimators']}, lr={params['learning_rate']})...")

    model = XGBRegressor(**params, verbosity=0)
    model.fit(
        X_train, y_train,
        verbose=False,
    )

    predictions = model.predict(X_test)
    predictions = np.maximum(predictions, 0)

    metrics = compute_metrics(y_test.values, predictions)

    # Feature importance
    feature_importance = pd.DataFrame({
        "feature": X_train.columns,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    print(f"   MAPE: {metrics['MAPE (%)']}%  |  RMSE: {metrics['RMSE']}  |  R²: {metrics['R²']}")

    return {
        "model_name": "XGBoost",
        "model": model,
        "predictions": predictions,
        "metrics": metrics,
        "feature_importance": feature_importance,
        "params": params,
    }


# ──────────────────────────────────────────────
# Time Series Cross-Validation
# ──────────────────────────────────────────────

def time_series_cv(
    X: pd.DataFrame,
    y: pd.Series,
    model_type: str = "xgboost",
    n_splits: int = None,
    params: dict = None,
) -> dict:
    """
    Perform time-series cross-validation with expanding window.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target vector
    model_type : str
        One of 'xgboost', 'random_forest'
    n_splits : int
        Number of CV splits. Defaults from config
    params : dict, optional
        Model hyperparameters

    Returns
    -------
    dict
        CV results with per-fold and aggregate metrics
    """
    n_splits = n_splits or config.CV_SPLITS

    print(f"\n🔄 Time-series CV ({n_splits} splits) for {model_type}...")

    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_metrics = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        if model_type == "xgboost":
            result = train_xgboost(X_train, y_train, X_test, y_test, params)
        elif model_type == "random_forest":
            result = train_random_forest(X_train, y_train, X_test, y_test, params)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        result["metrics"]["fold"] = fold + 1
        fold_metrics.append(result["metrics"])
        print(f"   Fold {fold + 1}: MAPE={result['metrics']['MAPE (%)']}%")

    # Aggregate metrics
    metrics_df = pd.DataFrame(fold_metrics)
    avg_metrics = {
        "MAPE (%)": round(metrics_df["MAPE (%)"].mean(), 2),
        "RMSE": round(metrics_df["RMSE"].mean(), 2),
        "MAE": round(metrics_df["MAE"].mean(), 2),
        "R²": round(metrics_df["R²"].mean(), 4),
        "MAPE_std": round(metrics_df["MAPE (%)"].std(), 2),
    }

    print(f"\n   📊 Avg MAPE: {avg_metrics['MAPE (%)']}% ± {avg_metrics['MAPE_std']}%")

    return {
        "model_type": model_type,
        "n_splits": n_splits,
        "fold_metrics": metrics_df,
        "avg_metrics": avg_metrics,
    }


# ──────────────────────────────────────────────
# Model Comparison
# ──────────────────────────────────────────────

def compare_models(results: Dict[str, dict]) -> pd.DataFrame:
    """
    Compare multiple model results in a summary table.

    Parameters
    ----------
    results : dict
        Dictionary of model name → result dict (from train_* functions)

    Returns
    -------
    pd.DataFrame
        Comparison table
    """
    rows = []
    for name, result in results.items():
        row = {"Model": name}
        row.update(result["metrics"])
        rows.append(row)

    comparison = pd.DataFrame(rows).set_index("Model")

    # Compute delta MAPE vs baseline (first model)
    baseline_mape = comparison.iloc[0]["MAPE (%)"]
    comparison["Δ-MAPE vs Baseline"] = round(baseline_mape - comparison["MAPE (%)"], 2)

    print("\n" + "=" * 70)
    print("📊 MODEL COMPARISON")
    print("=" * 70)
    print(comparison.to_string())
    print("=" * 70)

    return comparison


# ──────────────────────────────────────────────
# Train-Test Split (Time-series aware)
# ──────────────────────────────────────────────

def temporal_train_test_split(
    df: pd.DataFrame,
    target_col: str = "Weekly_Sales",
    feature_cols: list = None,
    test_weeks: int = None,
    date_col: str = "Date",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data temporally (no shuffle) for time-series.

    Parameters
    ----------
    df : pd.DataFrame
        Feature-engineered DataFrame, sorted by date
    target_col : str
        Target column
    feature_cols : list
        Feature columns to include
    test_weeks : int
        Number of weeks for test set. Defaults from config
    date_col : str
        Date column for sorting

    Returns
    -------
    tuple
        (X_train, X_test, y_train, y_test)
    """
    test_weeks = test_weeks or config.TEST_SIZE_WEEKS

    df = df.sort_values(date_col).reset_index(drop=True)

    split_idx = len(df) - test_weeks

    if feature_cols is None:
        from src.feature_engineering import get_feature_columns
        feature_cols = get_feature_columns(df, target_col)

    # Remove any non-numeric columns from features
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in feature_cols if c in numeric_cols]

    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # Rows with NaNs should be dropped during feature engineering, not zero-filled here

    print(f"📋 Train/Test split: {len(X_train)} train / {len(X_test)} test rows")
    print(f"   Features: {len(feature_cols)} columns")

    return X_train, X_test, y_train, y_test


# ──────────────────────────────────────────────
# Save / Load Models
# ──────────────────────────────────────────────

def save_model(model, name: str, metrics: dict = None) -> str:
    """
    Save a trained model to disk.

    Parameters
    ----------
    model : object
        Trained model object
    name : str
        Model name for the file
    metrics : dict, optional
        Metrics to save alongside

    Returns
    -------
    str
        Path to saved model
    """
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    model_path = os.path.join(config.MODELS_DIR, f"{name}.pkl")
    
    save_data = {"model": model, "metrics": metrics}
    joblib.dump(save_data, model_path)

    print(f"💾 Model saved: {model_path}")
    return model_path


def load_model(name: str) -> dict:
    """
    Load a trained model from disk.

    Parameters
    ----------
    name : str
        Model name

    Returns
    -------
    dict
        Dictionary with 'model' and 'metrics'
    """
    model_path = os.path.join(config.MODELS_DIR, f"{name}.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    data = joblib.load(model_path)
    print(f"✅ Model loaded: {model_path}")
    return data


# ──────────────────────────────────────────────
# Full Training Pipeline
# ──────────────────────────────────────────────

def run_full_training(
    df: pd.DataFrame,
    target_col: str = "Weekly_Sales",
    feature_cols: list = None,
    date_col: str = "Date",
    save: bool = True,
) -> dict:
    """
    Run the complete model training and comparison pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        Feature-engineered DataFrame
    target_col : str
        Target column
    feature_cols : list, optional
        Feature columns
    date_col : str
        Date column
    save : bool
        Whether to save trained models

    Returns
    -------
    dict
        All model results and comparison table
    """
    print("\n" + "=" * 70)
    print("🚀 TrendSense — Full Model Training Pipeline")
    print("=" * 70)

    # 1. Split data
    X_train, X_test, y_train, y_test = temporal_train_test_split(
        df, target_col, feature_cols, date_col=date_col
    )

    results = {}

    # 2. Naive Baseline (predicting lag_1w)
    # Using ARIMA on multi-store panel data without separating by store causes statsmodels to hang.
    print(f"📈 Evaluating Naive Baseline (Lag 1W)...")
    if "lag_1w" in X_test.columns:
        naive_preds = X_test["lag_1w"].values
    else:
        naive_preds = np.full(len(y_test), y_train.mean())
        
    metrics = compute_metrics(y_test.values, naive_preds)
    print(f"   MAPE: {metrics['MAPE (%)']}%  |  RMSE: {metrics['RMSE']}  |  R²: {metrics['R²']}")
    
    results["Naive Baseline"] = {
        "model_name": "Naive Baseline",
        "model": None,
        "predictions": naive_preds,
        "metrics": metrics
    }

    # 3. Random Forest
    rf_result = train_random_forest(X_train, y_train, X_test, y_test)
    results["Random Forest"] = rf_result

    # 4. XGBoost
    xgb_result = train_xgboost(X_train, y_train, X_test, y_test)
    results["XGBoost + TVI"] = xgb_result

    # 5. Compare models
    comparison = compare_models(results)

    # 6. Save best model
    if save:
        best_model_name = comparison["MAPE (%)"].idxmin()
        best_result = results[best_model_name]
        if best_result.get("model") is not None:
            save_model(best_result["model"], "best_model", best_result["metrics"])
            save_model(xgb_result["model"], "xgboost_tvi", xgb_result["metrics"])
            save_model(rf_result["model"], "random_forest", rf_result["metrics"])

    return {
        "results": results,
        "comparison": comparison,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }


# ──────────────────────────────────────────────
# Main execution for testing
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("TrendSense — Models Module Test")
    print("=" * 60)

    # Create synthetic data for testing
    np.random.seed(42)
    n = 200
    dates = pd.date_range("2020-01-01", periods=n, freq="W-SUN")

    X_data = pd.DataFrame({
        "week_of_year": dates.isocalendar().week.astype(int),
        "month": dates.month,
        "lag_1w": np.random.uniform(10000, 50000, n),
        "lag_2w": np.random.uniform(10000, 50000, n),
        "rolling_mean_4w": np.random.uniform(15000, 45000, n),
        "tvi_feature": np.random.normal(0, 20, n),
        "is_holiday": np.random.choice([0, 1], n, p=[0.85, 0.15]),
    })

    y_data = pd.Series(
        20000 + 5000 * np.sin(2 * np.pi * np.arange(n) / 52)
        + 3000 * X_data["is_holiday"]
        + np.random.normal(0, 2000, n),
        name="Weekly_Sales"
    )

    split = n - 24
    X_train, X_test = X_data.iloc[:split], X_data.iloc[split:]
    y_train, y_test = y_data.iloc[:split], y_data.iloc[split:]

    # Test Random Forest
    rf = train_random_forest(X_train, y_train, X_test, y_test)
    print(f"\nTop RF features:\n{rf['feature_importance'].head()}\n")

    # Test XGBoost
    xgb = train_xgboost(X_train, y_train, X_test, y_test)
    print(f"\nTop XGB features:\n{xgb['feature_importance'].head()}\n")

    # Compare
    compare_models({"ARIMA (Baseline)": {"metrics": {"MAPE (%)": 18.5, "RMSE": 5200, "MAE": 4100, "R²": 0.62}},
                     "Random Forest": rf, "XGBoost + TVI": xgb})

    print("\n✅ Models module test complete!")
