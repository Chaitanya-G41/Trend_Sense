# TrendSense: Social Media Trend Prediction for Business Decision Making

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B.svg)](https://streamlit.io)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-brightgreen.svg)](https://xgboost.readthedocs.io)

> AI-driven Decision Support System integrating social media trend velocity with historical sales data for proactive inventory management.

**Course:** CI124TA — AI Foundations for Engineers  
**Team:** A S Nemitha (1RV25CD002) • Chaitanya G (1RV25CD013) • Inika Ranganath Prasad (1RV25CD019)  
**Faculty:** Dr Sindhu D V, Dept. of CSE (Data Science), RVCE  

---

## 🎯 Problem Statement

Retail businesses rely solely on historical sales data for demand forecasting, making inventory decisions **reactive rather than proactive**. Existing systems fail to capture real-time social media signals that precede actual purchases by 3–14 days. This results in persistent stockouts and overstock losses, especially during festive seasons like Diwali and Big Billion Day.

**TrendSense** solves this by fusing Google Trends data with sales history to produce actionable decisions:

| Decision | Trigger | Color |
|---|---|---|
| **HOLD** | Predicted change < 10% | 🟢 |
| **INCREASE STOCK** | 10% ≤ change < 40% | 🟡 |
| **URGENT RESTOCK** | change ≥ 40% or TVI spike | 🔴 |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    TrendSense Pipeline                   │
├──────────┬──────────┬──────────┬──────────┬─────────────┤
│   Data   │   TVI    │ Feature  │    ML    │  Decision   │
│ Ingestion│Computation│Engineering│ Models  │   Engine    │
│          │          │          │          │             │
│ Walmart  │ Rate of  │ Temporal │ Naive    │ HOLD /      │
│ Google   │ Change   │ Holiday  │ Random   │ INCREASE /  │
│ Trends   │ Spike    │ Lag/Roll │ Forest   │ URGENT      │
│          │ Detection│ TVI Merge│ XGBoost  │ RESTOCK     │
└──────────┴──────────┴──────────┴──────────┴─────────────┘
                           │
                    ┌──────┴──────┐
                    │  Streamlit  │
                    │  Dashboard  │
                    └─────────────┘
```

---

## 🔬 Key Innovation: Trend Velocity Index (TVI)

Unlike raw search volume, TVI measures the **rate of change (acceleration)** of social interest:

```
TVI(t) = [Trend(t) - Trend(t-1)] / Trend(t-1) × 100
```

- Search volume growing **100→400 in 3 days** = viral spike needing urgent restocking
- Gradual growth 100→110→120 = normal seasonal pattern

**Spike Detection:** Flagged when `TVI(t) > μ + 2σ`
- **MILD** (>1σ) • **MODERATE** (>2σ) • **SEVERE** (>3σ)

---

## 📂 Project Structure

```text
Trend_Sense/
├── config.py                    # Global configuration
├── requirements.txt             # Python dependencies
├── run_pipeline.py              # End-to-end pipeline runner
├── README.md                    # This file
│
├── src/                         # Core ML pipeline
│   ├── data_ingestion.py        # Kaggle + Google Trends data loading
│   ├── tvi.py                   # Trend Velocity Index computation
│   ├── feature_engineering.py   # Feature creation pipeline
│   ├── models.py                # XGBoost, Random Forest, ARIMA
│   ├── lag_optimizer.py         # Category-specific lag discovery
│   ├── decision_engine.py       # HOLD / INCREASE / URGENT logic
│   └── utils.py                 # Plotting and helper functions
│
├── dashboard/                   # Streamlit web dashboard
│   ├── app.py                   # Main app with navigation
│   └── pages/                   # Multi-page dashboard
│       ├── page_01_overview.py
│       ├── page_02_trend_analysis.py
│       ├── page_03_predictions.py
│       └── page_04_decision_support.py
│
├── notebooks/                   # Jupyter notebooks for analysis
│   ├── 01_data_exploration.ipynb
│   ├── 02_tvi_analysis.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_results_visualization.ipynb
│
├── data/                        # Datasets (auto-created)
│   ├── raw/                     # Downloaded CSVs
│   ├── trends/                  # Google Trends cached data
│   └── processed/               # Feature-engineered data
│
├── models/                      # Saved trained models (.pkl)
├── reports/figures/             # Generated plots
└── tests/                       # Unit tests
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up Kaggle API (for dataset download)

1. Go to [kaggle.com/settings](https://www.kaggle.com/settings) → "Create New Token"
2. Place `kaggle.json` in `~/.kaggle/` (Linux/Mac) or `C:\Users\<you>\.kaggle\` (Windows)

### 3. Run the Pipeline

```bash
# Full pipeline (processes data + trains models natively)
python run_pipeline.py
```

### 4. Launch Dashboard

```bash
streamlit run dashboard/app.py
```

The dashboard opens at `http://localhost:8501` with 4 pages:
- **Overview** — KPI cards, sales trends, store performance
- **Trend Analysis** — TVI visualization, spike detection
- **Predictions** — Model comparison (Naive vs RF vs XGBoost)
- **Decision Support** — Live, category-specific store dashboard with organic TVI injection

---

## 📊 Datasets

| Dataset | Source | Rows | Use |
|---|---|---|---|
| Walmart Store Sales | [Kaggle](https://www.kaggle.com/datasets/yasserh/walmart-dataset) | ~6,400 | Primary sales baseline |
| Rossmann Store Sales | [Kaggle](https://www.kaggle.com/datasets/pratyushakar/rossmann-store-sales) | ~1M | *Planned for V2 Generalisation* |
| Google Trends India | [trends.google.com](https://trends.google.com) | Weekly | TVI computation |

---

## 🤖 Models

| Model | Role | Expected MAPE |
|---|---|---|
| Naive Baseline (Lag 1W) | Baseline (no TVI) | ~15–20% |
| Random Forest | Intermediate | ~10–14% |
| **XGBoost + TVI** | **Primary** | **<12%** |

**Δ-MAPE**: XGBoost+TVI achieves measurable improvement over the baseline.

---

## ⚙️ Category-Specific Lag Optimization

| Category | Default Lag | Rationale |
|---|---|---|
| Fashion | 3 days | Impulse buying, social influence |
| Electronics | 14 days | Research-driven, price comparison |
| Festival | 7 days | Moderate decision time |
| General | 7 days | Average behavior |
| Groceries | 2 days | Immediate need |

---

## 📄 License

This project is developed for academic purposes as part of CI124TA — AI Foundations for Engineers, RVCE.

---

*Built with ❤️ by Nemitha, Chaitanya & Inika*
