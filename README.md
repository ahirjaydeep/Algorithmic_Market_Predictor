# 📈 Algorithmic Market Predictor v2
### S&P 500 · No-Leakage Architecture · XGBoost · Live Inference

> **Author:** Jaydeep Ahir  
> *Lead Machine Learning Engineer*

---

## Overview

A production-grade machine learning pipeline that predicts **next-day stock closing prices** using a rigorously engineered, **data-leakage-free** architecture. Built on XGBoost with chronological train/test splitting to faithfully simulate real-world trading conditions.

Most amateur stock prediction notebooks achieve artificially inflated R² scores (99%+) by leaking same-day price data into the model. This project was built to solve that problem from the ground up.

---

## The Core Problem: Data Leakage

Early iterations of stock prediction models often achieve suspiciously high accuracy by using **same-day boundaries** (e.g., today's Open/High/Low) to predict today's Close. Because intraday prices are tightly correlated, the model learns a trivial "persistence" pattern rather than any genuine predictive signal.

**This project explicitly prevents leakage by:**
- Shifting the target variable exactly one trading day forward
- Deriving all features strictly from historical (past) data
- Splitting train/test sets **chronologically**, not randomly

---

## Pipeline Architecture

```
Raw Price Data (yfinance)
        │
        ▼
┌─────────────────────────────────┐
│  Phase 1: Feature Engineering   │  ← No-leakage technical indicators
│  • MA Ratios (5d, 20d)          │
│  • RSI (14-period)              │
│  • MACD                         │
│  • Bollinger Bands              │
│  • ATR (Volatility)             │
│  • Target: Next-day % Return    │
└─────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────┐
│  Phase 2: Chronological Split   │  ← 80% train / 20% test (time-ordered)
│  + XGBoost Regressor Training   │
│  • L1 & L2 Regularization       │
│  • Depth & child-weight limits  │
└─────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────┐
│  Phase 3: Price Reconstruction  │  ← Return → Actual price projection
│  • Visualization (last 100 days)│
│  • Error metrics (RMSE, MAE, R²)│
└─────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────┐
│  Phase 4: Live Inference        │  ← Predicts next trading session
│  • Latest close as input        │
│  • Outputs projected price + Δ  │
└─────────────────────────────────┘
```

---

## Features & Technical Indicators

| Feature | Type | Description |
|---|---|---|
| `MA5_Ratio` | Trend | Price relative to 5-day moving average |
| `MA20_Ratio` | Trend | Price relative to 20-day moving average |
| `RSI_14` | Momentum | 14-period Relative Strength Index |
| `MACD` | Momentum | EMA(12) − EMA(26) crossover signal |
| `Bollinger_Width` | Volatility | Band width as a % of price |
| `ATR_14` | Volatility | 14-period Average True Range |
| `Target_Return` | **Target** | Next day's % return (shifted −1) |

---

## Model Configuration

```python
xgb.XGBRegressor(
    n_estimators   = 150,
    learning_rate  = 0.05,
    max_depth      = 4,
    min_child_weight = 3,
    subsample      = 0.8,
    reg_alpha      = 0.1,   # L1 regularization
    reg_lambda     = 1.0,   # L2 regularization
    random_state   = 42
)
```

Heavy regularization is applied specifically to combat overfitting to financial noise — a common failure mode in time-series ML.

---

## Installation & Usage

### Prerequisites

```bash
pip install yfinance xgboost scikit-learn pandas numpy matplotlib seaborn
```

### Run in Google Colab (Recommended)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

1. Upload `Algorithmic_Market_Predictor_v2.ipynb` to Colab
2. Run all cells in order (`Runtime → Run all`)
3. Live data is fetched automatically via `yfinance`

### Run Locally

```bash
git clone https://github.com/ahirjaydeep/Algorithmic_Market_Predictor.git
cd algorithmic-market-predictor
jupyter notebook Algorithmic_Market_Predictor_v2.ipynb
```

---

## Live Inference Output

After training, the notebook produces a real-time prediction for the **next active trading session**:

```
==================================================
 LIVE MARKET INFERENCE FOR AAPL
==================================================
Data Current As Of:    2025-XX-XX
Last Closing Price:    $XXX.XX
--------------------------------------------------
Projected Next Close:  $XXX.XX
Expected Movement:     $+X.XX (+X.XX%)
==================================================
```

---

## Key Design Decisions

**Why predict returns instead of raw prices?**  
Raw prices are non-stationary — they trend over time, which makes them unsuitable targets for most ML models. Percentage returns are approximately stationary and are the correct financial quantity to model.

**Why chronological splitting?**  
Random splitting causes future data to bleed into training, simulating unrealistic "oracle" conditions. A chronological 80/20 split ensures the model is evaluated on data it has never seen and that was generated *after* training.

**Why XGBoost over deep learning?**  
Tabular financial data with engineered features is well-suited to gradient boosting. XGBoost with strong regularization generalizes better than neural networks on short financial time series without architectural tuning.

---

## Project Structure

```
📂 algorithmic-market-predictor/
├── 📓 Algorithmic_Market_Predictor_v2.ipynb   # Main notebook
└── 📄 README.md
```

---

## Disclaimer

> This project is built for **educational and research purposes only**. It is not financial advice. Past price patterns do not guarantee future results. Do not make investment decisions based on this model.

---

## Author

**Jaydeep Ahir** — Lead Machine Learning Engineer

*Note: The ML architecture, feature engineering logic, and code execution in this project are original work. Explanatory markdown text and documentation were drafted with the assistance of an AI language model.*
