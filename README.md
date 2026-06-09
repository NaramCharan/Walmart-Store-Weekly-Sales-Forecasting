# 🛒 Walmart Store Weekly Sales Forecasting 📈

**`Machine Learning · Time Series Forecasting · Retail Demand Prediction`**

A retail demand forecasting system that predicts weekly Walmart store sales across multiple Store–Department combinations using feature engineering, custom recursive walk-forward forecasting, and ensemble tree-based models — with temporal leakage prevention enforced throughout.

Unlike traditional single-series forecasting projects, this system implements a **global forecasting approach**: a single model simultaneously learns sales patterns across all Walmart store and department combinations.



## Problem Statement

Accurate weekly sales forecasting is a critical business capability in retail operations. Without it, organizations face:

- Overstocking or understocking inventory
- Missed revenue during holiday and promotional periods
- Inefficient supply chain operations
- Poor allocation of store resources

This project addresses these challenges by building an end-to-end ML forecasting pipeline using Walmart's historical weekly sales data, holiday indicators, promotional markdowns, and macroeconomic features.

---

## Project Highlights

- **Global Multi-Series Forecasting** — one model trained across all Store–Department combinations simultaneously
- **Custom Recursive Walk-Forward Forecasting** — built from scratch without forecasting libraries, for multi-entity time series
- **Temporal Leakage Caught & Fixed** — detected artificially inflated R² ≈ 0.98, diagnosed root cause, and redesigned the pipeline
- **Rolling Feature Leakage Fixed** — corrected `rolling().mean()` to `shift(1).rolling()` to prevent current-week data from contaminating features
- **Comparative Model Benchmarking** — Random Forest, XGBoost, and LightGBM evaluated under identical leakage-free conditions
- **Modular Pipeline** — preprocessing, feature engineering, and forecasting cleanly separated into reusable components

---

## Key Engineering Challenges & How They Were Solved

This project involved two distinct mistakes that were caught, diagnosed, and fixed. Both are worth understanding.

### ❌ Challenge 1 — Temporal Data Leakage

**What happened:** Lag and rolling features were computed on the full dataset *before* splitting into train and validation sets.

**Why it's a problem:** The validation set inadvertently had access to future sales values through the lag/rolling features — information that would not exist in real deployment.

**Result:** All models showed R² ≈ 0.98 — suspiciously high for a first forecasting attempt.

**Fix:** Split data by date *first*, then compute features strictly on the training partition. Validation features are computed week-by-week inside the recursive forecasting loop.

---

### ❌ Challenge 2 — Rolling Feature Leakage

**What happened:** Rolling features were computed as `rolling().mean()`, which included the *current week's* sales inside the rolling window.

**Why it's a problem:** During forecasting, the current week's sales don't exist yet. Using them in features is another form of leakage.

**Fix:** Changed to `shift(1).rolling()` — ensuring only *past* weeks contribute to rolling statistics. Every rolling feature is now anchored one week behind the prediction target.

---

### ❌ Challenge 3 — Future Feature Generation

**What happened:** After fixing leakage, a new problem emerged. For future validation weeks, there are no actual sales values to compute lag and rolling features from.

**Fix:** Built a custom recursive walk-forward forecasting engine. Each week's prediction is injected back into the dataset as if it were a real sales value, enabling lag and rolling features to be computed for the following week. This simulates real-world deployment conditions.

---
## 📦 Dataset

**Source:** [Walmart Recruiting — Store Sales Forecasting](https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting) — Kaggle Competition

| File | Description | Rows | Key Columns |
|---|---|---|---|
| `train.csv` | Historical weekly sales per Store–Dept | ~421,570 | Store, Dept, Date, Weekly_Sales, IsHoliday |
| `test.csv` | Future weeks to predict (no Weekly_Sales) | ~115,064 | Store, Dept, Date, IsHoliday |
| `stores.csv` | Metadata for all 45 stores | 45 | Store, Type (A/B/C), Size |
| `features.csv` | External indicators per store per week | ~8,190 | Temperature, Fuel_Price, MarkDown1–5, CPI, Unemployment |

**Coverage:** 45 stores · 99 departments · 2010-02-05 to 2012-11-01 · ~2.5 years of weekly data

**Holiday weeks tracked:** Super Bowl · Labor Day · Thanksgiving · Christmas
*(Holiday weeks are weighted 5× higher in Kaggle's official evaluation metric)*

**Notable data challenges handled:**
- `MarkDown` data only available after November 2011 — missing values filled with `0`
- `CPI` and `Unemployment` missing for future test weeks — handled via forward-fill
- Store Type is anonymized (A / B / C) representing different store formats and sizes

## Architecture Overview

```
raw data (train.csv, test.csv, stores.csv, features.csv)
         │
         ▼
   MakeFeatures (feature_creation.py)
   ─────────────────────────────────────────────────────
   · merge store and external features onto training data
   · extract date parts: Day, Month, ISO Week
   · compute lag features (lag_1, lag_2, lag_4)
   · compute rolling features (mean/std over 4 and 8 weeks)
   · drop rows with NaN lag/rolling values
         │
         ▼
      main.py
   ─────────────────────────────────────────────────────
   · fill MarkDown nulls with 0; forward-fill CPI and Unemployment
   · temporal split: train < 2012-05-25, validation >= 2012-05-25
   · lag/rolling features computed on training set ONLY
   · OneHotEncode: Type, IsHoliday
   · train XGBoost, LightGBM, Random Forest inside sklearn Pipelines
   · recursive walk-forward forecasting on validation set
   · benchmark all three models; select best
   · retrain best model on full data; forecast test set
         │
         ▼
   Recursive_Forecasting_ (recursive_forecasting.py)
   ─────────────────────────────────────────────────────
   · concatenate train + validation, sorted by Store, Dept, Date
   · iterate week-by-week from validation start date
   · recompute lag and rolling features for current week
   · predict Weekly_Sales for all Store–Dept pairs
   · inject predictions back into combined_df['Weekly_Sales']
   · advance date by 7 days → repeat
```

---

## Feature Engineering

Features were engineered to capture historical patterns, short-term trends, weekly seasonality, and sales volatility.

**Date Features**

| Feature | Description |
|---|---|
| `Day` | Day of the month |
| `Month` | Month of the year |
| `Week` | ISO week number |

**Lag Features** — computed per `(Store, Dept)` group

| Feature | Description |
|---|---|
| `lag_1` | Sales from 1 week prior |
| `lag_2` | Sales from 2 weeks prior |
| `lag_4` | Sales from 4 weeks prior |

**Rolling Statistical Features** — computed from `lag_1` via `shift(1).rolling()` to prevent leakage

| Feature | Purpose | Description |
|---|---|---|
| `rolling_4_mean` | Trend | 4-week rolling mean (`min_periods=2`) |
| `rolling_8_mean` | Trend | 8-week rolling mean (`min_periods=4`) |
| `rolling_4_std` | Volatility | 4-week rolling std (`min_periods=2`) |
| `rolling_8_std` | Volatility | 8-week rolling std (`min_periods=4`) |

**External Business Features:** `IsHoliday`, `Type`, `Size`, `MarkDown1–5`, `CPI`, `Fuel_Price`, `Unemployment`, `Temperature`

---

## Recursive Walk-Forward Forecasting

Generating future predictions is the central engineering challenge. Future weeks have no actual sales values, so lag and rolling features cannot be computed directly.

`Recursive_Forecasting_` in `recursive_forecasting.py` solves this:

```
Historical Training Data (with known Weekly_Sales)
        │
        ▼
Concatenate with Validation Set (Weekly_Sales = NaN initially)
Sort by Store, Dept, Date
        │
        ▼
For each week in validation horizon:
    ├── Recompute lag_1, lag_2, lag_4 using current Weekly_Sales column
    ├── Recompute rolling_4/8_mean and rolling_4/8_std via shift(1).rolling()
    ├── Predict Weekly_Sales for all (Store, Dept) pairs this week
    ├── Write predictions back into combined_df['Weekly_Sales']
    └── Advance date by 7 days → repeat
        │
        ▼
Return combined_df with forecasted Weekly_Sales for all future weeks
```

**Concrete example of recursive logic:**

| Step | Predicting Week | Uses History From |
|---|---|---|
| 1 | 2012-05-25 | 2012-05-18, 2012-05-11, 2012-04-27 (actual) |
| 2 | 2012-06-01 | 2012-05-25 (predicted), 2012-05-18, 2012-05-11 (actual) |
| 3 | 2012-06-08 | 2012-06-01 (predicted), 2012-05-25 (predicted), 2012-05-18 (actual) |

Each prediction feeds the next. This is how real-world deployment works.

**Why not use `skforecast`?**

`skforecast` is designed for single-series forecasting. This project required recursive forecasting across ~3,000 Store–Department pairs simultaneously, which necessitated a custom global multi-series implementation.

---

## Temporal Leakage Detection & Resolution

**Initial (broken) pipeline:**
```
Full Dataset → Create Lag Features → Create Rolling Features → Train/Validation Split
```
Result: R² ≈ 0.98 — looks great, completely wrong.

**Fixed pipeline:**
```
Train/Validation Split → Feature Engineering on Train Only → Recursive Forecasting on Validation
```

**Two separate leakage issues fixed:**

| Issue | Cause | Fix |
|---|---|---|
| Temporal leakage | Features computed before split | Split first, engineer after |
| Rolling window leakage | `rolling().mean()` included current week | Changed to `shift(1).rolling()` |

---

## Model Training & Hyperparameter Decisions

Three models were trained using `sklearn.Pipeline` with `OneHotEncoder` for `Type` and `IsHoliday`.

Hyperparameters were tuned manually through experimentation rather than automated search (Optuna, GridSearch). Key observation during tuning:

> Increasing `max_depth` and `num_leaves` beyond a threshold caused clear overfitting — validation error rose while training error dropped. This was a direct, observable signal used to select final parameters.

| Model | Key Hyperparameters |
|---|---|
| Random Forest | `n_estimators=1000`, `max_depth=12`, `min_samples_split=2` |
| XGBoost | `n_estimators=1000`, `max_depth=14`, `learning_rate=0.1` |
| LightGBM | `n_estimators=1000`, `max_depth=16`, `num_leaves=70`, `learning_rate=0.2` |

---

## Model Performance

All models evaluated on the held-out validation set (`Date >= 2012-05-25`) after recursive walk-forward forecasting with both leakage issues fully resolved:

| Model | MSE | RMSE | R² |
|---|---|---|---|
| XGBoost | 46,932,840.00 | 6,850.75 | 90.33% |
| Random Forest | 30,911,100.00 | 5,559.78 | 93.63% |
| **LightGBM** | **21,593,720.00** | **4,646.90** | **95.55%** |

**Selected Model: LightGBM Regressor**

LightGBM was selected as the final forecasting model based on:
- Highest R² score (95.55%)
- Lowest RMSE (4,646.90)
- Best generalization under multi-entity recursive forecasting conditions

> These are the honest post-leakage-removal metrics. The inflated R² ≈ 0.98 figures seen during initial experimentation were a direct result of temporal data leakage and are not representative of real forecasting performance.

---

## Key Learnings

**1. Feature engineering matters more than model complexity**
Performance improved more from better lag and rolling features than from switching between algorithms.

**2. High validation scores can be a red flag**
R² ≈ 0.98 triggered suspicion, not celebration. That instinct to question results is more valuable than the result itself.

**3. There are multiple ways leakage can hide in a time-series pipeline**
Temporal leakage (split order) and rolling window leakage (current-week inclusion) are separate bugs that look identical from the outside — both inflate metrics silently.

**4. Forecasting is fundamentally different from standard ML**
Future features don't exist. You have to simulate the real world during evaluation, not just hold out rows.

**5. Debugging is the majority of the work**
A large portion of this project was diagnosing incorrect pipelines, not training models.

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-EB4C42?style=for-the-badge&logo=xgboost&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-02569B?style=for-the-badge&logo=lightgbm&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=matplotlib&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![Git](https://img.shields.io/badge/Git-F05032?style=for-the-badge&logo=git&logoColor=white)

---

## Project Structure

```
Walmart-Store-Weekly-Sales-Forecasting/
│
├── Data/
│   ├── train.csv
│   ├── test.csv
│   ├── stores.csv
│   └── features.csv
│
├── outputs/
│   └── metrics_table.png
│
├── src/
│   ├── feature_creation.py
│   └── recursive_forecasting.py
│
├── main.py
├── requirements.txt
└── README.md
```

---

## Installation & Usage

**1. Clone the repository**

```bash
git clone https://github.com/NaramCharan/Walmart-Store-Weekly-Sales-Forecasting.git
cd Walmart-Store-Weekly-Sales-Forecasting
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Run the pipeline**

```bash
python main.py
```

> 🔁 **Pipeline Execution Flow**
>
> `📂 Data Loading` → `🔗 Merging` → `📅 Date Feature Extraction` → `✂️ Temporal Split` → `⚙️ Lag/Rolling Feature Engineering` → `🏋️ Model Training` → `🔄 Recursive Walk-Forward Forecasting` → `📊 Benchmarking` → `🔃 Full-Data Retraining` → `🎯 Test Set Prediction` → `📄 overall_predictions.csv`

---

## 🔮 Future Roadmap

- [ ] Optimize recursive forecasting loop — recompute features only for current week rows instead of full DataFrame
- [ ] Deep learning forecasting with LSTM or Temporal Fusion Transformer
- [ ] MLOps pipeline with automated retraining, monitoring, and drift detection
- [ ] Probabilistic forecasting with prediction intervals

---

## Research Reference

> *"Walmart Sales Prediction Based on Machine Learning" — DR Press*

Domain understanding and methodological context were drawn from this paper. All implementation — including the recursive forecasting engine, leakage detection and resolution, rolling feature correction, and global multi-series pipeline — was independently developed through experimentation.

---

## Contact

**Naram Charan**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://www.linkedin.com/in/naramcharan/)
[![Email](https://img.shields.io/badge/Email-charannaram1710@gmail.com-red?logo=gmail)](mailto:charannaram1710@gmail.com)

---

*If this project helped you, consider giving it a star on GitHub.*
