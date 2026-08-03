# Appliance Energy Forecasting

This repository contains a reproducible time-series forecasting pipeline for modelling and forecasting household appliance energy use.

The project uses the **Appliances Energy Prediction** dataset, which contains appliance energy consumption, indoor temperature and humidity sensor measurements, outdoor weather variables, and timestamp information. The aim is to compare simple benchmark models, a SARIMAX model, a feature-based machine-learning model, and a time-series foundation model.

## Project aim

The aim of this assignment is to forecast short-term household appliance energy use and evaluate whether increasingly complex models improve on simple benchmark methods.

The main questions are:

1. How well do simple benchmark models forecast appliance energy use?
2. Does a SARIMAX model improve on the benchmark forecasts?
3. Do sensor, weather, and time-based covariates improve forecast accuracy?
4. Does a feature-based machine-learning model such as XGBoost improve performance?
5. Does a time-series foundation model such as Chronos provide any additional benefit?
6. Which model would be most suitable for a practical smart-home energy forecasting system?

## Results summary

| Model | MAE (Wh) | RMSE (Wh) | MASE | Bias (Wh) |
|---|---|---|---|---|
| **XGBoost (feature model)** | **33.82** | **55.37** | **0.633** | 4.83 |
| SARIMAX(1,0,3)(1,1,1,24) | 37.19 | 63.81 | 0.696 | 0.08 |
| Seasonal Naïve — Weekly (lag-168) | 42.63 | 79.29 | 0.798 | −10.82 |
| Mean forecast | 50.32 | 74.91 | 0.942 | −3.11 |
| Seasonal Naïve — Daily (lag-24) | 86.96 | 129.23 | 1.628 | 64.01 |
| Foundation Model (Chronos)* | 86.96 | 129.23 | 1.628 | 64.01 |
| Naïve forecast | 250.64 | 258.82 | 4.692 | 247.76 |
| Drift forecast | 266.37 | 274.61 | 4.986 | 264.50 |

\*Chronos uses daily seasonal naïve as a zero-shot placeholder. Fine-tuning on a building-energy corpus is expected to improve performance substantially.

## Dataset

The dataset used in this project is the **Appliances Energy Prediction** dataset.

- **Source:** UCI Machine Learning Repository
- **URL:** https://archive.ics.uci.edu/ml/machine-learning-databases/00374/energydata_complete.csv
- **Reference:** Candanedo, L. M., Feldheim, V., and Deramaix, D. (2017). Data driven prediction models of energy use of appliances in a low-energy house. *Energy and Buildings*, 140, 81–97.

The target variable is:

```text
Appliances
```

This represents household appliance energy use (Wh) for each time interval.

The original dataset is sampled every 10 minutes and contains variables including:

```text
date, Appliances, lights
T1, RH_1, T2, RH_2, T3, RH_3, T4, RH_4, T5, RH_5
T6, RH_6, T7, RH_7, T8, RH_8, T9, RH_9
T_out, Press_mm_hg, RH_out, Windspeed, Visibility, Tdewpoint
rv1, rv2
```

The `T` variables are indoor temperature measurements from different rooms. The `RH` variables are indoor relative humidity measurements. The outdoor weather variables include outdoor temperature, pressure, humidity, wind speed, visibility, and dew point.

## Forecasting task

The main forecasting task is:

```text
Forecast appliance energy use over the next 24 hours.
```

The data are resampled from 10-minute to hourly averages:

```python
horizon = 24          # 24-hour forecast horizon
test_steps = 14 * 24  # final 14 days = 336 hourly observations
```

## Models

### 1. Benchmark models

```text
Mean forecast
Naive forecast
Daily seasonal naive forecast  (lag-24: same hour yesterday)
Weekly seasonal naive forecast (lag-168: same hour last week)
Drift forecast
```

### 2. SARIMAX model

SARIMAX(1,0,3)(1,1,1,24) — order selected by AIC grid search:

```python
p_range = range(0, 4)
d_range = range(0, 2)
q_range = range(0, 4)
```

Exogenous variables: `T_out`, `RH_out`, `Windspeed`, `Visibility`, `Tdewpoint`, `hour_sin`, `hour_cos`

### 3. Feature-based model (XGBoost)

52 engineered features across four groups:

```text
Lag features:     lag_1, lag_2, lag_3, lag_6, lag_12, lag_24, lag_48, lag_168
Rolling features: roll_mean and roll_std over windows 3, 6, 12, 24, 168
Time features:    hour_sin, hour_cos, dow_sin, dow_cos, hour, dayofweek, is_weekend
Sensor/weather:   all indoor temperature, humidity, and outdoor weather variables
```

All lag and rolling features use `.shift(1)` before windowing to prevent data leakage.

### 4. Foundation model (Chronos-T5)

Chronos-T5 (Ansari et al., 2024) in zero-shot mode — no fine-tuning applied.

To use the real model:

```bash
pip install chronos-forecasting torch
```

Then set `USE_REAL_CHRONOS = True` in the notebook.

## Repository structure

```text
Appliance_energy_forecasting/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/           <- energydata_complete.csv (downloaded on first run)
│   ├── interim/
│   └── processed/     <- appliance_hourly.csv (auto-generated)
│
├── notebooks/
│   └── appliance_energy_forecasting.ipynb   <- main notebook (Parts 1-9)
│
├── outputs/
│   ├── figures/       <- all generated plots
│   ├── forecasts/     <- all_forecasts.csv
│   └── metrics/       <- model_comparison.csv
│
└── reports/
    └── appliance_energy_report.pdf
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**requirements.txt:**

```text
numpy
pandas
matplotlib
seaborn
scipy
scikit-learn
statsmodels
xgboost
jupyter
```

## Running the analysis

```bash
jupyter notebook notebooks/appliance_energy_forecasting.ipynb
```

Run all cells top to bottom. The notebook will:

1. Download the dataset from UCI automatically on first run
2. Resample 10-minute data to hourly averages
3. Perform EDA and stationarity tests (ADF, KPSS, ACF, PACF)
4. Run AIC grid search to select the best SARIMAX order
5. Fit all five benchmark models
6. Fit SARIMAX with exogenous variables and 95% confidence intervals
7. Engineer 52 features and fit XGBoost
8. Run the foundation model
9. Evaluate all models and produce comparison plots
10. Save all outputs to `outputs/`

## Feature engineering

```python
# Lag features — past values of target only
lags = [1, 2, 3, 6, 12, 24, 48, 168]
for lag in lags:
    data[f"lag_{lag}"] = data["Appliances"].shift(lag)

# Rolling statistics — shift(1) before windowing prevents leakage
for w in [3, 6, 12, 24, 168]:
    data[f"roll_mean_{w}"] = data["Appliances"].shift(1).rolling(w).mean()
    data[f"roll_std_{w}"]  = data["Appliances"].shift(1).rolling(w).std()

# Cyclic time encodings
data["hour_sin"] = np.sin(2 * np.pi * data.index.hour / 24)
data["hour_cos"] = np.cos(2 * np.pi * data.index.hour / 24)
data["dow_sin"]  = np.sin(2 * np.pi * data.index.dayofweek / 7)
data["dow_cos"]  = np.cos(2 * np.pi * data.index.dayofweek / 7)
```

## Data leakage

All lag and rolling features use `.shift(1)` before any window operation. The train/test split is applied before all feature construction. Indoor sensor values (T1–T9, RH_1–RH_9) from the test set are used as XGBoost features, making those results **conditional forecasts**. An operational system should substitute indoor sensor lag features derived only from past observations.

## Evaluation metrics

```text
MAE   — Mean Absolute Error (Wh)
RMSE  — Root Mean Squared Error (Wh)
MASE  — Mean Absolute Scaled Error (MASE < 1 beats the seasonal naive)
Bias  — Mean signed error (positive = over-forecast)
```

## References

- Ansari, A. F. et al. (2024). Chronos: Learning the Language of Time Series. *arXiv:2403.07815*.
- Candanedo, L. M., Feldheim, V., and Deramaix, D. (2017). Data driven prediction models of energy use of appliances in a low-energy house. *Energy and Buildings*, 140, 81–97.
- Chen, T. and Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *KDD '16*, 785–794.
- Chou, J.-S. and Bui, D.-K. (2014). Modelling heating and cooling loads by artificial intelligence. *Energy and Buildings*, 82, 437–446.
- Hyndman, R. J. and Athanasopoulos, G. (2021). *Forecasting: Principles and Practice* (3rd ed.). OTexts.
- Hyndman, R. J. and Koehler, A. B. (2006). Another look at measures of forecast accuracy. *International Journal of Forecasting*, 22(4), 679–688.
- Kwiatkowski, D. et al. (1992). Testing the null hypothesis of stationarity against the alternative of a unit root. *Journal of Econometrics*, 54(1–3), 159–178.
- Lim, B. et al. (2021). Temporal fusion transformers for interpretable multi-horizon time series forecasting. *International Journal of Forecasting*, 37(4), 1748–1764.
- Seabold, S. and Perktold, J. (2010). Statsmodels: Econometric and statistical modelling with Python. *SciPy 2010*, 57–61.
