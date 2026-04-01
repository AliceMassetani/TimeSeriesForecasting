import pandas as pd
import numpy as np
import os
import holidays
import logging

# Suppress Prophet's noisy cmdstanpy logger from spamming on exit
cmdstanpy_logger = logging.getLogger('cmdstanpy')
cmdstanpy_logger.addHandler(logging.NullHandler())
cmdstanpy_logger.propagate = False
cmdstanpy_logger.setLevel(logging.CRITICAL)

from darts import TimeSeries, set_option
from darts.models import Prophet, ARIMA
from darts.metrics import mae, mse, rmse, r2_score
from darts.utils.missing_values import fill_missing_values


base_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_dir, '..', 'datasets', 'Dati_processed.csv')
outputs_dir = os.path.join(base_dir, '..', 'outputs')
os.makedirs(outputs_dir, exist_ok=True)
report_path = os.path.join(outputs_dir, 'hyperparameter_results.txt')

print("--- 1. Data Loading and Preparation ---")
df = pd.read_csv(data_path)
df['TimeStamp'] = pd.to_datetime(df['TimeStamp'])

# 1h interval data
df_resampled = df

# Italian Holidays
it_holidays = holidays.Italy()
df_resampled['is_holiday'] = df_resampled['TimeStamp'].apply(lambda x: 1 if x in it_holidays else 0)

target_col = 'InUseCapacity'
future_cols = ['hour', 'day_of_week', 'is_holiday']
past_cols = [col for col in df_resampled.columns if col not in [target_col, 'TimeStamp', 'item_id'] and col not in future_cols]

# Create TimeSeries objects
series = fill_missing_values(TimeSeries.from_dataframe(df_resampled, time_col='TimeStamp', value_cols=target_col))
future_cov = fill_missing_values(TimeSeries.from_dataframe(df_resampled, time_col='TimeStamp', value_cols=future_cols))
past_cov = fill_missing_values(TimeSeries.from_dataframe(df_resampled, time_col='TimeStamp', value_cols=past_cols))

# For GridSearch, we use the last 30 days as our cross-validation/evaluation split
# to keep the search process reasonably fast while still being representative.
prediction_length = 30 * 24 
train_series, val_series = series[:-prediction_length], series[-prediction_length:]

print(f" Total Series: {len(series)} steps")
print(f" Validation Start Point: {len(train_series)} steps")

# ==========================================
# --- 2. ARIMA Gridsearch ---
# ==========================================
print("\n--- 2. Starting ARIMA Gridsearch ---")
parameters_arima = {
    "p": [1, 2, 3, 5],
    "d": [0, 1],
    "q": [0, 1, 2]
}

print(f" Evaluating ARIMA parameter combinations...")
# Darts gridsearch evaluates combinations using a rolling window from the `start` point
best_arima, best_params_arima, min_error_arima = ARIMA.gridsearch(
    parameters=parameters_arima,
    series=train_series,
    start=len(train_series)-prediction_length,
    forecast_horizon=24,
    stride=24,
    metric=rmse,
    n_jobs=-1,
    verbose=True
)
print(f"   Best ARIMA Parameters: {best_params_arima}")
print(f"   Best ARIMA RMSE Score: {min_error_arima:.4f}")

# Calculate additional metrics for the winning model
other_metrics_arima = best_arima.backtest(
    series=train_series,
    start=len(train_series)-prediction_length,
    forecast_horizon=24,
    stride=24,
    metric=[mae, mse, r2_score],
    verbose=False
)
mae_arima, mse_arima, r2_arima = other_metrics_arima
print(f"   Best ARIMA MAE Score: {mae_arima:.4f}")
print(f"   Best ARIMA MSE Score: {mse_arima:.4f}")
print(f"   Best ARIMA R2 Score: {r2_arima:.4f}")

# ==========================================
# --- 3. PROPHET Gridsearch ---
# ==========================================
print("\n--- 3. Starting Prophet Gridsearch ---")
parameters_prophet = {
    "country_holidays": ["IT"],
    "seasonality_prior_scale": [0.01, 0.1, 1.0, 10.0],
    "changepoint_prior_scale": [0.001, 0.01, 0.1, 0.5]
}

print(f" Evaluating Prophet parameter combinations...")
best_prophet, best_params_prophet, min_error_prophet = Prophet.gridsearch(
    parameters=parameters_prophet,
    series=train_series,
    future_covariates=future_cov,
    start=len(train_series)-prediction_length,
    forecast_horizon=24,
    stride=24,
    metric=rmse,
    n_jobs=-1,
    verbose=True
)
print(f"   Best Prophet Parameters: {best_params_prophet}")
print(f"   Best Prophet RMSE Score: {min_error_prophet:.4f}")

# Calculate additional metrics for the winning model
other_metrics_prophet = best_prophet.backtest(
    series=train_series,
    future_covariates=future_cov,
    start=len(train_series)-prediction_length,
    forecast_horizon=24,
    stride=24,
    metric=[mae, mse, r2_score],
    verbose=False
)
mae_prophet, mse_prophet, r2_prophet = other_metrics_prophet
print(f"   Best Prophet MAE Score: {mae_prophet:.4f}")
print(f"   Best Prophet MSE Score: {mse_prophet:.4f}")
print(f"   Best Prophet R2 Score: {r2_prophet:.4f}")


# ==========================================
# --- 4. CHRONOS-2 Gridsearch ---
# ==========================================
print("\n--- 4. Starting Chronos-2 Gridsearch ---")
from darts.models import Chronos2Model

# We define the hyperparameter space. 
parameters_chronos = {
    "hub_model_name": ["amazon/chronos-2"],
    "input_chunk_length": [256, 512, 1024],
    "output_chunk_length": [128, 512, 1024]
}

print(f" Evaluating Chronos-2 parameter combinations on GPU...")
try:
    best_chronos, best_params_chronos, min_error_chronos = Chronos2Model.gridsearch(
        parameters=parameters_chronos,
        series=train_series,
        past_covariates=past_cov,
        future_covariates=future_cov,
        start=len(train_series)-prediction_length,
        forecast_horizon=24,
        stride=24,
        metric=rmse,
        n_jobs=1,
        verbose=True
    )

    print(f"   Best Chronos-2 Parameters: {best_params_chronos}")
    print(f"   Best Chronos-2 RMSE Score: {min_error_chronos:.4f}")

    # Calculate additional metrics for the winning model using matching rolling window
    other_metrics_chronos = best_chronos.backtest(
        series=train_series,
        past_covariates=past_cov,
        future_covariates=future_cov,
        start=len(train_series)-prediction_length,
        forecast_horizon=24,
        stride=24,
        metric=[mae, mse, r2_score],
        verbose=False
    )
    mae_chronos, mse_chronos, r2_chronos = other_metrics_chronos

    print(f"   Best Chronos-2 MAE Score: {mae_chronos:.4f}")
    print(f"   Best Chronos-2 MSE Score: {mse_chronos:.4f}")
    print(f"   Best Chronos-2 R2 Score: {r2_chronos:.4f}")

except Exception as e:
    print(f"\n Chronos-2 GridSearch failed: {e}")
    best_params_chronos = "N/A"
    min_error_chronos = np.nan
    mae_chronos = np.nan
    mse_chronos = np.nan
    r2_chronos = np.nan


# ==========================================

# Save metrics to txt file
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("--- Hyperparameter Search Results ---\n\n")
    f.write("ARIMA:\n")
    f.write(f"Best Parameters: {best_params_arima}\n")
    f.write(f"Best RMSE Score (Optimized): {min_error_arima:.4f}\n")
    f.write(f"Best MAE Score: {mae_arima:.4f}\n")
    f.write(f"Best MSE Score: {mse_arima:.4f}\n")
    f.write(f"Best R2 Score: {r2_arima:.4f}\n\n")
    
    f.write("Prophet:\n")
    f.write(f"Best Parameters: {best_params_prophet}\n")
    f.write(f"Best RMSE Score (Optimized): {min_error_prophet:.4f}\n")
    f.write(f"Best MAE Score: {mae_prophet:.4f}\n")
    f.write(f"Best MSE Score: {mse_prophet:.4f}\n")
    f.write(f"Best R2 Score: {r2_prophet:.4f}\n\n")
    
    f.write("Chronos-2:\n")
    if best_params_chronos != "N/A":
        f.write(f"Best Parameters: {best_params_chronos}\n")
        f.write(f"Best RMSE Score (Optimized): {min_error_chronos:.4f}\n")
        f.write(f"Best MAE Score: {mae_chronos:.4f}\n")
        f.write(f"Best MSE Score: {mse_chronos:.4f}\n")
        f.write(f"Best R2 Score: {r2_chronos:.4f}\n")
    else:
        f.write("GridSearch failed.\n")

print(f"\n Hyperparameter search finished! Results saved to {report_path}")
