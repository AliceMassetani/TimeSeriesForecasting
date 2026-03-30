import pandas as pd
import numpy as np
import os
import holidays

from darts import TimeSeries, set_option
from darts.models import Prophet, ARIMA
from darts.metrics import mae
from darts.utils.missing_values import fill_missing_values

def run_hyperparameter_search():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, '..', 'datasets', 'Dati_processed.csv')
    
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
    # past_cov = fill_missing_values(TimeSeries.from_dataframe(df_resampled, time_col='TimeStamp', value_cols=past_cols)) # Not needed for ARIMA/Prophet

    # For GridSearch, we use the last 30 days as our cross-validation/evaluation split
    # to keep the search process reasonably fast while still being representative.
    prediction_length = 30 * 24 
    train_series = series[:-prediction_length]

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
        series=series,
        start=len(train_series),           # Where validation begins
        forecast_horizon=24,               # Test 24-hours ahead repetitively
        metric=mae,
        n_jobs=-1,                         # Use all available CPU cores for speed
        verbose=True
    )
    print(f"  🏆 Best ARIMA Parameters: {best_params_arima}")
    print(f"  🏆 Best ARIMA MAE Score: {min_error_arima:.4f}")

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
        series=series,
        future_covariates=future_cov,      # Only passing future_cov since Prophet supports it
        start=len(train_series),
        forecast_horizon=24,
        metric=mae,
        n_jobs=-1,
        verbose=True
    )
    print(f"  🏆 Best Prophet Parameters: {best_params_prophet}")
    print(f"  🏆 Best Prophet MAE Score: {min_error_prophet:.4f}")

    print("\n🎉 Hyperparameter search finished! Update your models in the main scripts with these parameters.")

if __name__ == "__main__":
    run_hyperparameter_search()
