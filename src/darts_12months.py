import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import holidays

from darts import TimeSeries, set_option
from darts.models import Chronos2Model, Prophet, ARIMA
from darts.metrics import mae, rmse, r2_score
from darts.utils.missing_values import fill_missing_values


def run_native_darts_tournament_12m():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, '..', 'datasets', 'Dati_processed.csv')
    outputs_dir = os.path.join(base_dir, '..', 'outputs')
    os.makedirs(outputs_dir, exist_ok=True)

    print("--- 1. Data Loading and Preparation (12-Month Horizon) ---")
    df = pd.read_csv(data_path)
    df['TimeStamp'] = pd.to_datetime(df['TimeStamp'])
    
    # Data is already at 1h intervals (24 steps/day) — no resampling needed
    df_resampled = df

    # Add Italian Holidays
    it_holidays = holidays.Italy()
    df_resampled['is_holiday'] = df_resampled['TimeStamp'].apply(lambda x: 1 if x in it_holidays else 0)

    # --- 2. Target Definition and Covariate Split ---
    target_col = 'InUseCapacity'
    
    # Strict separation to prevent data leakage
    future_cols = ['hour', 'day_of_week', 'is_holiday']

    past_cols = [col for col in df_resampled.columns if col not in [target_col, 'TimeStamp', 'item_id'] and col not in future_cols]
    
    print(f" Resampling: 1h intervals (24 steps/day)")
    print(f" Total resampled steps: {len(df_resampled)}")
    print(f" Target: {target_col}")
    print(f" Future Covariates ({len(future_cols)}): {future_cols}")
    print(f" Past Covariates ({len(past_cols)}): {past_cols}")

    # --- 3. TimeSeries Creation and Data Imputation (Darts Native) ---
    series = fill_missing_values(TimeSeries.from_dataframe(df_resampled, time_col='TimeStamp', value_cols=target_col))
    future_cov = fill_missing_values(TimeSeries.from_dataframe(df_resampled, time_col='TimeStamp', value_cols=future_cols))
    past_cov = fill_missing_values(TimeSeries.from_dataframe(df_resampled, time_col='TimeStamp', value_cols=past_cols))

    # Split: 12 months = 365 days * 24 steps/day = 8760 steps
    prediction_length = 365 * 24  # 8760 steps
    
    # Safety check: ensure enough training data
    if len(series) <= prediction_length:
        print(f" WARNING: Series length ({len(series)}) <= prediction_length ({prediction_length}).")
        print(f" Adjusting prediction_length to use 20% of data for training.")
        prediction_length = int(len(series) * 0.8)
    
    train_series, test_series = series[:-prediction_length], series[-prediction_length:]
    print(f"\n Training steps: {len(train_series)}")
    print(f" Test/Prediction steps: {len(test_series)} ({prediction_length // 24} days)")

    # --- 4. Models Arena ---
    # Chronos-2 requires: input_chunk_length + output_chunk_length <= len(train_series)
    # We maximize output_chunk_length (up to 1024 limit) to minimize auto-regressive error compounding
    available_budget = len(train_series) - 1
    chronos_output_len = min(1024, int(available_budget * 0.6))
    chronos_input_len = min(1024, available_budget - chronos_output_len)
    
    print(f"\n Chronos-2 Config: input_chunk={chronos_input_len}, output_chunk={chronos_output_len}")
    print(f" Auto-regressive steps needed: {max(0, prediction_length - chronos_output_len)}")
    
    models = {
        "ARIMA": ARIMA(), 
        "Prophet": Prophet(country_holidays='IT'),
        "Chronos-2": Chronos2Model(
            input_chunk_length=chronos_input_len,
            output_chunk_length=chronos_output_len
        ) 
    }

    results = {}

    # --- 5. Training, Prediction, Backtesting ---
    for name, model in models.items():
        print(f"\n Running: {name}...")
        
        # Dynamic handling of model-supported covariates
        fit_kwargs = {}
        predict_kwargs = {}
        
        if model.supports_future_covariates:
            fit_kwargs['future_covariates'] = future_cov
            predict_kwargs['future_covariates'] = future_cov
            
        if model.supports_past_covariates:
            fit_kwargs['past_covariates'] = past_cov
            predict_kwargs['past_covariates'] = past_cov

        # Model Training
        model.fit(train_series, **fit_kwargs)
        
        # Prediction (Chronos-2 auto-regresses beyond output_chunk_length)
        if getattr(model, "is_probabilistic", False):
            forecast = model.predict(n=prediction_length, num_samples=100, **predict_kwargs).quantile(0.5)
        else:
            forecast = model.predict(n=prediction_length, **predict_kwargs)
        
        # Calculate Comprehensive Metrics
        mae_val = mae(test_series, forecast)
        rmse_val = rmse(test_series, forecast)
        r2_val = r2_score(test_series, forecast)
        
        print(f" {name} Results:")
        print(f"   -> MAE:   {mae_val:.2f}")
        print(f"   -> RMSE:  {rmse_val:.2f}")
        print(f"   -> R²:    {r2_val:.2f}")

        # Historical Forecasts (Darts Native Backtesting) - Prophet only
        # retrain=True is required for local models like Prophet
        if name in ["Prophet"]: 
            backtest_days = 30  # backtest over last 30 days of training set
            backtest_steps = backtest_days * 24  # 720 steps at 24 steps/day
            horizon_steps = 24  # 1-day forecast horizon (24 steps = 24h at 1h intervals)
            stride_steps = 24   # stride of 1 day
            
            if len(train_series) > backtest_steps + horizon_steps:
                print(f"   Starting Historical Backtesting for {name} (last {backtest_days} days of train set)...")
                backtest_forecast = model.historical_forecasts(
                    series=train_series,
                    future_covariates=future_cov if model.supports_future_covariates else None,
                    past_covariates=past_cov if model.supports_past_covariates else None,
                    start=len(train_series) - backtest_steps,
                    forecast_horizon=horizon_steps,
                    stride=stride_steps,
                    retrain=True,
                    verbose=False
                )
                backtest_mae = mae(train_series, backtest_forecast)
                print(f"    Backtest MAE ({name}): {backtest_mae:.2f}")
            else:
                print(f"   Skipping backtesting for {name}: insufficient training data.")

        # Store all metrics
        results[name] = {
            "forecast": forecast, 
            "mae": mae_val, 
            "rmse": rmse_val,
            "r2": r2_val
        }

    # --- 6. Benchmark Plots Generation ---
    print("\n--- 6. Generating Final Benchmark Plots ---")
    
    set_option("plotting.use_darts_style", True)
    # Plot 1: Full 12-Month Benchmark
    plt.figure(figsize=(20, 8))
    test_series.plot(label='Actual Data', color='green', linewidth=2)
    for name, data in results.items():
        label_str = f'{name} (MAE: {data["mae"]:.2f}, R²: {data["r2"]:.2f})'
        data["forecast"].plot(label=label_str)
    plt.title('12-Month Benchmark - Predictive Autoscaling (1h resolution)')
    plt.legend()
    plt.savefig(os.path.join(outputs_dir, 'darts_native_benchmark_12m_full.png'))
    plt.close()

    # Plot 2: Zoomed View (Last 30 Days)
    plt.figure(figsize=(20, 8))
    zoom_steps = 30 * 24  # 30 days * 24 steps/day = 720 steps
    test_series[-zoom_steps:].plot(label='Actual Data', color='green', linewidth=3)
    for name, data in results.items():
        data["forecast"][-zoom_steps:].plot(label=name, linewidth=2)
    plt.title('Monthly Detail (Zoom on Last 30 Days)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(outputs_dir, 'darts_native_benchmark_12m_zoom.png'))
    plt.close()

    # Plot 3: Quarterly View (Last 90 Days)
    plt.figure(figsize=(20, 8))
    zoom_q = 90 * 24  # 90 days * 24 steps/day = 2160 steps
    test_series[-zoom_q:].plot(label='Actual Data', color='green', linewidth=3)
    for name, data in results.items():
        data["forecast"][-zoom_q:].plot(label=name, linewidth=2)
    plt.title('Quarterly Detail (Zoom on Last 90 Days)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(outputs_dir, 'darts_native_benchmark_12m_quarterly.png'))
    plt.close()

    # --- Save Metrics Summary to CSV ---
    metrics_summary = pd.DataFrame({
        name: {
            "MAE": data["mae"],
            "RMSE": data["rmse"],
            "R-squared": data["r2"]
        } for name, data in results.items()
    }).T
    metrics_summary.to_csv(os.path.join(outputs_dir, 'darts_metrics_summary_12m.csv'))

    print("\n--- Final Metrics Summary (12-Month Forecast) ---")
    print(metrics_summary.to_string())
    print(f"\n 12-Month Darts analysis completed! Check the /outputs folder.")

if __name__ == "__main__":
    run_native_darts_tournament_12m()
