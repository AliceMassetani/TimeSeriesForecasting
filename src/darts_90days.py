import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import holidays

from darts import TimeSeries, set_option
from darts.models import Chronos2Model, Prophet, ARIMA
from darts.metrics import mae, rmse, r2_score
from darts.utils.missing_values import fill_missing_values
from darts.utils.statistics import plot_residuals_analysis


def run_native_darts_tournament():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, '..', 'datasets', 'Dati_processed.csv')
    outputs_dir = os.path.join(base_dir, '..', 'outputs')
    os.makedirs(outputs_dir, exist_ok=True)

    print("--- 1. Data Loading and Preparation ---")
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
    
    print(f" Target: {target_col}")
    print(f" Future Covariates ({len(future_cols)}): {future_cols}")
    print(f" Past Covariates ({len(past_cols)}): {past_cols}")

    # --- 3. TimeSeries Creation and Data Imputation (Darts Native) ---
    series = fill_missing_values(TimeSeries.from_dataframe(df_resampled, time_col='TimeStamp', value_cols=target_col))
    future_cov = fill_missing_values(TimeSeries.from_dataframe(df_resampled, time_col='TimeStamp', value_cols=future_cols))
    past_cov = fill_missing_values(TimeSeries.from_dataframe(df_resampled, time_col='TimeStamp', value_cols=past_cols))

    # Split 90 days = 90 * 24 steps/day = 2160 steps
    prediction_length = 90 * 24  # 2160 steps
    train_series, test_series = series[:-prediction_length], series[-prediction_length:]

    # --- 4. Models Arena ---
    models = {
        "ARIMA": ARIMA(), 
        "Prophet": Prophet(country_holidays='IT'),
        "Chronos-2": Chronos2Model(
            hub_model_name="amazon/chronos-2",
            input_chunk_length=1024, 
            output_chunk_length=1024
        ) 
    }

    results = {}

    # --- 5. Training, Prediction, Backtesting, and Residuals ---
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
        
        # Native Darts prediction
        if getattr(model, "is_probabilistic", False):
            forecast = model.predict(n=prediction_length, num_samples=100, **predict_kwargs).quantile(0.5)
        else:
            forecast = model.predict(n=prediction_length, **predict_kwargs)
        
        # --- NEW: Calculate 5 Comprehensive Metrics ---
        mae_val = mae(test_series, forecast)
        rmse_val = rmse(test_series, forecast)
        r2_val = r2_score(test_series, forecast)
        
        print(f" {name} Results:")
        print(f"   -> MAE:   {mae_val:.2f}")
        print(f"   -> RMSE:  {rmse_val:.2f}")
        print(f"   -> R²:    {r2_val:.2f}")

            
        # Historical Forecasts (Darts Native Backtesting)
        if name in ["Prophet"]: 
            print(f"   Starting Historical Backtesting for {name} (last 30 days of train set)...")
            backtest_forecast = model.historical_forecasts(
                series=train_series,
                future_covariates=future_cov if model.supports_future_covariates else None,
                past_covariates=past_cov if model.supports_past_covariates else None,
                start=len(train_series) - (30 * 24),  # last 30 days
                forecast_horizon=24,                   # 1-day horizon (24 steps)
                stride=24,                             # 1-day stride
                retrain=True,
                verbose=False
            )
            backtest_mae = mae(train_series, backtest_forecast)
            print(f"    Backtest MAE ({name}): {backtest_mae:.2f}")

        # Store all metrics for final plotting/reporting
        results[name] = {
            "forecast": forecast, 
            "mae": mae_val, 
            "rmse": rmse_val,
            "r2": r2_val
        }

    # --- 6. Benchmark Plots Generation ---
    print("\n--- 6. Generating Final Benchmark Plots ---")
    
    set_option("plotting.use_darts_style", True)
    
    # Plot 1: Full 90-Day Benchmark
    plt.figure(figsize=(18, 7))
    test_series.plot(label='Actual Data', color='green', linewidth=2)
    for name, data in results.items():
        # Display multiple key metrics in the legend
        label_str = f'{name} (MAE: {data["mae"]:.2f}, R²: {data["r2"]:.2f})'
        data["forecast"].plot(label=label_str)
    plt.title('90-Day Benchmark - Predictive Autoscaling')
    plt.legend()
    plt.savefig(os.path.join(outputs_dir, 'darts_native_benchmark_3m_full.png'))
    plt.close()

    # Plot 2: Zoomed View (Last 7 Days)
    plt.figure(figsize=(18, 7))
    zoom_steps = 7 * 24  # 7 days * 24 steps/day = 168 steps
    test_series[-zoom_steps:].plot(label='Actual Data', color='green', linewidth=3)
    for name, data in results.items():
        data["forecast"][-zoom_steps:].plot(label=name, linewidth=2)
    plt.title('Weekly Detail (Zoom on Last 7 Days)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(outputs_dir, 'darts_native_benchmark_3m_zoom.png'))
    plt.close()

    # --- NEW: Save Metrics Summary to CSV ---
    metrics_summary = pd.DataFrame({
        name: {
            "MAE": data["mae"],
            "RMSE": data["rmse"],
            "R-squared": data["r2"]
        } for name, data in results.items()
    }).T
    metrics_summary.to_csv(os.path.join(outputs_dir, 'darts_metrics_summary.csv'))

    print("\n--- Final Metrics Summary ---")
    print(metrics_summary.to_string())
    print(f"\n Native Darts analysis completed! Check the /outputs folder.")

if __name__ == "__main__":
    run_native_darts_tournament()


