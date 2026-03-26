import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error

def run_long_term_prophet():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, '..', 'datasets', 'Dati_processed.csv')

    print("--- 1. Loading and Resampling Data (3H Max) ---")
    df = pd.read_csv(data_path)
    df['TimeStamp'] = pd.to_datetime(df['TimeStamp'])
    
    # Resample every 3 hours taking the maximum peak for Capacity Planning
    # Use ffill() to maintain a continuous frequency without missing gaps
    df_resampled = df.set_index('TimeStamp').resample('3h').max()
    df_resampled = df_resampled.ffill().reset_index()

    # Prophet requires columns to be specifically named 'ds' and 'y'
    df_resampled = df_resampled.rename(columns={'TimeStamp': 'ds', 'CapacityUtilization': 'y'})

    print("--- 2. Chronological Split (Last 90 Days) ---")
    # 90 days * 8 records per day = 720 steps
    prediction_length = 720 
    train_df = df_resampled.iloc[:-prediction_length].copy()
    test_df = df_resampled.iloc[-prediction_length:].copy()

    train_df['cap'] = 100
    train_df['floor'] = 0

    print("--- 3. Running Prophet Model ---")
    # interval_width=0.80 sets the upper bound (yhat_upper) to the 90th percentile (P90)
    prophet_model = Prophet(
        growth='logistic',
        interval_width=0.80, 
        yearly_seasonality=True, 
        daily_seasonality=True, 
        weekly_seasonality=True
    )
    # Add Italian holidays to help the model handle special days
    prophet_model.add_country_holidays(country_name='IT')
    
    # Train the model on historical context
    prophet_model.fit(train_df)
    
    print("--- 4. Running Inference (Long Term) ---")
    future = test_df[['ds']].copy()

    future['cap'] = 100
    future['floor'] = 0
    
    prophet_forecast = prophet_model.predict(future)
    
    # Extract P50 (Median) and P90 (Upper bound of 80% interval)
    prophet_p50 = prophet_forecast['yhat'].values
    prophet_p90 = prophet_forecast['yhat_upper'].values
    y_true = test_df['y'].values

    print("--- 5. Evaluating Performance (3 Months) ---")
    prophet_mae = mean_absolute_error(y_true, prophet_p50)
    prophet_rmse = np.sqrt(mean_squared_error(y_true, prophet_p50))
    
    print(f"\n✅ PROPHET RESULTS (90 Days / 3H Resampling):")
    print(f"MAE: {prophet_mae:.2f}")
    print(f"RMSE: {prophet_rmse:.2f}")

    # Prepare outputs directory
    outputs_dir = os.path.join(base_dir, '..', 'outputs')
    os.makedirs(outputs_dir, exist_ok=True)

    print("--- 6. Saving Visual Evaluation (Full 90 Days) ---")
    plt.figure(figsize=(16, 8))
    
    # Visual context: last 30 days of history
    context_days = 30 * 8
    plt.plot(train_df['ds'].tail(context_days), train_df['y'].tail(context_days), label='History (Last 30 days)', color='black', linewidth=1)
    
    # Actual Data (Green)
    plt.plot(test_df['ds'], test_df['y'], label='Actual Data (Reality)', color='green', linewidth=2)
    
    # Prophet P50 (Orange) and P90 (Red)
    plt.plot(test_df['ds'], prophet_p50, label=f'Prophet P50 (MAE: {prophet_mae:.2f})', color='orange', linewidth=2)
    plt.plot(test_df['ds'], prophet_p90, label='Safety Threshold P90', color='red', linestyle=':', linewidth=1.5)
    plt.fill_between(test_df['ds'], prophet_p50, prophet_p90, color='red', alpha=0.1)

    plt.title('Capacity Planning: Prophet 90-Day Forecast (3H Peaks)')
    plt.xlabel('Time')
    plt.ylabel('Max Capacity Utilization (3H)')
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    full_plot_path = os.path.join(outputs_dir, 'long_term_prophet_90days.png')
    plt.savefig(full_plot_path)
    print(f"✅ Prophet full chart saved successfully at: {full_plot_path}")

    print("--- 7. Saving Visual Evaluation (30-Day ZOOM) ---")
    plt.figure(figsize=(16, 8))
    
    # Zoom on the last 30 days of the test set (240 steps)
    zoom_steps = 30 * 8
    test_zoom = test_df.tail(zoom_steps)
    prophet_forecast_zoom = prophet_forecast.tail(zoom_steps)
    
    # Actual Data (Green)
    plt.plot(test_zoom['ds'], test_zoom['y'], label='Actual Data (Reality)', color='green', linewidth=2, marker='o', markersize=3)
    
    # Prophet P50 (Orange) and P90 (Red)
    plt.plot(test_zoom['ds'], prophet_forecast_zoom['yhat'], label='Prophet P50 (Median)', color='orange', linewidth=2.5)
    plt.plot(test_zoom['ds'], prophet_forecast_zoom['yhat_upper'], label='Safety Threshold P90', color='red', linestyle=':', linewidth=1.5)
    plt.fill_between(test_zoom['ds'], prophet_forecast_zoom['yhat'], prophet_forecast_zoom['yhat_upper'], color='red', alpha=0.1)

    plt.title('Capacity Planning: Prophet Forecast - ZOOM on Last 30 Days (3H Peaks)')
    plt.xlabel('Time')
    plt.ylabel('Max Capacity Utilization (3H)')
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    zoom_plot_path = os.path.join(outputs_dir, 'long_term_prophet_90days_ZOOM.png')
    plt.savefig(zoom_plot_path)
    print(f"✅ Prophet ZOOM chart saved successfully at: {zoom_plot_path}")

if __name__ == "__main__":
    run_long_term_prophet()