import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error

def run_prophet_backtesting():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, '..', 'datasets', 'Dati_processed.csv')

    print("--- 1. Loading and Formatting Data for Prophet ---")
    df = pd.read_csv(data_path)
    df['TimeStamp'] = pd.to_datetime(df['TimeStamp'])
    df = df.sort_values('TimeStamp').reset_index(drop=True)

    # Prophet requires strict column names: 'ds' (datestamp) and 'y' (target)
    prophet_df = pd.DataFrame({
        'ds': df['TimeStamp'],
        'y': df['CapacityUtilization']
    })

    print("--- 2. Chronological Split (Last 1000 Hours) ---")
    prediction_length = 1000 
    train_df = prophet_df.iloc[:-prediction_length].copy()
    test_df = prophet_df.iloc[-prediction_length:].copy()

    print("--- 3. Training Prophet Model ---")
    # interval_width=0.80 means yhat_upper will represent the 90th percentile (P90)
    model = Prophet(interval_width=0.80, daily_seasonality=True, weekly_seasonality=True)
    
    # Adding Italian holidays automatically
    model.add_country_holidays(country_name='IT')
    
    # Fit the model on historical data
    model.fit(train_df)

    print("--- 4. Running Inference ---")
    # We create a dataframe for the future dates to predict
    future = test_df[['ds']].copy()
    forecast = model.predict(future)

    print("--- 5. Evaluating Performance ---")
    y_pred = forecast['yhat'].values
    y_true = test_df['y'].values

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    print("\n✅ PROPHET RESULTS:")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

    print("--- 6. Saving Visual Evaluation (7-Day Zoom) ---")
    plt.figure(figsize=(14, 7))
    
    # Zooming into the last 168 hours (7 days) for the chart
    zoom_hours = 168
    test_zoom = test_df.tail(zoom_hours)
    forecast_zoom = forecast.tail(zoom_hours)

    # Reality vs Forecast
    plt.plot(test_zoom['ds'], test_zoom['y'], label='Actual Test Data (Reality)', color='green', linewidth=1.5, marker='o', markersize=2)
    plt.plot(forecast_zoom['ds'], forecast_zoom['yhat'], label='Forecast P50 (Prophet)', color='blue', linewidth=2)
    
    # Safety Threshold P90 (yhat_upper with 80% interval)
    plt.plot(forecast_zoom['ds'], forecast_zoom['yhat_upper'], label='Safety Threshold P90', color='red', linestyle='--', linewidth=1.5)
    plt.fill_between(forecast_zoom['ds'], forecast_zoom['yhat'], forecast_zoom['yhat_upper'], color='red', alpha=0.1)

    plt.title('Prophet Evaluation: Zoom on Last 7 Days (168h)')
    plt.xlabel('Time')
    plt.ylabel('Capacity Utilization')
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    outputs_dir = os.path.join(base_dir, '..', 'outputs')
    os.makedirs(outputs_dir, exist_ok=True)
    plt.savefig(os.path.join(outputs_dir, 'prophet_backtesting_7days_zoom.png'))
    print("✅ Prophet chart saved successfully.")

if __name__ == "__main__":
    run_prophet_backtesting()