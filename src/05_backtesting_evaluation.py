import pandas as pd
import torch
import holidays
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from chronos import Chronos2Pipeline

def run_backtesting():
    # 1. Paths configuration
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, '..', 'datasets', 'Dati_processed.csv')

    print("--- 1. Loading Processed Dataset ---")
    df = pd.read_csv(data_path)
    df['TimeStamp'] = pd.to_datetime(df['TimeStamp'])
    
    # Sort by time to strictly enforce chronological order (Crucial for Time Series)
    df = df.sort_values('TimeStamp').reset_index(drop=True)

    # Adding item_id and covariates
    df['item_id'] = 'appstream_fleet'
    df['hour'] = df['TimeStamp'].dt.hour
    df['day_of_week'] = df['TimeStamp'].dt.dayofweek
    
    it_holidays = holidays.Italy()
    df['is_holiday'] = df['TimeStamp'].apply(lambda x: int(x in it_holidays))

    print("--- 2. Performing Chronological Split ---")
    # We use exactly the last 1000 hours (~41 days) for testing.
    prediction_length = 1000 
    
    # Train = Everything EXCEPT the last 1000 hours
    # Test = Exactly the last 1000 hours
    train_df = df.iloc[:-prediction_length].copy()
    test_df = df.iloc[-prediction_length:].copy()
    
    print(f"Total historical rows: {len(df)}")
    print(f"Training Set (Context): {len(train_df)} rows")
    print(f"Testing Set (Prediction Horizon): {len(test_df)} rows")

    print("--- 3. Creating Future Covariates for the Test Set ---")
    # We feed the model the timestamps and covariates of the test set, 
    # but NOT the actual CapacityUtilization (it has to guess that!)
    future_df = test_df[['TimeStamp', 'item_id', 'hour', 'day_of_week', 'is_holiday']].copy()

    print("--- 4. Initializing Chronos-2 Model ---")
    pipeline = Chronos2Pipeline.from_pretrained(
        "amazon/chronos-2",
        device_map="auto",
        dtype=torch.bfloat16,
    )

    print("--- 5. Running Inference on the Test Set ---")
    # Warning: Predicting a long 20% horizon might take a couple of minutes to process
    forecast_df = pipeline.predict_df(
        df=train_df,
        future_df=future_df,
        id_column="item_id",
        timestamp_column="TimeStamp",
        target="CapacityUtilization",
        prediction_length=prediction_length,
        quantile_levels=[0.5, 0.9] 
    )

    print("--- 6. Evaluating Performance Metrics ---")
    # We compare the median forecast (P50) against the actual values hidden in the test set
    y_pred = forecast_df['0.5'].values
    y_true = test_df['CapacityUtilization'].values

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    print("\n✅ BACKTESTING RESULTS:")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

    print("--- 7. Saving Visual Evaluation (Zoom: Last 7 Days) ---")
    plt.figure(figsize=(14, 7))
    
    # We zoom into the last 168 hours (7 days) to better visualize daily patterns
    zoom_hours = 168
    test_df_zoom = test_df.tail(zoom_hours)
    forecast_df_zoom = forecast_df.tail(zoom_hours)
    
    # Plotting the actual data (Reality)
    plt.plot(test_df_zoom['TimeStamp'], test_df_zoom['CapacityUtilization'], 
             label='Actual Test Data (Reality)', color='green', linewidth=1.5, marker='o', markersize=2)

    # 1. Median forecast line (P50)
    plt.plot(forecast_df_zoom['TimeStamp'], forecast_df_zoom['0.5'], 
             label='Forecast P50 (Expected)', color='blue', linewidth=2)
    
    # 2. Distinct P90 line (Safety Threshold for the Autoscaler)
    plt.plot(forecast_df_zoom['TimeStamp'], forecast_df_zoom['0.9'], 
             label='Safety Threshold P90', color='red', linestyle='--', linewidth=1.5)
    
    # 3. Uncertainty area between P50 and P90 (Visual buffer)
    plt.fill_between(forecast_df_zoom['TimeStamp'], forecast_df_zoom['0.5'], 
                     forecast_df_zoom['0.9'], color='red', alpha=0.1)

    plt.title('Backtesting Evaluation: Zoom on Last 7 Days (168h)')
    plt.xlabel('Time')
    plt.ylabel('Capacity Utilization')
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    outputs_dir = os.path.join(base_dir, '..', 'outputs')
    os.makedirs(outputs_dir, exist_ok=True)
    
    # Use a specific name for the zoomed chart
    plot_path = os.path.join(outputs_dir, 'chronos_backtesting_7days_zoom.png')
    plt.savefig(plot_path)
    print(f"✅ Evaluation chart saved successfully at: {plot_path}")

if __name__ == "__main__":
    run_backtesting()