import pandas as pd
import torch
import holidays
import os
import matplotlib.pyplot as plt
from chronos import Chronos2Pipeline

def run_inference():
    # 1. Paths configuration
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, '..', 'datasets', 'Dati_processed.csv')

    print("--- 1. Loading Processed Dataset ---")
    df = pd.read_csv(data_path)
    df['TimeStamp'] = pd.to_datetime(df['TimeStamp'])
    
    # CRITICAL: We must add the item_id and the covariates to the historical dataset.
    # Chronos-2 needs to learn from the past data how these variables influence CapacityUtilization.
    df['item_id'] = 'appstream_fleet'
    df['hour'] = df['TimeStamp'].dt.hour
    df['day_of_week'] = df['TimeStamp'].dt.dayofweek
    
    it_holidays = holidays.Italy()
    df['is_holiday'] = df['TimeStamp'].apply(lambda x: int(x in it_holidays))
    
    # We want to forecast the next 24 hours for the autoscaler
    prediction_length = 24

    print("--- 2. Creating Future Covariates (future_df) ---")
    # Chronos-2 natively supports known future covariates. 
    # We generate a dataframe for the next 24 hours with our temporal features.
    last_date = df['TimeStamp'].max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(hours=1), periods=prediction_length, freq='h')
    
    future_df = pd.DataFrame({'TimeStamp': future_dates})
    future_df['item_id'] = 'appstream_fleet'
    
    # Extracting the exact same covariates used in the past data
    future_df['hour'] = future_df['TimeStamp'].dt.hour
    future_df['day_of_week'] = future_df['TimeStamp'].dt.dayofweek
    future_df['is_holiday'] = future_df['TimeStamp'].apply(lambda x: int(x in it_holidays))

    print("--- 3. Initializing Chronos-2 Model ---")
    # Loading the chosen model
    pipeline = Chronos2Pipeline.from_pretrained(
        "amazon/chronos-2",
        device_map="auto",
        dtype=torch.bfloat16,
    )

    print("--- 4. Running Zero-Shot Inference ---")
    # The predict_df API automatically maps columns present in both df and future_df as known covariates.
    # It uses 'item_id' to understand it's processing a single time series.
    forecast_df = pipeline.predict_df(
        df=df,
        future_df=future_df,
        id_column="item_id",
        timestamp_column="TimeStamp",
        target="CapacityUtilization",
        prediction_length=prediction_length,
        quantile_levels=[0.5, 0.9] 
    )

    print("\n✅ Predictions generated successfully!")
    print("Preview of the P50 and P90 quantiles for the next hours:")
    print(forecast_df[['TimeStamp', '0.5', '0.9']].head())

    print("--- 5. Plotting the Forecast ---")
    
    # We select only the last 72 hours of historical data to keep the chart readable
    # Otherwise, 15 months of data would compress the 24-hour forecast into a tiny invisible line
    history_plot = df.tail(72)

    plt.figure(figsize=(12, 6))
    
    # 1. Plot the historical actual usage (Black line)
    plt.plot(history_plot['TimeStamp'], history_plot['CapacityUtilization'], 
             label='Historical Capacity (Last 72h)', color='black', linewidth=1.5)
    
    # 2. Plot the P50 median forecast (Blue line)
    plt.plot(forecast_df['TimeStamp'], forecast_df['0.5'], 
             label='Forecast P50 (Expected)', color='blue', linewidth=2)
    
    # 3. Plot the P90 upper bound forecast (Red dashed line)
    plt.plot(forecast_df['TimeStamp'], forecast_df['0.9'], 
             label='Forecast P90 (Safety Margin)', color='red', linestyle='dashed', linewidth=1.5)
    
    # 4. Fill the area between P50 and P90 to visually represent the uncertainty/buffer zone
    plt.fill_between(forecast_df['TimeStamp'], forecast_df['0.5'], forecast_df['0.9'], 
                     color='blue', alpha=0.1, label='Autoscaler Buffer Zone')

    # Formatting the chart
    plt.title('AWS AppStream Capacity Utilization: Chronos-2 Forecast')
    plt.xlabel('Time')
    plt.ylabel('Capacity Utilization')
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save the plot as an image file in your workspace
    outputs_dir = os.path.join(base_dir, '..', 'outputs')
    os.makedirs(outputs_dir, exist_ok=True)
    
    plot_path = os.path.join(outputs_dir, 'forecast_visualization.png')
    plt.savefig(plot_path)
    print(f"✅ Chart saved successfully at: {plot_path}")

if __name__ == "__main__":
    run_inference()