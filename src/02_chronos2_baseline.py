import pandas as pd
import torch
import os
import numpy as np
from chronos import Chronos2Pipeline

print("--- Starting Chronos-2 Forecasting ---")

# 1. Data path resolution
path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'datasets', 'Dati.csv'))

print(f"Loading data from: {path}")
df = pd.read_csv(path, skiprows=4)

# 2. Context preparation
context_length = 168
prediction_length = 24

print(f"Extracting the last {context_length} steps of CapacityUtilization...")

# Ensure the column is numeric and remove any potential NaN values
df['CapacityUtilization'] = pd.to_numeric(df['CapacityUtilization'], errors='coerce')
context_data = df['CapacityUtilization'].tail(context_length).dropna().to_numpy(dtype=float)

# Chronos-2 requires a 3D input tensor: (n_series, n_variates, history_length)
# .unsqueeze(0) adds n_series, .unsqueeze(1) adds n_variates
context_tensor = torch.tensor(context_data, dtype=torch.float32).unsqueeze(0).unsqueeze(1)

# 3. Load the Amazon Chronos-2 model
print("Loading Amazon Chronos-2 model...")
pipeline = Chronos2Pipeline.from_pretrained(
    "amazon/chronos-2",
    device_map="cpu",         # Using CPU for compatibility
    dtype=torch.float32, # Using float32 to avoid BFloat16 CPU errors
)

# 4. Generate forecasts
print(f"Generating scenarios for the next {prediction_length} steps...")
# Removed num_samples as it's not a direct argument in this pipeline version
forecast = pipeline.predict(
    context_tensor,
    prediction_length=prediction_length
)

# 5. Result analysis for AWS AppStream scaling
print("\n--- RESULTS FOR AUTOSCALER ---")
# Extract raw forecast samples [batch, samples, prediction_length]
forecast_samples = forecast[0].numpy() 

# Calculate quantiles for the next immediate time step
p50_next_step = np.quantile(forecast_samples[:, 0], 0.50)
p90_next_step = np.quantile(forecast_samples[:, 0], 0.90)

print(f"Current Value (last context): {context_data[-1]:.2f}")
print(f"Forecast P50 (Next Hour): {p50_next_step:.2f}")
print(f"Forecast P90 (Next Hour): {p90_next_step:.2f}")