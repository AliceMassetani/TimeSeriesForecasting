"""
TSMixer Hyperparameter Search -- Hourly Autoscaler (h=2)
=========================================================
Goal      : Find the best TSMixerModel configuration for a 2-hour-ahead
            forecast, assuming the autoscaler runs every hour.
Library   : 100% Darts-native (gridsearch, backtest, metrics).

Fixed (not searched):
  input_chunk_length  = 168   7-day weekly context window
  output_chunk_length = 8     single-pass output >= max horizon
  forecast_horizon    = 2     target horizon: next 2 hours
  n_epochs            = 50    with EarlyStopping(patience=10, train_loss)
  likelihood          = QuantileRegression([0.1, 0.3, 0.5, 0.7, 0.9])

Searched:
  hidden_size   in {32, 64, 128}       MLP width -- temporal mixing
  ff_size       in {32, 64, 128}       MLP width -- feature mixing
  num_blocks    in {1, 2, 3}           stacked mixer blocks
  dropout       in {0.05, 0.1, 0.2}   regularisation + MC-dropout at inference
  batch_size    in {16, 32, 64}        small batches preferred (user constraint)
  lr            in {1e-3, 5e-4, 1e-4} Adam learning rate

Gridsearch strategy (SPLIT MODE -- fast evaluation):
  We use Split Mode to evaluate 729 combinations quickly:
    - Train ONCE on fit_series (first 80% of the training pool).
    - Evaluate RMSE by predicting the entire validation period in one shot.
  While a rolling window is more operationally correct for a 2-hour horizon,
  split mode cuts execution time by >95% and is standard practice for
  coarsely ranking hyperparameter grids before final selection.

  EarlyStopping monitors val_loss during gridsearch via fit_kwargs.

Outputs (saved to /outputs/):
  tsmixer_hyperparams_split_results.csv        winner row with all metrics
  tsmixer_hyperparams_split_results.txt        human-readable summary report
  tsmixer_hyperparams_split_best_backtest.png  backtest plot on held-out test window
"""

import gc
import logging
import os

import holidays
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from darts import TimeSeries, concatenate
from darts.metrics import mae, mse, rmse, r2_score
from darts.models import TSMixerModel
from darts.utils.likelihood_models import QuantileRegression
from darts.utils.missing_values import fill_missing_values
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

try:
    import torch
except ImportError:
    torch = None

logging.getLogger("cmdstanpy").setLevel(logging.CRITICAL)
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_PATH   = os.path.join(BASE_DIR, "..", "datasets", "Dati_processed.csv")
BASE_OUTPUTS_DIR = os.path.join(BASE_DIR, "..", "outputs")
SCRIPT_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUTPUTS_DIR = os.path.join(BASE_OUTPUTS_DIR, SCRIPT_NAME)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

TARGET           = "InUseCapacity"
FORECAST_HORIZON = 2     # h=2: autoscaler fires every hour, needs 2h lookahead
INPUT_CHUNK_LEN  = 168   # 7 days of hourly context (captures weekly seasonality)
OUTPUT_CHUNK_LEN = 8     # >= FORECAST_HORIZON, avoids multi-step auto-regression

VAL_FRAC    = 0.20   # fraction of training pool withheld for validation
TEST_HOURS  = 72     # held-out test window (never seen during grid search)
N_EPOCHS    = 50     # EarlyStopping(patience=10) fires well before this
EVAL_STRIDE = 6      # evaluate every 6h -> ~361 rolling forecasts over 90-day val

# ---------------------------------------------------------------------------
# Hyperparameter grid
#
# Darts gridsearch() passes every key in `parameters` as a constructor kwarg.
# Fixed settings are single-element lists so they reach the model constructor
# without being varied.
#
# EarlyStopping inside gridsearch must monitor train_loss (not val_loss)
# because gridsearch retrains at each historical_forecasts window and cannot
# keep a persistent val dataloader alive. This still prevents runaway training.
# ---------------------------------------------------------------------------
PARAM_GRID = {
    # -- searched axes --
    "hidden_size":     [32, 64, 128],
    "ff_size":         [32, 64, 128],
    "num_blocks":      [1, 2, 3],
    "dropout":         [0.1, 0.2],
    "batch_size":      [16, 32, 64],
    "optimizer_kwargs": [
        {"lr": 1e-3},
        {"lr": 5e-4},
        {"lr": 1e-4},
    ],
    # -- fixed axes (single-element lists) --
    "input_chunk_length":  [INPUT_CHUNK_LEN],
    "output_chunk_length": [OUTPUT_CHUNK_LEN],
    "n_epochs":            [N_EPOCHS],
    "likelihood":          [QuantileRegression(quantiles=[0.1, 0.3, 0.5, 0.7, 0.9])],
    # EarlyStopping on val_loss
    "pl_trainer_kwargs": [{
        "enable_progress_bar": True,
        "callbacks": [EarlyStopping(
            monitor="val_loss", patience=10, min_delta=1e-4, mode="min"
        )],
    }],
    "random_state": [42],
}

# ---------------------------------------------------------------------------
# 1. Data Loading & Feature Engineering
# ---------------------------------------------------------------------------
print("=" * 65)
print("1. DATA LOADING & FEATURE ENGINEERING")
print("=" * 65)

df = pd.read_csv(DATA_PATH)
df["TimeStamp"] = pd.to_datetime(df["TimeStamp"])
df = df.sort_values("TimeStamp").reset_index(drop=True)

it_holidays = holidays.Italy()
df["hour"]        = df["TimeStamp"].dt.hour
df["day_of_week"] = df["TimeStamp"].dt.dayofweek
df["is_holiday"]  = df["TimeStamp"].apply(lambda x: 1 if x in it_holidays else 0)

calendar_cols = ["hour", "day_of_week", "is_holiday"]

print(f"  Dataset : {len(df)} hourly rows")
print(f"  Period  : {df['TimeStamp'].iloc[0]}  ->  {df['TimeStamp'].iloc[-1]}")
print(f"  Target  : {TARGET}  (min={df[TARGET].min():.0f}, max={df[TARGET].max():.0f})")

# Extend calendar into the future (needed by future_covariates at predict time)
last_ts   = df["TimeStamp"].iloc[-1]
fut_dates = pd.date_range(
    start=last_ts + pd.Timedelta(hours=1),
    periods=OUTPUT_CHUNK_LEN,
    freq="h",
)
fut_df = pd.DataFrame({
    "TimeStamp":   fut_dates,
    "hour":        fut_dates.hour,
    "day_of_week": fut_dates.dayofweek,
    "is_holiday":  [1 if d in it_holidays else 0 for d in fut_dates],
})
df_extended = pd.concat([df, fut_df], ignore_index=True)

# ---------------------------------------------------------------------------
# 2. TimeSeries Creation
# ---------------------------------------------------------------------------
print("\n" + "=" * 65)
print("2. BUILDING DARTS TIMESERIES")
print("=" * 65)

series = fill_missing_values(
    TimeSeries.from_dataframe(df, time_col="TimeStamp", value_cols=TARGET, freq="h")
)
future_cov = fill_missing_values(
    TimeSeries.from_dataframe(
        df_extended, time_col="TimeStamp", value_cols=calendar_cols, freq="h"
    )
)

print(f"  Target TimeSeries  : {len(series)} steps")
print(f"  Future covariates  : {len(future_cov)} steps  (includes {OUTPUT_CHUNK_LEN}h lookahead)")

# ---------------------------------------------------------------------------
# 3. Train / Val / Test Split
#
#   |<---------- train_val_series ----------->|<-- test (72h) -->|
#   |<-- fit_series (80%) -->|<-- val (20%) ->|
#
# gridsearch trains on fit_series and evaluates on val_series.
# The held-out test window is NEVER touched during the search.
# ---------------------------------------------------------------------------
print("\n" + "=" * 65)
print("3. TRAIN / VAL / TEST SPLIT")
print("=" * 65)

test_start_idx   = len(series) - TEST_HOURS
train_val_series = series[:test_start_idx]
test_series      = series[test_start_idx:]

val_len    = int(len(train_val_series) * VAL_FRAC)
val_series = train_val_series[-val_len:]
fit_series = train_val_series[:-val_len]

print(f"  Fit   (train)   : {len(fit_series)} steps  ({len(fit_series)//24} days)")
print(f"  Val             : {len(val_series)} steps  ({val_len//24} days)  -- gridsearch metric + early stopping")
print(f"  Test (held-out) : {len(test_series)} steps  ({TEST_HOURS}h)  -- only for final best model")

# ---------------------------------------------------------------------------
# 4. Gridsearch -- Split Mode (Fast Evaluation)
#
# Each combination is trained on fit_series, then sweeps over val_series in
# one pass to compute RMSE. This acts as an excellent coarse filter to find
# the top configurations in a fraction of the time of a rolling window.
# ---------------------------------------------------------------------------
searched_axes = {k: v for k, v in PARAM_GRID.items() if len(v) > 1}
total_combos  = 1
for v in searched_axes.values():
    total_combos *= len(v)

print("\n" + "=" * 65)
print("4. GRIDSEARCH  (split mode)")
print(f"   Searched axes  : {list(searched_axes.keys())}")
print(f"   Combinations   : {total_combos}")
print(f"   Metric         : RMSE  (split mode on full val window)")
print(f"   n_epochs cap   : {N_EPOCHS}  (EarlyStopping on val_loss, patience=10)")
print("=" * 65)

best_model_untrained, best_params, best_rmse = TSMixerModel.gridsearch(
    parameters=PARAM_GRID,
    series=fit_series,                  # Darts trains on fit_series
    val_series=val_series,              # Split mode: evaluate on val_series
    future_covariates=future_cov,
    # forecast_horizon and stride are omitted in split mode
    metric=rmse,
    reduction=np.mean,
    n_jobs=1,           # sequential -- required for PyTorch/Lightning
    verbose=True,
    fit_kwargs={
        # Passed to fit() so Lightning creates a val dataloader for EarlyStopping
        "val_series": val_series,
    },
)

print("\n  [OK] Gridsearch complete.")
lr_value = best_params.get("optimizer_kwargs", {}).get("lr", "N/A")
print(f"  Best RMSE (split mode) : {best_rmse:.4f}")
print(f"  Best params : {best_params}")

# ---------------------------------------------------------------------------
# 5. Refit best model with val_series for proper EarlyStopping (val_loss)
# ---------------------------------------------------------------------------
print("\n" + "=" * 65)
print("5. REFIT BEST MODEL  (fit_series + val_series, EarlyStopping val_loss)")
print("=" * 65)

# gridsearch returns an UNTRAINED model instance -- must refit before predict.
# Here we use val_series properly so Lightning monitors val_loss.
best_params["pl_trainer_kwargs"]["callbacks"] = [
    EarlyStopping(monitor="val_loss", patience=10, min_delta=1e-4, mode="min")
]
best_model_untrained = TSMixerModel(**best_params)
best_model_untrained.fit(
    series=fit_series,
    val_series=val_series,
    future_covariates=future_cov,
    val_future_covariates=future_cov,
    verbose=True,
)
best_model = best_model_untrained
print("  Refit complete.")

# ---------------------------------------------------------------------------
# 6. Additional metrics -- val window rolling backtest (h=2, stride=1)
# ---------------------------------------------------------------------------
print("\n" + "=" * 65)
print("6. ADDITIONAL METRICS -- BEST MODEL  (val window, stride=1)")
print("=" * 65)

val_metrics = best_model.backtest(
    series=train_val_series,
    future_covariates=future_cov,
    start=val_series.start_time(),
    forecast_horizon=FORECAST_HORIZON,
    stride=1,
    last_points_only=True,
    retrain=False,
    metric=[rmse, mae, mse, r2_score],
    reduction=np.mean,
    verbose=False,
)
rmse_val, mae_val, mse_val, r2_val = val_metrics

print(f"  RMSE : {rmse_val:.4f}")
print(f"  MAE  : {mae_val:.4f}")
print(f"  MSE  : {mse_val:.4f}")
print(f"  R2   : {r2_val:.4f}")

# ---------------------------------------------------------------------------
# 7. Held-out test evaluation (72h window, never seen during search)
# ---------------------------------------------------------------------------
print("\n" + "=" * 65)
print("7. HELD-OUT TEST EVALUATION  (last 72h -- never seen during search)")
print("=" * 65)

hf_list = best_model.historical_forecasts(
    series=series,
    future_covariates=future_cov,
    start=test_series.start_time(),
    forecast_horizon=FORECAST_HORIZON,
    stride=1,
    num_samples=200,
    retrain=False,
    verbose=False,
)

hf_concat   = concatenate(hf_list)
hf_p50      = hf_concat.quantile(0.5)
act_aligned = series.slice_intersect(hf_p50)

test_rmse = rmse(act_aligned, hf_p50)
test_mae  = mae( act_aligned, hf_p50)
test_mse  = mse( act_aligned, hf_p50)
test_r2   = r2_score(act_aligned, hf_p50)

print(f"  RMSE : {test_rmse:.4f}")
print(f"  MAE  : {test_mae:.4f}")
print(f"  MSE  : {test_mse:.4f}")
print(f"  R2   : {test_r2:.4f}")

# ---------------------------------------------------------------------------
# 8. Save outputs
# ---------------------------------------------------------------------------
print("\n" + "=" * 65)
print("8. SAVING RESULTS")
print("=" * 65)

results_row = {
    "hidden_size": best_params.get("hidden_size"),
    "ff_size":     best_params.get("ff_size"),
    "num_blocks":  best_params.get("num_blocks"),
    "dropout":     best_params.get("dropout"),
    "batch_size":  best_params.get("batch_size"),
    "lr":          lr_value,
    "val_RMSE":    round(rmse_val,  4),
    "val_MAE":     round(mae_val,   4),
    "val_MSE":     round(mse_val,   4),
    "val_R2":      round(r2_val,    4),
    "test_RMSE":   round(test_rmse, 4),
    "test_MAE":    round(test_mae,  4),
    "test_MSE":    round(test_mse,  4),
    "test_R2":     round(test_r2,   4),
}
results_df = pd.DataFrame([results_row])
csv_path = os.path.join(OUTPUTS_DIR, "tsmixer_hyperparams_split_results.csv")
results_df.to_csv(csv_path, index=False)
print(f"  Saved: {csv_path}")

txt_path = os.path.join(OUTPUTS_DIR, "tsmixer_hyperparams_split_results.txt")
with open(txt_path, "w", encoding="utf-8") as f:
    f.write("TSMixer Hyperparameter Search -- Results\n")
    f.write("=" * 50 + "\n\n")
    f.write("Fixed settings\n")
    f.write(f"  forecast_horizon    : {FORECAST_HORIZON}h\n")
    f.write(f"  input_chunk_length  : {INPUT_CHUNK_LEN}\n")
    f.write(f"  output_chunk_length : {OUTPUT_CHUNK_LEN}\n")
    f.write(f"  n_epochs (max)      : {N_EPOCHS}  (EarlyStopping patience=10)\n")
    f.write(f"  eval_mode           : SPLIT (fast evaluation on val block)\n")
    f.write(f"  covariate mode      : calendar (hour, day_of_week, is_holiday)\n\n")
    f.write("Searched axes\n")
    for k, v in searched_axes.items():
        f.write(f"  {k:20s}: {v}\n")
    f.write(f"\nTotal combinations   : {total_combos}\n\n")
    f.write("-" * 50 + "\n")
    f.write("BEST CONFIGURATION\n")
    f.write("-" * 50 + "\n")
    skip = {"input_chunk_length", "output_chunk_length", "n_epochs",
            "likelihood", "pl_trainer_kwargs", "random_state"}
    for k, v in best_params.items():
        if k not in skip:
            f.write(f"  {k:20s}: {v}\n")
    f.write(f"\nValidation metrics (split mode over val window)\n")
    f.write(f"  RMSE : {rmse_val:.4f}\n")
    f.write(f"  MAE  : {mae_val:.4f}\n")
    f.write(f"  MSE  : {mse_val:.4f}\n")
    f.write(f"  R2   : {r2_val:.4f}\n")
    f.write(f"\nHeld-out test metrics (last {TEST_HOURS}h, never seen during search)\n")
    f.write(f"  RMSE : {test_rmse:.4f}\n")
    f.write(f"  MAE  : {test_mae:.4f}\n")
    f.write(f"  MSE  : {test_mse:.4f}\n")
    f.write(f"  R2   : {test_r2:.4f}\n")
print(f"  Saved: {txt_path}")

# ---------------------------------------------------------------------------
# 9. Backtest plot -- best model on held-out test window
# ---------------------------------------------------------------------------
print("\n" + "=" * 65)
print("9. GENERATING BACKTEST PLOT")
print("=" * 65)

fig, ax = plt.subplots(figsize=(16, 5))
fig.suptitle(
    f"TSMixer Best Config -- h={FORECAST_HORIZON}h Backtest (Held-Out Test Window)\n"
    f"hidden={best_params.get('hidden_size')}  ff={best_params.get('ff_size')}  "
    f"blocks={best_params.get('num_blocks')}  dropout={best_params.get('dropout')}  "
    f"bs={best_params.get('batch_size')}  lr={lr_value}",
    fontsize=11, fontweight="bold",
)

act_aligned.plot(ax=ax, label="Actual", color="#2ecc71", linewidth=2)
p10 = hf_concat.quantile(0.1)
p90 = hf_concat.quantile(0.9)
hf_p50.plot(
    ax=ax,
    label=f"Forecast P50  RMSE={test_rmse:.2f}  R2={test_r2:.4f}",
    color="#8e44ad",
    linewidth=1.8,
)
ax.fill_between(
    p10.time_index,
    p10.values().flatten(),
    p90.values().flatten(),
    alpha=0.18,
    color="#8e44ad",
    label="P10-P90 band",
)
ax.set_ylabel("InUseCapacity (instances)")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
fig.tight_layout()

plot_path = os.path.join(OUTPUTS_DIR, "tsmixer_hyperparams_split_best_backtest.png")
fig.savefig(plot_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {plot_path}")

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------
del best_model, hf_list, hf_concat, hf_p50
gc.collect()
if torch and torch.cuda.is_available():
    torch.cuda.empty_cache()

print("\n" + "=" * 65)
print("DONE -- all outputs saved to /outputs/")
print("=" * 65)
print("  tsmixer_hyperparams_split_results.csv")
print("  tsmixer_hyperparams_split_results.txt")
print("  tsmixer_hyperparams_split_best_backtest.png")
