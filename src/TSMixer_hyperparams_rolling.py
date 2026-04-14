"""
TSMixer Hyperparameter Search -- Hourly Autoscaler (h=2)
=========================================================
Goal      : Find the best TSMixerModel configuration for a 2-hour-ahead
            forecast using a single rolling-window evaluation over the
            validation set.

  For every combination in PARAM_GRID:
    1. Train on fit_series (with EarlyStopping on val_loss).
    2. Run `.backtest(forecast_horizon=2, stride=1, retrain=False)` over
       val_series -- exactly how the autoscaler will operate in production.
    3. Select the rolling-RMSE champion as the final best model.

  FINAL -- Refit on full train_val:
    Retrain the champion on the full train_val_series.
    Evaluate on the held-out test window (last 72h).

Library   : 100% Darts-native.

Outputs (saved to /outputs/):
  tsmixer_hyperparams_rolling_results.csv        winner row with all metrics
  tsmixer_hyperparams_rolling_results.txt        human-readable summary report
  tsmixer_hyperparams_rolling_best_backtest.png  backtest plot on held-out test
"""

import gc
import itertools
import logging
import os

import holidays
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

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
OUTPUTS_DIR = os.path.join(BASE_DIR, "..", "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

TARGET           = "InUseCapacity"
FORECAST_HORIZON = 2     # h=2: autoscaler fires every hour, needs 2h lookahead
INPUT_CHUNK_LEN  = 168   # 7 days of hourly context (captures weekly seasonality)
OUTPUT_CHUNK_LEN = 8     # >= FORECAST_HORIZON, avoids multi-step auto-regression

VAL_FRAC   = 0.20   # fraction of training pool withheld for validation
TEST_HOURS = 72     # held-out test window (never seen during search)
N_EPOCHS   = 50     # EarlyStopping(patience=10) fires well before this

# ---------------------------------------------------------------------------
# Hyperparameter grid
# ---------------------------------------------------------------------------
PARAM_GRID = {
    # -- searched axes --
    "hidden_size":     [16, 32, 64],
    "ff_size":         [16, 32, 64],
    "num_blocks":      [1, 2, 3],
    "dropout":         [0.1, 0.2],
    "batch_size":      [32, 64, 128],
    "optimizer_kwargs": [
        {"lr": 1e-3},
        {"lr": 5e-4},
        {"lr": 1e-4},
    ],
}

# Fixed params applied to every model (not part of the grid search axes)
FIXED_PARAMS = {
    "input_chunk_length":  INPUT_CHUNK_LEN,
    "output_chunk_length": OUTPUT_CHUNK_LEN,
    "n_epochs":            N_EPOCHS,
    "likelihood":          QuantileRegression(quantiles=[0.1, 0.3, 0.5, 0.7, 0.9]),
    "random_state":        42,
    # (Empty pl_trainer_kwargs base; it gets freshly instantiated in the loop)
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
print(f"  Val             : {len(val_series)} steps  ({val_len//24} days)")
print(f"  Test (held-out) : {len(test_series)} steps  ({TEST_HOURS}h)")

# ---------------------------------------------------------------------------
# 4. Rolling Window Hyperparameter Search
#    Train every combination on fit_series (EarlyStopping on val_loss).
#    Evaluate with a strict rolling backtest (forecast_horizon=2, stride=1)
#    over the val window -- exactly how the autoscaler operates in production.
#    Pick the combination with the lowest rolling RMSE as the champion.
# ---------------------------------------------------------------------------
keys   = list(PARAM_GRID.keys())
combos = list(itertools.product(*[PARAM_GRID[k] for k in keys]))
total  = len(combos)

print("\n" + "=" * 65)
print("4. ROLLING WINDOW HYPERPARAMETER SEARCH")
print(f"   Searched axes  : {keys}")
print(f"   Combinations   : {total}")
print(f"   Metric         : RMSE  (rolling backtest, forecast_horizon={FORECAST_HORIZON}, stride=1)")
print("=" * 65)

best_params    = None
best_roll_rmse = float("inf")
best_roll_mae  = float("inf")
best_roll_mse  = float("inf")
best_roll_r2   = float("inf")


for i, values in enumerate(tqdm(combos, desc="RollingSearch", unit="combo")):
    params      = dict(zip(keys, values))
    full_params = {
        **params,
        **FIXED_PARAMS,
        "pl_trainer_kwargs": {
            "enable_progress_bar": False,
            "callbacks": [EarlyStopping(
                monitor="val_loss", patience=10, min_delta=1e-4, mode="min"
            )],
        },
    }

    roll_rmse = roll_mae = roll_mse = roll_r2 = float("inf")

    try:
        model = TSMixerModel(**full_params)
        model.fit(
            series=fit_series,
            val_series=val_series,
            future_covariates=future_cov,
            val_future_covariates=future_cov,
            verbose=False,
        )

        # Rolling window backtest -- simulates autoscaler at h=2, every hour
        rolling_metrics = model.backtest(
            series=train_val_series,
            future_covariates=future_cov,
            start=val_series.start_time(),
            forecast_horizon=FORECAST_HORIZON,
            stride=1,
            last_points_only=True,
            retrain=False,
            num_samples=1,                  # deterministic for speed
            metric=[rmse, mae, mse, r2_score],
            reduction=np.mean,
            verbose=False,
        )
        roll_rmse, roll_mae, roll_mse, roll_r2 = rolling_metrics

    except Exception:
        pass  # bad combo; scores stay inf

    if roll_rmse < best_roll_rmse:
        best_roll_rmse = roll_rmse
        best_roll_mae  = roll_mae
        best_roll_mse  = roll_mse
        best_roll_r2   = roll_r2
        best_params    = params

    # Free memory immediately -- no array to accumulate
    try:
        del model
    except NameError:
        pass
    gc.collect()
    if torch and torch.cuda.is_available():
        torch.cuda.empty_cache()


# -----------------------------------------------------------------------------

lr_value = best_params.get("optimizer_kwargs", {}).get("lr", "N/A")

print(f"\n  [OK] Search complete. Champion (rolling RMSE={best_roll_rmse:.4f}):")
print(
    f"    hidden={best_params['hidden_size']}  ff={best_params['ff_size']}  "
    f"blocks={best_params['num_blocks']}  dropout={best_params['dropout']}  "
    f"bs={best_params['batch_size']}  lr={lr_value}"
)

# ---------------------------------------------------------------------------
# 5. Refit champion on full train_val (fit_series + val_series)
# ---------------------------------------------------------------------------
print("\n" + "=" * 65)
print("5. REFIT CHAMPION ON FULL TRAIN+VAL")
print("=" * 65)

champion_params_full = {
    **best_params,
    **FIXED_PARAMS,
    "pl_trainer_kwargs": {
        "enable_progress_bar": True,
        "callbacks": [EarlyStopping(
            monitor="train_loss", patience=10, min_delta=1e-4, mode="min"
        )],
    },
}
final_model = TSMixerModel(**champion_params_full)
final_model.fit(
    series=train_val_series,
    future_covariates=future_cov,
    verbose=True,
)
print("  Refit complete.")

# ---------------------------------------------------------------------------
# 6. Held-out test evaluation (72h, never seen during search)
# ---------------------------------------------------------------------------
print("\n" + "=" * 65)
print("6. HELD-OUT TEST EVALUATION  (last 72h)")
print("=" * 65)

hf_list = final_model.historical_forecasts(
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
# 7. Save outputs
# ---------------------------------------------------------------------------
print("\n" + "=" * 65)
print("7. SAVING RESULTS")
print("=" * 65)

results_row = {
    "hidden_size": best_params.get("hidden_size"),
    "ff_size":     best_params.get("ff_size"),
    "num_blocks":  best_params.get("num_blocks"),
    "dropout":     best_params.get("dropout"),
    "batch_size":  best_params.get("batch_size"),
    "lr":          lr_value,
    "roll_RMSE":   round(best_roll_rmse, 4),
    "roll_MAE":    round(best_roll_mae,  4),
    "roll_MSE":    round(best_roll_mse,  4),
    "roll_R2":     round(best_roll_r2,   4),
    "test_RMSE":   round(test_rmse, 4),
    "test_MAE":    round(test_mae,  4),
    "test_MSE":    round(test_mse,  4),
    "test_R2":     round(test_r2,   4),
}
results_df = pd.DataFrame([results_row])
csv_path = os.path.join(OUTPUTS_DIR, "tsmixer_hyperparams_rolling_results.csv")
results_df.to_csv(csv_path, index=False)
print(f"  Saved: {csv_path}")

txt_path = os.path.join(OUTPUTS_DIR, "tsmixer_hyperparams_rolling_results.txt")
with open(txt_path, "w", encoding="utf-8") as f:
    f.write("TSMixer Hyperparameter Search -- Rolling Window\n")
    f.write("=" * 55 + "\n\n")
    f.write("Fixed settings\n")
    f.write(f"  forecast_horizon    : {FORECAST_HORIZON}h\n")
    f.write(f"  input_chunk_length  : {INPUT_CHUNK_LEN}\n")
    f.write(f"  output_chunk_length : {OUTPUT_CHUNK_LEN}\n")
    f.write(f"  n_epochs (max)      : {N_EPOCHS}  (EarlyStopping patience=10)\n")
    f.write(f"  covariate mode      : calendar (hour, day_of_week, is_holiday)\n\n")
    f.write("Searched axes\n")
    for k, v in PARAM_GRID.items():
        f.write(f"  {k:20s}: {v}\n")
    f.write(f"\nTotal combinations   : {total}\n\n")
    f.write("-" * 55 + "\n")
    f.write("CHAMPION CONFIGURATION\n")
    f.write("-" * 55 + "\n")
    for k, v in best_params.items():
        f.write(f"  {k:20s}: {v}\n")
    f.write(f"\nRolling metrics (val window, h={FORECAST_HORIZON}, stride=1)\n")
    f.write(f"  RMSE : {best_roll_rmse:.4f}\n")
    f.write(f"  MAE  : {best_roll_mae:.4f}\n")
    f.write(f"  MSE  : {best_roll_mse:.4f}\n")
    f.write(f"  R2   : {best_roll_r2:.4f}\n")
    f.write(f"\nHeld-out test metrics (last {TEST_HOURS}h)\n")
    f.write(f"  RMSE : {test_rmse:.4f}\n")
    f.write(f"  MAE  : {test_mae:.4f}\n")
    f.write(f"  MSE  : {test_mse:.4f}\n")
    f.write(f"  R2   : {test_r2:.4f}\n")
print(f"  Saved: {txt_path}")

# ---------------------------------------------------------------------------
# 8. Backtest plot -- champion on held-out test window
# ---------------------------------------------------------------------------
print("\n" + "=" * 65)
print("8. GENERATING BACKTEST PLOT")
print("=" * 65)

fig, ax = plt.subplots(figsize=(16, 5))
fig.suptitle(
    f"TSMixer Champion (Rolling Window Search) -- h={FORECAST_HORIZON}h Backtest\n"
    f"hidden={best_params.get('hidden_size')}  ff={best_params.get('ff_size')}  "
    f"blocks={best_params.get('num_blocks')}  dropout={best_params.get('dropout')}  "
    f"bs={best_params.get('batch_size')}  lr={lr_value}  "
    f"rolling_RMSE={best_roll_rmse:.4f}",
    fontsize=11, fontweight="bold",
)

act_aligned.plot(ax=ax, label="Actual", color="#2ecc71", linewidth=2)
p10 = hf_concat.quantile(0.1)
p90 = hf_concat.quantile(0.9)
hf_p50.plot(
    ax=ax,
    label=f"Forecast P50  test_RMSE={test_rmse:.2f}  R2={test_r2:.4f}",
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

plot_path = os.path.join(OUTPUTS_DIR, "tsmixer_hyperparams_rolling_best_backtest.png")
fig.savefig(plot_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {plot_path}")

# ---------------------------------------------------------------------------
# 9. Comparison Plots (TSMixer Hourly vs TimesFM vs Rolling)
# ---------------------------------------------------------------------------
print("\n" + "=" * 65)
print("9. COMPARISON PLOT: TSMixer Hourly vs TimesFM")
print("=" * 65)

# Load existing csv backtest predictions and convert to TimeSeries
tsm_hourly_df = pd.read_csv(os.path.join(OUTPUTS_DIR, "tsmixer_backtest_p50_h2.csv"))
tsm_ds = TimeSeries.from_dataframe(tsm_hourly_df, time_col="TimeStamp", value_cols="InUseCapacity_P50")

tfm_hourly_df = pd.read_csv(os.path.join(OUTPUTS_DIR, "timesfm2p5_backtest_p50_h2.csv"))
tfm_ds = TimeSeries.from_dataframe(tfm_hourly_df, time_col="TimeStamp", value_cols="InUseCapacity_P50")

fig, ax = plt.subplots(figsize=(16, 5))
fig.suptitle("Comparison: Rolling Gridsearch vs Baseline TSMixer vs TimesFM (h=2)", fontsize=12, fontweight="bold")

act_aligned.plot(ax=ax, label="Actual Capacity", color="black", linewidth=2.5, linestyle="dashed")
tsm_ds.plot(ax=ax, label="TSMixer Hourly Baseline", color="#3498db", linewidth=1.5, alpha=0.8)
tfm_ds.plot(ax=ax, label="TimesFM 2.5", color="#e74c3c", linewidth=1.5, alpha=0.8)
hf_p50.plot(ax=ax, label=f"TSMixer Rolling (RMSE={test_rmse:.2f})", color="#2ecc71", linewidth=2.0)

ax.set_ylabel("InUseCapacity")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
fig.tight_layout()

plot9_path = os.path.join(OUTPUTS_DIR, "plot_comparison_tsmixer_rolling_vs_baselines.png")
fig.savefig(plot9_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {plot9_path}")

# ---------------------------------------------------------------------------
# 10. Comparison Plot (Rolling vs Split Search)
# ---------------------------------------------------------------------------
print("\n" + "=" * 65)
print("10. COMPARISON PLOT: Rolling vs Split Search")
print("=" * 65)

# Best params for Split Model (haded to load from file previously)
split_params = {
    'hidden_size': 32, 
    'ff_size': 32, 
    'num_blocks': 1, 
    'dropout': 0.1, 
    'batch_size': 16, 
    'optimizer_kwargs': {'lr': 0.0005}
}
print("  Training mocked Split champion...")
split_params_full = {
    **split_params,
    **FIXED_PARAMS,
    "pl_trainer_kwargs": {"enable_progress_bar": False},
}
split_model = TSMixerModel(**split_params_full)
split_model.fit(series=train_val_series, future_covariates=future_cov, verbose=False)

split_hf_list = split_model.historical_forecasts(
    series=series,
    future_covariates=future_cov,
    start=test_series.start_time(),
    forecast_horizon=FORECAST_HORIZON,
    stride=1,
    num_samples=200,
    retrain=False,
    verbose=False,
)
split_hf_p50 = concatenate(split_hf_list).quantile(0.5)
split_test_rmse = rmse(act_aligned, split_hf_p50)

fig, ax = plt.subplots(figsize=(16, 5))
fig.suptitle("Comparison: Rolling Search vs Split Search (h=2 Test)", fontsize=12, fontweight="bold")
act_aligned.plot(ax=ax, label="Actual Capacity", color="black", linewidth=2.5, linestyle="dashed")
hf_p50.plot(ax=ax, label=f"Rolling Search (Test RMSE={test_rmse:.2f})", color="#2ecc71", linewidth=2.0)
split_hf_p50.plot(ax=ax, label=f"Split Search (Test RMSE={split_test_rmse:.2f})", color="#e67e22", linewidth=2.0)

ax.set_ylabel("InUseCapacity")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
fig.tight_layout()

plot10_path = os.path.join(OUTPUTS_DIR, "plot_comparison_tsmixer_rolling_vs_split.png")
fig.savefig(plot10_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {plot10_path}")

# ---------------------------------------------------------------------------
# 11. EXTRA: Bar Chart Comparison (Metrics vs Baselines)
# ---------------------------------------------------------------------------
print("\n" + "=" * 65)
print("11. EXTRA: BAR CHART METRICS COMPARISON (vs Baselines)")
print("=" * 65)

# Fast-evaluate TSMixer Target-Only Baseline using ONLY previous results (NO re-training!)
print("  Extracting Baseline TSMixer (Target-Only) metrics from previously saved CSV...")
tsm_target_df = pd.read_csv(os.path.join(OUTPUTS_DIR, "tsmixer_backtest_p50_h2.csv"))
p50_preds = tsm_target_df["InUseCapacity_P50"]
actuals   = tsm_target_df["InUseCapacity_Actual"]

base_tgt_rmse = ((p50_preds - actuals)**2).mean()**0.5
base_tgt_mae  = (p50_preds - actuals).abs().mean()
base_tgt_mse  = ((p50_preds - actuals)**2).mean()
# R2 formula: 1 - (SS_res / SS_tot)
ss_res = ((p50_preds - actuals)**2).sum()
ss_tot = ((actuals - actuals.mean())**2).sum()
base_tgt_r2 = 1 - (ss_res / ss_tot)

print(f"    -> Extracted Target-Only Baseline: RMSE={base_tgt_rmse:.4f}, R2={base_tgt_r2:.4f}")

print("  Extracting Baseline TSMixer (Calendar Covariates) metrics from previously saved CSV...")
try:
    tsm_cal_df = pd.read_csv(os.path.join(OUTPUTS_DIR, "tsmixer_calendar_backtest_p50_h2.csv"))
    p50_cal = tsm_cal_df["InUseCapacity_P50"]
    act_cal = tsm_cal_df["InUseCapacity_Actual"]
    base_cov_rmse = ((p50_cal - act_cal)**2).mean()**0.5
    base_cov_mae  = (p50_cal - act_cal).abs().mean()
    base_cov_mse  = ((p50_cal - act_cal)**2).mean()
    base_cov_r2   = 1 - (((p50_cal - act_cal)**2).sum() / ((act_cal - act_cal.mean())**2).sum())
    print(f"    -> Extracted Calendar Baseline: RMSE={base_cov_rmse:.4f}, R2={base_cov_r2:.4f}")
except Exception as e:
    print(f"    -> WARNING: Unable to load Calendar baseline CSV. ({e})")
    base_cov_rmse = base_cov_mae = base_cov_mse = base_cov_r2 = 0.0

# Load TimesFM
tfm_metrics_df = pd.read_csv(os.path.join(OUTPUTS_DIR, "timesfm2p5_hourly_metrics.csv"))
tfm_h2 = tfm_metrics_df[tfm_metrics_df["horizon_h"] == FORECAST_HORIZON].iloc[0]
tfm_rmse = float(tfm_h2["RMSE"])
tfm_mae  = float(tfm_h2["MAE"])
tfm_mse  = float(tfm_h2["MSE"])
tfm_r2   = float(tfm_h2.iloc[-1]) # R2 is the last column

metrics_data = [
    ("Champion Rolling",        "#2ecc71", [test_mse, test_rmse, test_mae, test_r2]),
    ("Baseline TSMixer (Cov)",  "#3498db", [base_cov_mse, base_cov_rmse, base_cov_mae, base_cov_r2]),
    ("Baseline TSMixer (Tgt)",  "#8e44ad", [base_tgt_mse, base_tgt_rmse, base_tgt_mae, base_tgt_r2]),
    ("TimesFM 2.5",             "#e67e22", [tfm_mse, tfm_rmse, tfm_mae, tfm_r2]),
]

metric_labels = ["MSE", "RMSE", "MAE", "R²"]

fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(18, 5))
fig.suptitle(f"Metrics Comparison at Horizon h={FORECAST_HORIZON} (From Static CSVs)", fontsize=14, fontweight="bold")

n_bars = len(metrics_data)
bw = 0.8 / n_bars
x = np.arange(1)
offsets = np.linspace(-(n_bars - 1) / 2 * bw, (n_bars - 1) / 2 * bw, n_bars)

all_legend_bars = []
for idx, ax in enumerate(axes):
    for model_idx, (model_name, color, values) in enumerate(metrics_data):
        val = values[idx]
        bar = ax.bar(x + offsets[model_idx], [val], bw, color=color, alpha=0.85)
        if idx == 0:
            all_legend_bars.append((bar, model_name))
        
        ax.text(
            bar[0].get_x() + bar[0].get_width() / 2.0,
            val,
            f"{val:.3f}",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
        )
    
    ax.set_title(metric_labels[idx], fontweight="bold")
    ax.set_xticks([])
    ax.grid(axis="y", alpha=0.3)

fig.legend([b[0] for b in all_legend_bars], [n[1] for n in all_legend_bars], loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.05), fontsize=10)
fig.tight_layout(rect=[0, 0.05, 1, 0.96])

plot11_path = os.path.join(OUTPUTS_DIR, "plot_comparison_metrics_bars.png")
fig.savefig(plot11_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {plot11_path}")

# ---------------------------------------------------------------------------
# 12. EXTRA: Forecast Curve Comparison (Champion vs Base)
# ---------------------------------------------------------------------------
print("\n" + "=" * 65)
print("12. EXTRA: FORECAST CURVE COMPARISON (Champion vs Base)")
print("=" * 65)

fig, ax = plt.subplots(figsize=(16, 5))
fig.suptitle(f"Forecast Comparison: Champion vs Baselines (h={FORECAST_HORIZON})", fontsize=12, fontweight="bold")

# Plot Actuals
act_aligned.plot(ax=ax, label="Actual Capacity", color="black", linewidth=2.5, linestyle="dashed")

# Plot Base Target
_base_tgt_dt = pd.to_datetime(tsm_target_df.get("TimeStamp", tsm_target_df.index))
_df_tgt = pd.DataFrame({"TimeStamp": _base_tgt_dt, "InUseCapacity_P50": p50_preds.values})
base_tgt_ts = TimeSeries.from_dataframe(_df_tgt, time_col="TimeStamp", value_cols="InUseCapacity_P50")
base_tgt_ts.plot(ax=ax, label=f"Baseline Target-Only (RMSE={base_tgt_rmse:.2f}, R²={base_tgt_r2:.2f})", color="#6c3483", linewidth=1.5, alpha=0.7)

# Plot Base Covariates
if "tsm_cal_df" in locals():
    _base_cov_dt = pd.to_datetime(tsm_cal_df.get("TimeStamp", tsm_cal_df.index))
    _df_cov = pd.DataFrame({"TimeStamp": _base_cov_dt, "InUseCapacity_P50": p50_cal.values})
    base_cov_ts = TimeSeries.from_dataframe(_df_cov, time_col="TimeStamp", value_cols="InUseCapacity_P50")
    base_cov_ts.plot(ax=ax, label=f"Baseline Calendar Cov. (RMSE={base_cov_rmse:.2f}, R²={base_cov_r2:.2f})", color="#3498db", linewidth=1.5, alpha=0.9)

# Plot Champion
hf_p50.plot(ax=ax, label=f"Champion Rolling (RMSE={test_rmse:.2f}, R²={test_r2:.2f})", color="#2ecc71", linewidth=2.5)

ax.set_ylabel("InUseCapacity")
ax.legend(fontsize=9, loc='upper left')
ax.grid(True, alpha=0.3)
fig.tight_layout()

plot12_path = os.path.join(OUTPUTS_DIR, "plot_comparison_forecast_champion_vs_base.png")
fig.savefig(plot12_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {plot12_path}")

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------
try:
    del final_model, hf_list, hf_concat, hf_p50, split_model, split_hf_list, split_hf_p50
    del base_target_model, base_cov_model
except:
    pass
gc.collect()
if torch and torch.cuda.is_available():
    torch.cuda.empty_cache()

print("\n" + "=" * 65)
print("DONE -- all outputs saved to /outputs/")
print("=" * 65)
print("  tsmixer_hyperparams_rolling_results.csv")
print("  tsmixer_hyperparams_rolling_results.txt")
print("  tsmixer_hyperparams_rolling_best_backtest.png")
print("  plot_comparison_tsmixer_rolling_vs_baselines.png")
print("  plot_comparison_tsmixer_rolling_vs_split.png")
