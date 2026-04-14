import os
import gc
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from darts import TimeSeries, concatenate, set_option
from darts.models import TSMixerModel
from darts.utils.likelihood_models import QuantileRegression
from darts.metrics import mse, rmse, mae, r2_score
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import torch

warnings.filterwarnings("ignore")

try:
    set_option("plotting.use_darts_style", True)
except Exception:
    pass

# =============================================================================
# 1. CONFIGURATION
# =============================================================================

# Paths
DATA_DIR    = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "datasets"))
OUTPUTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs"))
os.makedirs(OUTPUTS_DIR, exist_ok=True)

CSV_FILE = os.path.join(DATA_DIR, "Dati_processed.csv")

TARGET           = "InUseCapacity"
INPUT_CHUNK_LEN  = 168
OUTPUT_CHUNK_LEN = 8
FORECAST_HORIZON = 2  # Operational goal (p50)
STRIDE           = 1
TEST_HOURS       = 72
VAL_SPLIT        = 0.2

# Best Champion Hyperparameters (from Rolling Gridsearch)
CHAMP_HIDDEN_SIZE = 32
CHAMP_FF_SIZE     = 32
CHAMP_NUM_BLOCKS  = 3
CHAMP_DROPOUT     = 0.1
CHAMP_BATCH_SIZE  = 32
CHAMP_LR          = 0.001
CHAMP_EPOCHS      = 100

print("=" * 65)
print("1. EVALUATING TSMIXER CHAMPION MODEL (Rolling Params)")
print("=" * 65)

# =============================================================================
# 2. DATA PREPARATION
# =============================================================================
print("  Loading data from:", CSV_FILE)
df = pd.read_csv(CSV_FILE, parse_dates=["TimeStamp"])
df = df.sort_values("TimeStamp").reset_index(drop=True)

df["hour"] = df["TimeStamp"].dt.hour
df["day_of_week"] = df["TimeStamp"].dt.dayofweek
import holidays
it_holidays = holidays.Italy(years=[2024, 2025])
df["is_holiday"] = df["TimeStamp"].apply(lambda d: 1 if d in it_holidays else 0)

last_ts = df["TimeStamp"].iloc[-1]
fut_dates = pd.date_range(start=last_ts + pd.Timedelta(hours=1), periods=OUTPUT_CHUNK_LEN, freq="h")
fut_df = pd.DataFrame({
    "TimeStamp": fut_dates,
    "hour": fut_dates.hour,
    "day_of_week": fut_dates.dayofweek,
    "is_holiday": [1 if d in it_holidays else 0 for d in fut_dates],
})
df_extended = pd.concat([df, fut_df], ignore_index=True)

def fill_missing_values(ts: TimeSeries) -> TimeSeries:
    from darts.dataprocessing.transformers import MissingValuesFiller
    return MissingValuesFiller().transform(ts)

series = fill_missing_values(TimeSeries.from_dataframe(df, time_col="TimeStamp", value_cols=TARGET, freq="h"))
future_cov = fill_missing_values(TimeSeries.from_dataframe(df_extended, time_col="TimeStamp", value_cols=["hour", "day_of_week", "is_holiday"], freq="h"))

# Splits
test_start_idx = len(series) - TEST_HOURS
train_series   = series[:test_start_idx]
test_series    = series[test_start_idx:]
val_len        = int(len(train_series) * VAL_SPLIT)
val_series     = train_series[-val_len:]
fit_series     = train_series[:-val_len]

# Actual labels for h=2 backtest
act_aligned = test_series[FORECAST_HORIZON-1 : -1]

# =============================================================================
# 3. BUILD AND EVALUATE CHAMPION MODELS
# =============================================================================

def build_champion():
    early_stop = EarlyStopping(monitor="val_loss", patience=10, min_delta=1e-4, mode="min")
    return TSMixerModel(
        input_chunk_length=INPUT_CHUNK_LEN,
        output_chunk_length=OUTPUT_CHUNK_LEN,
        hidden_size=CHAMP_HIDDEN_SIZE,
        ff_size=CHAMP_FF_SIZE,
        num_blocks=CHAMP_NUM_BLOCKS,
        dropout=CHAMP_DROPOUT,
        batch_size=CHAMP_BATCH_SIZE,
        n_epochs=CHAMP_EPOCHS,
        likelihood=QuantileRegression(quantiles=[0.1, 0.3, 0.5, 0.7, 0.9]),
        optimizer_kwargs={"lr": CHAMP_LR},
        random_state=42,
        pl_trainer_kwargs={"enable_progress_bar": False, "callbacks": [early_stop]},
    )

# ── 3.1 Champion (Covariates) ──
print("\n-> Fitting Champion (Calendar Covariates)...")
champ_cov_model = build_champion()
champ_cov_model.fit(series=fit_series, val_series=val_series, future_covariates=future_cov, val_future_covariates=future_cov, verbose=False)

print("   Backtesting Champion Covariates...")
champ_cov_hfs = champ_cov_model.historical_forecasts(
    series=series, future_covariates=future_cov, start=test_series.start_time(),
    forecast_horizon=FORECAST_HORIZON, stride=STRIDE, num_samples=200, retrain=False, verbose=False
)
champ_cov_p50 = concatenate(champ_cov_hfs).quantile(0.5)

cc_rmse = rmse(act_aligned, champ_cov_p50)
cc_mae  = mae(act_aligned, champ_cov_p50)
cc_mse  = mse(act_aligned, champ_cov_p50)
cc_r2   = r2_score(act_aligned, champ_cov_p50)

# ── 3.2 Champion (Target-Only) ──
print("\n-> Fitting Champion (Target-Only)...")
champ_tgt_model = build_champion()
champ_tgt_model.fit(series=fit_series, val_series=val_series, verbose=False)

print("   Backtesting Champion Target-Only...")
champ_tgt_hfs = champ_tgt_model.historical_forecasts(
    series=series, start=test_series.start_time(), forecast_horizon=FORECAST_HORIZON, 
    stride=STRIDE, num_samples=200, retrain=False, verbose=False
)
champ_tgt_p50 = concatenate(champ_tgt_hfs).quantile(0.5)

ct_rmse = rmse(act_aligned, champ_tgt_p50)
ct_mae  = mae(act_aligned, champ_tgt_p50)
ct_mse  = mse(act_aligned, champ_tgt_p50)
ct_r2   = r2_score(act_aligned, champ_tgt_p50)

# Match axes perfectly to avoid pandas length mismatch
champ_cov_p50 = champ_cov_p50.slice_intersect(act_aligned)
act_cov_aligned = act_aligned.slice_intersect(champ_cov_p50)

cc_df = pd.DataFrame({"InUseCapacity_P50": champ_cov_p50.values().flatten(), "InUseCapacity_Actual": act_cov_aligned.values().flatten()}, index=champ_cov_p50.time_index)
cc_df.index.name = "TimeStamp"
cc_df.to_csv(os.path.join(OUTPUTS_DIR, "champion_cov_backtest_p50_h2.csv"))

champ_tgt_p50 = champ_tgt_p50.slice_intersect(act_aligned)
act_tgt_aligned = act_aligned.slice_intersect(champ_tgt_p50)

ct_df = pd.DataFrame({"InUseCapacity_P50": champ_tgt_p50.values().flatten(), "InUseCapacity_Actual": act_tgt_aligned.values().flatten()}, index=champ_tgt_p50.time_index)
ct_df.index.name = "TimeStamp"
ct_df.to_csv(os.path.join(OUTPUTS_DIR, "champion_tgt_backtest_p50_h2.csv"))

# =============================================================================
# 4. LOAD BASELINE METRICS FROM CSVS
# =============================================================================
print("\n" + "=" * 65)
print("2. LOADING BASELINE METRICS")
print("=" * 65)

# ── Baseline Target-Only ──
tsm_target_df = pd.read_csv(os.path.join(OUTPUTS_DIR, "tsmixer_backtest_p50_h2.csv"))
p50_tgt = tsm_target_df["InUseCapacity_P50"]
act_tgt = tsm_target_df["InUseCapacity_Actual"]
bt_rmse = ((p50_tgt - act_tgt)**2).mean()**0.5
bt_mae  = (p50_tgt - act_tgt).abs().mean()
bt_mse  = ((p50_tgt - act_tgt)**2).mean()
bt_r2   = 1 - (((p50_tgt - act_tgt)**2).sum() / ((act_tgt - act_tgt.mean())**2).sum())

base_tgt_dt = pd.to_datetime(tsm_target_df.get("TimeStamp", tsm_target_df.index))
base_tgt_ts = TimeSeries.from_dataframe(pd.DataFrame({"TimeStamp": base_tgt_dt, "InUseCapacity_P50": p50_tgt.values}), time_col="TimeStamp", value_cols="InUseCapacity_P50")

# ── Baseline Covariates ──
try:
    tsm_cal_df = pd.read_csv(os.path.join(OUTPUTS_DIR, "tsmixer_calendar_backtest_p50_h2.csv"))
    p50_cov = tsm_cal_df["InUseCapacity_P50"]
    act_cov = tsm_cal_df["InUseCapacity_Actual"]
    bc_rmse = ((p50_cov - act_cov)**2).mean()**0.5
    bc_mae  = (p50_cov - act_cov).abs().mean()
    bc_mse  = ((p50_cov - act_cov)**2).mean()
    bc_r2   = 1 - (((p50_cov - act_cov)**2).sum() / ((act_cov - act_cov.mean())**2).sum())
    
    base_cov_dt = pd.to_datetime(tsm_cal_df.get("TimeStamp", tsm_cal_df.index))
    base_cov_ts = TimeSeries.from_dataframe(pd.DataFrame({"TimeStamp": base_cov_dt, "InUseCapacity_P50": p50_cov.values}), time_col="TimeStamp", value_cols="InUseCapacity_P50")
    print("  Loaded: Baseline Covariates from CSV")
except Exception as e:
    print(f"  WARNING: Unable to load Calendar baseline CSV. ({e})")
    bc_rmse = bc_mae = bc_mse = bc_r2 = 0.0
    base_cov_ts = None

# ── TimesFM ──
tfm_metrics_df = pd.read_csv(os.path.join(OUTPUTS_DIR, "timesfm2p5_hourly_metrics.csv"))
tfm_h2  = tfm_metrics_df[tfm_metrics_df["horizon_h"] == FORECAST_HORIZON].iloc[0]
tfm_rmse = float(tfm_h2["RMSE"])
tfm_mae  = float(tfm_h2["MAE"])
tfm_mse  = float(tfm_h2["MSE"])
tfm_r2   = float(tfm_h2.iloc[-1])
print("  Loaded: TimesFM 2.5 metrics")

# =============================================================================
# 5. GENERATE PLOTS
# =============================================================================
print("\n" + "=" * 65)
print("3. GENERATING COMPARISON PLOTS")
print("=" * 65)


def plot_metric_bars(metrics_data, title, filename):
    metric_labels = ["MSE", "RMSE", "MAE", "R²"]
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(18, 5))
    fig.suptitle(title, fontsize=14, fontweight="bold")

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

    fig.legend([b[0] for b in all_legend_bars], [n[1] for n in all_legend_bars], loc='lower center', ncol=len(metrics_data), bbox_to_anchor=(0.5, -0.05), fontsize=11)
    fig.tight_layout(rect=[0, 0.05, 1, 0.96])
    
    path = os.path.join(OUTPUTS_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Plot A: Champion vs Best Baselines ──
metrics_a = [
    ("Champion TSMixer (Cov)", "#00cec9", [cc_mse, cc_rmse, cc_mae, cc_r2]),
    ("Baseline TSMixer (Cov)", "#0984e3", [bc_mse, bc_rmse, bc_mae, bc_r2]),
    ("TimesFM 2.5",            "#e67e22", [tfm_mse, tfm_rmse, tfm_mae, tfm_r2]),
]
plot_metric_bars(metrics_a, "Metrics Comparison: Best TSMixer vs Baseline vs TimesFM (h=2)", "eval_metrics_best_baseline.png")

# ── Plot B: All 4 TSMixer Variants ──
metrics_b = [
    ("Champion (Cov)", "#00cec9", [cc_mse, cc_rmse, cc_mae, cc_r2]),
    ("Champion (Tgt)", "#9b59b6", [ct_mse, ct_rmse, ct_mae, ct_r2]),
    ("Baseline (Cov)", "#0984e3", [bc_mse, bc_rmse, bc_mae, bc_r2]),
    ("Baseline (Tgt)", "#FF1493", [bt_mse, bt_rmse, bt_mae, bt_r2]),
]
plot_metric_bars(metrics_b, "Metrics Comparison: Champion (Cov/Tgt) vs Baseline (Cov/Tgt) (h=2)", "eval_metrics_4_variants.png")


# ── Plot C: Forecast Curve (Champion Cov vs Base Cov) ──
fig, ax = plt.subplots(figsize=(16, 5))
fig.suptitle(f"Forecast Overlay: Champion (Covariates) vs Baseline (Covariates)", fontsize=13, fontweight="bold")
act_aligned.plot(ax=ax, label="Actual", color="#2ecc71", linewidth=2.5)
if base_cov_ts is not None:
    base_cov_ts.plot(ax=ax, label=f"Baseline Cov (RMSE={bc_rmse:.2f})", color="#00BFFF", linewidth=1.5, alpha=0.9)
champ_cov_p50.plot(ax=ax, label=f"Champion Cov (RMSE={cc_rmse:.2f})", color="#FF8C00", linewidth=1.5)

ax.set_ylabel("InUseCapacity")
ax.legend(fontsize=10, loc='upper left')
ax.grid(True, alpha=0.3)
fig.tight_layout()
plot_c_path = os.path.join(OUTPUTS_DIR, "eval_forecast_champ_vs_basecov.png")
fig.savefig(plot_c_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {plot_c_path}")

# ── Plot D: Forecast Curve (All 4 Variants) ──
fig, ax = plt.subplots(figsize=(16, 6))
fig.suptitle(f"Forecast Overlay: All 4 TSMixer Variants (h={FORECAST_HORIZON})", fontsize=13, fontweight="bold")

act_aligned.plot(ax=ax, label="Actual", color="#2ecc71", linewidth=2.5)
if base_cov_ts is not None:
    base_cov_ts.plot(ax=ax, label=f"Baseline Cov (RMSE={bc_rmse:.2f})", color="#00BFFF", linewidth=1.5, alpha=0.9)
base_tgt_ts.plot(ax=ax, label=f"Baseline Tgt (RMSE={bt_rmse:.2f})", color="#FF1493", linewidth=1.5, alpha=0.9)

champ_tgt_p50.plot(ax=ax, label=f"Champion Tgt (RMSE={ct_rmse:.2f})", color="#8A2BE2", linewidth=1.5, alpha=0.9)
champ_cov_p50.plot(ax=ax, label=f"Champion Cov (RMSE={cc_rmse:.2f})", color="#FF8C00", linewidth=1.5, alpha=0.9)

ax.set_ylabel("InUseCapacity")
ax.legend(fontsize=9, loc='upper left')
ax.grid(True, alpha=0.3)
fig.tight_layout()
plot_d_path = os.path.join(OUTPUTS_DIR, "eval_forecast_4_variants.png")
fig.savefig(plot_d_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {plot_d_path}")

try:
    del champ_cov_model, champ_tgt_model
except:
    pass
gc.collect()
if torch and torch.cuda.is_available():
    torch.cuda.empty_cache()

print("\n" + "=" * 65)
print("DONE -- Evaluation Plots Saved.")
print("=" * 65)
