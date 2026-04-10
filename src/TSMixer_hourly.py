"""
TSMixer Hourly Short-Term Forecaster for AWS AppStream Autoscaler
=================================================================
Goal      : Predict InUseCapacity h ∈ {2, 3, 4} steps (hours) ahead.
Frequency : Triggered every hour by the autoscaler daemon.
Library   : 100% Darts-native (TimeSeries, historical_forecasts, metrics, plots).

Model
-----
TSMixerModel (Time-Series Mixer) — an All-MLP architecture (MixedCovariatesTorchModel).
  supports_past_covariates   = True
  supports_future_covariates = True
  supports_static_covariates = True
  Trainable (requires fit with gradient descent). Not zero-shot.

Covariate Strategy
------------------
Dropped (data-leakage or effect, not cause) — same policy as chronos2_hourly.py:
  - ActualCapacity       = AvailableCapacity + InUseCapacity  (linear combo)
  - AvailableCapacity    = ActualCapacity - InUseCapacity     (same leakage)
  - CapacityUtilization  = InUseCapacity / ActualCapacity     (derived from target)
  - DesiredCapacity      = native scaler output, not user-demand predictor
  - InsufficientCapacityError / InsufficientConcurrencyLimitError  (sparse)

Two modes compared (same as Chronos-2):
  "none"     → target-only (no covariates, purest comparison)
  "calendar" → + hour / day_of_week / is_holiday as future covariates

Fair-comparison guarantees vs chronos2_hourly.py
-------------------------------------------------
  ✓ Same target column      : InUseCapacity
  ✓ Same input_chunk_length : 168 (7 days @ 1h — weekly seasonality)
  ✓ Same output_chunk_length: 8   (≥ max horizon, no auto-regression needed)
  ✓ Same test window        : 72h rolling backtest, stride=1
  ✓ Same horizons evaluated : h ∈ {2, 3, 4}
  ✓ Same num_samples        : 200 Monte-Carlo draws
  ✓ Same metrics            : MSE, RMSE, MAE, R²
  ✓ Same covariate modes    : "none" and "calendar"
  ✓ Same plot style         : backtest bands, mode comparison, bar charts
  ✗ Training                : TSMixer requires gradient-descent training
                              (Chronos-2 is zero-shot). Val split = last 20%
                              of train set used for early stopping.
"""

import os
import logging

import holidays
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from darts import TimeSeries, concatenate, set_option
from darts.metrics import mae, mse, rmse, r2_score
from darts.models import TSMixerModel
from darts.utils.likelihood_models import QuantileRegression
from darts.utils.missing_values import fill_missing_values

logging.getLogger("cmdstanpy").setLevel(logging.CRITICAL)
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Configuration — keep identical to chronos2_hourly.py for a fair comparison
# ---------------------------------------------------------------------------
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_PATH   = os.path.join(BASE_DIR, "..", "datasets", "Dati_processed.csv")
OUTPUTS_DIR = os.path.join(BASE_DIR, "..", "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

TARGET          = "InUseCapacity"
HORIZONS        = [2, 3, 4]            # hours ahead to evaluate
COVARIATE_MODES = ["none", "calendar"] # both modes benchmarked (same as Chronos-2)

# Context window / single-pass output size — identical to Chronos-2
INPUT_CHUNK_LEN  = 168   # 7 days of hourly context (weekly seasonality)
OUTPUT_CHUNK_LEN = 8     # ≥ max(HORIZONS) — no auto-regression needed

# Rolling-backtest window (last N hours used as held-out test)
TEST_HOURS  = 72    # 3 days
NUM_SAMPLES = 200   # Monte-Carlo samples for probabilistic quantiles

# TSMixer training parameters
# Val split: last 20% of training data is used for early stopping.
# n_epochs capped with early stopping (patience=10) to avoid overfitting.
N_EPOCHS         = 50
BATCH_SIZE       = 32
HIDDEN_SIZE      = 64
FF_SIZE          = 64
NUM_BLOCKS       = 2
DROPOUT          = 0.1
VAL_SPLIT        = 0.2   # fraction of train used for validation / early stopping
LEARNING_RATE    = 1e-3

# Columns to drop from any covariate consideration (leakage / effect-not-cause)
LEAKAGE_COLS = {
    "ActualCapacity",
    "AvailableCapacity",
    "CapacityUtilization",
    "DesiredCapacity",
    "InsufficientCapacityError",
    "InsufficientConcurrencyLimitError",
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

# Calendar features — same as Chronos-2
it_holidays = holidays.Italy()
df["hour"]        = df["TimeStamp"].dt.hour
df["day_of_week"] = df["TimeStamp"].dt.dayofweek
df["is_holiday"]  = df["TimeStamp"].apply(lambda x: 1 if x in it_holidays else 0)

print(f"  Dataset: {len(df)} hourly rows")
print(f"  Period : {df['TimeStamp'].iloc[0]}  →  {df['TimeStamp'].iloc[-1]}")
print(f"  Target : {TARGET}  (min={df[TARGET].min():.0f}, max={df[TARGET].max():.0f})")

# ---------------------------------------------------------------------------
# 2. Correlation Analysis (printed for transparency)
# ---------------------------------------------------------------------------
print("\n" + "=" * 65)
print("2. COVARIATE CORRELATION ANALYSIS vs InUseCapacity")
print("=" * 65)

numeric_cols = df.select_dtypes(include="number").columns.tolist()
corr = df[numeric_cols].corr()[TARGET].drop(TARGET).sort_values(key=abs, ascending=False)
print(corr.to_string(float_format="{:.4f}".format))

calendar_cols = ["hour", "day_of_week", "is_holiday"]
surviving_past_cols = [
    c for c in numeric_cols
    if c not in LEAKAGE_COLS and c not in calendar_cols and c != TARGET
]
print(f"\n  Surviving past-covariate candidates after leakage filter: {surviving_past_cols}")
print(f"  → All dropped by policy (non-causal). Using calendar covariates only.")

# ---------------------------------------------------------------------------
# 3. TimeSeries Creation
# ---------------------------------------------------------------------------
print("\n" + "=" * 65)
print("3. BUILDING DARTS TIMESERIES")
print("=" * 65)

# Extend calendar into the future (needed for future_covariates in predict())
# TSMixer needs future covariates to be known up to n steps ahead at prediction time.
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

series = fill_missing_values(
    TimeSeries.from_dataframe(df, time_col="TimeStamp", value_cols=TARGET, freq="h")
)
future_cov = fill_missing_values(
    TimeSeries.from_dataframe(df_extended, time_col="TimeStamp", value_cols=calendar_cols, freq="h")
)

print(f"  Target TimeSeries  : {len(series)} steps")
print(f"  Future covariates  : {len(future_cov)} steps  (includes {OUTPUT_CHUNK_LEN}h lookahead)")

# ---------------------------------------------------------------------------
# 4. Train / Test Split
# ---------------------------------------------------------------------------
print("\n" + "=" * 65)
print("4. TRAIN / TEST SPLIT")
print("=" * 65)

test_start_idx = len(series) - TEST_HOURS
train_series   = series[:test_start_idx]
test_series    = series[test_start_idx:]

# Internal validation split for early stopping (carved from training only,
# never from the backtest test window — avoids data leakage into evaluation)
val_len        = int(len(train_series) * VAL_SPLIT)
val_series     = train_series[-val_len:]      # last 20% of train
fit_series     = train_series[:-val_len]      # first 80% of train

print(f"  Train (fit)  : {len(fit_series)} steps  ({len(fit_series)//24} days)")
print(f"  Train (val)  : {len(val_series)} steps  ({val_len//24} days)  — for early stopping")
print(f"  Test         : {len(test_series)} steps  ({TEST_HOURS}h rolling backtest window)")
print(f"  Test starts at: {test_series.start_time()}")

# ---------------------------------------------------------------------------
# 5. Model Factory
# ---------------------------------------------------------------------------

def build_model() -> TSMixerModel:
    """Instantiate TSMixerModel with QuantileRegression likelihood.

    TSMixer key parameters:
    - hidden_size / ff_size: MLP widths in the mixer blocks
    - num_blocks           : number of stacked mixer blocks
    - dropout              : regularisation (also enables MC-dropout at inference)
    - likelihood           : QuantileRegression → probabilistic quantile output
    - n_epochs / batch_size: standard PyTorch-Lightning training knobs

    Early stopping is handled by passing val_series to fit(); PyTorch-Lightning
    will monitor val_loss and stop if it does not improve for `patience` epochs.
    """
    from pytorch_lightning.callbacks.early_stopping import EarlyStopping
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=10,
        min_delta=1e-4,
        mode="min",
    )
    return TSMixerModel(
        input_chunk_length=INPUT_CHUNK_LEN,
        output_chunk_length=OUTPUT_CHUNK_LEN,
        hidden_size=HIDDEN_SIZE,
        ff_size=FF_SIZE,
        num_blocks=NUM_BLOCKS,
        dropout=DROPOUT,
        likelihood=QuantileRegression(quantiles=[0.1, 0.3, 0.5, 0.7, 0.9]),
        n_epochs=N_EPOCHS,
        batch_size=BATCH_SIZE,
        optimizer_kwargs={"lr": LEARNING_RATE},
        pl_trainer_kwargs={
            "enable_progress_bar": True,
            "callbacks": [early_stop],
        },
        random_state=42,
    )

# ---------------------------------------------------------------------------
# 6. Backtesting Loop — one run per (covariate_mode, horizon)
# ---------------------------------------------------------------------------
print("\n" + "=" * 65)
print("5. BACKTESTING  (historical_forecasts, stride=1)")
print("=" * 65)

all_results    = {}   # (mode, h) → dict
backtest_preds = {}   # (mode, h) → dict of TimeSeries

for mode in COVARIATE_MODES:
    print(f"\n  ── Mode: '{mode}' ──")

    model = build_model()

    fit_kwargs     = {}
    predict_kwargs = {}
    val_kwargs     = {}
    if mode == "calendar":
        fit_kwargs["future_covariates"]     = future_cov
        val_kwargs["val_future_covariates"] = future_cov
        predict_kwargs["future_covariates"] = future_cov

    print(f"    Fitting TSMixer on {len(fit_series)} steps "
          f"(+ {len(val_series)} val steps, early-stop patience=10)…")
    model.fit(
        series=fit_series,
        val_series=val_series,
        **fit_kwargs,
        **val_kwargs,
        verbose=True,
    )

    for h in HORIZONS:
        print(f"    Backtesting h={h}…", end=" ", flush=True)

        hf_list = model.historical_forecasts(
            series=series,
            **predict_kwargs,
            start=test_series.start_time(),
            forecast_horizon=h,
            stride=1,
            num_samples=NUM_SAMPLES,
            retrain=False,
            verbose=False,
        )

        hf_concat = concatenate(hf_list)
        hf_p50    = hf_concat.quantile(0.5)   # point estimate for metrics

        aligned_actual = series.slice_intersect(hf_p50)

        mae_val  = mae( aligned_actual, hf_p50)
        mse_val  = mse( aligned_actual, hf_p50)
        rmse_val = rmse(aligned_actual, hf_p50)
        r2_val   = r2_score(aligned_actual, hf_p50)

        print(f"MSE={mse_val:.2f}  RMSE={rmse_val:.2f}  MAE={mae_val:.2f}  R²={r2_val:.4f}")

        key = (mode, h)
        all_results[key] = {
            "mae":  mae_val,
            "mse":  mse_val,
            "rmse": rmse_val,
            "r2":   r2_val,
        }
        backtest_preds[key] = {
            "concat": hf_concat,
            "p50":    hf_p50,
            "actual": aligned_actual,
        }

# ---------------------------------------------------------------------------
# 7. Load Chronos-2 Hourly Benchmark for comparison
# ---------------------------------------------------------------------------
chronos_bench_path = os.path.join(OUTPUTS_DIR, "chronos2_hourly_metrics.csv")
chronos_bench = None
if os.path.exists(chronos_bench_path):
    cdf = pd.read_csv(chronos_bench_path, index_col=[0, 1])
    if "none" in cdf.index.get_level_values(0):
        chronos_bench = cdf.loc["none"]
        print(f"\n  Loaded Chronos-2 hourly benchmark (mode='none') for comparison.")

# ---------------------------------------------------------------------------
# 8. Metrics Summary Table
# ---------------------------------------------------------------------------
print("\n" + "=" * 65)
print("6. METRICS SUMMARY")
print("=" * 65)

rows = []
for (mode, h), metrics in all_results.items():
    rows.append({
        "mode":       mode,
        "horizon_h":  h,
        "MSE":  round(metrics["mse"],  4),
        "RMSE": round(metrics["rmse"], 4),
        "MAE":  round(metrics["mae"],  4),
        "R²":   round(metrics["r2"],   4),
    })

metrics_df = pd.DataFrame(rows).set_index(["mode", "horizon_h"])
print(metrics_df.to_string())

# Side-by-side comparison with Chronos-2 (target-only) if available
if chronos_bench is not None:
    print("\n  ── Head-to-head TSMixer (none) vs Chronos-2 (none) ──")
    header = f"  {'h':>3}  {'Metric':>6}  {'TSMixer':>10}  {'Chronos-2':>10}  {'Δ':>10}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for h in HORIZONS:
        for metric_label, mk in [("RMSE", "rmse"), ("MAE", "mae"), ("R²", "r2")]:
            tfm_val = all_results[("none", h)][mk]
            try:
                col = "R²" if metric_label == "R²" else metric_label
                c2_val = float(chronos_bench.loc[h, col])
                delta  = tfm_val - c2_val
                better = (metric_label == "R²" and delta > 0) or (metric_label != "R²" and delta < 0)
                symbol = "↓ BETTER" if better else "↑ WORSE"
                print(f"  {h:>3}  {metric_label:>6}  {tfm_val:>10.4f}  {c2_val:>10.4f}  {delta:>+10.4f}  {symbol}")
            except Exception:
                print(f"  {h:>3}  {metric_label:>6}  {tfm_val:>10.4f}  {'N/A':>10}")

metrics_csv = os.path.join(OUTPUTS_DIR, "tsmixer_hourly_metrics.csv")
metrics_df.to_csv(metrics_csv)
print(f"\n  Saved: {metrics_csv}")

# ---------------------------------------------------------------------------
# 9. Operational Forecast — next h steps from the very last observation
# ---------------------------------------------------------------------------
print("\n" + "=" * 65)
print("7. OPERATIONAL FORECAST (next steps from now)")
print("=" * 65)

# Use the 'none' (target-only) model as the operational default (no lookahead needed)
op_model = build_model()
print(f"  Fitting on full series ({len(series)} steps)…")
op_model.fit(series=series, verbose=True)

op_horizon  = max(HORIZONS)   # predict up to the longest horizon = 4 steps
op_forecast = op_model.predict(n=op_horizon, num_samples=NUM_SAMPLES)

f_vals = op_forecast.all_values()   # shape (n_steps, 1, n_samples)
df_forecast = pd.DataFrame(
    {
        "InUseCapacity_P50": np.quantile(f_vals, 0.50, axis=-1).flatten(),
        "InUseCapacity_P70": np.quantile(f_vals, 0.70, axis=-1).flatten(),
        "InUseCapacity_P90": np.quantile(f_vals, 0.90, axis=-1).flatten(),
    },
    index=op_forecast.time_index,
)
df_forecast.index.name = "TimeStamp"

forecast_csv = os.path.join(OUTPUTS_DIR, "tsmixer_hourly_next_forecast.csv")
df_forecast.to_csv(forecast_csv)
print(f"  Saved : {forecast_csv}")
print(df_forecast.to_string())

# ---------------------------------------------------------------------------
# 10. Plots (Darts-native + matplotlib overlay) — same style as Chronos-2
# ---------------------------------------------------------------------------
print("\n" + "=" * 65)
print("8. GENERATING PLOTS")
print("=" * 65)

set_option("plotting.use_darts_style", True)


def save_fig(fig, fname):
    path = os.path.join(OUTPUTS_DIR, fname)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Plot 1: Rolling Backtest — one subplot per horizon (mode='none') ────────
fig, axes = plt.subplots(nrows=len(HORIZONS), ncols=1, figsize=(18, 5 * len(HORIZONS)), sharex=True)
fig.suptitle("TSMixer Short-Term Backtesting (target-only mode)", fontsize=14, fontweight="bold")

for ax, h in zip(axes, HORIZONS):
    key    = ("none", h)
    preds  = backtest_preds[key]
    actual = preds["actual"]
    p50    = preds["p50"]
    full   = preds["concat"]

    p10 = full.quantile(0.1)
    p90 = full.quantile(0.9)

    actual.plot(ax=ax, label="Actual", color="#2ecc71", linewidth=2)
    p50.plot(ax=ax, label=f"Forecast P50 (h={h})", color="#8e44ad", linewidth=1.5)

    ax.fill_between(
        p10.time_index,
        p10.values().flatten(),
        p90.values().flatten(),
        alpha=0.25,
        color="#8e44ad",
        label="P10–P90 band",
    )

    m = all_results[key]
    ax.set_title(
        f"h={h}h  |  MSE={m['mse']:.2f}   RMSE={m['rmse']:.2f}   "
        f"MAE={m['mae']:.2f}   R²={m['r2']:.4f}",
        fontsize=10,
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

fig.tight_layout(rect=[0, 0, 1, 0.97])
save_fig(fig, "tsmixer_hourly_backtest_none.png")

# ── Plot 2: Mode Comparison per horizon ────────────────────────────────────
fig, axes = plt.subplots(nrows=len(HORIZONS), ncols=1, figsize=(18, 5 * len(HORIZONS)), sharex=True)
fig.suptitle("TSMixer: target-only vs calendar-covariates", fontsize=14, fontweight="bold")

mode_styles = {"none": ("#8e44ad", "Target-Only"), "calendar": ("#2980b9", "Calendar Cov.")}

for ax, h in zip(axes, HORIZONS):
    actual = backtest_preds[("none", h)]["actual"]
    actual.plot(ax=ax, label="Actual", color="#2ecc71", linewidth=2)

    for mode, (color, label) in mode_styles.items():
        key = (mode, h)
        m   = all_results[key]
        backtest_preds[key]["p50"].plot(
            ax=ax,
            label=f"{label}  RMSE={m['rmse']:.2f}  R²={m['r2']:.4f}",
            color=color,
            linewidth=1.5,
        )

    ax.set_title(f"Horizon h={h}h — mode comparison", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

fig.tight_layout(rect=[0, 0, 1, 0.97])
save_fig(fig, "tsmixer_hourly_mode_comparison.png")

# ── Plot 3: Metrics bar chart (per mode × horizon) ──────────────────────────
bar_width = 0.35
x         = np.arange(len(HORIZONS))

# Optionally add Chronos-2 bars for cross-model comparison
show_c2 = chronos_bench is not None

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 10))
fig.suptitle(
    "TSMixer Metrics by Horizon & Mode" + (" (vs Chronos-2 target-only)" if show_c2 else ""),
    fontsize=14,
    fontweight="bold",
)

metric_defs = [("MSE", "mse"), ("RMSE", "rmse"), ("MAE", "mae"), ("R²", "r2")]

for ax, (metric_label, metric_key) in zip(axes.flatten(), metric_defs):
    col_label = "R²" if metric_label == "R²" else metric_label

    vals_none = [all_results[("none",     h)][metric_key] for h in HORIZONS]
    vals_cal  = [all_results[("calendar", h)][metric_key] for h in HORIZONS]

    if show_c2:
        # Three bars: TSMixer-none, TSMixer-calendar, Chronos-2-none
        offset = bar_width
        try:
            vals_c2 = [float(chronos_bench.loc[h, col_label]) for h in HORIZONS]
        except Exception:
            vals_c2 = [0.0] * len(HORIZONS)

        bars1 = ax.bar(x - offset, vals_none, bar_width, label="TSMixer Target-Only",   color="#8e44ad", alpha=0.85)
        bars2 = ax.bar(x,          vals_cal,  bar_width, label="TSMixer Calendar Cov.", color="#2980b9", alpha=0.85)
        bars3 = ax.bar(x + offset, vals_c2,   bar_width, label="Chronos-2 Target-Only", color="#e74c3c", alpha=0.85)
        all_bars = list(bars1) + list(bars2) + list(bars3)
        ax.set_xticks(x)
        ax.set_xticklabels([f"h={h}" for h in HORIZONS])
    else:
        bars1 = ax.bar(x - bar_width / 2, vals_none, bar_width, label="Target-Only",   color="#8e44ad", alpha=0.85)
        bars2 = ax.bar(x + bar_width / 2, vals_cal,  bar_width, label="Calendar Cov.", color="#2980b9", alpha=0.85)
        all_bars = list(bars1) + list(bars2)
        ax.set_xticks(x)
        ax.set_xticklabels([f"h={h}" for h in HORIZONS])

    ax.set_title(metric_label, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    for bar in all_bars:
        h_val = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            h_val,
            f"{h_val:.2f}",
            ha="center",
            va="bottom",
            fontsize=7,
            fontweight="bold",
        )

fig.tight_layout(rect=[0, 0, 1, 0.96])
save_fig(fig, "tsmixer_hourly_metrics_bars.png")

# ── Plot 4: Operational forecast (next 4 steps, P50/P70/P90) ───────────────
fig, ax = plt.subplots(figsize=(12, 5))
fig.suptitle(
    f"TSMixer Operational Forecast — Next {op_horizon}h from {series.end_time()}",
    fontsize=13,
    fontweight="bold",
)

context_ts = series[-48:]
context_ts.plot(ax=ax, label="Historical (last 48h)", color="#2ecc71", linewidth=2)

op_p50 = op_forecast.quantile(0.5)
op_p70 = op_forecast.quantile(0.7)
op_p90 = op_forecast.quantile(0.9)

op_p50.plot(ax=ax, label="Forecast P50", color="#8e44ad", linewidth=2.5)
op_p70.plot(ax=ax, label="Forecast P70", color="#9b59b6", linewidth=1.5, linestyle="--")
op_p90.plot(ax=ax, label="Forecast P90", color="#6c3483", linewidth=1.5, linestyle=":")

ax.fill_between(
    op_p50.time_index,
    op_p50.values().flatten(),
    op_p90.values().flatten(),
    alpha=0.2,
    color="#8e44ad",
    label="P50–P90 band",
)

ax.axvline(x=series.end_time(), color="gray", linestyle="--", linewidth=1, label="Now")
ax.set_ylabel("InUseCapacity (instances)")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
fig.tight_layout()
save_fig(fig, "tsmixer_hourly_operational_forecast.png")

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
print("\n" + "=" * 65)
print("DONE — all outputs saved to /outputs/")
print("=" * 65)
print(f"  tsmixer_hourly_metrics.csv")
print(f"  tsmixer_hourly_next_forecast.csv")
print(f"  tsmixer_hourly_backtest_none.png")
print(f"  tsmixer_hourly_mode_comparison.png")
print(f"  tsmixer_hourly_metrics_bars.png")
print(f"  tsmixer_hourly_operational_forecast.png")
