"""
TimesFM-2.5 Hourly Short-Term Forecaster for AWS AppStream Autoscaler
======================================================================
Goal      : Predict InUseCapacity h ∈ {2, 3, 4} steps (hours) ahead.
Frequency : Triggered every hour by the autoscaler daemon.
Library   : 100% Darts-native (TimeSeries, historical_forecasts, metrics, plots).

Covariate Strategy
------------------
TimesFM 2.5 does NOT support covariates natively (past or future).
  supports_past_covariates   = False
  supports_future_covariates = False

The Darts note states: "The source implementation uses Xreg to fit a ridge
regression between covariates and the target series (or forecast residuals)
as a pre/post-processing step."  We run pure target-only, which is the direct
apples-to-apples equivalent of the Chronos-2 "none" (target-only) mode.

Dropping (data-leakage or effect, not cause) — same policy as Chronos-2:
  - ActualCapacity       = AvailableCapacity + InUseCapacity  (linear combo)
  - AvailableCapacity    = ActualCapacity - InUseCapacity     (same leakage)
  - CapacityUtilization  = InUseCapacity / ActualCapacity     (derived from target)
  - DesiredCapacity      = native scaler output, not user-demand predictor
  - InsufficientCapacityError / InsufficientConcurrencyLimitError  (sparse)

Fair-comparison guarantees vs chronos2_8hours.py
-------------------------------------------------
  ✓ Same target column      : InUseCapacity
  ✓ Same input_chunk_length : 168  (7 days @ 1h — weekly seasonality)
  ✓ Same output_chunk_length: 8    (≥ max horizon, ≤ TimesFM hard limit of 128)
  ✓ Same test window        : 72h rolling backtest, stride=1
  ✓ Same horizons evaluated : h ∈ {2, 3, 4}
  ✓ Same num_samples        : 200 Monte-Carlo draws for probabilistic quantiles
  ✓ Same metrics            : MSE, RMSE, MAE, R²
  ✓ Same plot style         : backtest bands, operational forecast
  ✗ Covariates              : None (TimesFM 2.5 architectural constraint)
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
from darts.models import TimesFM2p5Model
from darts.utils.likelihood_models import QuantileRegression
from darts.utils.missing_values import fill_missing_values

logging.getLogger("cmdstanpy").setLevel(logging.CRITICAL)
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_PATH   = os.path.join(BASE_DIR, "..", "datasets", "Dati_processed.csv")
BASE_OUTPUTS_DIR = os.path.join(BASE_DIR, "..", "outputs")
SCRIPT_NAME = os.path.splitext(os.path.basename(__file__))[0].replace(" ", "")
OUTPUTS_DIR = os.path.join(BASE_OUTPUTS_DIR, SCRIPT_NAME)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

TARGET   = "InUseCapacity"
HORIZONS = [2, 3, 4]   # hours ahead to evaluate

# TimesFM-2.5 context window / single-pass output size
# Hard constraints from the model config:
#   context_limit      = 16 384
#   output_patch_len   = 128   → output_chunk_length MUST be ≤ 128
INPUT_CHUNK_LEN  = 168   # 7 days of hourly context (weekly seasonality)
OUTPUT_CHUNK_LEN = 8     # ≥ max(HORIZONS); same as Chronos-2 script

# Rolling-backtest window — identical to Chronos-2
TEST_HOURS  = 72    # 3 days
NUM_SAMPLES = 200   # Monte-Carlo draws for probabilistic quantiles

# Columns to drop from any covariate consideration (leakage / effect-not-cause)
LEAKAGE_COLS = {
    "ActualCapacity",
    "AvailableCapacity",
    "CapacityUtilization",
    "DesiredCapacity",
    "InsufficientCapacityError",
    "InsufficientConcurrencyLimitError",
}

# TimesFM 2.5 pre-trained quantiles (full set: 0.1 … 0.9 in steps of 0.1).
# We must use a strict subset; pick the same four that Chronos-2 uses.
TIMESFM_QUANTILES = [0.1, 0.3, 0.5, 0.7, 0.9]

# ---------------------------------------------------------------------------
# 1. Data Loading & Feature Engineering
# ---------------------------------------------------------------------------
print("=" * 65)
print("1. DATA LOADING & FEATURE ENGINEERING")
print("=" * 65)

df = pd.read_csv(DATA_PATH)
df["TimeStamp"] = pd.to_datetime(df["TimeStamp"])
df = df.sort_values("TimeStamp").reset_index(drop=True)

# Calendar features (informational / correlation analysis only —
# NOT passed to TimesFM as covariates since the model doesn't support them)
it_holidays = holidays.Italy()
df["hour"]        = df["TimeStamp"].dt.hour
df["day_of_week"] = df["TimeStamp"].dt.dayofweek
df["is_holiday"]  = df["TimeStamp"].apply(lambda x: 1 if x in it_holidays else 0)

print(f"  Dataset: {len(df)} hourly rows")
print(f"  Period : {df['TimeStamp'].iloc[0]}  →  {df['TimeStamp'].iloc[-1]}")
print(f"  Target : {TARGET}  (min={df[TARGET].min():.0f}, max={df[TARGET].max():.0f})")

# ---------------------------------------------------------------------------
# 2. Correlation Analysis (printed for transparency — same as Chronos-2)
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
print(
    f"  → All dropped: TimesFM 2.5 does not support covariates natively.\n"
    f"    (Model constraint: supports_past_covariates=False, "
    f"supports_future_covariates=False)"
)

# ---------------------------------------------------------------------------
# 3. TimeSeries Creation  — target series only (no covariate series needed)
# ---------------------------------------------------------------------------
print("\n" + "=" * 65)
print("3. BUILDING DARTS TIMESERIES")
print("=" * 65)

series = fill_missing_values(
    TimeSeries.from_dataframe(df, time_col="TimeStamp", value_cols=TARGET, freq="h")
)

print(f"  Target TimeSeries  : {len(series)} steps")
print(f"  No covariate series built (TimesFM 2.5 architectural constraint).")

# ---------------------------------------------------------------------------
# 4. Train / Test Split  
# ---------------------------------------------------------------------------
print("\n" + "=" * 65)
print("4. TRAIN / TEST SPLIT")
print("=" * 65)

test_start_idx = len(series) - TEST_HOURS
train_series   = series[:test_start_idx]
test_series    = series[test_start_idx:]

print(f"  Train : {len(train_series)} steps  ({len(train_series)//24} days)")
print(f"  Test  : {len(test_series)} steps  ({TEST_HOURS}h rolling backtest window)")
print(f"  Test starts at: {test_series.start_time()}")

# ---------------------------------------------------------------------------
# 5. Model Factory
# ---------------------------------------------------------------------------

def build_model() -> TimesFM2p5Model:
    """Instantiate TimesFM2p5Model with QuantileRegression likelihood.

    Key differences from Chronos-2:
    - hub_model_name defaults to 'google/timesfm-2.5-200m-pytorch'
    - Only QuantileRegression is supported as likelihood
    - Quantiles must be a strict subset of [0.1, 0.2, ..., 0.9]
    - No covariates accepted (past or future)
    - fit() is mandatory but training-free (weights are frozen / pre-trained)
    """
    return TimesFM2p5Model(
        input_chunk_length=INPUT_CHUNK_LEN,
        output_chunk_length=OUTPUT_CHUNK_LEN,
        likelihood=QuantileRegression(quantiles=TIMESFM_QUANTILES),
    )

# ---------------------------------------------------------------------------
# 6. Backtesting Loop — one single mode: target-only (no covariates possible)
# ---------------------------------------------------------------------------
print("\n" + "=" * 65)
print("5. BACKTESTING  (historical_forecasts, stride=1)")
print("=" * 65)
print(
    "  Mode: 'target-only'  (TimesFM 2.5 has no covariate support;\n"
    "        this is the fair equivalent of Chronos-2 mode='none')"
)

all_results    = {}   # h → dict
backtest_preds = {}   # h → dict

# Fit once, reuse across horizons — identical to Chronos-2 approach
model = build_model()
print(f"\n  Fitting TimesFM-2.5 on {len(train_series)} training steps…")
print(f"  (Note: TimesFM 2.5 is training-free; fit() only configures the model)")
model.fit(series=train_series)

for h in HORIZONS:
    print(f"  Backtesting h={h}…", end=" ", flush=True)

    hf_list = model.historical_forecasts(
        series=series,
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

    all_results[h] = {
        "mae":  mae_val,
        "mse":  mse_val,
        "rmse": rmse_val,
        "r2":   r2_val,
    }
    backtest_preds[h] = {
        "concat": hf_concat,
        "p50":    hf_p50,
        "actual": aligned_actual,
    }

# Save backtest P50 CSVs for cross-model line comparison
for h in HORIZONS:
    p50_s = backtest_preds[h]["p50"]
    act_s = backtest_preds[h]["actual"]
    out_df = pd.DataFrame({
        "InUseCapacity_P50":    p50_s.values().flatten(),
        "InUseCapacity_Actual": act_s.values().flatten(),
    }, index=p50_s.time_index)
    out_df.index.name = "TimeStamp"
    out_df.to_csv(os.path.join(OUTPUTS_DIR, f"timesfm2p5_backtest_p50_h{h}.csv"))
print("  Saved TimesFM-2.5 backtest P50 CSVs (timesfm2p5_backtest_p50_h*.csv)")

# ---------------------------------------------------------------------------
# 7. Load Chronos-2 Hourly Benchmark for direct comparison
# ---------------------------------------------------------------------------
chronos_bench_path = os.path.join(OUTPUTS_DIR, "chronos2_hourly_metrics.csv")
chronos_bench = None
if os.path.exists(chronos_bench_path):
    cdf = pd.read_csv(chronos_bench_path, index_col=[0, 1])
    # Extract the target-only ('none') mode rows, same horizons
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
for h, metrics in all_results.items():
    row = {
        "model":      "TimesFM-2.5",
        "horizon_h":  h,
        "MSE":  round(metrics["mse"],  4),
        "RMSE": round(metrics["rmse"], 4),
        "MAE":  round(metrics["mae"],  4),
        "R²":   round(metrics["r2"],   4),
    }
    rows.append(row)

metrics_df = pd.DataFrame(rows).set_index(["model", "horizon_h"])
print(metrics_df.to_string())

# Side-by-side comparison with Chronos-2 if available
if chronos_bench is not None:
    print("\n  ── Head-to-head vs Chronos-2 (target-only / mode='none') ──")
    header = f"  {'h':>3}  {'Metric':>6}  {'TimesFM-2.5':>12}  {'Chronos-2':>10}  {'Δ (TFM-C2)':>12}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for h in HORIZONS:
        for metric_label, metric_key in [("RMSE", "RMSE"), ("MAE", "MAE"), ("R²", "R²")]:
            tfm_val = all_results[h][metric_key.lower().replace("²", "2").replace("rmse","rmse").replace("mae","mae").replace("r²","r2")]
            if metric_label == "R²":
                tfm_val = all_results[h]["r2"]
            elif metric_label == "RMSE":
                tfm_val = all_results[h]["rmse"]
            else:
                tfm_val = all_results[h]["mae"]
            try:
                c2_val = float(chronos_bench.loc[h, metric_label])
                delta  = tfm_val - c2_val
                symbol = "↓ BETTER" if (metric_label == "R²" and delta > 0) or (metric_label != "R²" and delta < 0) else "↑ WORSE"
                print(f"  {h:>3}  {metric_label:>6}  {tfm_val:>12.4f}  {c2_val:>10.4f}  {delta:>+12.4f}  {symbol}")
            except Exception:
                print(f"  {h:>3}  {metric_label:>6}  {tfm_val:>12.4f}  {'N/A':>10}")

metrics_csv = os.path.join(OUTPUTS_DIR, "timesfm2p5_hourly_metrics.csv")
metrics_df.to_csv(metrics_csv)
print(f"\n  Saved: {metrics_csv}")

# ---------------------------------------------------------------------------
# 9. Operational Forecast — next h steps from the very last observation
# ---------------------------------------------------------------------------
print("\n" + "=" * 65)
print("7. OPERATIONAL FORECAST (next steps from now)")
print("=" * 65)

op_model = build_model()
print(f"  Fitting on full series ({len(series)} steps)…")
op_model.fit(series=series)

op_horizon  = max(HORIZONS)   # predict up to longest horizon = 4 steps
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

forecast_csv = os.path.join(OUTPUTS_DIR, "timesfm2p5_hourly_next_forecast.csv")
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


# ── Plot 1: Rolling Backtest — one subplot per horizon ─────────────────────
fig, axes = plt.subplots(nrows=len(HORIZONS), ncols=1, figsize=(18, 5 * len(HORIZONS)), sharex=True)
fig.suptitle("TimesFM-2.5 Short-Term Backtesting (target-only)", fontsize=14, fontweight="bold")

for ax, h in zip(axes, HORIZONS):
    preds  = backtest_preds[h]
    actual = preds["actual"]
    p50    = preds["p50"]
    full   = preds["concat"]

    p10 = full.quantile(0.1)
    p90 = full.quantile(0.9)

    actual.plot(ax=ax, label="Actual", color="#2ecc71", linewidth=2)
    p50.plot(ax=ax, label=f"Forecast P50 (h={h})", color="#e67e22", linewidth=1.5)

    ax.fill_between(
        p10.time_index,
        p10.values().flatten(),
        p90.values().flatten(),
        alpha=0.25,
        color="#e67e22",
        label="P10–P90 band",
    )

    m = all_results[h]
    ax.set_title(
        f"h={h}h  |  MSE={m['mse']:.2f}   RMSE={m['rmse']:.2f}   "
        f"MAE={m['mae']:.2f}   R²={m['r2']:.4f}",
        fontsize=10,
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

fig.tight_layout(rect=[0, 0, 1, 0.97])
save_fig(fig, "timesfm2p5_hourly_backtest.png")

# ── Plot 2: Metrics bar chart (MSE / RMSE / MAE / R² per horizon) ──────────
bar_width = 0.35
x = np.arange(len(HORIZONS))

# Build Chronos-2 comparison arrays if available
labels_bar     = ["TimesFM-2.5"]
colors_bar     = ["#e67e22"]
metric_arrays  = {k: [all_results[h][k] for h in HORIZONS] for k in ["mse", "rmse", "mae", "r2"]}

if chronos_bench is not None:
    labels_bar.append("Chronos-2")
    colors_bar.append("#e74c3c")
    for mk, col in [("mse", "MSE"), ("rmse", "RMSE"), ("mae", "MAE"), ("r2", "R²")]:
        try:
            metric_arrays[mk + "_c2"] = [float(chronos_bench.loc[h, col]) for h in HORIZONS]
        except Exception:
            metric_arrays[mk + "_c2"] = [0.0] * len(HORIZONS)

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 10))
fig.suptitle("TimesFM-2.5 Metrics by Horizon" + (" vs Chronos-2" if chronos_bench is not None else ""),
             fontsize=14, fontweight="bold")

metric_defs = [("MSE", "mse"), ("RMSE", "rmse"), ("MAE", "mae"), ("R²", "r2")]

for ax, (metric_label, metric_key) in zip(axes.flatten(), metric_defs):
    vals_tfm = metric_arrays[metric_key]
    bars1 = ax.bar(
        x - bar_width / 2 if chronos_bench is not None else x,
        vals_tfm, bar_width, label="TimesFM-2.5", color="#e67e22", alpha=0.85
    )

    if chronos_bench is not None:
        vals_c2 = metric_arrays.get(metric_key + "_c2", [0.0] * len(HORIZONS))
        bars2 = ax.bar(x + bar_width / 2, vals_c2, bar_width, label="Chronos-2", color="#e74c3c", alpha=0.85)
        all_bars = list(bars1) + list(bars2)
    else:
        all_bars = list(bars1)

    ax.set_title(metric_label, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"h={h}" for h in HORIZONS])
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
save_fig(fig, "timesfm2p5_hourly_metrics_bars.png")

# ── Plot 4: Operational forecast (next 4 steps, P50/P70/P90) ───────────────
fig, ax = plt.subplots(figsize=(12, 5))
fig.suptitle(
    f"TimesFM-2.5 Operational Forecast — Next {op_horizon}h from {series.end_time()}",
    fontsize=13,
    fontweight="bold",
)

context_ts = series[-48:]
context_ts.plot(ax=ax, label="Historical (last 48h)", color="#2ecc71", linewidth=2)

op_p50 = op_forecast.quantile(0.5)
op_p70 = op_forecast.quantile(0.7)
op_p90 = op_forecast.quantile(0.9)

op_p50.plot(ax=ax, label="Forecast P50", color="#e67e22", linewidth=2.5)
op_p70.plot(ax=ax, label="Forecast P70", color="#d35400", linewidth=1.5, linestyle="--")
op_p90.plot(ax=ax, label="Forecast P90", color="#9b59b6", linewidth=1.5, linestyle=":")

ax.fill_between(
    op_p50.time_index,
    op_p50.values().flatten(),
    op_p90.values().flatten(),
    alpha=0.2,
    color="#9b59b6",
    label="P50–P90 band",
)

ax.axvline(x=series.end_time(), color="gray", linestyle="--", linewidth=1, label="Now")
ax.set_ylabel("InUseCapacity (instances)")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
fig.tight_layout()
save_fig(fig, "timesfm2p5_hourly_operational_forecast.png")

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
print("\n" + "=" * 65)
print("DONE — all outputs saved to /outputs/")
print("=" * 65)
print(f"  timesfm2p5_hourly_metrics.csv")
print(f"  timesfm2p5_hourly_next_forecast.csv")
print(f"  timesfm2p5_hourly_backtest.png")

print(f"  timesfm2p5_hourly_metrics_bars.png")
print(f"  timesfm2p5_hourly_operational_forecast.png")