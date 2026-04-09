"""
Chronos-2 Hourly Short-Term Forecaster for AWS AppStream Autoscaler
====================================================================
Goal      : Predict InUseCapacity h ∈ {2, 3, 4} steps (hours) ahead.
Frequency : Triggered every hour by the autoscaler daemon.
Library   : 100% Darts-native (TimeSeries, historical_forecasts, metrics, plots).

Covariate Strategy
------------------
Dropped (data-leakage or effect, not cause):
  - ActualCapacity       = AvailableCapacity + InUseCapacity  (linear combo)
  - AvailableCapacity    = ActualCapacity - InUseCapacity     (same leakage)
  - CapacityUtilization  = InUseCapacity / ActualCapacity     (derived from target)
  - DesiredCapacity      = native scaler output, not user-demand predictor
  - InsufficientCapacityError / InsufficientConcurrencyLimitError  (sparse, consequential)

Two modes compared:
  "none"     → target-only (pure Chronos-2 zero-shot, fastest)  [default]
  "calendar" → + hour / day_of_week / is_holiday as future covariates
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
from darts.metrics import mae, smape, rmse, r2_score
from darts.models import Chronos2Model
from darts.utils.likelihood_models import QuantileRegression
from darts.utils.missing_values import fill_missing_values

logging.getLogger("cmdstanpy").setLevel(logging.CRITICAL)
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_PATH   = os.path.join(BASE_DIR, "..", "datasets", "Dati_processed.csv")
OUTPUTS_DIR = os.path.join(BASE_DIR, "..", "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

TARGET          = "InUseCapacity"
HORIZONS        = [2, 3, 4]           # hours ahead to evaluate
COVARIATE_MODES = ["none", "calendar"] # both modes are benchmarked

# Chronos-2 context window / single-pass output size
INPUT_CHUNK_LEN  = 168   # 7 days of hourly context (weekly seasonality)
OUTPUT_CHUNK_LEN = 8     # ≥ max(HORIZONS) — no auto-regression needed

# Rolling-backtest window (last N hours used as held-out test)
TEST_HOURS = 72   # 3 days
NUM_SAMPLES = 200  # Monte-Carlo samples for probabilistic quantiles

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

# Calendar features
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

# Identify columns that survive leakage filter (informational only — we enforce
# the LEAKAGE_COLS exclusion list regardless of correlation magnitude)
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
last_ts  = df["TimeStamp"].iloc[-1]
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

print(f"  Train : {len(train_series)} steps  ({len(train_series)//24} days)")
print(f"  Test  : {len(test_series)} steps  ({TEST_HOURS}h rolling backtest window)")
print(f"  Test starts at: {test_series.start_time()}")

# ---------------------------------------------------------------------------
# 5. Model Factory
# ---------------------------------------------------------------------------

def build_model():
    """Instantiate Chronos2Model with QuantileRegression likelihood.

    Passing a QuantileRegression instance enables native probabilistic output
    so that num_samples > 1 and quantile_timeseries() both work correctly.
    Quantiles must be a subset of those used during Chronos-2 pre-training:
    [0.1, 0.2, ..., 0.9]. We use the four we need for P10/P50/P70/P90.
    """
    return Chronos2Model(
        input_chunk_length=INPUT_CHUNK_LEN,
        output_chunk_length=OUTPUT_CHUNK_LEN,
        hub_model_name="amazon/chronos-2",
        likelihood=QuantileRegression(quantiles=[0.1, 0.3, 0.5, 0.7, 0.9]),
    )

# ---------------------------------------------------------------------------
# 6. Backtesting Loop — one run per (covariate_mode, horizon)
# ---------------------------------------------------------------------------
print("\n" + "=" * 65)
print("5. BACKTESTING  (historical_forecasts, stride=1)")
print("=" * 65)

all_results   = {}   # (mode, h) → dict
backtest_preds = {}  # (mode, h) → concatenated TimeSeries

for mode in COVARIATE_MODES:
    print(f"\n  ── Mode: '{mode}' ──")

    # Fit once per mode, reuse across horizons
    model = build_model()

    fit_kwargs = {}
    predict_kwargs = {}
    if mode == "calendar":
        fit_kwargs["future_covariates"]     = future_cov
        predict_kwargs["future_covariates"] = future_cov

    print(f"    Fitting Chronos-2 on {len(train_series)} training steps…")
    model.fit(series=train_series, **fit_kwargs)

    for h in HORIZONS:
        print(f"    Backtesting h={h}…", end=" ", flush=True)

        # With QuantileRegression likelihood, historical_forecasts is fully
        # probabilistic and supports num_samples > 1 normally.
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

        # Concatenate the list into a single probabilistic TimeSeries
        hf_concat  = concatenate(hf_list)
        hf_p50     = hf_concat.quantile(0.5)   # point estimate for metrics

        # Align actuals to the forecast index
        aligned_actual = series.slice_intersect(hf_p50)

        mae_val  = mae( aligned_actual, hf_p50)
        mape_val = smape(aligned_actual, hf_p50)
        rmse_val = rmse(aligned_actual, hf_p50)
        r2_val   = r2_score(aligned_actual, hf_p50)

        print(f"sMAPE={mape_val:.2f}%  RMSE={rmse_val:.2f}  MAE={mae_val:.2f}  R\u00b2={r2_val:.4f}")

        key = (mode, h)
        all_results[key] = {
            "mae":  mae_val,
            "mape": mape_val,
            "rmse": rmse_val,
            "r2":   r2_val,
        }
        backtest_preds[key] = {
            "concat": hf_concat,    # full probabilistic series for band plots
            "p50":    hf_p50,       # P50 line for metrics
            "actual": aligned_actual,
        }

# ---------------------------------------------------------------------------
# 7. Load 90-Day Benchmark (if available) for comparison
# ---------------------------------------------------------------------------
benchmark_path = os.path.join(OUTPUTS_DIR, "darts_metrics_summary.csv")
benchmark_rmse = None
if os.path.exists(benchmark_path):
    bm_df = pd.read_csv(benchmark_path, index_col=0)
    if "Chronos-2" in bm_df.index and "RMSE" in bm_df.columns:
        benchmark_rmse = bm_df.loc["Chronos-2", "RMSE"]
        print(f"\n  90-Day Benchmark (Chronos-2 RMSE): {benchmark_rmse:.4f}")

# ---------------------------------------------------------------------------
# 8. Metrics Summary Table
# ---------------------------------------------------------------------------
print("\n" + "=" * 65)
print("6. METRICS SUMMARY")
print("=" * 65)

rows = []
for (mode, h), metrics in all_results.items():
    rows.append({
        "mode":  mode,
        "horizon_h": h,
        "sMAPE (%)": round(metrics["mape"], 4),
        "RMSE":     round(metrics["rmse"], 4),
        "MAE":      round(metrics["mae"],  4),
        "R²":       round(metrics["r2"],   4),
    })

metrics_df = pd.DataFrame(rows).set_index(["mode", "horizon_h"])
print(metrics_df.to_string())

if benchmark_rmse is not None:
    print(f"\n  Reference 90-Day Benchmark Chronos-2 RMSE : {benchmark_rmse:.4f}")
    best_rmse = metrics_df["RMSE"].min()
    delta  = best_rmse - benchmark_rmse
    symbol = "✓ BETTER" if delta < 0 else "✗ WORSE"
    print(f"  Best short-term RMSE (h∈{{2,3,4}})        : {best_rmse:.4f}  ({delta:+.4f})  {symbol}")

metrics_csv = os.path.join(OUTPUTS_DIR, "chronos2_hourly_metrics.csv")
metrics_df.to_csv(metrics_csv)
print(f"\n  Saved: {metrics_csv}")

# ---------------------------------------------------------------------------
# 9. Operational Forecast — next h steps from the very last observation
# ---------------------------------------------------------------------------
print("\n" + "=" * 65)
print("7. OPERATIONAL FORECAST (next steps from now)")
print("=" * 65)

# Use the 'none' (target-only) model as the operational default (fastest, no lookahead needed)
op_model = build_model()
op_model.fit(series=series)

op_horizon = max(HORIZONS)   # predict up to the longest horizon = 4 steps
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

forecast_csv = os.path.join(OUTPUTS_DIR, "chronos2_hourly_next_forecast.csv")
df_forecast.to_csv(forecast_csv)
print(f"  Saved : {forecast_csv}")
print(df_forecast.to_string())

# ---------------------------------------------------------------------------
# 10. Plots (Darts-native + matplotlib overlay)
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
fig.suptitle("Chronos-2 Short-Term Backtesting (target-only mode)", fontsize=14, fontweight="bold")

for ax, h in zip(axes, HORIZONS):
    key    = ("none", h)
    preds  = backtest_preds[key]
    actual = preds["actual"]
    p50    = preds["p50"]
    full   = preds["concat"]

    p10 = full.quantile(0.1)
    p90 = full.quantile(0.9)

    # Use Darts .plot() then overlay quantile band via matplotlib
    actual.plot(ax=ax, label="Actual", color="#2ecc71", linewidth=2)
    p50.plot(ax=ax, label=f"Forecast P50 (h={h})", color="#e74c3c", linewidth=1.5)

    # Confidence band
    ax.fill_between(
        p10.time_index,
        p10.values().flatten(),
        p90.values().flatten(),
        alpha=0.25,
        color="#e74c3c",
        label="P10–P90 band",
    )

    m = all_results[key]
    ax.set_title(
        f"h={h}h  |  MAPE={m['mape']:.2f}%   RMSE={m['rmse']:.2f}   "
        f"MAE={m['mae']:.2f}   R²={m['r2']:.4f}",
        fontsize=10,
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

fig.tight_layout(rect=[0, 0, 1, 0.97])
save_fig(fig, "chronos2_hourly_backtest_none.png")

# ── Plot 2: Mode Comparison per horizon ────────────────────────────────────
fig, axes = plt.subplots(nrows=len(HORIZONS), ncols=1, figsize=(18, 5 * len(HORIZONS)), sharex=True)
fig.suptitle("Chronos-2: target-only vs calendar-covariates", fontsize=14, fontweight="bold")

mode_styles = {"none": ("#e74c3c", "Target-Only"), "calendar": ("#3498db", "Calendar Cov.")}

for ax, h in zip(axes, HORIZONS):
    actual = backtest_preds[("none", h)]["actual"]
    actual.plot(ax=ax, label="Actual", color="#2ecc71", linewidth=2)

    for mode, (color, label) in mode_styles.items():
        key = (mode, h)
        m   = all_results[key]
        backtest_preds[key]["p50"].plot(
            ax=ax,
            label=f"{label}  sMAPE={m['mape']:.2f}%  R\u00b2={m['r2']:.4f}",
            color=color,
            linewidth=1.5,
        )

    ax.set_title(f"Horizon h={h}h — mode comparison", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

fig.tight_layout(rect=[0, 0, 1, 0.97])
save_fig(fig, "chronos2_hourly_mode_comparison.png")

# ── Plot 3: Metrics bar chart (MAPE per mode × horizon) ────────────────────
bar_width = 0.35
x = np.arange(len(HORIZONS))

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 10))
fig.suptitle("Metrics by Horizon & Covariate Mode", fontsize=14, fontweight="bold")

metric_keys = [("MAPE (%)", "mape"), ("RMSE", "rmse"), ("MAE", "mae"), ("R²", "r2")]

for ax, (metric_label, metric_key) in zip(axes.flatten(), metric_keys):
    vals_none = [all_results[("none", h)][metric_key] for h in HORIZONS]
    vals_cal  = [all_results[("calendar", h)][metric_key] for h in HORIZONS]

    bars1 = ax.bar(x - bar_width / 2, vals_none, bar_width, label="Target-Only",  color="#e74c3c", alpha=0.85)
    bars2 = ax.bar(x + bar_width / 2, vals_cal,  bar_width, label="Calendar Cov.", color="#3498db", alpha=0.85)

    ax.set_title(metric_label, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"h={h}" for h in HORIZONS])
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # Value labels on bars
    for bar in list(bars1) + list(bars2):
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
save_fig(fig, "chronos2_hourly_metrics_bars.png")

# ── Plot 4: Operational forecast (next 4 steps, P50/P70/P90) ───────────────
fig, ax = plt.subplots(figsize=(12, 5))
fig.suptitle(
    f"Operational Forecast — Next {op_horizon}h from {series.end_time()}",
    fontsize=13,
    fontweight="bold",
)

# Context: last 48h of actual data plotted with Darts
context_ts = series[-48:]
context_ts.plot(ax=ax, label="Historical (last 48h)", color="#2ecc71", linewidth=2)

# Forecast quantile bands via Darts
op_p50 = op_forecast.quantile(0.5)
op_p70 = op_forecast.quantile(0.7)
op_p90 = op_forecast.quantile(0.9)

op_p50.plot(ax=ax, label="Forecast P50", color="#e74c3c",  linewidth=2.5)
op_p70.plot(ax=ax, label="Forecast P70", color="#e67e22",  linewidth=1.5, linestyle="--")
op_p90.plot(ax=ax, label="Forecast P90", color="#9b59b6",  linewidth=1.5, linestyle=":")

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
save_fig(fig, "chronos2_hourly_operational_forecast.png")

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
print("\n" + "=" * 65)
print("DONE — all outputs saved to /outputs/")
print("=" * 65)
print(f"  chronos2_hourly_metrics.csv")
print(f"  chronos2_hourly_next_forecast.csv")
print(f"  chronos2_hourly_backtest_none.png")
print(f"  chronos2_hourly_mode_comparison.png")
print(f"  chronos2_hourly_metrics_bars.png")
print(f"  chronos2_hourly_operational_forecast.png")