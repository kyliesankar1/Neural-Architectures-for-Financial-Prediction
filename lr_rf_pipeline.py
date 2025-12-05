"""
This script loads the universe, fetches price data,
builds features, trains Logistic Regression and Random Forest models,
performs walk-forward validation, and generates comprehensive performance analysis
including predictive and economic performance metrics.
"""

import pandas as pd
import numpy as np
from pathlib import Path

from src.data.panel import (
    build_adj_close_panel,
    compute_returns,
    compute_momentum,
    compute_volatility,
    rank_cross_sectional,
)
from src.models.logistic_regression import train_logistic_regression
from src.models.random_forest import train_random_forest
from src.evaluation.analysis import (
    get_sklearn_feature_importance,
    walk_forward_validation,
    compute_daily_accuracy,
    plot_daily_accuracy,
    plot_rolling_accuracy,
    backtest_long_short_from_proba,
    plot_equity_curve,
)


# load the universe of stocks
universe_path = Path("data/universe/us_universe_full_filtered.csv")
universe = pd.read_csv(universe_path)
tickers = universe["ticker"].dropna().unique().tolist()

print(f"Loaded {len(tickers)} tickers from {universe_path}")
print("First few tickers:", tickers[:10])


# load prices (cached after first run)
prices = build_adj_close_panel(
    tickers,
    start="2014-01-01",
    end="2020-12-31",
)

print("Price panel shape:", prices.shape)

if prices.empty:
    raise SystemExit(
        "ERROR: prices DataFrame is empty.\n"
        "This means no price history was loaded for your tickers in the "
        "date range 2014-01-01 to 2020-12-31.\n"
        "Check your cached price files or adjust the date range."
    )

print("First few rows of prices:\n", prices.head())
print("First few columns (tickers):", list(prices.columns)[:10])

# create basic features
returns_1d = compute_returns(prices, periods=1)
momentum_126d = compute_momentum(prices, lookback=126)
vol_20d = compute_volatility(returns_1d, window=20, annualize=True)
mom_rank = rank_cross_sectional(momentum_126d)

# combine into one long DataFrame (date × ticker)
dataset = pd.DataFrame({
    "ret_1d": returns_1d.stack(),
    "momentum_126d": momentum_126d.stack(),
    "vol_20d": vol_20d.stack(),
    "mom_rank": mom_rank.stack(),
})

dataset.index.names = ["date", "ticker"]
dataset = dataset.dropna()


# make target variables
y_class = (returns_1d.shift(-1).stack() > 0).astype(int)   # up/down
y_class.name = "target"

y_reg = returns_1d.shift(-1).stack()                       # next-day return
y_reg.name = "future_return"

# join everything together
df = dataset.join([y_class, y_reg]).reset_index()
df["date"] = pd.to_datetime(df["date"])

# make dates timezone-naive so we can compare to split_date
df["date"] = df["date"].dt.tz_localize(None)

# these are the features I'm using in the models
feature_cols = ["ret_1d", "momentum_126d", "vol_20d", "mom_rank"]

# drop any rows where features or targets are NaN
df = df.dropna(subset=feature_cols + ["target", "future_return"])

print("Final dataset shape after dropping NaNs:", df.shape)


# ============================================================================
# WALK-FORWARD VALIDATION
# ============================================================================

print("\n" + "=" * 80)
print("WALK-FORWARD VALIDATION")
print("=" * 80)

# Define split dates for walk-forward validation
split_dates = [
    "2017-01-01",
    "2018-01-01",
    "2019-01-01",
    "2020-01-01",
    "2021-01-01",
]

# Walk-forward validation for Logistic Regression
print("\n" + "-" * 80)
print("Logistic Regression - Walk-Forward Validation")
print("-" * 80)
lr_wf_test, lr_wf_metrics = walk_forward_validation(
    df,
    train_logistic_regression,
    feature_cols=feature_cols,
    target_col="target",
    date_col="date",
    split_dates=split_dates,
)

# Walk-forward validation for Random Forest
print("\n" + "-" * 80)
print("Random Forest - Walk-Forward Validation")
print("-" * 80)
rf_wf_test, rf_wf_metrics = walk_forward_validation(
    df,
    train_random_forest,
    feature_cols=feature_cols,
    target_col="target",
    date_col="date",
    split_dates=split_dates,
)


# ============================================================================
# SUMMARY METRICS TABLE
# ============================================================================

print("\n" + "=" * 80)
print("SUMMARY METRICS TABLE")
print("=" * 80)

summary_metrics = pd.DataFrame({
    "Logistic Regression": lr_wf_metrics,
    "Random Forest": rf_wf_metrics,
})
summary_metrics.index.name = "Metric"

print("\nWalk-Forward Validation Summary:")
print(summary_metrics.round(4))


# ============================================================================
# DAILY ACCURACY ANALYSIS
# ============================================================================

print("\n" + "=" * 80)
print("DAILY ACCURACY ANALYSIS")
print("=" * 80)

# Compute daily accuracy for both models
lr_daily_acc = compute_daily_accuracy(lr_wf_test, date_col="date", y_true_col="y_true", y_proba_col="y_proba")
rf_daily_acc = compute_daily_accuracy(rf_wf_test, date_col="date", y_true_col="y_true", y_proba_col="y_proba")

print(f"\nLogistic Regression - Mean daily accuracy: {lr_daily_acc['daily_accuracy'].mean():.4f}")
print(f"Random Forest - Mean daily accuracy: {rf_daily_acc['daily_accuracy'].mean():.4f}")


# ============================================================================
# ECONOMIC PERFORMANCE: LONG-SHORT BACKTEST
# ============================================================================

print("\n" + "=" * 80)
print("ECONOMIC PERFORMANCE: LONG-SHORT BACKTEST")
print("=" * 80)

# Ensure future_return is in test predictions (it should already be there from df)
if "future_return" not in lr_wf_test.columns:
    lr_wf_test = lr_wf_test.merge(
        df[["date", "ticker", "future_return"]],
        on=["date", "ticker"],
        how="left",
    )
if "future_return" not in rf_wf_test.columns:
    rf_wf_test = rf_wf_test.merge(
        df[["date", "ticker", "future_return"]],
        on=["date", "ticker"],
        how="left",
    )

# Long-short backtest using probability scores
# Use smaller k since we may not have many stocks per day
k = 5  # Number of stocks to long/short (reduced from 20)
lr_backtest = backtest_long_short_from_proba(
    lr_wf_test,
    date_col="date",
    proba_col="y_proba",
    future_return_col="future_return",
    k=k,
)

rf_backtest = backtest_long_short_from_proba(
    rf_wf_test,
    date_col="date",
    proba_col="y_proba",
    future_return_col="future_return",
    k=k,
)

print(f"\nLogistic Regression Long-Short Strategy:")
print(f"  Sharpe Ratio: {lr_backtest.attrs.get('sharpe_estimate', np.nan):.4f}")
print(f"  Total Return: {lr_backtest['cum_return'].iloc[-1]:.4f}" if not lr_backtest.empty else "  No data")

print(f"\nRandom Forest Long-Short Strategy:")
print(f"  Sharpe Ratio: {rf_backtest.attrs.get('sharpe_estimate', np.nan):.4f}")
print(f"  Total Return: {rf_backtest['cum_return'].iloc[-1]:.4f}" if not rf_backtest.empty else "  No data")


# ============================================================================
# SAVE RESULTS AND GENERATE PLOTS
# ============================================================================

out = Path("results")
out.mkdir(exist_ok=True)

# Save predictions and metrics
lr_wf_test.to_csv(out / "lr_walkforward_predictions.csv", index=False)
rf_wf_test.to_csv(out / "rf_walkforward_predictions.csv", index=False)
summary_metrics.to_csv(out / "lr_rf_summary_metrics.csv")

# Save backtest results
if not lr_backtest.empty:
    lr_backtest.to_csv(out / "lr_backtest_results.csv", index=False)
if not rf_backtest.empty:
    rf_backtest.to_csv(out / "rf_backtest_results.csv", index=False)

print("\n" + "=" * 80)
print("GENERATING VISUALIZATIONS")
print("=" * 80)

# Plot daily accuracy
plot_daily_accuracy(
    lr_daily_acc,
    save_path=out / "lr_daily_accuracy.png",
    title="Logistic Regression - Daily Accuracy Over Time",
)
print("Saved: lr_daily_accuracy.png")

plot_daily_accuracy(
    rf_daily_acc,
    save_path=out / "rf_daily_accuracy.png",
    title="Random Forest - Daily Accuracy Over Time",
)
print("Saved: rf_daily_accuracy.png")

# Plot rolling accuracy (30-day window)
plot_rolling_accuracy(
    lr_daily_acc,
    window=30,
    save_path=out / "lr_rolling_accuracy.png",
    title="Logistic Regression - Rolling Accuracy (30-day window)",
)
print("Saved: lr_rolling_accuracy.png")

plot_rolling_accuracy(
    rf_daily_acc,
    window=30,
    save_path=out / "rf_rolling_accuracy.png",
    title="Random Forest - Rolling Accuracy (30-day window)",
)
print("Saved: rf_rolling_accuracy.png")

# Plot equity curves
if not lr_backtest.empty:
    plot_equity_curve(
        lr_backtest,
        save_path=out / "lr_equity_curve.png",
        title="Logistic Regression - Long-Short Equity Curve",
    )
    print("Saved: lr_equity_curve.png")

if not rf_backtest.empty:
    plot_equity_curve(
        rf_backtest,
        save_path=out / "rf_equity_curve.png",
        title="Random Forest - Long-Short Equity Curve",
    )
    print("Saved: rf_equity_curve.png")

# Combined comparison plot
import matplotlib.pyplot as plt

# Combined daily accuracy plot
plt.figure(figsize=(14, 6))
plt.plot(lr_daily_acc["date"], lr_daily_acc["daily_accuracy"], alpha=0.7, label="Logistic Regression", linewidth=1)
plt.plot(rf_daily_acc["date"], rf_daily_acc["daily_accuracy"], alpha=0.7, label="Random Forest", linewidth=1)
plt.axhline(y=0.5, color="r", linestyle="--", label="Random (0.5)")
plt.xlabel("Date")
plt.ylabel("Daily Accuracy")
plt.title("Daily Accuracy Comparison")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(out / "combined_daily_accuracy.png", dpi=150)
plt.close()
print("Saved: combined_daily_accuracy.png")

# Combined equity curve plot
if not lr_backtest.empty and not rf_backtest.empty:
    plt.figure(figsize=(14, 6))
    plt.plot(lr_backtest["date"], lr_backtest["cum_return"], label=f"Logistic Regression (Sharpe: {lr_backtest.attrs.get('sharpe_estimate', np.nan):.2f})", linewidth=2)
    plt.plot(rf_backtest["date"], rf_backtest["cum_return"], label=f"Random Forest (Sharpe: {rf_backtest.attrs.get('sharpe_estimate', np.nan):.2f})", linewidth=2)
    plt.axhline(y=0, color="k", linestyle="--", alpha=0.5)
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.title("Long-Short Equity Curve Comparison")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / "combined_equity_curve.png", dpi=150)
    plt.close()
    print("Saved: combined_equity_curve.png")


print("\n" + "=" * 80)
print("PIPELINE COMPLETE!")
print("=" * 80)
print("\nAll results saved to 'results/' folder:")
print("  - lr_walkforward_predictions.csv")
print("  - rf_walkforward_predictions.csv")
print("  - lr_rf_summary_metrics.csv")
print("  - lr_backtest_results.csv")
print("  - rf_backtest_results.csv")
print("  - lr_daily_accuracy.png")
print("  - rf_daily_accuracy.png")
print("  - lr_rolling_accuracy.png")
print("  - rf_rolling_accuracy.png")
print("  - lr_equity_curve.png")
print("  - rf_equity_curve.png")
print("  - combined_daily_accuracy.png")
print("  - combined_equity_curve.png")
