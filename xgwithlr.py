import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Project helpers
from src.data.panel import (
    build_adj_close_panel,
    compute_returns,
    compute_momentum,
    compute_volatility,
    rank_cross_sectional,
)

from src.models.xgb_classifier import train_xgb_classifier
from src.models.xgb_ranker import train_xgb_ranker
from src.evaluation.analysis import backtest_long_short


# =========================================================
# CONFIG
# =========================================================

START = "2014-01-01"
END = "2020-12-31"
SPLIT = "2018-01-01"        # train < SPLIT, test >= SPLIT

result_dir = Path("results")
result_dir.mkdir(exist_ok=True)


# =========================================================
# 1. LOAD UNIVERSE
# =========================================================

universe_path = Path("data/universe/us_universe_full_filtered.csv")
universe = pd.read_csv(universe_path)
tickers = universe["ticker"].dropna().unique().tolist()

print(f"Loaded {len(tickers)} tickers.")


# =========================================================
# 2. LOAD PRICE PANEL
# =========================================================

prices = build_adj_close_panel(tickers, start=START, end=END)
if prices.empty:
    raise ValueError("ERROR: Could not load price panel.")

print("Price panel shape:", prices.shape)


# =========================================================
# 3. FEATURE ENGINEERING
# =========================================================

returns_1d = compute_returns(prices, periods=1)
momentum_126d = compute_momentum(prices, lookback=126)
vol_20d = compute_volatility(returns_1d, window=20, annualize=True)
mom_rank = rank_cross_sectional(momentum_126d)

dataset = pd.DataFrame({
    "ret_1d": returns_1d.stack(),
    "momentum_126d": momentum_126d.stack(),
    "vol_20d": vol_20d.stack(),
    "mom_rank": mom_rank.stack(),
})

dataset.index.names = ["date", "ticker"]
dataset = dataset.dropna()


# =========================================================
# 4. TARGET VARIABLES
# =========================================================

y_class = (returns_1d.shift(-1).stack() > 0).astype(int)
y_class.name = "target"

y_reg = returns_1d.shift(-1).stack()
y_reg.name = "future_return"

df = dataset.join([y_class, y_reg]).reset_index()
df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)

# Base features
feature_cols = ["ret_1d", "momentum_126d", "vol_20d", "mom_rank"]

# Clean data
df = df.dropna(subset=feature_cols + ["target", "future_return"]).copy()
print("Final dataset shape:", df.shape)


# =========================================================
# 5. TRAIN/TEST SPLIT
# =========================================================

train_df = df[df["date"] < SPLIT].copy()
test_df  = df[df["date"] >= SPLIT].copy()

print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")


# =========================================================
# 6. TRAIN LINEAR REGRESSION ON TRAIN ONLY (NO LEAKAGE)
# =========================================================

print("\nTraining Linear Regression (NO LEAK)...")

lr = LinearRegression()
lr.fit(train_df[feature_cols], train_df["future_return"])

# Predict on ENTIRE dataset
df["lr_pred"] = lr.predict(df[feature_cols])

# Add to features
feature_cols = feature_cols + ["lr_pred"]

print("Updated feature set:", feature_cols)


# =========================================================
# 7. Train XGBOOST CLASSIFIER
# =========================================================

clf_model, clf_metrics, clf_test = train_xgb_classifier(
    df=df,
    feature_cols=feature_cols,
    target_col="target",
    date_col="date",
    split_date=SPLIT,
)

print("\n=== CLASSIFIER METRICS ===")
for k, v in clf_metrics.items():
    print(f"{k}: {v:.4f}")

# Feature importance
clf_importance = clf_model.get_booster().get_score(importance_type="gain")
fi_clf = pd.DataFrame(
    clf_importance.items(), columns=["feature", "gain"]
).sort_values("gain", ascending=False)

print("\nClassifier Feature Importance:")
print(fi_clf)


# =========================================================
# 8. TRAIN XGBOOST RANKER
# =========================================================

rank_model, rank_test = train_xgb_ranker(
    df=df,
    feature_cols=feature_cols,
    target_col="future_return",
    date_col="date",
    split_date=SPLIT,
)

rank_importance = rank_model.get_booster().get_score(importance_type="gain")
fi_rank = pd.DataFrame(
    rank_importance.items(), columns=["feature", "gain"]
).sort_values("gain", ascending=False)

print("\nRanker Feature Importance:")
print(fi_rank)


# =========================================================
# 9. LONG–SHORT BACKTEST
# =========================================================

bt = backtest_long_short(
    df=rank_test,
    date_col="date",
    score_col="rank_score",
    future_return_col="future_return",
    k=5,
)

sharpe = bt.attrs["sharpe_estimate"]
print(f"\n=== BACKTEST SHARPE === {sharpe:.4f}")
print(bt.tail())


# =========================================================
# 10. SAVE EVERYTHING
# =========================================================

save_map = {
    "clf_predictions_xg_lr.csv": clf_test,
    "rank_predictions_xg_lr.csv": rank_test,
    "backtest_xg_lr.csv": bt,
    "fi_classifier_xg_lr.csv": fi_clf,
    "fi_ranker_xg_lr.csv": fi_rank,
}

for fname, data in save_map.items():
    data.to_csv(result_dir / fname, index=False)
    print(f"Saved {fname}")

print("\nAll results saved successfully! 🎉")
