import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

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
# 1. LOAD UNIVERSE
# =========================================================

universe_path = Path("data/universe/us_universe_full_filtered.csv")
universe = pd.read_csv(universe_path)
tickers = universe["ticker"].dropna().unique().tolist()

print(f"Loaded {len(tickers)} tickers.")


# =========================================================
# 2. LOAD PRICE PANEL
# =========================================================

prices = build_adj_close_panel(
    tickers,
    start="2014-01-01",
    end="2020-12-31",
)

if prices.empty:
    raise ValueError("ERROR: Could not load price panel.")

print("Price panel shape:", prices.shape)
print(prices.head())


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


# =========================================================
# 5. CLEAN DATA
# =========================================================

feature_cols = ["ret_1d", "momentum_126d", "vol_20d", "mom_rank"]

df = df.dropna(subset=feature_cols + ["target", "future_return"]).copy()
print("Dataset shape after dropping NaNs:", df.shape)


# =========================================================
# 6. LINEAR REGRESSION STACKING FEATURE
# =========================================================

print("\nTraining Linear Regression...")

lr = LinearRegression()
lr.fit(df[feature_cols], df["future_return"])

df["lr_pred"] = lr.predict(df[feature_cols])
feature_cols.append("lr_pred")

print("New feature set:", feature_cols)


# =========================================================
# 7. TRAIN XGBOOST CLASSIFIER
# =========================================================

clf_model, clf_metrics, clf_test = train_xgb_classifier(
    df=df,
    feature_cols=feature_cols,
    target_col="target",
    date_col="date",
    split_date="2018-01-01",
)

print("\n=== XGBOOST CLASSIFIER METRICS ===")
for k, v in clf_metrics.items():
    print(f"{k}: {v:.4f}")

importance_dict = clf_model.get_booster().get_score(importance_type="gain")

fi_clf = (
    pd.DataFrame(list(importance_dict.items()), columns=["feature", "gain"])
    .sort_values("gain", ascending=False)
    .reset_index(drop=True)
)

print("\n=== CLASSIFIER FEATURE IMPORTANCE ===")
print(fi_clf)


# =========================================================
# 8. TRAIN XGBOOST RANKER
# =========================================================

rank_model, rank_test = train_xgb_ranker(
    df=df,
    feature_cols=feature_cols,
    target_col="future_return",
    date_col="date",
    split_date="2018-01-01",
)

rank_importance = rank_model.get_booster().get_score(importance_type="gain")

fi_rank = (
    pd.DataFrame(list(rank_importance.items()), columns=["feature", "gain"])
    .sort_values("gain", ascending=False)
    .reset_index(drop=True)
)

print("\n=== RANKER FEATURE IMPORTANCE ===")
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
print(f"\n=== BACKTEST SHARPE ===\n{sharpe}")
print(bt.tail())


# =========================================================
# 10. SAVE ALL RESULTS (fully guaranteed)
# =========================================================

results_dir = Path("results")
results_dir.mkdir(exist_ok=True)

save_dict = {
    "clf_predictions_xg_lr.csv": clf_test,
    "rank_predictions_xg_lr.csv": rank_test,
    "backtest_xg_lr.csv": bt,
    "fi_classifier_xg_lr.csv": fi_clf,
    "fi_ranker_xg_lr.csv": fi_rank,
}

for filename, data in save_dict.items():
    path = results_dir / filename
    data.to_csv(path, index=False)
    print(f"Saved {filename} → {path}")

print("\nAll results saved successfully! 🎉")
