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


# ---------------------------------------------------------
# Load universe
# ---------------------------------------------------------

universe_path = Path("data/universe/us_universe_full_filtered.csv")
universe = pd.read_csv(universe_path)
tickers = universe["ticker"].dropna().unique().tolist()

print(f"Loaded {len(tickers)} tickers.")


# ---------------------------------------------------------
# Load price panel
# ---------------------------------------------------------

prices = build_adj_close_panel(
    tickers,
    start="2014-01-01",
    end="2020-12-31",
)

if prices.empty:
    raise ValueError("ERROR: Could not load price panel.")

print("Price panel shape:", prices.shape)
print(prices.head())


# ---------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# Targets
# ---------------------------------------------------------

y_class = (returns_1d.shift(-1).stack() > 0).astype(int)
y_class.name = "target"

y_reg = returns_1d.shift(-1).stack()
y_reg.name = "future_return"

df = dataset.join([y_class, y_reg]).reset_index()
df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)


# ---------------------------------------------------------
# Feature columns (defined BEFORE NaN dropping)
# ---------------------------------------------------------

feature_cols = ["ret_1d", "momentum_126d", "vol_20d", "mom_rank"]


# ---------------------------------------------------------
# Drop NaNs BEFORE training LR or XGBoost
# ---------------------------------------------------------

df = df.dropna(subset=feature_cols + ["target", "future_return"]).copy()
print("Dataset shape after dropping NaNs:", df.shape)


# ---------------------------------------------------------
# Linear Regression Stacking Feature
# ---------------------------------------------------------

print("\nTraining Linear Regression...")

lr = LinearRegression()
lr.fit(df[feature_cols], df["future_return"])

df["lr_pred"] = lr.predict(df[feature_cols])
feature_cols.append("lr_pred")

print("New feature set:", feature_cols)


# ---------------------------------------------------------
# Train XGBoost Classifier
# ---------------------------------------------------------

clf_model, clf_metrics, clf_test = train_xgb_classifier(
    df=df,
    feature_cols=feature_cols,
    target_col="target",
    date_col="date",
    split_date="2018-01-01",
)

print("\nXGBoost Classifier Metrics:")
for k, v in clf_metrics.items():
    print(f"{k}: {v:.4f}")

importance_dict = clf_model.get_booster().get_score(importance_type="gain")

fi_clf = (
    pd.DataFrame(list(importance_dict.items()), columns=["feature", "gain"])
    .sort_values("gain", ascending=False)
    .reset_index(drop=True)
)

print("\nClassifier Feature Importance:")
print(fi_clf)

