
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

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

# Targets
y_class = (returns_1d.shift(-1).stack() > 0).astype(int)
y_class.name = "target"

y_reg = returns_1d.shift(-1).stack()
y_reg.name = "future_return"

df = dataset.join([y_class, y_reg]).reset_index()
df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)

# ---------------------------------------------------------
# Add Linear Regression Prediction (Stacking Feature)
# ---------------------------------------------------------

feature_cols = ["ret_1d", "momentum_126d", "vol_20d", "mom_rank"]

print("\nTraining Linear Regression on base features...")
lr = LinearRegression()
lr.fit(df[feature_cols], df["future_return"])

df["lr_pred"] = lr.predict(df[feature_cols])
feature_cols.append("lr_pred")

print("Added 'lr_pred' to feature set.")
print("New feature set:", feature_cols)


# ---------------------------------------------------------
# Drop NaNs
# ---------------------------------------------------------

df = df.dropna(subset=feature_cols + ["target", "future_return"])
print("Final dataset shape:", df.shape)


# ---------------------------------------------------------
# Train XGBoost Classifier
# ---------------------------------------------------------

importance = model.get_booster().get_score(importance_type="gain")

clf_model, clf_metrics, clf_test = train_xgb_classifier(
    df,
    feature_cols=feature_cols,
    target_col="target",
    date_col="date",
    split_date="2018-01-01",
)

print("\nXGBoost Classifier Metrics:")
for k, v in clf_metrics.items():
    print(f"{k}: {v:.4f}")
importance_df = (
    pd.DataFrame.from_dict(importance_dict, orient="index", columns=["gain"])
    .rename_axis("feature")
    .sort_values("gain", ascending=False)
)

print("\nFeature Importance (Gain):")
print(importance_df)



# ---------------------------------------------------------
# Confusion Matrix
# ---------------------------------------------------------

cm = confusion_matrix(clf_test["y_true"], clf_test["y_pred"])
tn, fp, fn, tp = cm.ravel()

print("\nConfusion Matrix:")
print(cm)

accuracy = accuracy_score(clf_test["y_true"], clf_test["y_pred"])
precision = precision_score(clf_test["y_true"], clf_test["y_pred"])
recall_up = recall_score(clf_test["y_true"], clf_test["y_pred"])
recall_down = tn / (tn + fp)
f1 = f1_score(clf_test["y_true"], clf_test["y_pred"])

print("\nStats:")
print(f"Accuracy = {accuracy:.4f}")
print(f"Precision = {precision:.4f}")
print(f"Recall (Up) = {recall_up:.4f}")
print(f"Recall (Down) = {recall_down:.4f}")
print(f"F1 Score = {f1:.4f}")


# ---------------------------------------------------------
# Train Ranking Model (XGBRanker)
# ---------------------------------------------------------

rank_model, rank_test = train_xgb_ranker(
    df,
    feature_cols=feature_cols,
    target_col="future_return",
    date_col="date",
    split_date="2018-01-01",
)
# --- Ranker Feature Importance (using XGBoost Booster API) ---
booster = rank_model.get_booster()
score_dict = booster.get_score(importance_type="gain")

fi_rank = (
    pd.DataFrame({
        "feature": list(score_dict.keys()),
        "gain": list(score_dict.values())
    })
    .sort_values("gain", ascending=False)
    .reset_index(drop=True)
)

print("\nRanker Feature Importance:")
print(fi_rank)



# ---------------------------------------------------------
# Long/Short Backtest
# ---------------------------------------------------------

bt = backtest_long_short(
    df=rank_test,
    date_col="date",
    score_col="rank_score",
    future_return_col="future_return",
    k=5,
)

print("\nEstimated Sharpe:", bt.attrs["sharpe_estimate"])
print(bt.tail())


# ---------------------------------------------------------
# Save Outputs
# ---------------------------------------------------------

out = Path("results")
out.mkdir(exist_ok=True)

clf_test.to_csv(out / "clf_predictions_xg_lr.csv", index=False)
rank_test.to_csv(out / "rank_predictions_xg_lr.csv", index=False)
bt.to_csv(out / "backtest_xg_lr.csv", index=False)
fi_clf.to_csv(out / "fi_classifier_xg_lr.csv")
fi_rank.to_csv(out / "fi_ranker_xg_lr.csv")

print("\nAll results saved to /results/")
