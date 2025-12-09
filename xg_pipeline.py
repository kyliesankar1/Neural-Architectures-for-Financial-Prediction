"""
Full XGBoost pipeline:
- Load universe and prices
- Build features
- Train classifier (up/down)
- Train ranker (cross-sectional ranking)
- Compute feature importance
- Run long/short backtest
- Save results
"""

import pandas as pd
from pathlib import Path

# Project modules
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


# ============================================================
# 1. Load Universe
# ============================================================

universe_path = Path("data/universe/us_universe_full_filtered.csv")
universe = pd.read_csv(universe_path)
tickers = universe["ticker"].dropna().unique().tolist()

print(f"Loaded {len(tickers)} tickers.")
print("Example tickers:", tickers[:10])


# ============================================================
# 2. Load Price Panel
# ============================================================

prices = build_adj_close_panel(
    tickers,
    start="2014-01-01",
    end="2020-12-31",
)

if prices.empty:
    raise ValueError("Price panel is EMPTY — no data loaded.")

print("Price panel shape:", prices.shape)
print(prices.head())


# ============================================================
# 3. Feature Engineering
# ============================================================

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

feature_cols = ["ret_1d", "momentum_126d", "vol_20d", "mom_rank"]

# Drop any rows with missing values
df = df.dropna(subset=feature_cols + ["target", "future_return"])
print("Final dataset size:", df.shape)


# ============================================================
# 4. Train Classifier (Up/Down)
# ============================================================

clf_model, clf_metrics, clf_test = train_xgb_classifier(
    df=df,
    feature_cols=feature_cols,
    target_col="target",
    date_col="date",
    split_date="2018-01-01",
)

print("\n=== CLASSIFIER METRICS ===")
for k, v in clf_metrics.items():
    print(f"{k}: {v:.4f}")


# --- Classifier Feature Importance ---
booster_clf = clf_model.get_booster()
importance_dict_clf = booster_clf.get_score(importance_type="gain")

fi_clf = (
    pd.DataFrame({
        "feature": feature_cols,
        "gain": [importance_dict_clf.get(f"f{i}", 0) for i in range(len(feature_cols))]
    })
    .sort_values("gain", ascending=False)
    .reset_index(drop=True)
)

print("\nClassifier Feature Importance:")
print(fi_clf)


# ============================================================
# 5. Train Ranker (Predict Higher Returns)
# ============================================================

rank_model, rank_test = train_xgb_ranker(
    df=df,
    feature_cols=feature_cols,
    target_col="future_return",
    date_col="date",
    split_date="2018-01-01",
)

# --- Ranker Feature Importance ---
booster_rank = rank_model.get_booster()
importance_dict_rank = booster_rank.get_score(importance_type="gain")

fi_rank = (
    pd.DataFrame({
        "feature": feature_cols,
        "gain": [importance_dict_rank.get(f"f{i}", 0) for i in range(len(feature_cols))]
    })
    .sort_values("gain", ascending=False)
    .reset_index(drop=True)
)

print("\nRanker Feature Importance:")
print(fi_rank)


# ============================================================
# 6. Long/Short Backtest
# ============================================================

bt = backtest_long_short(
    df=rank_test,
    date_col="date",
    score_col="rank_score",
    future_return_col="future_return",
    k=5,
)

print("\nBacktest Sharpe:", bt.attrs["sharpe_estimate"])
print(bt.tail())


# ============================================================
# 7. Save Results
# ============================================================

output_dir = Path("results")
output_dir.mkdir(exist_ok=True)

clf_test.to_csv(output_dir / "classifier_predictions.csv", index=False)
rank_test.to_csv(output_dir / "ranker_predictions.csv", index=False)
bt.to_csv(output_dir / "backtest_results.csv", index=False)
fi_clf.to_csv(output_dir / "feature_importance_classifier.csv", index=False)
fi_rank.to_csv(output_dir / "feature_importance_ranker.csv", index=False)

print("\nAll results saved to 'results/' folder.")
print("Pipeline finished successfully! 🎉")
