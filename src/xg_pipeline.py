"""

This script loads the universe, fetches price data,
builds features, trains two XGBoost models,
checks feature importance, and runs a simple long/short backtest.
"""

import pandas as pd
from pathlib import Path

from src.data.panel import (
    build_adj_close_panel,
    compute_returns,
    compute_momentum,
    compute_volatility,
    rank_cross_sectional,
)
from src.models.xgb_classifier import train_xgb_classifier
from src.models.xgb_ranker import train_xgb_ranker
from src.evaluation.analysis import get_feature_importance_gain, backtest_long_short


# load the universe of stocks
universe_path = Path("data/universe/us_universe_sample_filtered.csv")
universe = pd.read_csv(universe_path)
tickers = universe["ticker"].tolist()

print(f"Loaded {len(tickers)} tickers.")


# load prices (cached after first run)
prices = build_adj_close_panel(
    tickers,
    start="2014-01-01",
    end="2020-12-31",
)

print("Price panel shape:", prices.shape)


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

# these are the features I'm using in the models
feature_cols = ["ret_1d", "momentum_126d", "vol_20d", "mom_rank"]


# train the classifier (predict up/down movement)
clf_model, clf_metrics, clf_test = train_xgb_classifier(
    df,
    feature_cols=feature_cols,
    target_col="target",
    date_col="date",
    split_date="2018-01-01",
)

print("\nClassifier metrics:")
for k, v in clf_metrics.items():
    print(f"{k}: {v:.4f}")

# classifier feature importance
fi_clf = get_feature_importance_gain(clf_model, feature_cols)
print("\nClassifier feature importance:\n", fi_clf)


# train the ranking model (predict which stocks have higher returns)
rank_model, rank_test = train_xgb_ranker(
    df,
    feature_cols=feature_cols,
    target_col="future_return",
    date_col="date",
    split_date="2018-01-01",
)

# ranker feature importance
fi_rank = get_feature_importance_gain(rank_model, feature_cols)
print("\nRanker feature importance:\n", fi_rank)


# simple long/short backtest: long top-k predictions and short bottom-k
bt = backtest_long_short(
    df=rank_test,
    date_col="date",
    score_col="rank_score",
    future_return_col="future_return",
    k=5,
)

print("\nEstimated Sharpe:", bt.attrs["sharpe_estimate"])
print(bt.tail())


# save everything
out = Path("results")
out.mkdir(exist_ok=True)

clf_test.to_csv(out / "classifier_predictions.csv", index=False)
rank_test.to_csv(out / "ranker_predictions.csv", index=False)
bt.to_csv(out / "backtest_results.csv", index=False)

print("\nSaved results to 'results/' folder.")
print("Pipeline finished!")
