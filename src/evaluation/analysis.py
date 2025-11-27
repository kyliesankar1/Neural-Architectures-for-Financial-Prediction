from __future__ import annotations
import pandas as pd
import numpy as np
from typing import List
from xgboost import XGBModel


def get_feature_importance_gain(
    model: XGBModel,
    feature_names: List[str],
) -> pd.Series:
    """
    Feature importances from XGBoost ('gain').
    """
    booster = model.get_booster()
    raw_scores = booster.get_score(importance_type="gain")

    mapped = {}
    for i, fname in enumerate(feature_names):
        key = f"f{i}"
        mapped[fname] = raw_scores.get(key, 0.0)

    return pd.Series(mapped).sort_values(ascending=False)


def backtest_long_short(
    df: pd.DataFrame,
    date_col: str = "date",
    score_col: str = "rank_score",
    future_return_col: str = "future_return",
    k: int = 20,
) -> pd.DataFrame:
    """
    Long top-k, short bottom-k by score each day.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    daily = []
    for dt, group in df.groupby(date_col):
        group = group.dropna(subset=[score_col, future_return_col])
        if group.empty:
            continue

        top = group.nlargest(k, score_col)
        bottom = group.nsmallest(k, score_col)

        long_ret = top[future_return_col].mean()
        short_ret = bottom[future_return_col].mean()
        port_ret = long_ret - short_ret

        daily.append({"date": dt, "port_return": port_ret})

    res = pd.DataFrame(daily).sort_values("date")
    res["cum_return"] = (1.0 + res["port_return"]).cumprod() - 1.0

    mean = res["port_return"].mean()
    std = res["port_return"].std()
    sharpe = np.nan
    if std > 0:
        sharpe = mean / std * np.sqrt(252)

    res.attrs["sharpe_estimate"] = sharpe
    return res
