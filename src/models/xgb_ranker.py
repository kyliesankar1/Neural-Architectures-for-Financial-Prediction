from __future__ import annotations
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
from xgboost import XGBRanker


def _groups_by_date(df: pd.DataFrame, date_col: str) -> np.ndarray:
    """
    Return group sizes (rows per date), sorted by date.
    """
    df_sorted = df.sort_values(date_col)
    return df_sorted.groupby(date_col).size().values


def train_xgb_ranker(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "future_return",
    date_col: str = "date",
    split_date: str = "2018-01-01",
    xgb_params: Dict[str, Any] | None = None,
) -> Tuple[XGBRanker, pd.DataFrame]:
    """
    Train an XGBoost ranking model that ranks stocks within each date.

    Returns:
        model
        test_df with an extra 'rank_score' column.
    """
    if xgb_params is None:
        xgb_params = dict(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="rank:pairwise",
            tree_method="hist",
        )

    df = df.sort_values(date_col)
    split_ts = pd.to_datetime(split_date)
    train_df = df[df[date_col] < split_ts].copy()
    test_df = df[df[date_col] >= split_ts].copy()

    X_train = train_df[feature_cols].values
    y_train = train_df[target_col].values

    train_groups = _groups_by_date(train_df, date_col)

    model = XGBRanker(**xgb_params)
    model.fit(X_train, y_train, group=train_groups)

    X_test = test_df[feature_cols].values
    scores = model.predict(X_test)

    out = test_df.copy()
    out["rank_score"] = scores
    return model, out
