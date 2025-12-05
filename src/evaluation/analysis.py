from __future__ import annotations
import pandas as pd
import numpy as np
from typing import List, Callable, Dict, Any, Tuple
import matplotlib.pyplot as plt
from pathlib import Path

# XGBoost import commented out - not needed for LR/RF pipeline
# Uncomment if you need XGBoost feature importance:
# from xgboost import XGBModel
# 
# def get_feature_importance_gain(
#     model: XGBModel,
#     feature_names: List[str],
# ) -> pd.Series:
#     """
#     Feature importances from XGBoost ('gain').
#     """
#     booster = model.get_booster()
#     raw_scores = booster.get_score(importance_type="gain")
#
#     mapped = {}
#     for i, fname in enumerate(feature_names):
#         key = f"f{i}"
#         mapped[fname] = raw_scores.get(key, 0.0)
#
#     return pd.Series(mapped).sort_values(ascending=False)


def get_sklearn_feature_importance(
    model,
    feature_names: List[str],
) -> pd.Series:
    """
    Feature importances from sklearn models (RandomForest, etc.).
    """
    importances = model.feature_importances_
    return pd.Series(
        dict(zip(feature_names, importances))
    ).sort_values(ascending=False)


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


def backtest_long_short_from_proba(
    df: pd.DataFrame,
    date_col: str = "date",
    proba_col: str = "y_proba",
    future_return_col: str = "future_return",
    k: int = 20,
) -> pd.DataFrame:
    """
    Long top-k (highest proba), short bottom-k (lowest proba) each day.
    Uses probability scores instead of rank scores.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    daily = []
    for dt, group in df.groupby(date_col):
        group = group.dropna(subset=[proba_col, future_return_col])
        if len(group) < 2 * k:  # Need at least 2k stocks to long/short k each
            continue

        top = group.nlargest(k, proba_col)
        bottom = group.nsmallest(k, proba_col)

        long_ret = top[future_return_col].mean()
        short_ret = bottom[future_return_col].mean()
        port_ret = long_ret - short_ret

        daily.append({"date": dt, "port_return": port_ret})

    if not daily:
        # Return empty DataFrame with proper structure
        res = pd.DataFrame(columns=["date", "port_return"])
        res.attrs["sharpe_estimate"] = np.nan
        return res
    
    res = pd.DataFrame(daily).sort_values("date")

    res["cum_return"] = (1.0 + res["port_return"]).cumprod() - 1.0

    mean = res["port_return"].mean()
    std = res["port_return"].std()
    sharpe = np.nan
    if std > 0:
        sharpe = mean / std * np.sqrt(252)

    res.attrs["sharpe_estimate"] = sharpe
    return res


def compute_daily_accuracy(
    df: pd.DataFrame,
    date_col: str = "date",
    y_true_col: str = "y_true",
    y_proba_col: str = "y_proba",
    threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Compute daily accuracy from predictions.
    Returns DataFrame with date and daily_accuracy columns.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df["y_pred"] = (df[y_proba_col] >= threshold).astype(int)

    daily_acc = []
    for dt, group in df.groupby(date_col):
        group = group.dropna(subset=[y_true_col, y_proba_col])
        if group.empty:
            continue
        acc = (group[y_true_col] == group["y_pred"]).mean()
        daily_acc.append({"date": dt, "daily_accuracy": acc})

    return pd.DataFrame(daily_acc).sort_values("date")


def compute_rolling_accuracy(
    daily_acc_df: pd.DataFrame,
    window: int = 30,
    date_col: str = "date",
    acc_col: str = "daily_accuracy",
) -> pd.Series:
    """
    Compute rolling window accuracy.
    Returns Series with rolling accuracy.
    """
    daily_acc_df = daily_acc_df.sort_values(date_col)
    return daily_acc_df[acc_col].rolling(window=window, min_periods=1).mean()


def walk_forward_validation(
    df: pd.DataFrame,
    train_fn: Callable,
    feature_cols: List[str],
    target_col: str = "target",
    date_col: str = "date",
    split_dates: List[str] | None = None,
    **train_kwargs,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Perform walk-forward validation with multiple train/test splits.
    
    Args:
        df: DataFrame with features and target
        train_fn: Function that takes (df, feature_cols, ...) and returns (model, metrics, test_df)
        feature_cols: List of feature column names
        target_col: Target column name
        date_col: Date column name
        split_dates: List of split dates (train < split_date, test >= split_date)
        **train_kwargs: Additional kwargs to pass to train_fn
    
    Returns:
        Tuple of (all_test_predictions_df, overall_metrics_dict)
    """
    if split_dates is None:
        # Default: yearly splits from 2017 to 2021
        split_dates = [
            "2017-01-01",
            "2018-01-01",
            "2019-01-01",
            "2020-01-01",
            "2021-01-01",
        ]

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)

    all_test_dfs = []
    window_metrics = []

    print(f"\nWalk-Forward Validation with {len(split_dates)-1} windows:")
    print("=" * 80)

    for i in range(len(split_dates) - 1):
        train_end = split_dates[i]
        test_start = split_dates[i]
        test_end = split_dates[i + 1]

        train_df = df[df[date_col] < pd.to_datetime(train_end)].copy()
        test_df = df[
            (df[date_col] >= pd.to_datetime(test_start))
            & (df[date_col] < pd.to_datetime(test_end))
        ].copy()

        if train_df.empty or test_df.empty:
            print(f"Window {i+1}: Skipping (train={len(train_df)}, test={len(test_df)})")
            continue

        print(f"\nWindow {i+1}: Train < {train_end}, Test {test_start} to {test_end}")
        print(f"  Train samples: {len(train_df)}, Test samples: {len(test_df)}")

        # Combine train and test for the train_fn (it will split internally)
        # Use a split_date that ensures all train_df is used for training
        combined_df = pd.concat([train_df, test_df], ignore_index=True)
        
        # Use a date just before test_start to ensure train_df is training set
        split_date_for_fn = (pd.to_datetime(test_start) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        
        # Train model
        model, metrics, _ = train_fn(
            combined_df,
            feature_cols=feature_cols,
            target_col=target_col,
            date_col=date_col,
            split_date=split_date_for_fn,
            **train_kwargs,
        )

        # Make predictions on actual test set
        X_test = test_df[feature_cols].values
        y_test = test_df[target_col].values
        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)

        # Compute metrics for this window
        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
            roc_auc_score,
            log_loss,
        )

        window_metrics_dict = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, y_proba)),
            "log_loss": float(log_loss(y_test, y_proba)),
        }

        test_df = test_df.copy()
        test_df["y_true"] = y_test
        test_df["y_proba"] = y_proba
        test_df["y_pred"] = y_pred

        all_test_dfs.append(test_df)
        window_metrics.append(window_metrics_dict)

        print(f"  Accuracy: {window_metrics_dict['accuracy']:.4f}, ROC-AUC: {window_metrics_dict['roc_auc']:.4f}")

    if not all_test_dfs:
        raise ValueError("No valid windows found for walk-forward validation")

    # Combine all test predictions
    combined_test = pd.concat(all_test_dfs, ignore_index=True)

    # Compute overall metrics
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
        log_loss,
    )

    overall_metrics = {
        "accuracy": float(accuracy_score(combined_test["y_true"], combined_test["y_pred"])),
        "precision": float(precision_score(combined_test["y_true"], combined_test["y_pred"], zero_division=0)),
        "recall": float(recall_score(combined_test["y_true"], combined_test["y_pred"], zero_division=0)),
        "f1": float(f1_score(combined_test["y_true"], combined_test["y_pred"], zero_division=0)),
        "roc_auc": float(roc_auc_score(combined_test["y_true"], combined_test["y_proba"])),
        "log_loss": float(log_loss(combined_test["y_true"], combined_test["y_proba"])),
    }

    print("\n" + "=" * 80)
    print("Overall Walk-Forward Metrics:")
    for k, v in overall_metrics.items():
        print(f"  {k}: {v:.4f}")

    return combined_test, overall_metrics


def plot_daily_accuracy(
    daily_acc_df: pd.DataFrame,
    save_path: Path | str,
    date_col: str = "date",
    acc_col: str = "daily_accuracy",
    title: str = "Daily Accuracy Over Time",
):
    """Plot daily accuracy over time."""
    plt.figure(figsize=(12, 5))
    plt.plot(daily_acc_df[date_col], daily_acc_df[acc_col], alpha=0.7, linewidth=1)
    plt.axhline(y=0.5, color="r", linestyle="--", label="Random (0.5)")
    plt.xlabel("Date")
    plt.ylabel("Daily Accuracy")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150)
    plt.close()


def plot_rolling_accuracy(
    daily_acc_df: pd.DataFrame,
    save_path: Path | str,
    date_col: str = "date",
    acc_col: str = "daily_accuracy",
    window: int = 30,
    title: str = "Rolling Accuracy Over Time",
):
    """Plot rolling window accuracy."""
    rolling_acc = compute_rolling_accuracy(daily_acc_df, window=window, date_col=date_col, acc_col=acc_col)

    plt.figure(figsize=(12, 5))
    plt.plot(daily_acc_df[date_col], daily_acc_df[acc_col], alpha=0.3, linewidth=0.5, label="Daily")
    plt.plot(daily_acc_df[date_col], rolling_acc, linewidth=2, label=f"{window}-day rolling")
    plt.axhline(y=0.5, color="r", linestyle="--", label="Random (0.5)")
    plt.xlabel("Date")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150)
    plt.close()


def plot_equity_curve(
    backtest_df: pd.DataFrame,
    save_path: Path | str,
    date_col: str = "date",
    cum_return_col: str = "cum_return",
    title: str = "Long-Short Equity Curve",
):
    """Plot cumulative return (equity curve) from backtest."""
    plt.figure(figsize=(12, 6))
    plt.plot(backtest_df[date_col], backtest_df[cum_return_col], linewidth=2)
    plt.axhline(y=0, color="k", linestyle="--", alpha=0.5)
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.title(f"{title} (Sharpe: {backtest_df.attrs.get('sharpe_estimate', 'N/A'):.2f})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150)
    plt.close()
