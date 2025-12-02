"""
LSTM baseline for stock return direction (daily or weekly)

What it does:
- Load universe from data/universe/us_universe_full_filtered.csv
- Load adjusted close prices (splits/dividends handled)
- Optional weekly resample (W-FRI)
- Compute features:
    * 1-period returns
    * 5-period returns
    * momentum
    * volatility
    * cross-sectional momentum rank
    * cross-sectional momentum z-score
    * MACD indicators (line, signal, histogram)
- Build sliding windows of WINDOW_SIZE time steps
- Label = 1 if next return > 0 else 0
- Train an LSTM classifier
- Save test predictions + summary
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

# Project helpers
from src.data.panel import (
    build_adj_close_panel,
    compute_returns,
    compute_momentum,
    compute_volatility,
    rank_cross_sectional,
    zscore_cross_sectional,
)

UNIVERSE_PATH = Path("data/universe/us_universe_full_filtered.csv")
RESULTS_DIR = Path("results")

START_DATE = "2014-01-01"
END_DATE = "2021-12-31"
SPLIT_DATE = "2018-01-01"

MAX_TICKERS = 30
WINDOW_SIZE = 40
BATCH_SIZE = 64
NUM_EPOCHS = 20
HIDDEN_SIZE = 64
NUM_LAYERS = 1
LEARNING_RATE = 1e-3
SEED = 42

# "daily" or "weekly"
FREQUENCY = "weekly"


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_macd(
    prices: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Compute MACD for each column of prices.
    Returns:
        macd_line, signal_line, macd_hist (DataFrames)
    """
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    macd_hist = macd_line - signal_line
    return macd_line, signal_line, macd_hist


def build_lstm_dataset(
    prices: pd.DataFrame,
    window_size: int = 20,
    frequency: str = "daily",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build LSTM samples using multiple technical features including MACD.
    """

    # Base features
    rets_1 = compute_returns(prices, periods=1)
    rets_5 = compute_returns(prices, periods=5)

    # Adjust lookbacks for weekly
    if frequency == "weekly":
        mom_lookback = 26
        vol_window = 10
    else:
        mom_lookback = 20
        vol_window = 20

    mom = compute_momentum(prices, lookback=mom_lookback)
    vol = compute_volatility(rets_1, window=vol_window, annualize=True)
    mom_rank = rank_cross_sectional(mom)
    mom_z = zscore_cross_sectional(mom)

    macd_line, macd_signal, macd_hist = compute_macd(prices)

    X_list: List[np.ndarray] = []
    y_list: List[int] = []
    date_list: List[pd.Timestamp] = []
    ticker_list: List[str] = []

    for ticker in prices.columns:
        df_feat = pd.DataFrame(
            {
                "ret1": rets_1[ticker],
                "ret5": rets_5[ticker],
                "mom": mom[ticker],
                "vol20": vol[ticker],
                "mom_rank": mom_rank[ticker],
                "mom_z": mom_z[ticker],
                "macd_line": macd_line[ticker],
                "macd_signal": macd_signal[ticker],
                "macd_hist": macd_hist[ticker],
            }
        )

        df_feat = df_feat.dropna(how="any")
        if len(df_feat) <= window_size + 1:
            continue

        future_ret = rets_1[ticker].reindex(df_feat.index).shift(-1)
        df_feat["future_ret"] = future_ret
        df_feat = df_feat.dropna(subset=["future_ret"])

        if len(df_feat) <= window_size:
            continue

        # Normalize per ticker
        feat_cols = [
            "ret1",
            "ret5",
            "mom",
            "vol20",
            "mom_rank",
            "mom_z",
            "macd_line",
            "macd_signal",
            "macd_hist",
        ]

        means = df_feat[feat_cols].mean()
        stds = df_feat[feat_cols].std()

        if (stds == 0).any() or stds.isna().any():
            continue

        df_feat_norm = df_feat.copy()
        df_feat_norm[feat_cols] = (df_feat[feat_cols] - means) / stds

        for idx in range(window_size - 1, len(df_feat_norm)):
            window = df_feat_norm.iloc[idx - window_size + 1 : idx + 1]

            X_window = window[feat_cols].values.astype(np.float32)
            future_ret_val = df_feat_norm["future_ret"].iloc[idx]
            label = 1 if future_ret_val > 0 else 0
            label_date = df_feat_norm.index[idx]

            X_list.append(X_window)
            y_list.append(label)
            date_list.append(label_date)
            ticker_list.append(ticker)

    if len(X_list) == 0:
        raise SystemExit("ERROR: No LSTM samples built.")

    return (
        np.stack(X_list, axis=0),
        np.array(y_list, dtype=np.int64),
        np.array(date_list),
        np.array(ticker_list),
    )


class LSTMBinaryClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last_hidden = out[:, -1, :]
        return self.fc(last_hidden)


# ----------------------------------------------------------------------
# SINGLE-SPLIT BASELINE (original main)
# ----------------------------------------------------------------------
def main() -> None:
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("LSTM pipeline (Kylie) - with MACD [single split]")
    print("=" * 80)

    # Load universe
    universe = pd.read_csv(UNIVERSE_PATH)
    all_tickers = universe["ticker"].dropna().unique().tolist()
    tickers = all_tickers[:MAX_TICKERS]

    print(f"Using {len(tickers)} tickers:", tickers[:10])

    # Load adjusted close
    prices = build_adj_close_panel(
        tickers, start=START_DATE, end=END_DATE
    )

    if prices.empty:
        raise SystemExit("ERROR: prices empty")

    # Daily/weekly
    freq = FREQUENCY.lower()
    if freq == "weekly":
        prices_used = prices.resample("W-FRI").last()
        print("\nUsing WEEKLY closes.")
    else:
        prices_used = prices
        print("\nUsing DAILY closes.")

    # Dataset
    print("\nBuilding LSTM dataset...")
    X, y, dates, ticker_arr = build_lstm_dataset(
        prices_used,
        window_size=WINDOW_SIZE,
        frequency=freq,
    )

    # Train/test split
    split_ts = pd.to_datetime(SPLIT_DATE)
    if getattr(dates[0], "tzinfo", None):
        split_ts = split_ts.tz_localize(dates[0].tzinfo)

    train_mask = dates < split_ts
    test_mask = dates >= split_ts

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    # Dataloaders
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    # Model
    num_features = X_train.shape[2]
    model = LSTMBinaryClassifier(
        input_size=num_features,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training
    print("\nTraining (single-split)...")
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for xb, yb in train_loader:
            xb = xb.float().to(device)
            yb = yb.float().view(-1, 1).to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * xb.size(0)

        epoch_loss = total_loss / len(train_ds)
        print(f"Epoch {epoch:02d}/{NUM_EPOCHS} - loss: {epoch_loss:.4f}")

    # Evaluation
    print("\nEvaluating (single-split)...")
    model.eval()
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for xb, yb in test_loader:
            logits = model(xb.float().to(device))
            all_logits.append(logits.cpu().numpy())
            all_labels.append(yb.numpy())

    all_logits = np.vstack(all_logits).reshape(-1)
    all_labels = np.concatenate(all_labels)

    probs = 1 / (1 + np.exp(-all_logits))
    preds = (probs >= 0.5).astype(int)

    accuracy = (preds == all_labels).mean()
    print(f"\nTest accuracy (single-split): {accuracy:.4f}")

    # Save predictions
    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / "lstm_predictions_single_split.csv"

    pd.DataFrame(
        {
            "date": dates[test_mask],
            "ticker": ticker_arr[test_mask],
            "y_true": all_labels,
            "y_proba": probs,
            "y_pred": preds,
        }
    ).to_csv(out_path, index=False)

    print(f"Saved single-split predictions to {out_path}")

    # Summary file
    summary_path = RESULTS_DIR / "lstm_summary_single_split.txt"
    with open(summary_path, "w") as f:
        f.write("LSTM w/ MACD Summary (Single split)\n")
        f.write(f"Frequency: {FREQUENCY}\n")
        f.write(f"Test accuracy: {accuracy:.4f}\n")
        f.write(f"Window size: {WINDOW_SIZE}\n")
        f.write(f"Tickers: {len(tickers)}\n")

    print("Single-split run done!")


# ----------------------------------------------------------------------
# WALK-FORWARD EVALUATION
# ----------------------------------------------------------------------
def run_walkforward() -> None:
    """
    Run a walk-forward (rolling) backtest on the same data as main(),
    but using multiple time windows instead of a single split.
    """
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("LSTM pipeline (Kylie) - WALK-FORWARD")
    print("=" * 80)

    # ------------ 1) Load universe + prices (same as in main) ------------
    universe = pd.read_csv(UNIVERSE_PATH)
    all_tickers = universe["ticker"].dropna().unique().tolist()
    tickers = all_tickers[:MAX_TICKERS]

    print(f"Using {len(tickers)} tickers:", tickers[:10])

    prices = build_adj_close_panel(
        tickers, start=START_DATE, end=END_DATE
    )
    if prices.empty:
        raise SystemExit("ERROR: prices empty")

    freq = FREQUENCY.lower()
    if freq == "weekly":
        prices_used = prices.resample("W-FRI").last()
        print("\nUsing WEEKLY closes.")
    else:
        prices_used = prices
        print("\nUsing DAILY closes.")

    print("\nBuilding LSTM dataset (for walk-forward)...")
    X, y, dates, ticker_arr = build_lstm_dataset(
        prices_used,
        window_size=WINDOW_SIZE,
        frequency=freq,
    )

    num_features = X.shape[2]

    # ------------ 2) Define year boundaries for windows ------------
    # We will train < 2017-01-01, test 2017; train < 2018-01-01, test 2018; etc.
    split_points = pd.to_datetime(
        [
            "2017-01-01",
            "2018-01-01",
            "2019-01-01",
            "2020-01-01",
            "2021-01-01",
            "2022-01-01",  # one past END_DATE
        ]
    )

    # Match timezone if needed
    if getattr(dates[0], "tzinfo", None):
        tz = dates[0].tzinfo
        split_points = [sp.tz_localize(tz) for sp in split_points]
    else:
        split_points = list(split_points)

    # Containers to collect all test predictions from all windows
    all_probs_list: List[np.ndarray] = []
    all_true_list: List[np.ndarray] = []
    all_pred_list: List[np.ndarray] = []
    all_dates_list: List[np.ndarray] = []
    all_tickers_list: List[np.ndarray] = []
    window_accuracies: List[tuple] = []

    print("\nStarting WALK-FORWARD evaluation...\n")

    # ------------ 3) Loop over windows ------------
    for i in range(len(split_points) - 1):
        train_end = split_points[i]
        test_start = train_end
        test_end = split_points[i + 1]

        # Masks for this window
        train_mask = dates < train_end
        test_mask = (dates >= test_start) & (dates < test_end)

        if not train_mask.any() or not test_mask.any():
            print(
                f"Skipping window {i+1}: "
                f"train_end={train_end.date()}, "
                f"test_start={test_start.date()}, "
                f"test_end={test_end.date()} (no data)"
            )
            continue

        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]
        dates_test = dates[test_mask]
        tickers_test = ticker_arr[test_mask]

        print("=" * 80)
        print(
            f"[Window {i+1}] "
            f"train < {train_end.date()}, "
            f"test {test_start.date()}–{(test_end - pd.Timedelta(days=1)).date()}"
        )
        print(
            f"  Train samples: {len(X_train)}, "
            f"Test samples: {len(X_test)}"
        )

        # Build loaders
        train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
        test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

        # ------------ 4) New model for this window ------------
        model = LSTMBinaryClassifier(
            input_size=num_features,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
        ).to(device)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

        print("\n  Training...")
        for epoch in range(1, NUM_EPOCHS + 1):
            model.train()
            total_loss = 0.0

            for xb, yb in train_loader:
                xb = xb.float().to(device)
                yb = yb.float().view(-1, 1).to(device)

                optimizer.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * xb.size(0)

            epoch_loss = total_loss / len(train_ds)
            print(f"    Epoch {epoch:02d}/{NUM_EPOCHS} - loss: {epoch_loss:.4f}")

        # ------------ 5) Evaluate on this window ------------
        print("  Evaluating...")
        model.eval()
        fold_logits: List[np.ndarray] = []
        fold_labels: List[np.ndarray] = []

        with torch.no_grad():
            for xb, yb in test_loader:
                logits = model(xb.float().to(device))
                fold_logits.append(logits.cpu().numpy())
                fold_labels.append(yb.numpy())

        fold_logits_arr = np.vstack(fold_logits).reshape(-1)
        fold_labels_arr = np.concatenate(fold_labels)

        fold_probs = 1 / (1 + np.exp(-fold_logits_arr))
        fold_preds = (fold_probs >= 0.5).astype(int)
        fold_acc = (fold_preds == fold_labels_arr).mean()

        print(f"  Window {i+1} accuracy: {fold_acc:.4f}\n")

        window_accuracies.append(
            (
                i + 1,
                train_end.date(),
                test_start.date(),
                (test_end - pd.Timedelta(days=1)).date(),
                float(fold_acc),
            )
        )

        # Save results for this window into global lists
        all_probs_list.append(fold_probs)
        all_true_list.append(fold_labels_arr)
        all_pred_list.append(fold_preds)
        all_dates_list.append(dates_test)
        all_tickers_list.append(tickers_test)

    # ------------ 6) Combine results from all windows ------------
    if not all_probs_list:
        raise SystemExit("ERROR: No walk-forward windows produced predictions.")

    probs = np.concatenate(all_probs_list)
    all_labels = np.concatenate(all_true_list)
    preds = np.concatenate(all_pred_list)
    dates_all = np.concatenate(all_dates_list)
    tickers_all = np.concatenate(all_tickers_list)

    overall_accuracy = (preds == all_labels).mean()
    print("=" * 80)
    print(f"\n[Walk-forward] Overall test accuracy: {overall_accuracy:.4f}")

    # Save predictions
    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / "lstm_predictions_walkforward.csv"

    pd.DataFrame(
        {
            "date": dates_all,
            "ticker": tickers_all,
            "y_true": all_labels,
            "y_proba": probs,
            "y_pred": preds,
        }
    ).to_csv(out_path, index=False)

    print(f"[Walk-forward] Saved predictions to {out_path}")

    # Save summary
    summary_path = RESULTS_DIR / "lstm_summary_walkforward.txt"
    with open(summary_path, "w") as f:
        f.write("LSTM w/ MACD - Walk-forward Summary\n")
        f.write(f"Frequency: {FREQUENCY}\n")
        f.write(f"Overall walk-forward accuracy: {overall_accuracy:.4f}\n")
        f.write(f"Window size: {WINDOW_SIZE}\n")
        f.write(f"Tickers: {len(tickers)}\n")
        f.write("\nPer-window accuracies:\n")
        f.write("window_idx,train_end,test_start,test_end,accuracy\n")
        for w_idx, train_end_d, test_start_d, test_end_d, acc in window_accuracies:
            f.write(
                f"{w_idx},{train_end_d},{test_start_d},{test_end_d},{acc:.4f}\n"
            )

    print(f"[Walk-forward] Saved walk-forward summary to {summary_path}")
    print("Walk-forward run done!")


# ----------------------------------------------------------------------
# ANALYSIS: per-year / per-ticker performance
# ----------------------------------------------------------------------
def analyze_predictions(pred_csv_path: Path) -> None:
    """
    Load a predictions CSV and compute:
      - overall accuracy
      - accuracy by year
      - accuracy by ticker
      - accuracy by (year, ticker)
    Save results to separate CSVs in RESULTS_DIR.
    """

    print(f"\n[Analysis] Loading predictions from {pred_csv_path}")
    df = pd.read_csv(pred_csv_path)

    # Make sure these columns exist
    required_cols = {"date", "ticker", "y_true", "y_pred"}
    missing = required_cols - set(df.columns)
    if missing:
        raise SystemExit(f"ERROR: predictions file missing columns: {missing}")

    # Parse year from date
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year

    # Overall accuracy
    overall_acc = (df["y_true"] == df["y_pred"]).mean()
    print(f"[Analysis] Overall accuracy (from file): {overall_acc:.4f}")

    # Accuracy by year
    acc_by_year = (
        df.groupby("year")
        .apply(lambda g: (g["y_true"] == g["y_pred"]).mean())
        .reset_index(name="accuracy")
    )
    print("\n[Analysis] Accuracy by year:")
    print(acc_by_year)

    # Accuracy by ticker
    acc_by_ticker = (
        df.groupby("ticker")
        .apply(lambda g: (g["y_true"] == g["y_pred"]).mean())
        .reset_index(name="accuracy")
    )

    print("\n[Analysis] Best tickers (top 10):")
    print(acc_by_ticker.sort_values("accuracy", ascending=False).head(10))

    print("\n[Analysis] Worst tickers (bottom 10):")
    print(acc_by_ticker.sort_values("accuracy", ascending=True).head(10))

    # Accuracy by (year, ticker)
    acc_by_year_ticker = (
        df.groupby(["year", "ticker"])
        .apply(lambda g: (g["y_true"] == g["y_pred"]).mean())
        .reset_index(name="accuracy")
    )

    # Make sure results dir exists
    RESULTS_DIR.mkdir(exist_ok=True)

    # Save summaries
    acc_by_year_path = RESULTS_DIR / "accuracy_by_year.csv"
    acc_by_ticker_path = RESULTS_DIR / "accuracy_by_ticker.csv"
    acc_by_year_ticker_path = RESULTS_DIR / "accuracy_by_year_ticker.csv"

    acc_by_year.to_csv(acc_by_year_path, index=False)
    acc_by_ticker.to_csv(acc_by_ticker_path, index=False)
    acc_by_year_ticker.to_csv(acc_by_year_ticker_path, index=False)

    print(f"\n[Analysis] Saved accuracy_by_year to {acc_by_year_path}")
    print(f"[Analysis] Saved accuracy_by_ticker to {acc_by_ticker_path}")
    print(f"[Analysis] Saved accuracy_by_year_ticker to {acc_by_year_ticker_path}")
    print("[Analysis] Done.")

def analyze_sharpe(pred_csv_path: Path) -> None:
    """
    Take a predictions CSV (with date, ticker, y_pred) and compute:
      - overall strategy Sharpe (long/flat based on y_pred)
      - Sharpe by year
      - Sharpe by ticker

    Strategy:
      position_t = 1 if y_pred_t == 1 else 0   (long/flat)
      strategy_ret_t = position_t * future_ret_t

    where future_ret_t is the return from t -> t+1, matching how labels were defined.
    """

    print(f"\n[Sharpe] Loading predictions from {pred_csv_path}")
    df_pred = pd.read_csv(pred_csv_path)

    required_cols = {"date", "ticker", "y_pred"}
    missing = required_cols - set(df_pred.columns)
    if missing:
        raise SystemExit(f"ERROR: predictions file missing columns: {missing}")

    # Parse dates
    df_pred["date"] = pd.to_datetime(df_pred["date"])

    # ------------------------------------------------------------------
    # Rebuild the same price panel and FUTURE returns used for labels
    # ------------------------------------------------------------------
    tickers = df_pred["ticker"].dropna().unique().tolist()

    prices = build_adj_close_panel(
        tickers, start=START_DATE, end=END_DATE
    )
    if prices.empty:
        raise SystemExit("ERROR: prices empty when rebuilding for Sharpe analysis")

    freq = FREQUENCY.lower()
    if freq == "weekly":
        prices_used = prices.resample("W-FRI").last()
        annual_factor = 52
    else:
        prices_used = prices
        annual_factor = 252

    # 1-period returns (t-1 -> t)
    rets_1 = compute_returns(prices_used, periods=1)

    # FUTURE returns (t -> t+1), matching how labels were created in build_lstm_dataset
    future_rets = rets_1.shift(-1)

    # Long form: date, ticker, future_ret
    df_ret = (
        future_rets
        .stack()
        .reset_index()
    )
    df_ret.columns = ["date", "ticker", "future_ret"]

    # Merge predictions with realized FUTURE returns on (date, ticker)
    df = pd.merge(df_pred, df_ret, on=["date", "ticker"], how="inner")
    df = df.dropna(subset=["future_ret"])

    if df.empty:
        raise SystemExit("ERROR: no overlap between predictions and future returns for Sharpe analysis")

    # ------------------------------------------------------------------
    # Build strategy returns: long when y_pred==1, flat otherwise
    # ------------------------------------------------------------------
    df["position"] = (df["y_pred"] == 1).astype(float)
    df["strategy_ret"] = df["position"] * df["future_ret"]

    mean_ret = df["strategy_ret"].mean()
    vol_ret = df["strategy_ret"].std()

    if vol_ret > 0:
        sharpe = np.sqrt(annual_factor) * mean_ret / vol_ret
    else:
        sharpe = np.nan

    print(f"[Sharpe] Overall annualized Sharpe (long/flat): {sharpe:.4f}")

    # ------------------------------------------------------------------
    # Sharpe by year
    # ------------------------------------------------------------------
    df["year"] = df["date"].dt.year

    def _sharpe(group: pd.DataFrame) -> float:
        m = group["strategy_ret"].mean()
        s = group["strategy_ret"].std()
        return np.sqrt(annual_factor) * m / s if s > 0 else np.nan

    sharpe_by_year = (
        df.groupby("year")
        .apply(_sharpe)
        .reset_index(name="sharpe")
    )
    print("\n[Sharpe] Sharpe by year:")
    print(sharpe_by_year)

    # ------------------------------------------------------------------
    # Sharpe by ticker
    # ------------------------------------------------------------------
    sharpe_by_ticker = (
        df.groupby("ticker")
        .apply(_sharpe)
        .reset_index(name="sharpe")
    )

    print("\n[Sharpe] Best tickers by Sharpe (top 10):")
    print(sharpe_by_ticker.sort_values("sharpe", ascending=False).head(10))

    print("\n[Sharpe] Worst tickers by Sharpe (bottom 10):")
    print(sharpe_by_ticker.sort_values("sharpe", ascending=True).head(10))

    # Save to CSVs
    RESULTS_DIR.mkdir(exist_ok=True)
    sharpe_year_path = RESULTS_DIR / "sharpe_by_year.csv"
    sharpe_ticker_path = RESULTS_DIR / "sharpe_by_ticker.csv"

    sharpe_by_year.to_csv(sharpe_year_path, index=False)
    sharpe_by_ticker.to_csv(sharpe_ticker_path, index=False)

    print(f"\n[Sharpe] Saved sharpe_by_year to {sharpe_year_path}")
    print(f"[Sharpe] Saved sharpe_by_ticker to {sharpe_ticker_path}")
    print("[Sharpe] Done.")

# ----------------------------------------------------------------------
# ENTRY POINT
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Run single-split baseline
    main()

    # Run walk-forward evaluation
    run_walkforward()

    # Analyze walk-forward predictions (change path if you want single-split instead)
    analyze_predictions(RESULTS_DIR / "lstm_predictions_walkforward.csv")

    analyze_sharpe(RESULTS_DIR / "lstm_predictions_walkforward.csv")
