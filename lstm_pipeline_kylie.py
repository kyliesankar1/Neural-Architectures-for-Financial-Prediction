"""
LSTM baseline for stock return direction (daily or weekly).

What it does:
- Load universe from data/universe/us_universe_full_filtered.csv
- Load adjusted close prices with build_adj_close_panel (handles splits/dividends)
- Optionally resample to weekly closes (W-FRI)
- Build multiple features:
    * 1-period returns
    * 5-period returns
    * momentum (long lookback)
    * 20-period volatility
    * cross-sectional momentum rank
    * cross-sectional momentum z-score
- For each stock, build sliding windows of length WINDOW_SIZE over these features
- Label = 1 if next-period return > 0, else 0
- Train an LSTM to predict up/down
- Split train/test by date (time-based split, like xg_pipeline.py)
- Save test predictions & summary under results/
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

# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------

UNIVERSE_PATH = Path("data/universe/us_universe_full_filtered.csv")
RESULTS_DIR = Path("results")

START_DATE = "2014-01-01"
END_DATE = "2021-12-31"
SPLIT_DATE = "2018-01-01"

MAX_TICKERS = 30        # limit tickers for speed
WINDOW_SIZE = 40          # how many past periods the LSTM sees
BATCH_SIZE = 64
NUM_EPOCHS = 15
HIDDEN_SIZE = 32
NUM_LAYERS = 1
LEARNING_RATE = 1e-3
SEED = 42

# "daily" or "weekly"
FREQUENCY = "daily"    


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -------------------------------------------------------------------
# Dataset construction
# -------------------------------------------------------------------

def build_lstm_dataset(
    prices: pd.DataFrame,
    window_size: int = 20,
    frequency: str = "daily",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build LSTM samples from price data with MULTIPLE features.

    Features per (date, ticker):
      - 1-period return
      - 5-period return
      - momentum
      - 20-period volatility
      - momentum cross-sectional rank
      - momentum cross-sectional z-score

    For each ticker, we:
      - build a feature DataFrame over time
      - add 'future_ret' = next-period 1-step return
      - slide a window of length `window_size` over the features
      - label = 1 if future_ret > 0 else 0
      - we associate each sample with the "label date" (the last date of the window)

    Returns:
      X: (N, window_size, num_features) float32
      y: (N,) int64
      dates: (N,) array of pd.Timestamp for label day
      tickers: (N,) array of ticker strings
    """

    # 1. Compute base features
    rets_1 = compute_returns(prices, periods=1)
    rets_5 = compute_returns(prices, periods=5)

    # adjust lookbacks a bit if weekly
    if frequency == "weekly":
        mom_lookback = 26   # ~6-month momentum in weeks
        vol_window = 10     # 10 weeks
    else:
        mom_lookback = 126  # ~6-month momentum in days
        vol_window = 20

    mom = compute_momentum(prices, lookback=mom_lookback)
    vol = compute_volatility(rets_1, window=vol_window, annualize=True)
    mom_rank = rank_cross_sectional(mom)
    mom_z = zscore_cross_sectional(mom)

    X_list: List[np.ndarray] = []
    y_list: List[int] = []
    date_list: List[pd.Timestamp] = []
    ticker_list: List[str] = []

    for ticker in prices.columns:
        # Build feature DataFrame for THIS ticker
        df_feat = pd.DataFrame(
            {
                "ret1": rets_1[ticker],
                "ret5": rets_5[ticker],
                "mom": mom[ticker],
                "vol20": vol[ticker],
                "mom_rank": mom_rank[ticker],
                "mom_z": mom_z[ticker],
            }
        )

        # Drop any rows with NaN so that each row has a full feature vector
        df_feat = df_feat.dropna(how="any")

        if len(df_feat) <= window_size + 1:
            # not enough history for this ticker
            continue

        # Add future 1-step return as label source
        future_ret = rets_1[ticker].reindex(df_feat.index).shift(-1)
        df_feat["future_ret"] = future_ret

        df_feat = df_feat.dropna(subset=["future_ret"])

        if len(df_feat) <= window_size:
            continue

        # OPTIONAL: normalize features per ticker (z-score each column except future_ret)
        feat_cols = ["ret1", "ret5", "mom", "vol20", "mom_rank", "mom_z"]
        feat_values = df_feat[feat_cols]
        means = feat_values.mean()
        stds = feat_values.std()

        # skip tickers with zero variance in any feature
        if (stds == 0).any() or stds.isna().any():
            continue

        df_feat_norm = df_feat.copy()
        df_feat_norm[feat_cols] = (feat_values - means) / stds

        # Sliding window:
        # for index idx, we use rows [idx-window_size+1 .. idx] as input,
        # and future_ret at idx as the label (direction of next return)
        for idx in range(window_size - 1, len(df_feat_norm)):
            window_data = df_feat_norm.iloc[idx - window_size + 1 : idx + 1]

            # features only (exclude future_ret)
            window_X = window_data[feat_cols].values.astype(np.float32)  # (window_size, num_features)

            future_ret_label = df_feat_norm["future_ret"].iloc[idx]
            label = 1 if future_ret_label > 0 else 0
            label_date = df_feat_norm.index[idx]

            X_list.append(window_X)
            y_list.append(label)
            date_list.append(label_date)
            ticker_list.append(ticker)

    if len(X_list) == 0:
        raise SystemExit("ERROR: No LSTM samples built. Check your data or window size.")

    X = np.stack(X_list, axis=0)  # (N, window_size, num_features)
    y = np.array(y_list, dtype=np.int64)
    dates = np.array(date_list)
    tickers = np.array(ticker_list)

    return X, y, dates, tickers


# -------------------------------------------------------------------
# Model
# -------------------------------------------------------------------

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
        """
        x: (batch_size, seq_len, input_size)
        returns: logits (batch_size, 1)
        """
        out, _ = self.lstm(x)          # (batch, seq, hidden)
        last_hidden = out[:, -1, :]    # use last time-step
        logits = self.fc(last_hidden)  # (batch, 1)
        return logits


# -------------------------------------------------------------------
# Main pipeline
# -------------------------------------------------------------------

def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("LSTM pipeline (Kylie) - multifeature, daily/weekly toggle")
    print("=" * 80)

    # 1. Load universe
    print(f"\nLoading universe from {UNIVERSE_PATH} ...")
    universe = pd.read_csv(UNIVERSE_PATH)
    all_tickers = universe["ticker"].dropna().unique().tolist()
    tickers = all_tickers[:MAX_TICKERS]
    print(f"Total tickers in file: {len(all_tickers)}; using first {len(tickers)}")
    print("Some tickers:", tickers[:10])

    # 2. Load adjusted close prices
    print("\nLoading adjusted close prices via build_adj_close_panel...")
    prices = build_adj_close_panel(
        tickers,
        start=START_DATE,
        end=END_DATE,
    )
    print("Raw daily price panel shape:", prices.shape)
    print("First few rows of prices:\n", prices.head())

    if prices.empty:
        raise SystemExit(
            "ERROR: prices DataFrame is empty. "
            "Check your data/universe files or date range."
        )

    # 2b. Daily vs Weekly
    freq = FREQUENCY.lower()
    if freq == "weekly":
        prices_used = prices.resample("W-FRI").last()
        print("\nUsing WEEKLY closes (W-FRI resample).")
    elif freq == "daily":
        prices_used = prices
        print("\nUsing DAILY closes.")
    else:
        raise ValueError(f"FREQUENCY must be 'daily' or 'weekly', got: {FREQUENCY}")

    print("Price panel USED shape:", prices_used.shape)
    print("First few rows of prices_used:\n", prices_used.head())

    # 3. Build LSTM dataset (sliding windows of features)
    print("\nBuilding LSTM dataset (sliding windows of multi-feature returns)...")
    X, y, dates, ticker_arr = build_lstm_dataset(
        prices_used, window_size=WINDOW_SIZE, frequency=freq
    )
    print(f"Total samples built: {X.shape[0]}")
    print("X shape:", X.shape, "y shape:", y.shape)

    # 4. Time-based train/test split by label date
    split_ts = pd.to_datetime(SPLIT_DATE)
    # dates from prices are tz-aware; align split_ts to that if necessary
    if getattr(dates[0], "tzinfo", None) is not None and dates[0].tzinfo is not None:
        split_ts = split_ts.tz_localize(dates[0].tzinfo)

    train_mask = dates < split_ts
    test_mask = dates >= split_ts

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    dates_test = dates[test_mask]
    tickers_test = ticker_arr[test_mask]

    print(f"\nTrain samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

    # 5. Create PyTorch datasets/dataloaders
    X_train_t = torch.from_numpy(X_train)
    y_train_t = torch.from_numpy(y_train)
    X_test_t = torch.from_numpy(X_test)
    y_test_t = torch.from_numpy(y_test)

    train_ds = TensorDataset(X_train_t, y_train_t)
    test_ds = TensorDataset(X_test_t, y_test_t)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    # 6. Build model
    num_features = X_train.shape[2]
    print(f"\nInput features per time step: {num_features}")

    model = LSTMBinaryClassifier(
        input_size=num_features,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 7. Training loop
    print("\nTraining LSTM...")
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device).float()
            y_batch = y_batch.to(device).float().view(-1, 1)  # (batch, 1)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * X_batch.size(0)

        epoch_loss /= len(train_ds)
        print(f"Epoch {epoch:02d}/{NUM_EPOCHS} - train loss: {epoch_loss:.4f}")

    # 8. Evaluation on test set
    print("\nEvaluating on test set...")
    model.eval()
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device).float()
            logits = model(X_batch)
            all_logits.append(logits.cpu().numpy())
            all_labels.append(y_batch.numpy())

    all_logits = np.vstack(all_logits).reshape(-1)
    all_labels = np.concatenate(all_labels)

    probs = 1 / (1 + np.exp(-all_logits))  # sigmoid
    preds = (probs >= 0.5).astype(int)

    accuracy = (preds == all_labels).mean()
    print(f"Test accuracy: {accuracy:.4f}")

    # 9. Save predictions to results/
    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / "lstm_predictions.csv"

    out_df = pd.DataFrame(
        {
            "date": dates_test,
            "ticker": tickers_test,
            "y_true": all_labels,
            "y_proba": probs,
            "y_pred": preds,
        }
    ).sort_values("date")

    out_df.to_csv(out_path, index=False)
    print(f"\nSaved LSTM predictions to {out_path}")

    # Tiny summary file
    summary_path = RESULTS_DIR / "lstm_summary.txt"
    with open(summary_path, "w") as f:
        f.write("LSTM binary classifier summary\n")
        f.write(f"Universe file: {UNIVERSE_PATH}\n")
        f.write(f"Tickers used: {len(tickers)}\n")
        f.write(f"Frequency: {FREQUENCY}\n")
        f.write(f"Date range: {START_DATE} to {END_DATE}\n")
        f.write(f"Train/test split: {SPLIT_DATE}\n")
        f.write(f"Window size: {WINDOW_SIZE}\n")
        f.write(f"Hidden size: {HIDDEN_SIZE}, Num layers: {NUM_LAYERS}\n")
        f.write(f"Num epochs: {NUM_EPOCHS}, LR: {LEARNING_RATE}\n")
        f.write(f"Train samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}\n")
        f.write(f"Test accuracy: {accuracy:.4f}\n")

    print(f"Saved LSTM summary to {summary_path}")
    print("\nLSTM pipeline finished!")


if __name__ == "__main__":
    main()
