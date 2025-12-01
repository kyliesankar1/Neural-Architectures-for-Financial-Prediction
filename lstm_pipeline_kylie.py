"""
LSTM baseline for stock return direction.

What it does:
- Load universe from data/universe/us_universe_full_filtered.csv
- Load adjusted close prices with build_adj_close_panel (handles splits/dividends)
- Compute daily returns with compute_returns
- Build sliding windows of returns per stock (sequence length = window_size)
- Train an LSTM to predict whether the next day's return is > 0 (up/down)
- Split train/test by date (time-based split, like the XGBoost pipeline)
- Save test predictions to results/lstm_predictions.csv
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

# Project helpers 
from src.data.panel import build_adj_close_panel, compute_returns

UNIVERSE_PATH = Path("data/universe/us_universe_full_filtered.csv")
RESULTS_DIR = Path("results")

START_DATE = "2014-01-01"
END_DATE = "2020-12-31"
SPLIT_DATE = "2018-01-01"

MAX_TICKERS = 30          # limit due to speed
WINDOW_SIZE = 20          # how many past days the LSTM sees
BATCH_SIZE = 64
NUM_EPOCHS = 10
HIDDEN_SIZE = 32
NUM_LAYERS = 1
LEARNING_RATE = 1e-3
SEED = 42

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_lstm_dataset(
    prices: pd.DataFrame,
    window_size: int = 20,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build LSTM samples from price data.

    Steps:
      - Compute daily returns using project helper compute_returns.
      - For each ticker, create sliding windows of length `window_size`.
      - Input X: sequence of normalized returns (window_size, 1)
      - Label y: 1 if next-day return > 0, else 0
      - Also track the date and ticker for each sample so we can do
        a time-based train/test split and interpret outputs.

    Returns:
      X: (N, window_size, 1) float32
      y: (N,) int64
      dates: (N,) array of pd.Timestamp (for label day)
      tickers: (N,) array of ticker strings
    """
    # Compute daily returns using the same helper the XGBoost pipeline uses
    rets = compute_returns(prices, periods=1).dropna(how="all")

    X_list: List[np.ndarray] = []
    y_list: List[int] = []
    date_list: List[pd.Timestamp] = []
    ticker_list: List[str] = []

    for ticker in rets.columns:
        s = rets[ticker].dropna()
        if len(s) <= window_size:
            continue

        # Normalize returns per ticker (z-score)
        mean = s.mean()
        std = s.std()
        if std == 0 or np.isnan(std):
            # skip tickers with no variation
            continue
        s_norm = (s - mean) / std

        # Create sliding windows
        for i in range(len(s_norm) - window_size):
            window = s_norm.iloc[i : i + window_size].values.astype(np.float32)
            window = window.reshape(-1, 1)  # (window_size, 1)

            future_ret = s.iloc[i + window_size]  # unnormalized return for label
            label = 1 if future_ret > 0 else 0
            label_date = s.index[i + window_size]

            X_list.append(window)
            y_list.append(label)
            date_list.append(label_date)
            ticker_list.append(ticker)

    X = np.stack(X_list, axis=0)  # (N, window_size, 1)
    y = np.array(y_list, dtype=np.int64)
    dates = np.array(date_list)
    tickers = np.array(ticker_list)

    return X, y, dates, tickers


# -----------------------------
# PyTorch model
# -----------------------------
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


def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("LSTM pipeline (Kylie)")
    print("=" * 80)

    # 1. Load universe
    print(f"\nLoading universe from {UNIVERSE_PATH} ...")
    universe = pd.read_csv(UNIVERSE_PATH)
    all_tickers = universe["ticker"].dropna().unique().tolist()
    tickers = all_tickers[:MAX_TICKERS]
    print(f"Total tickers in file: {len(all_tickers)}; using first {len(tickers)}")
    print("Some tickers:", tickers[:10])

    # 2. Load prices via project helper (adjusted close)
    print("\nLoading adjusted close prices via build_adj_close_panel...")
    prices = build_adj_close_panel(
        tickers,
        start=START_DATE,
        end=END_DATE,
    )
    print("Price panel shape:", prices.shape)
    print("First few rows of prices:\n", prices.head())

    if prices.empty:
        raise SystemExit(
            "ERROR: prices DataFrame is empty. "
            "Check your data/universe files or date range."
        )

    # 3. Build LSTM dataset (sliding windows of returns)
    print("\nBuilding LSTM dataset (sliding windows of returns)...")
    X, y, dates, ticker_arr = build_lstm_dataset(prices, window_size=WINDOW_SIZE)
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
    train_ds = TensorDataset(
        torch.from_numpy(X_train), torch.from_numpy(y_train)
    )
    test_ds = TensorDataset(
        torch.from_numpy(X_test), torch.from_numpy(y_test)
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    # 6. Build model
    model = LSTMBinaryClassifier(
        input_size=1,
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
            X_batch = X_batch.to(device)
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
            X_batch = X_batch.to(device)
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

    # 10. Tiny summary file
    summary_path = RESULTS_DIR / "lstm_summary.txt"
    with open(summary_path, "w") as f:
        f.write("LSTM binary classifier summary\n")
        f.write(f"Universe file: {UNIVERSE_PATH}\n")
        f.write(f"Tickers used: {len(tickers)}\n")
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
