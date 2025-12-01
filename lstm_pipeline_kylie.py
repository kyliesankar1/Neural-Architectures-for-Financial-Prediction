"""
LSTM baseline for stock return direction (daily or weekly) with MACD added.

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
NUM_EPOCHS = 25
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

def compute_macd(prices: pd.DataFrame,
                 fast: int = 12,
                 slow: int = 26,
                 signal: int = 9):
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
        mom_lookback = 126
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
            "ret1", "ret5", "mom", "vol20",
            "mom_rank", "mom_z",
            "macd_line", "macd_signal", "macd_hist",
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



def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("LSTM pipeline (Kylie) - with MACD")
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
    print("\nTraining...")
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
    print("\nEvaluating...")
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
    print(f"\nTest accuracy: {accuracy:.4f}")

    # Save predictions
    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / "lstm_predictions.csv"

    pd.DataFrame(
        {
            "date": dates[test_mask],
            "ticker": ticker_arr[test_mask],
            "y_true": all_labels,
            "y_proba": probs,
            "y_pred": preds,
        }
    ).to_csv(out_path, index=False)

    print(f"Saved predictions to {out_path}")

    # Summary file
    with open(RESULTS_DIR / "lstm_summary.txt", "w") as f:
        f.write("LSTM w/ MACD Summary\n")
        f.write(f"Frequency: {FREQUENCY}\n")
        f.write(f"Test accuracy: {accuracy:.4f}\n")
        f.write(f"Window size: {WINDOW_SIZE}\n")
        f.write(f"Tickers: {len(tickers)}\n")

    print("Done!")


if __name__ == "__main__":
    main()
