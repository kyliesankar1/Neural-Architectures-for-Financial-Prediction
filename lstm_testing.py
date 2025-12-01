"""
LSTM pipeline with MACD for stock return direction (daily or weekly).

Features used per (date, ticker):
- 1-period return
- 5-period return
- momentum (long lookback)
- volatility
- cross-sectional momentum rank
- cross-sectional momentum zscore
- MACD (standard: EMA12 - EMA26)
- MACD signal (EMA9 of MACD)
- MACD histogram (MACD - signal)

Outputs:
- Test accuracy
- Confusion matrix
- ROC curve
- Training loss curve
- Probability histogram
- Per-ticker accuracy
- Rank Information Coefficient (IC) over time
- L/S equity curve for predicting top/bottom stocks

All files saved under results/.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve,
    auc,
    confusion_matrix,
)

# Project imports
from src.data.panel import (
    build_adj_close_panel,
    compute_returns,
    compute_momentum,
    compute_volatility,
    rank_cross_sectional,
    zscore_cross_sectional,
)

# --------------------------- CONFIG -----------------------------------

UNIVERSE_PATH = Path("data/universe/us_universe_full_filtered.csv")
RESULTS_DIR = Path("results")

START_DATE = "2014-01-01"
END_DATE = "2020-12-31"
SPLIT_DATE = "2018-01-01"

MAX_TICKERS = 30
WINDOW_SIZE = 20
BATCH_SIZE = 64
NUM_EPOCHS = 20
HIDDEN_SIZE = 32
NUM_LAYERS = 1
LR = 1e-3
SEED = 42

FREQUENCY = "weekly"   # "daily" or "weekly"


# --------------------------- HELPERS -----------------------------------

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_macd(price_series: pd.Series) -> pd.DataFrame:
    """Compute standard MACD (EMA12 - EMA26) + signal + histogram."""
    ema12 = price_series.ewm(span=12, adjust=False).mean()
    ema26 = price_series.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal
    return pd.DataFrame({"macd": macd, "macd_signal": signal, "macd_hist": hist})


# --------------------------- DATASET BUILDER ---------------------------

def build_lstm_dataset(prices: pd.DataFrame, window_size=20, frequency="daily"):
    """
    Build LSTM dataset for ALL features + MACD.

    Returns:
        X: (N, window, num_features)
        y: (N,)
        dates: (N,)
        tickers: (N,)
    """

    rets_1 = compute_returns(prices, periods=1)
    rets_5 = compute_returns(prices, periods=5)

    if frequency == "weekly":
        mom_lb = 26
        vol_win = 10
    else:
        mom_lb = 126
        vol_win = 20

    mom = compute_momentum(prices, lookback=mom_lb)
    vol = compute_volatility(rets_1, window=vol_win, annualize=True)
    mom_rank = rank_cross_sectional(mom)
    mom_z = zscore_cross_sectional(mom)

    X_list, y_list, date_list, ticker_list = [], [], [], []

    for ticker in prices.columns:

        macd_df = compute_macd(prices[ticker])

        df_feat = pd.DataFrame({
            "ret1": rets_1[ticker],
            "ret5": rets_5[ticker],
            "mom": mom[ticker],
            "vol": vol[ticker],
            "mom_rank": mom_rank[ticker],
            "mom_z": mom_z[ticker],
            "macd": macd_df["macd"],
            "macd_signal": macd_df["macd_signal"],
            "macd_hist": macd_df["macd_hist"],
        }).dropna()

        if len(df_feat) <= window_size + 1:
            continue

        # label = next-period 1-day/1-week return direction
        future_ret = rets_1[ticker].reindex(df_feat.index).shift(-1)
        df_feat["future_ret"] = future_ret
        df_feat = df_feat.dropna(subset=["future_ret"])

        feat_cols = [
            "ret1", "ret5", "mom", "vol",
            "mom_rank", "mom_z",
            "macd", "macd_signal", "macd_hist",
        ]

        # per-ticker zscore normalization
        means = df_feat[feat_cols].mean()
        stds = df_feat[feat_cols].std()

        if (stds == 0).any():
            continue

        df_feat_norm = df_feat.copy()
        df_feat_norm[feat_cols] = (df_feat[feat_cols] - means) / stds

        for i in range(window_size, len(df_feat_norm)):
            window = df_feat_norm.iloc[i - window_size : i][feat_cols].values.astype(np.float32)
            label = 1 if df_feat_norm["future_ret"].iloc[i] > 0 else 0
            date = df_feat_norm.index[i]

            X_list.append(window)
            y_list.append(label)
            date_list.append(date)
            ticker_list.append(ticker)

    X = np.stack(X_list)
    y = np.array(y_list)
    dates = np.array(date_list)
    tickers = np.array(ticker_list)

    return X, y, dates, tickers


# --------------------------- MODEL -------------------------------------

class LSTMBinary(nn.Module):
    def __init__(self, input_size, hidden, layers):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.fc(last)


# --------------------------- MAIN PIPELINE -----------------------------

def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    RESULTS_DIR.mkdir(exist_ok=True)

    print("\n=== LSTM MACD Pipeline ===")

    # 1. Universe
    uni = pd.read_csv(UNIVERSE_PATH)
    tickers = uni["ticker"].dropna().unique().tolist()[:MAX_TICKERS]
    print(f"Using {len(tickers)} tickers")

    # 2. Prices
    prices = build_adj_close_panel(tickers, start=START_DATE, end=END_DATE)

    if FREQUENCY == "weekly":
        prices_used = prices.resample("W-FRI").last()
        print("Using weekly resample")
    else:
        prices_used = prices

    # 3. Dataset
    print("\nBuilding dataset...")
    X, y, dates, tickers_arr = build_lstm_dataset(
        prices_used,
        window_size=WINDOW_SIZE,
        frequency=FREQUENCY
    )

    split_ts = pd.to_datetime(SPLIT_DATE)

    mask_train = dates < split_ts
    mask_test = dates >= split_ts

    X_train, y_train = X[mask_train], y[mask_train]
    X_test, y_test = X[mask_test], y[mask_test]
    dates_test = dates[mask_test]
    tickers_test = tickers_arr[mask_test]

    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    # 4. DataLoaders
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    # 5. Model
    num_features = X_train.shape[2]
    model = LSTMBinary(num_features, HIDDEN_SIZE, NUM_LAYERS).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.BCEWithLogitsLoss()

    # 6. Train
    print("\nTraining...")
    losses = []

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total = 0
        for xb, yb in train_loader:
            xb = xb.to(device).float()
            yb = yb.to(device).float().view(-1, 1)

            opt.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            opt.step()
            total += loss.item() * xb.size(0)

        total /= len(train_ds)
        losses.append(total)
        print(f"Epoch {epoch}: loss={total:.4f}")

    # 7. Predict
    print("\nEvaluating...")
    model.eval()
    logits_list = []
    labels_list = []

    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device).float()
            logits = model(xb).cpu().numpy().reshape(-1)
            logits_list.append(logits)
            labels_list.append(yb.numpy())

    logits = np.concatenate(logits_list)
    y_true = np.concatenate(labels_list)
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs >= 0.5).astype(int)

    acc = (preds == y_true).mean()
    print(f"Test accuracy: {acc:.4f}")

    # Save predictions
    out_df = pd.DataFrame({
        "date": dates_test,
        "ticker": tickers_test,
        "y_true": y_true,
        "y_proba": probs,
        "y_pred": preds,
    }).sort_values("date")

    out_df.to_csv(RESULTS_DIR / "lstm_macd_predictions.csv", index=False)

    # ----------------------- PLOTS ----------------------------------

    # 1. Loss curve
    plt.figure()
    plt.plot(losses)
    plt.title("Training Loss Curve")
    plt.savefig(RESULTS_DIR / "lstm_loss.png")

    # 2. Confusion Matrix
    cm = confusion_matrix(y_true, preds)
    plt.figure()
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.savefig(RESULTS_DIR / "lstm_confusion.png")

    # 3. ROC Curve
    fpr, tpr, _ = roc_curve(y_true, probs)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.legend()
    plt.title("ROC Curve")
    plt.savefig(RESULTS_DIR / "lstm_roc.png")

    # 4. Probability histogram
    plt.figure()
    plt.hist(probs, bins=30)
    plt.title("Predicted Probabilities")
    plt.savefig(RESULTS_DIR / "lstm_prob_hist.png")

    # 5. Per-ticker accuracy
    ticker_acc = out_df.groupby("ticker").apply(
        lambda df: (df["y_true"] == df["y_pred"]).mean()
    )
    ticker_acc.to_csv(RESULTS_DIR / "lstm_ticker_accuracy.csv")
    plt.figure()
    ticker_acc.sort_values().plot(kind="bar")
    plt.title("Per-Ticker Accuracy")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "lstm_ticker_accuracy.png")

    # 6. Rank IC
    rank_df = out_df.copy()
    rank_df["rank_pred"] = rank_df.groupby("date")["y_proba"].rank(ascending=False)
    rank_df["rank_true"] = rank_df.groupby("date")["y_true"].rank(ascending=False)

    ic_daily = rank_df.groupby("date").apply(
        lambda df: df["rank_pred"].corr(df["rank_true"])
    )
    ic_daily.to_csv(RESULTS_DIR / "lstm_rank_ic.csv")

    plt.figure()
    ic_daily.plot()
    plt.title("Rank IC Over Time")
    plt.savefig(RESULTS_DIR / "lstm_rank_ic.png")

    # 7. L/S equity curve from top/bottom signals
    long = rank_df.groupby("date").apply(lambda df: df.nlargest(5, "y_proba"))["y_true"].mean(level=0)
    short = rank_df.groupby("date").apply(lambda df: df.nsmallest(5, "y_proba"))["y_true"].mean(level=0)

    ls_series = long - short
    plt.figure()
    ls_series.cumsum().plot()
    plt.title("L/S Strategy Equity Curve")
    plt.savefig(RESULTS_DIR / "lstm_ls_equity.png")

    print("\nSaved all LSTM results with MACD!")
    print("Files in results/:")
    for f in sorted(RESULTS_DIR.glob("lstm_*")):
        print("  -", f)


if __name__ == "__main__":
    main()
