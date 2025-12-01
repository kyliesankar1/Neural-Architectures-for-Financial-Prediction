"""
 analysis script for the XGBoost pipeline.

- Loads classifier, ranker, and backtest CSVs from results/
- Computes metrics for the classifier (accuracy, ROC AUC, etc.)
- Creates predicted labels from y_proba
- Plots:
    * ROC curve (classifier)
    * Confusion matrix (classifier)
    * Equity curve (backtest)
    * Histogram of daily returns (backtest)
All plots are saved into the results/ folder.
"""

import os
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss,
    confusion_matrix,
    roc_curve,
)

# -----------------------
# 1. Load the CSV files
# -----------------------

project_root = Path(__file__).resolve().parent
results_dir = project_root / "results"

print(f"Loading results from: {results_dir}")

clf_path = results_dir / "classifier_predictions.csv"
rank_path = results_dir / "ranker_predictions.csv"
bt_path = results_dir / "backtest_results.csv"

clf_df = pd.read_csv(clf_path, parse_dates=["date"])
rank_df = pd.read_csv(rank_path, parse_dates=["date"])
bt_df = pd.read_csv(bt_path, parse_dates=["date"])

print(f"Classifier predictions shape: {clf_df.shape}")
print(f"Ranker predictions shape: {rank_df.shape}")
print(f"Backtest results shape: {bt_df.shape}\n")

print("Classifier columns:\n", list(clf_df.columns))

# ------------------------------------
# 2. CLASSIFIER: metrics + ROC + CM
# ------------------------------------

# Your file already has:
#   y_true  = actual 0/1
#   y_proba = model's predicted probability of class 1
y_true = clf_df["y_true"].astype(int)
y_proba = clf_df["y_proba"].astype(float)

# Create predicted label by thresholding at 0.5
clf_df["pred_label"] = (y_proba >= 0.5).astype(int)
y_pred = clf_df["pred_label"]

print("\n=== Classifier Metrics ===")
acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred)
rec = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
roc = roc_auc_score(y_true, y_proba)
ll = log_loss(y_true, y_proba)

print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1-score : {f1:.4f}")
print(f"ROC AUC  : {roc:.4f}")
print(f"Log loss : {ll:.4f}")

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
print("\nConfusion matrix (rows = true, cols = predicted):")
print(cm)

plt.figure(figsize=(4, 4))
plt.imshow(cm, interpolation="nearest")
plt.title("Confusion Matrix (Classifier)")
plt.colorbar()
tick_marks = [0, 1]
plt.xticks(tick_marks, ["Down (0)", "Up (1)"])
plt.yticks(tick_marks, ["Down (0)", "Up (1)"])
plt.xlabel("Predicted label")
plt.ylabel("True label")

# Annotate counts on the image
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center",
            color="white" if cm[i, j] > cm.max() / 2 else "black",
        )

plt.tight_layout()
cm_path = results_dir / "confusion_matrix_classifier.png"
plt.savefig(cm_path)
plt.close()
print(f"Saved confusion matrix to {cm_path}")

# ROC curve
fpr, tpr, _ = roc_curve(y_true, y_proba)
plt.figure(figsize=(5, 5))
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc:.3f})")
plt.plot([0, 1], [0, 1], "k--", label="Random guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (Classifier)")
plt.legend(loc="lower right")
plt.tight_layout()
roc_path = results_dir / "roc_curve_classifier.png"
plt.savefig(roc_path)
plt.close()
print(f"Saved ROC curve to {roc_path}")

# ------------------------------------
# 3. BACKTEST: equity curve + histogram
# ------------------------------------

print("\n=== Backtest summary ===")
print(bt_df.describe()[["port_return", "cum_return"]])

# Equity curve (cumulative return over time)
plt.figure(figsize=(8, 4))
plt.plot(bt_df["date"], bt_df["cum_return"])
plt.xlabel("Date")
plt.ylabel("Cumulative return")
plt.title("Equity Curve (Long/Short Strategy)")
plt.tight_layout()
eq_path = results_dir / "equity_curve_backtest.png"
plt.savefig(eq_path)
plt.close()
print(f"Saved equity curve to {eq_path}")

# Histogram of daily portfolio returns
plt.figure(figsize=(5, 4))
plt.hist(bt_df["port_return"], bins=40)
plt.xlabel("Daily portfolio return")
plt.ylabel("Frequency")
plt.title("Distribution of Daily Returns")
plt.tight_layout()
hist_path = results_dir / "hist_daily_returns.png"
plt.savefig(hist_path)
plt.close()
print(f"Saved daily return histogram to {hist_path}")

print("\nAnalysis finished successfully.")
