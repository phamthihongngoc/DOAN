from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def latest_history(search_dir: Path) -> Path:
    histories = sorted(search_dir.rglob("history.csv"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not histories:
        raise FileNotFoundError(f"No history.csv found under {search_dir}")
    return histories[0]


def plot_history(history_path: Path, output_path: Path, table_path: Path) -> None:
    history = pd.read_csv(history_path)
    required = {"epoch", "train_loss", "train_acc", "val_loss", "val_acc"}
    missing = required - set(history.columns)
    if missing:
        raise ValueError(f"Missing columns in {history_path}: {sorted(missing)}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    history.to_csv(table_path, index=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    axes[0].plot(history["epoch"], history["train_loss"], label="Train loss", color="#3B82F6", linewidth=2)
    axes[0].plot(history["epoch"], history["val_loss"], label="Validation loss", color="#F59E0B", linewidth=2)
    axes[0].set_title("Ham mat mat theo epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)

    axes[1].plot(history["epoch"], history["train_acc"] * 100.0, label="Train accuracy", color="#22C55E", linewidth=2)
    axes[1].plot(history["epoch"], history["val_acc"] * 100.0, label="Validation accuracy", color="#EC4899", linewidth=2)
    axes[1].set_title("Do chinh xac theo epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_ylim(0, 105)
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)

    fig.suptitle("Qua trinh huan luyen 1D-CNN tren tap du lieu 50Hz", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--history",
        default=None,
        help="Path to history.csv. If omitted, uses the latest history under checkpoints/cnn_curve_100epoch.",
    )
    parser.add_argument(
        "--search-dir",
        default=str(ROOT / "checkpoints" / "cnn_curve_100epoch"),
        help="Directory to search for history.csv when --history is omitted.",
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "results" / "plots" / "cnn_training_loss_accuracy_curve.png"),
    )
    parser.add_argument(
        "--table",
        default=str(ROOT / "results" / "tables" / "cnn_training_history_curve.csv"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    history_path = Path(args.history) if args.history else latest_history(Path(args.search_dir))
    output_path = Path(args.output)
    table_path = Path(args.table)
    plot_history(history_path, output_path, table_path)
    print(f"History: {history_path}")
    print(f"Saved plot: {output_path}")
    print(f"Saved table: {table_path}")


if __name__ == "__main__":
    main()
