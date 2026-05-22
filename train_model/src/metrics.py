from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support


def normalize_confusion(matrix: np.ndarray, mode: str = "percent") -> np.ndarray:
    if mode == "none":
        return matrix.astype(np.float32)
    row_sum = matrix.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    normalized = matrix / row_sum
    if mode == "percent":
        return normalized * 100.0
    return normalized


def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_list: list[str],
    output_dir: Path,
    model_name: str,
    normalize: str = "percent",
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = list(range(len(label_list)))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    normalized = normalize_confusion(matrix, normalize)

    csv_path = output_dir / f"confusion_{model_name}.csv"
    pd.DataFrame(normalized, index=label_list, columns=label_list).to_csv(csv_path, index=True)

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(normalized, interpolation="nearest", cmap="Blues")
    ax.set_title(f"Confusion Matrix ({model_name})")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(label_list))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(label_list, rotation=45, ha="right")
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(label_list)
    threshold = normalized.max() * 0.6 if normalized.size else 0.0
    for i in range(normalized.shape[0]):
        for j in range(normalized.shape[1]):
            value = normalized[i, j]
            if normalize == "percent":
                text = f"{value:.1f}%"
            elif normalize == "row":
                text = f"{value:.2f}"
            else:
                text = f"{value:.0f}"
            color = "white" if value > threshold else "black"
            ax.text(j, i, text, ha="center", va="center", fontsize=6, color=color)
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    fig.tight_layout()
    png_path = output_dir / f"confusion_{model_name}.png"
    fig.savefig(png_path, dpi=220)
    plt.close(fig)
    return csv_path, png_path


def save_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_list: list[str],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for true_idx, pred_idx in zip(y_true, y_pred):
        rows.append(
            {
                "true_index": int(true_idx),
                "pred_index": int(pred_idx),
                "true_label": label_list[int(true_idx)],
                "pred_label": label_list[int(pred_idx)],
                "correct": int(true_idx == pred_idx),
            }
        )
    pd.DataFrame(rows).to_csv(output_path, index=False)


def save_per_label_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_list: list[str],
    output_path: Path,
) -> pd.DataFrame:
    labels = list(range(len(label_list)))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    row_sum = matrix.sum(axis=1)
    class_acc = np.divide(
        np.diag(matrix),
        np.maximum(row_sum, 1),
        out=np.zeros(len(label_list), dtype=np.float64),
        where=row_sum > 0,
    )
    df = pd.DataFrame(
        {
            "label": label_list,
            "support": support.astype(int),
            "accuracy": class_acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def save_tsne(
    features: np.ndarray,
    labels: np.ndarray,
    label_list: list[str],
    output_dir: Path,
    model_name: str,
    seed: int = 42,
) -> tuple[Path, Path] | None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if len(features) < 3:
        return None

    perplexity = min(30, max(2, (len(features) - 1) // 3))
    coords = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=seed,
    ).fit_transform(features)

    csv_path = output_dir / f"tsne_{model_name}.csv"
    pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1], "label_index": labels}).assign(
        label=lambda df: df["label_index"].map(lambda idx: label_list[int(idx)])
    ).to_csv(csv_path, index=False)

    fig, ax = plt.subplots(figsize=(11, 8))
    cmap = plt.get_cmap("tab20", len(label_list))
    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=labels,
        s=8,
        cmap=cmap,
        vmin=0,
        vmax=len(label_list) - 1,
        alpha=0.9,
    )
    legend_handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            markersize=6,
            markerfacecolor=cmap(i),
            markeredgecolor="none",
            label=label_list[i],
        )
        for i in range(len(label_list))
    ]
    ax.legend(
        handles=legend_handles,
        title="Label",
        fontsize=8,
        title_fontsize=9,
        ncol=2,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0,
        frameon=False,
    )
    ax.set_title(f"t-SNE ({model_name})")
    fig.tight_layout()
    png_path = output_dir / f"tsne_{model_name}.png"
    fig.savefig(png_path, dpi=220)
    plt.close(fig)
    return csv_path, png_path
