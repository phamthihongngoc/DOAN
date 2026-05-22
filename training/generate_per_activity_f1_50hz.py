from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader

from dataset import DatasetConfig, GestureDataset, build_label_list, parse_subjects
from models import ModelConfig, create_model


def resolve_layout() -> tuple[Path, Path, dict[str, Path]]:
    script_dir = Path(__file__).resolve().parent
    root = script_dir.parent if script_dir.name == "src" else script_dir
    results_dir = root / "results"

    if (root / "results_cnn_100epoch").exists():
        checkpoints = {
            "cnn": root / "results_cnn_100epoch" / "cnn_20260518_215132" / "best.pt",
            "lstm": root / "results_lstm_100epoch" / "lstm_20260518_222258" / "best.pt",
            "transformer": root / "results_transformer_100epoch" / "transformer_20260519_050409" / "best.pt",
        }
    else:
        checkpoints = {
            "cnn": root / "checkpoints" / "cnn_100epoch" / "cnn_20260518_215132" / "best.pt",
            "lstm": root / "checkpoints" / "lstm_100epoch" / "lstm_20260518_222258" / "best.pt",
            "transformer": root / "checkpoints" / "transformer_100epoch" / "transformer_20260519_050409" / "best.pt",
        }

    return root, results_dir, checkpoints


def load_normalization(checkpoint_path: Path) -> tuple[np.ndarray, np.ndarray]:
    norm_path = checkpoint_path.parent / "normalization.json"
    with norm_path.open("r", encoding="utf-8") as f:
        norm = json.load(f)
    return np.asarray(norm["mean"], dtype=np.float32), np.asarray(norm["std"], dtype=np.float32)


def predict_model(
    model_name: str,
    checkpoint_path: Path,
    manifest_path: Path,
    data_root: Path,
    label_to_index: dict[str, int],
) -> tuple[np.ndarray, np.ndarray]:
    mean, std = load_normalization(checkpoint_path)
    subjects = parse_subjects(None, ["S14", "S15"])
    dataset = GestureDataset(
        DatasetConfig(
            manifest_path=manifest_path,
            root_dir=data_root,
            subjects=subjects,
            label_to_index=label_to_index,
            mean=mean,
            std=std,
        )
    )
    loader = DataLoader(dataset, batch_size=128, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(ModelConfig(model_name, input_channels=6, num_classes=len(label_to_index))).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    preds = []
    labels = []
    with torch.no_grad():
        for inputs, target in loader:
            inputs = inputs.to(device)
            logits = model(inputs)
            preds.append(torch.argmax(logits, dim=1).cpu().numpy())
            labels.append(target.numpy())

    return np.concatenate(labels), np.concatenate(preds)


def save_f1_plot(metrics: pd.DataFrame, output_path: Path) -> None:
    pivot = metrics.pivot(index="label_id", columns="model", values="f1_percent").fillna(0.0)
    model_order = [name for name in ["cnn", "lstm", "transformer"] if name in pivot.columns]
    pivot = pivot[model_order]

    labels = list(pivot.index)
    x = np.arange(len(labels))
    width = min(0.8 / max(len(model_order), 1), 0.24)
    colors = {
        "cnn": "#3B82F6",
        "lstm": "#22C55E",
        "transformer": "#F59E0B",
    }

    fig, ax = plt.subplots(figsize=(14, 6))
    for idx, model_name in enumerate(model_order):
        offset = (idx - (len(model_order) - 1) / 2.0) * width
        ax.bar(
            x + offset,
            pivot[model_name],
            width=width,
            label=model_name.upper() if model_name != "cnn" else "1D-CNN",
            color=colors.get(model_name),
        )

    ax.set_title("So sanh F1-score theo tung hoat dong/cu chi")
    ax.set_ylabel("F1-score (%)")
    ax.set_xlabel("Hoat dong/cu chi")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=len(model_order))
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def save_accuracy_plot(metrics: pd.DataFrame, output_path: Path) -> None:
    pivot = metrics.pivot(index="label_id", columns="model", values="recognition_accuracy_percent").fillna(0.0)
    model_order = [name for name in ["cnn", "lstm", "transformer"] if name in pivot.columns]
    pivot = pivot[model_order]

    labels = list(pivot.index)
    x = np.arange(len(labels))
    width = min(0.8 / max(len(model_order), 1), 0.24)
    colors = {
        "cnn": "#3B82F6",
        "lstm": "#22C55E",
        "transformer": "#F59E0B",
    }

    fig, ax = plt.subplots(figsize=(14, 6))
    for idx, model_name in enumerate(model_order):
        offset = (idx - (len(model_order) - 1) / 2.0) * width
        ax.bar(
            x + offset,
            pivot[model_name],
            width=width,
            label=model_name.upper() if model_name != "cnn" else "1D-CNN",
            color=colors.get(model_name),
        )

    ax.set_title("So sanh Accuracy nhan dang theo tung hoat dong/cu chi")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlabel("Hoat dong/cu chi")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=len(model_order))
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def update_report_assets(report_path: Path, clean_layout: bool) -> None:
    if not report_path.exists():
        return
    text = report_path.read_text(encoding="utf-8")
    prefix = "training_50hz_clean" if clean_layout else "training"
    table_line = f"- Bang Precision/Recall/F1 theo tung hoat dong: `{prefix}/results/tables/per_activity_f1_score_vi.csv`"
    figure_line = f"- Bieu do so sanh F1-score theo tung hoat dong: `{prefix}/results/plots/per_activity_f1_score.png`"
    accuracy_table_line = f"- Bang Accuracy nhan dang theo tung cu chi: `{prefix}/results/tables/per_activity_accuracy_vi.csv`"
    accuracy_figure_line = f"- Bieu do so sanh Accuracy nhan dang theo tung cu chi: `{prefix}/results/plots/per_activity_accuracy.png`"

    if table_line not in text:
        text = text.replace(
            "- Bang so mau IMU theo nguoi thuc hien:",
            f"{table_line}\n- Bang so mau IMU theo nguoi thuc hien:",
        )
    if accuracy_table_line not in text:
        text = text.replace(
            "- Bang so mau IMU theo nguoi thuc hien:",
            f"{accuracy_table_line}\n- Bang so mau IMU theo nguoi thuc hien:",
        )
    if figure_line not in text:
        text = text.replace(
            "- Bieu do so mau IMU theo nguoi thuc hien:",
            f"{figure_line}\n- Bieu do so mau IMU theo nguoi thuc hien:",
        )
    if accuracy_figure_line not in text:
        text = text.replace(
            "- Bieu do so mau IMU theo nguoi thuc hien:",
            f"{accuracy_figure_line}\n- Bieu do so mau IMU theo nguoi thuc hien:",
        )
    report_path.write_text(text, encoding="utf-8")


def main() -> None:
    root, results_dir, checkpoints = resolve_layout()
    manifest_path = root.parent / "data_collection" / "data" / "processed_50hz" / "manifest_50hz.csv"
    data_root = root.parent / "data_collection"
    tables_dir = results_dir / "tables"
    plots_dir = results_dir / "plots"
    tables_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(manifest_path)
    label_list = build_label_list(manifest)
    label_to_index = {label: idx for idx, label in enumerate(label_list)}
    label_names = (
        manifest[["label_id", "label_name"]]
        .drop_duplicates()
        .set_index("label_id")["label_name"]
        .to_dict()
    )

    rows = []
    for model_name, checkpoint_path in checkpoints.items():
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Missing checkpoint for {model_name}: {checkpoint_path}")
        y_true, y_pred = predict_model(model_name, checkpoint_path, manifest_path, data_root, label_to_index)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=list(range(len(label_list))),
            zero_division=0,
        )
        for idx, label_id in enumerate(label_list):
            rows.append(
                {
                    "model": model_name,
                    "label_id": label_id,
                    "label_name": label_names.get(label_id, ""),
                    "support": int(support[idx]),
                    "precision_percent": round(float(precision[idx] * 100.0), 4),
                    "recall_percent": round(float(recall[idx] * 100.0), 4),
                    "f1_percent": round(float(f1[idx] * 100.0), 4),
                }
            )

    metrics = pd.DataFrame(rows)
    f1_table_path = tables_dir / "per_activity_f1_score_vi.csv"
    f1_plot_path = plots_dir / "per_activity_f1_score.png"
    accuracy_table_path = tables_dir / "per_activity_accuracy_vi.csv"
    accuracy_plot_path = plots_dir / "per_activity_accuracy.png"

    metrics.to_csv(f1_table_path, index=False)
    accuracy_metrics = metrics[
        ["model", "label_id", "label_name", "support", "recall_percent"]
    ].rename(columns={"recall_percent": "recognition_accuracy_percent"})
    accuracy_metrics.to_csv(accuracy_table_path, index=False)

    save_f1_plot(metrics, f1_plot_path)
    save_accuracy_plot(accuracy_metrics, accuracy_plot_path)
    update_report_assets(tables_dir / "report_assets.md", clean_layout=(root.name == "training_50hz_clean"))

    print(f"Saved table: {f1_table_path}")
    print(f"Saved plot: {f1_plot_path}")
    print(f"Saved table: {accuracy_table_path}")
    print(f"Saved plot: {accuracy_plot_path}")


if __name__ == "__main__":
    main()
