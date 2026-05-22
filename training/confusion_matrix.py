from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

from dataset import DatasetConfig, GestureDataset, build_label_list, parse_subjects
from models import ModelConfig, create_model
from utils import save_json


DEFAULT_TEST = ["S14", "S15"]


def normalize_matrix(matrix: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return matrix.astype(np.float32)
    row_sum = matrix.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    normalized = matrix / row_sum
    if mode == "percent":
        return normalized * 100.0
    return normalized


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["cnn", "lstm", "transformer"], default="cnn")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--manifest",
        default=r"g:\DOAN2\data_collection\data\processed_50hz\manifest_50hz.csv",
    )
    parser.add_argument("--root", default=r"g:\DOAN2\data_collection")
    parser.add_argument("--subjects", default=None)
    parser.add_argument("--output", default=r"g:\DOAN2\training\results\confusion")
    parser.add_argument(
        "--normalize",
        choices=["none", "row", "percent"],
        default="percent",
        help="Normalization mode: none, row (0-1), percent (0-100).",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    root_dir = Path(args.root)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_df = pd.read_csv(manifest_path)
    label_list = build_label_list(manifest_df)
    label_to_index = {label: idx for idx, label in enumerate(label_list)}
    subjects = parse_subjects(args.subjects, DEFAULT_TEST)

    norm_path = Path(args.checkpoint).parent / "normalization.json"
    with norm_path.open("r", encoding="utf-8") as f:
        norm = json.load(f)
    mean = np.array(norm["mean"], dtype=np.float32)
    std = np.array(norm["std"], dtype=np.float32)

    dataset = GestureDataset(
        DatasetConfig(
            manifest_path=manifest_path,
            root_dir=root_dir,
            subjects=subjects,
            label_to_index=label_to_index,
            mean=mean,
            std=std,
        )
    )
    loader = DataLoader(dataset, batch_size=128, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(ModelConfig(args.model, input_channels=6, num_classes=len(label_list))).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, target in loader:
            inputs = inputs.to(device)
            logits = model(inputs)
            preds = torch.argmax(logits, dim=1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(target.numpy())

    if not all_labels:
        raise RuntimeError("No samples found for the selected subjects.")

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(label_list))))
    normalized = normalize_matrix(matrix, args.normalize)

    df = pd.DataFrame(normalized, index=label_list, columns=label_list)
    out_csv = output_dir / f"confusion_{args.model}.csv"
    df.to_csv(out_csv, index=True)

    plt.figure(figsize=(12, 10))
    im = plt.imshow(normalized, interpolation="nearest", cmap="Blues")
    plt.title(f"Confusion Matrix ({args.model})")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(label_list))
    plt.xticks(tick_marks, label_list, rotation=45, ha="right")
    plt.yticks(tick_marks, label_list)
    threshold = normalized.max() * 0.6 if normalized.size else 0.0
    for i in range(normalized.shape[0]):
        for j in range(normalized.shape[1]):
            value = normalized[i, j]
            if args.normalize == "percent":
                text = f"{value:.1f}%"
            elif args.normalize == "row":
                text = f"{value:.2f}"
            else:
                text = f"{value:.0f}"
            color = "white" if value > threshold else "black"
            plt.text(j, i, text, ha="center", va="center", fontsize=6, color=color)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()

    out_png = output_dir / f"confusion_{args.model}.png"
    plt.savefig(out_png, dpi=200)
    plt.close()

    meta = {
        "model": args.model,
        "subjects": subjects,
        "normalize": args.normalize,
        "labels": label_list,
        "csv": str(out_csv),
        "png": str(out_png),
    }
    save_json(output_dir / f"confusion_{args.model}_meta.json", meta)

    print(f"Saved: {out_csv}")
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()
