import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import json
import torch
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader

from dataset import DatasetConfig, GestureDataset, build_label_list, parse_subjects
from models import ModelConfig, create_model
from utils import save_json


DEFAULT_TEST = ["S14", "S15"]


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
    parser.add_argument("--output", default=r"g:\DOAN2\training\results\tsne")
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

    features = []
    labels = []
    with torch.no_grad():
        for inputs, target in loader:
            inputs = inputs.to(device)
            logits, feats = model(inputs, return_features=True)
            features.append(feats.cpu().numpy())
            labels.append(target.numpy())

    features = np.concatenate(features)
    labels = np.concatenate(labels)

    tsne = TSNE(n_components=2, perplexity=30, init="pca", random_state=42)
    coords = tsne.fit_transform(features)

    out_csv = output_dir / f"tsne_{args.model}.csv"
    pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1], "label": labels}).to_csv(out_csv, index=False)

    plt.figure(figsize=(10, 8))
    cmap = plt.get_cmap("tab20", len(label_list))
    scatter = plt.scatter(
        coords[:, 0],
        coords[:, 1],
        c=labels,
        s=6,
        cmap=cmap,
        vmin=0,
        vmax=len(label_list) - 1,
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
    plt.legend(
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
    plt.title(f"t-SNE {args.model}")
    plt.tight_layout()
    out_png = output_dir / f"tsne_{args.model}.png"
    plt.savefig(out_png, dpi=200)
    plt.close()

    save_json(output_dir / f"tsne_{args.model}_labels.json", {"labels": label_list})
    print(f"Saved: {out_csv}")
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()
