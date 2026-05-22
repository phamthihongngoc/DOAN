from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from dataset import DEFAULT_COLS, build_label_list, load_manifest_data, parse_subjects
from utils import save_json, write_summary_xlsx


DEFAULT_TRAIN = [f"S{idx:02d}" for idx in range(1, 12)]
DEFAULT_VAL = ["S12", "S13"]
DEFAULT_TEST = ["S14", "S15"]


def extract_features(samples: np.ndarray) -> np.ndarray:
    # samples: (N, T, C)
    mean = samples.mean(axis=1)
    std = samples.std(axis=1)
    min_v = samples.min(axis=1)
    max_v = samples.max(axis=1)
    return np.concatenate([mean, std, min_v, max_v], axis=1)


def normalize_matrix(matrix: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return matrix.astype(np.float32)
    row_sum = matrix.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    normalized = matrix / row_sum
    if mode == "percent":
        return normalized * 100.0
    return normalized


def annotate_matrix(ax: plt.Axes, matrix: np.ndarray, normalize: str) -> None:
    threshold = matrix.max() * 0.6 if matrix.size else 0.0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if normalize == "percent":
                text = f"{value:.1f}%"
            elif normalize == "row":
                text = f"{value:.2f}"
            else:
                text = f"{value:.0f}"
            color = "white" if value > threshold else "black"
            ax.text(j, i, text, ha="center", va="center", fontsize=6, color=color)


def write_summary(summary_path: Path, row: list[str]) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    exists = summary_path.exists()
    with summary_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(
                [
                    "run_id",
                    "model",
                    "params",
                    "features",
                    "latency_ms_per_sample",
                    "test_acc",
                    "test_f1",
                ]
            )
        writer.writerow(row)


def build_model(name: str, seed: int):
    if name == "svm":
        return make_pipeline(StandardScaler(), SVC(C=10.0, gamma="scale", kernel="rbf"))
    if name == "gboost":
        return GradientBoostingClassifier(random_state=seed)
    if name == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="multi:softmax",
            eval_metric="mlogloss",
            random_state=seed,
            n_jobs=-1,
        )
    raise ValueError(f"Unknown model: {name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["svm", "gboost", "xgboost", "all"], default="all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--manifest",
        default=r"g:\DOAN2\data_collection\data\processed_50hz\manifest_50hz.csv",
    )
    parser.add_argument("--root", default=r"g:\DOAN2\data_collection")
    parser.add_argument("--train-subjects", default=None)
    parser.add_argument("--val-subjects", default=None)
    parser.add_argument("--test-subjects", default=None)
    parser.add_argument("--results-dir", default=r"g:\DOAN2\training\results_ml")
    parser.add_argument(
        "--normalize",
        choices=["none", "row", "percent"],
        default="percent",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    root_dir = Path(args.root)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    manifest_df, data = load_manifest_data(manifest_path, root_dir, DEFAULT_COLS)
    label_list = build_label_list(manifest_df)
    label_to_index = {label: idx for idx, label in enumerate(label_list)}

    subjects_train = parse_subjects(args.train_subjects, DEFAULT_TRAIN)
    subjects_val = parse_subjects(args.val_subjects, DEFAULT_VAL)
    subjects_test = parse_subjects(args.test_subjects, DEFAULT_TEST)

    subject_ids = manifest_df["subject_id"].astype(str).to_numpy()
    label_ids = manifest_df["label_id"].astype(str).to_numpy()

    def select(subjects: list[str]) -> tuple[np.ndarray, np.ndarray]:
        if subjects:
            mask = np.isin(subject_ids, np.asarray(subjects))
        else:
            mask = np.ones_like(subject_ids, dtype=bool)
        samples = data[mask]
        labels = label_ids[mask]
        x = extract_features(samples)
        y = np.array([label_to_index[label] for label in labels], dtype=np.int64)
        return x, y

    x_train, y_train = select(subjects_train)
    x_val, y_val = select(subjects_val)
    x_test, y_test = select(subjects_test)

    if len(x_train) == 0 or len(x_test) == 0:
        raise RuntimeError("Empty train/test split. Check subject filters.")

    models = ["svm", "gboost", "xgboost"] if args.model == "all" else [args.model]
    summary_csv = results_dir / "summary.csv"

    for name in models:
        run_id = f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        run_dir = results_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        model = build_model(name, args.seed)
        model.fit(x_train, y_train)

        start = time.perf_counter()
        preds = model.predict(x_test)
        elapsed = time.perf_counter() - start
        latency_ms = (elapsed / max(len(x_test), 1)) * 1000.0

        test_acc = float((preds == y_test).mean())
        test_f1 = float(f1_score(y_test, preds, average="macro"))

        matrix = confusion_matrix(y_test, preds, labels=list(range(len(label_list))))
        normalized = normalize_matrix(matrix, args.normalize)

        confusion_csv = run_dir / "confusion_matrix.csv"
        pd.DataFrame(normalized, index=label_list, columns=label_list).to_csv(confusion_csv, index=True)

        fig, ax = plt.subplots(figsize=(12, 10))
        im = ax.imshow(normalized, interpolation="nearest", cmap="Blues")
        ax.set_title(f"Confusion Matrix ({name})")
        fig.colorbar(im, fraction=0.046, pad=0.04)
        tick_marks = np.arange(len(label_list))
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(label_list, rotation=45, ha="right")
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(label_list)
        annotate_matrix(ax, normalized, args.normalize)
        ax.set_ylabel("True label")
        ax.set_xlabel("Predicted label")
        fig.tight_layout()
        confusion_png = run_dir / "confusion_matrix.png"
        fig.savefig(confusion_png, dpi=200)
        plt.close(fig)

        save_json(
            run_dir / "meta.json",
            {
                "run_id": run_id,
                "model": name,
                "features": "mean,std,min,max",
                "train_subjects": subjects_train,
                "val_subjects": subjects_val,
                "test_subjects": subjects_test,
                "test_acc": test_acc,
                "test_f1": test_f1,
                "latency_ms_per_sample": latency_ms,
                "labels": label_list,
                "confusion_csv": str(confusion_csv),
                "confusion_png": str(confusion_png),
            },
        )

        summary_row = [
            run_id,
            name,
            "",
            "mean,std,min,max",
            f"{latency_ms:.4f}",
            f"{test_acc:.4f}",
            f"{test_f1:.4f}",
        ]
        write_summary(summary_csv, summary_row)
        write_summary_xlsx(summary_csv, results_dir / "summary.xlsx")

        print(f"Run saved to: {run_dir}")
        print(f"Summary: {summary_csv}")
        print(f"Confusion: {confusion_png}")


if __name__ == "__main__":
    main()
