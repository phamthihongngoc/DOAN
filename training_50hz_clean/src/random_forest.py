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
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score

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
                    "n_estimators",
                    "max_depth",
                    "features",
                    "latency_ms_per_sample",
                    "test_acc",
                    "test_f1",
                ]
            )
        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--manifest",
        default=r"g:\DOAN2\data_collection\data\processed_50hz\manifest_50hz.csv",
    )
    parser.add_argument("--root", default=r"g:\DOAN2\data_collection")
    parser.add_argument("--train-subjects", default=None)
    parser.add_argument("--val-subjects", default=None)
    parser.add_argument("--test-subjects", default=None)
    parser.add_argument("--results-dir", default=r"G:\DOAN2\training_50hz_clean\baselines\random_forest")
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

    run_id = f"rf_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = results_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=args.seed,
        n_jobs=-1,
        class_weight="balanced",
    )
    model.fit(x_train, y_train)
    model_path = run_dir / "model.joblib"
    joblib.dump(model, model_path)
    save_json(run_dir / "labels.json", {"labels": label_list})

    start = time.perf_counter()
    preds = model.predict(x_test)
    elapsed = time.perf_counter() - start
    latency_ms = (elapsed / max(len(x_test), 1)) * 1000.0

    test_acc = float((preds == y_test).mean()) if len(y_test) else 0.0
    test_f1 = float(f1_score(y_test, preds, average="macro")) if len(y_test) else 0.0

    matrix = confusion_matrix(y_test, preds, labels=list(range(len(label_list))))
    normalized = normalize_matrix(matrix, args.normalize)

    save_json(
        run_dir / "meta.json",
        {
            "run_id": run_id,
            "model": "random_forest",
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "features": "mean,std,min,max",
            "train_subjects": subjects_train,
            "val_subjects": subjects_val,
            "test_subjects": subjects_test,
            "test_acc": test_acc,
            "test_f1": test_f1,
            "latency_ms_per_sample": latency_ms,
            "labels": label_list,
            "model_file": model_path.name,
        },
    )

    confusion_csv = run_dir / "confusion_matrix.csv"
    pd.DataFrame(normalized, index=label_list, columns=label_list).to_csv(confusion_csv, index=True)

    plt.figure(figsize=(12, 10))
    im = plt.imshow(normalized, interpolation="nearest", cmap="Blues")
    plt.title("Confusion Matrix (Random Forest)")
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
    confusion_png = run_dir / "confusion_matrix.png"
    plt.savefig(confusion_png, dpi=200)
    plt.close()

    summary_row = [
        run_id,
        "random_forest",
        str(args.n_estimators),
        "" if args.max_depth is None else str(args.max_depth),
        "mean,std,min,max",
        f"{latency_ms:.4f}",
        f"{test_acc:.4f}",
        f"{test_f1:.4f}",
    ]
    summary_csv = results_dir / "summary.csv"
    write_summary(summary_csv, summary_row)
    write_summary_xlsx(summary_csv, results_dir / "summary.xlsx")

    print(f"Run saved to: {run_dir}")
    print(f"Summary: {summary_csv}")
    print(f"Confusion: {confusion_png}")


if __name__ == "__main__":
    main()
