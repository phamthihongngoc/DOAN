from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

from src.dataset import DEFAULT_COLS, build_label_list, load_manifest_data
from src.metrics import (
    save_confusion_matrix,
    save_per_label_metrics,
    save_predictions,
    save_tsne,
)
from src.utils import (
    append_csv_row,
    config_path_default,
    ensure_output_dirs,
    load_config,
    output_root,
    save_json,
    seed_all,
    write_summary_xlsx,
)


def extract_features(samples: np.ndarray, feature_names: list[str]) -> np.ndarray:
    parts = []
    if "mean" in feature_names:
        parts.append(samples.mean(axis=1))
    if "std" in feature_names:
        parts.append(samples.std(axis=1))
    if "min" in feature_names:
        parts.append(samples.min(axis=1))
    if "max" in feature_names:
        parts.append(samples.max(axis=1))
    if not parts:
        raise ValueError("At least one feature is required for RandomForest.")
    return np.concatenate(parts, axis=1).astype(np.float32)


def select_split(
    manifest_df: pd.DataFrame,
    data: np.ndarray,
    subjects: list[str],
    label_to_index: dict[str, int],
    feature_names: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    subject_ids = manifest_df["subject_id"].astype(str).to_numpy()
    label_ids = manifest_df["label_id"].astype(str).to_numpy()
    mask = np.isin(subject_ids, np.asarray(subjects)) if subjects else np.ones_like(subject_ids, dtype=bool)
    samples = data[mask]
    labels = label_ids[mask]
    x = extract_features(samples, feature_names)
    y = np.array([label_to_index[label] for label in labels], dtype=np.int64)
    return x, y


def train_random_forest(cfg: dict, args: argparse.Namespace) -> dict[str, str]:
    ensure_output_dirs(cfg)
    dataset_cfg = cfg["dataset"]
    rf_cfg = cfg["random_forest"]
    seed = int(args.seed if args.seed is not None else cfg["training"]["seed"])
    seed_all(seed)

    manifest_path = Path(dataset_cfg["manifest"])
    root_dir = Path(dataset_cfg["root"])
    cols = dataset_cfg.get("columns", DEFAULT_COLS)
    feature_names = list(rf_cfg.get("features", ["mean", "std", "min", "max"]))
    manifest_df, data = load_manifest_data(manifest_path, root_dir, cols)
    label_list = build_label_list(manifest_df)
    label_to_index = {label: idx for idx, label in enumerate(label_list)}

    x_train, y_train = select_split(manifest_df, data, dataset_cfg["train_subjects"], label_to_index, feature_names)
    x_val, y_val = select_split(manifest_df, data, dataset_cfg["val_subjects"], label_to_index, feature_names)
    x_test, y_test = select_split(manifest_df, data, dataset_cfg["test_subjects"], label_to_index, feature_names)
    if len(x_train) == 0 or len(x_test) == 0:
        raise RuntimeError("Empty train/test split. Check subject filters in config.")

    out_root = output_root(cfg)
    run_id = f"random_forest_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = out_root / "runs" / run_id
    artifact_dir = run_dir / "artifacts"
    tsne_dir = run_dir / "tsne"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    tsne_dir.mkdir(parents=True, exist_ok=True)

    n_estimators = int(args.n_estimators if args.n_estimators is not None else rf_cfg["n_estimators"])
    max_depth = args.max_depth if args.max_depth is not None else rf_cfg.get("max_depth")
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=seed,
        n_jobs=-1,
        class_weight="balanced",
    )

    print(f"Training RandomForest | train={len(x_train)} val={len(x_val)} test={len(x_test)}", flush=True)
    fit_start = time.perf_counter()
    model.fit(x_train, y_train)
    fit_seconds = time.perf_counter() - fit_start

    val_pred = model.predict(x_val) if len(x_val) else np.array([], dtype=np.int64)
    val_acc = float((val_pred == y_val).mean()) if len(y_val) else 0.0
    val_f1 = float(f1_score(y_val, val_pred, average="macro", zero_division=0)) if len(y_val) else 0.0

    start = time.perf_counter()
    y_pred = model.predict(x_test)
    latency_ms = ((time.perf_counter() - start) / max(len(x_test), 1)) * 1000.0
    test_acc = float((y_pred == y_test).mean())
    test_f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))

    model_path = run_dir / "model.joblib"
    joblib.dump(model, model_path)
    save_predictions(y_test, y_pred, label_list, artifact_dir / "predictions.csv")
    save_per_label_metrics(y_test, y_pred, label_list, artifact_dir / "per_label_metrics.csv")
    save_confusion_matrix(y_test, y_pred, label_list, artifact_dir, "random_forest")

    if not args.skip_tsne:
        save_tsne(x_test, y_test, label_list, tsne_dir, "random_forest", seed=seed)

    save_json(
        run_dir / "meta.json",
        {
            "run_id": run_id,
            "model": "random_forest",
            "model_type": "classical_ml",
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "features": feature_names,
            "fit_seconds": fit_seconds,
            "val_acc": val_acc,
            "val_f1": val_f1,
            "test_acc": test_acc,
            "test_f1": test_f1,
            "latency_ms_per_sample": latency_ms,
            "labels": label_list,
            "model_path": str(model_path),
        },
    )

    summary_header = [
        "run_id",
        "model",
        "model_type",
        "params",
        "gflops",
        "latency_ms_per_sample",
        "test_acc",
        "test_f1",
        "val_acc",
        "val_f1",
        "epochs_completed",
        "stop_reason",
        "features",
        "run_dir",
    ]
    summary_path = out_root / "tables" / "model_summary.csv"
    append_csv_row(
        summary_path,
        summary_header,
        [
            run_id,
            "random_forest",
            "classical_ml",
            str(n_estimators),
            "",
            f"{latency_ms:.6f}",
            f"{test_acc:.6f}",
            f"{test_f1:.6f}",
            f"{val_acc:.6f}",
            f"{val_f1:.6f}",
            "1",
            "fit_complete",
            ",".join(feature_names),
            str(run_dir),
        ],
    )
    write_summary_xlsx(summary_path, out_root / "tables" / "model_summary.xlsx")

    print(f"Saved run: {run_dir}", flush=True)
    return {"run_id": run_id, "model": "random_forest", "run_dir": str(run_dir)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(config_path_default()))
    parser.add_argument("--n-estimators", type=int, default=None)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--skip-tsne", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cfg = load_config(args.config)
    train_random_forest(cfg, args)


if __name__ == "__main__":
    main()
