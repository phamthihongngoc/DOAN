from __future__ import annotations

import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.dataset import DEFAULT_COLS, DatasetConfig, GestureDataset, build_label_list
from src.metrics import (
    save_confusion_matrix,
    save_per_label_metrics,
    save_predictions,
    save_tsne,
)
from src.models import ModelConfig, create_model
from src.utils import (
    append_csv_row,
    compute_mean_std,
    config_path_default,
    ensure_output_dirs,
    evaluate_torch,
    inference_time_ms,
    load_config,
    output_root,
    profile_gflops,
    save_json,
    seed_all,
    write_summary_xlsx,
)


MODEL_NAMES = ["cnn", "lstm", "transformer"]


def train_one_epoch(model, loader, optimizer, criterion, device, epoch: int, epochs: int, show_progress: bool):
    model.train()
    total_loss = 0.0
    total = 0
    correct = 0
    progress = tqdm(
        loader,
        desc=f"Epoch {epoch:03d}/{epochs} train",
        unit="batch",
        dynamic_ncols=True,
        leave=False,
        disable=not show_progress,
    )
    for inputs, labels in progress:
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * labels.size(0)
        total += labels.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += int((preds == labels).sum().item())
        progress.set_postfix(
            loss=f"{total_loss / max(total, 1):.4f}",
            acc=f"{correct / max(total, 1):.4f}",
        )

    return total_loss / max(total, 1), correct / max(total, 1)


def evaluate_with_loss(
    model,
    loader,
    criterion,
    device,
    epoch: int,
    epochs: int,
    split: str,
    show_progress: bool,
) -> tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    all_preds = []
    all_labels = []
    progress = tqdm(
        loader,
        desc=f"Epoch {epoch:03d}/{epochs} {split}",
        unit="batch",
        dynamic_ncols=True,
        leave=False,
        disable=not show_progress,
    )
    with torch.no_grad():
        for inputs, labels in progress:
            inputs = inputs.to(device)
            labels = labels.to(device)
            logits = model(inputs)
            loss = criterion(logits, labels)
            total_loss += float(loss.item()) * labels.size(0)
            total += labels.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += int((preds == labels).sum().item())
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            progress.set_postfix(
                loss=f"{total_loss / max(total, 1):.4f}",
                acc=f"{correct / max(total, 1):.4f}",
            )

    if total == 0:
        return 0.0, 0.0, 0.0

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    return (
        total_loss / total,
        float((y_true == y_pred).mean()),
        float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    )


def extract_features(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    features = []
    labels = []
    with torch.no_grad():
        for inputs, target in loader:
            inputs = inputs.to(device)
            _, feats = model(inputs, return_features=True)
            features.append(feats.cpu().numpy())
            labels.append(target.numpy())
    return np.concatenate(features), np.concatenate(labels)


def build_datasets(cfg: dict, label_to_index: dict[str, int], mean: np.ndarray, std: np.ndarray):
    dataset_cfg = cfg["dataset"]
    cols = dataset_cfg.get("columns", DEFAULT_COLS)
    manifest_path = Path(dataset_cfg["manifest"])
    root_dir = Path(dataset_cfg["root"])
    train_subjects = dataset_cfg["train_subjects"]
    val_subjects = dataset_cfg["val_subjects"]
    test_subjects = dataset_cfg["test_subjects"]
    return (
        GestureDataset(DatasetConfig(manifest_path, root_dir, train_subjects, label_to_index, mean, std, cols)),
        GestureDataset(DatasetConfig(manifest_path, root_dir, val_subjects, label_to_index, mean, std, cols)),
        GestureDataset(DatasetConfig(manifest_path, root_dir, test_subjects, label_to_index, mean, std, cols)),
    )


def train_model(model_name: str, cfg: dict, args: argparse.Namespace) -> dict[str, str]:
    ensure_output_dirs(cfg)
    training_cfg = cfg["training"]
    dataset_cfg = cfg["dataset"]
    cols = dataset_cfg.get("columns", DEFAULT_COLS)
    seed = int(args.seed if args.seed is not None else training_cfg["seed"])
    seed_all(seed)

    manifest_path = Path(dataset_cfg["manifest"])
    root_dir = Path(dataset_cfg["root"])
    manifest_df = pd.read_csv(manifest_path)
    label_list = build_label_list(manifest_df)
    label_to_index = {label: idx for idx, label in enumerate(label_list)}
    mean, std = compute_mean_std(manifest_path, root_dir, dataset_cfg["train_subjects"], cols)

    out_root = output_root(cfg)
    run_id = f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = out_root / "runs" / run_id
    checkpoint_dir = run_dir / "checkpoints"
    artifact_dir = run_dir / "artifacts"
    tsne_dir = run_dir / "tsne"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    tsne_dir.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, test_ds = build_datasets(cfg, label_to_index, mean, std)
    batch_size = int(args.batch_size if args.batch_size is not None else training_cfg["batch_size"])
    num_workers = int(args.num_workers if args.num_workers is not None else training_cfg["num_workers"])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(ModelConfig(model_name, input_channels=len(cols), num_classes=len(label_list))).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr if args.lr is not None else training_cfg["learning_rate"]))
    criterion = nn.CrossEntropyLoss()

    epochs = int(args.epochs if args.epochs is not None else training_cfg["epochs"])
    min_epochs = int(training_cfg["min_epochs"])
    patience = int(training_cfg["patience"])
    early_stop_acc = float(training_cfg["early_stop_acc"])
    early_stop_f1 = float(training_cfg["early_stop_f1"])
    save_every_epoch = bool(training_cfg.get("save_every_epoch", True))
    show_progress = bool(training_cfg.get("show_progress", False))
    if bool(getattr(args, "progress", False)):
        show_progress = True
    if bool(getattr(args, "no_progress", False)):
        show_progress = False

    save_json(run_dir / "labels.json", {"labels": label_list})
    save_json(run_dir / "normalization.json", {"mean": mean.tolist(), "std": std.tolist(), "columns": cols})
    save_json(
        run_dir / "config_snapshot.json",
        {
            "model": model_name,
            "dataset": dataset_cfg,
            "training": training_cfg,
            "device": str(device),
            "run_id": run_id,
        },
    )

    history_path = run_dir / "history.csv"
    with history_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "val_f1", "elapsed_seconds"])

    best_score = -1.0
    best_val_acc = 0.0
    best_val_f1 = 0.0
    epochs_without_improvement = 0
    best_path = checkpoint_dir / "best.pt"
    last_path = checkpoint_dir / "last.pt"
    stop_reason = "max_epochs"
    epochs_completed = 0

    print(f"Training {model_name} | train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} device={device}", flush=True)
    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            epoch,
            epochs,
            show_progress,
        )
        val_loss, val_acc, val_f1 = evaluate_with_loss(
            model,
            val_loader,
            criterion,
            device,
            epoch,
            epochs,
            "val",
            show_progress,
        )
        elapsed = time.perf_counter() - epoch_start
        epochs_completed = epoch

        with history_path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    epoch,
                    f"{train_loss:.6f}",
                    f"{train_acc:.6f}",
                    f"{val_loss:.6f}",
                    f"{val_acc:.6f}",
                    f"{val_f1:.6f}",
                    f"{elapsed:.6f}",
                ]
            )

        checkpoint = {
            "model": model.state_dict(),
            "epoch": epoch,
            "val_acc": val_acc,
            "val_f1": val_f1,
            "model_name": model_name,
        }
        torch.save(checkpoint, last_path)
        if save_every_epoch:
            torch.save(checkpoint, checkpoint_dir / f"epoch_{epoch:03d}.pt")

        score = val_acc + val_f1
        if score > best_score + 1e-8:
            best_score = score
            best_val_acc = val_acc
            best_val_f1 = val_f1
            epochs_without_improvement = 0
            torch.save(checkpoint, best_path)
        else:
            epochs_without_improvement += 1

        print(
            f"Epoch {epoch:03d}/{epochs} "
            f"train_acc={train_acc:.4f} train_loss={train_loss:.4f} "
            f"val_acc={val_acc:.4f} val_f1={val_f1:.4f} val_loss={val_loss:.4f} "
            f"time={elapsed:.1f}s",
            flush=True,
        )

        if epoch >= min_epochs and val_acc >= early_stop_acc and val_f1 >= early_stop_f1:
            stop_reason = f"target_reached_acc_{early_stop_acc}_f1_{early_stop_f1}"
            break
        if epoch >= min_epochs and epochs_without_improvement >= patience:
            stop_reason = f"patience_{patience}"
            break

    if best_path.exists():
        checkpoint = torch.load(best_path, map_location=device)
        model.load_state_dict(checkpoint["model"])

    test_acc, test_f1, y_true, y_pred = evaluate_torch(model, test_loader, device)
    latency_ms = inference_time_ms(model, test_loader, device)
    params = sum(p.numel() for p in model.parameters())
    input_len = int(dataset_cfg["input_len"])
    gflops = profile_gflops(model, input_shape=(1, input_len, len(cols)))

    save_predictions(y_true, y_pred, label_list, artifact_dir / "predictions.csv")
    save_per_label_metrics(y_true, y_pred, label_list, artifact_dir / "per_label_metrics.csv")
    save_confusion_matrix(y_true, y_pred, label_list, artifact_dir, model_name)

    if not args.skip_tsne:
        features, tsne_labels = extract_features(model, test_loader, device)
        save_tsne(features, tsne_labels, label_list, tsne_dir, model_name, seed=seed)

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
            model_name,
            "deep_learning",
            str(params),
            f"{gflops:.6f}" if gflops is not None else "",
            f"{latency_ms:.6f}",
            f"{test_acc:.6f}",
            f"{test_f1:.6f}",
            f"{best_val_acc:.6f}",
            f"{best_val_f1:.6f}",
            str(epochs_completed),
            stop_reason,
            "raw_sequence",
            str(run_dir),
        ],
    )
    write_summary_xlsx(summary_path, out_root / "tables" / "model_summary.xlsx")

    save_json(
        run_dir / "meta.json",
        {
            "run_id": run_id,
            "model": model_name,
            "model_type": "deep_learning",
            "epochs_completed": epochs_completed,
            "stop_reason": stop_reason,
            "best_val_acc": best_val_acc,
            "best_val_f1": best_val_f1,
            "test_acc": test_acc,
            "test_f1": test_f1,
            "params": params,
            "gflops": gflops,
            "latency_ms_per_sample": latency_ms,
            "history_csv": str(history_path),
            "best_checkpoint": str(best_path),
        },
    )

    print(f"Saved run: {run_dir}", flush=True)
    return {"run_id": run_id, "model": model_name, "run_dir": str(run_dir)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(config_path_default()))
    parser.add_argument("--model", choices=[*MODEL_NAMES, "all"], default="all")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--skip-tsne", action="store_true")
    parser.add_argument("--progress", action="store_true", help="Show tqdm batch progress inside each epoch.")
    parser.add_argument("--no-progress", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cfg = load_config(args.config)
    models = MODEL_NAMES if args.model == "all" else [args.model]
    for model_name in models:
        train_model(model_name, cfg, args)


if __name__ == "__main__":
    main()
