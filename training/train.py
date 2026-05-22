from __future__ import annotations

import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader

from dataset import DEFAULT_COLS, DatasetConfig, GestureDataset, build_label_list, parse_subjects
from models import ModelConfig, create_model
from utils import compute_mean_std, evaluate, inference_time_ms, profile_gflops, save_json, seed_all, write_summary_xlsx


DEFAULT_TRAIN = [f"S{idx:02d}" for idx in range(1, 12)]
DEFAULT_VAL = ["S12", "S13"]
DEFAULT_TEST = ["S14", "S15"]
MODEL_NAMES = ["cnn", "lstm", "transformer"]


def terminal_style(text: str, code: str) -> str:
    if not sys.stdout.isatty():
        return text
    return f"\033[{code}m{text}\033[0m"


def format_elapsed(seconds: float) -> str:
    if seconds >= 60:
        minutes = int(seconds // 60)
        remainder = int(seconds % 60)
        return f"{minutes}m {remainder}s"
    if seconds >= 1:
        return f"{seconds:.0f}s"
    return f"{seconds * 1000:.0f}ms"


def format_step_time(seconds: float) -> str:
    if seconds >= 1:
        return f"{seconds:.0f}s/step"
    return f"{seconds * 1000:.0f}ms/step"


def format_epoch_metrics(
    steps: int,
    elapsed_seconds: float,
    train_acc: float,
    train_loss: float,
    val_acc: float,
    val_loss: float,
) -> str:
    bar = terminal_style("=" * 30, "32")
    step_count = terminal_style(f"{steps}/{steps}", "1")
    seconds_per_step = elapsed_seconds / max(steps, 1)
    return (
        f"{step_count} {bar} {format_elapsed(elapsed_seconds)} "
        f"{format_step_time(seconds_per_step)} - accuracy: {train_acc:.4f} - "
        f"loss: {train_loss:.4f} - val_accuracy: {val_acc:.4f} - val_loss: {val_loss:.4f}"
    )


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total = 0
    correct = 0
    for inputs, labels in loader:
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

    avg_loss = total_loss / max(total, 1)
    acc = correct / max(total, 1)
    return avg_loss, acc


def evaluate_with_loss(model, loader, criterion, device) -> tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            logits = model(inputs)
            loss = criterion(logits, labels)

            total_loss += float(loss.item()) * labels.size(0)
            total += labels.size(0)
            preds = torch.argmax(logits, dim=1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    if total == 0:
        return 0.0, 0.0, 0.0

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    avg_loss = total_loss / total
    acc = float((y_true == y_pred).mean())
    f1 = float(f1_score(y_true, y_pred, average="macro"))
    return avg_loss, acc, f1


def train_model(args: argparse.Namespace, model_name: str) -> dict[str, str | int | float]:
    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    manifest_path = Path(args.manifest)
    root_dir = Path(args.root)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    manifest_df = pd.read_csv(manifest_path)
    label_list = build_label_list(manifest_df)
    label_to_index = {label: idx for idx, label in enumerate(label_list)}

    train_subjects = parse_subjects(args.train_subjects, DEFAULT_TRAIN)
    val_subjects = parse_subjects(args.val_subjects, DEFAULT_VAL)
    test_subjects = parse_subjects(args.test_subjects, DEFAULT_TEST)

    mean, std = compute_mean_std(manifest_path, root_dir, train_subjects, DEFAULT_COLS)

    run_id = f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = results_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    save_json(run_dir / "labels.json", {"labels": label_list})
    save_json(run_dir / "normalization.json", {"mean": mean.tolist(), "std": std.tolist()})

    train_ds = GestureDataset(
        DatasetConfig(
            manifest_path=manifest_path,
            root_dir=root_dir,
            subjects=train_subjects,
            label_to_index=label_to_index,
            mean=mean,
            std=std,
        )
    )
    val_ds = GestureDataset(
        DatasetConfig(
            manifest_path=manifest_path,
            root_dir=root_dir,
            subjects=val_subjects,
            label_to_index=label_to_index,
            mean=mean,
            std=std,
        )
    )
    test_ds = GestureDataset(
        DatasetConfig(
            manifest_path=manifest_path,
            root_dir=root_dir,
            subjects=test_subjects,
            label_to_index=label_to_index,
            mean=mean,
            std=std,
        )
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    print(
        f"Training {model_name} | device {device} | "
        f"train {len(train_ds)} | val {len(val_ds)} | test {len(test_ds)}"
    )

    model = create_model(ModelConfig(model_name, input_channels=6, num_classes=len(label_list))).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_val = 0.0
    best_path = run_dir / "best.pt"
    history_path = run_dir / "history.csv"

    with history_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch",
            "train_loss",
            "train_acc",
            "val_loss",
            "val_acc",
            "val_f1",
            "elapsed_seconds",
        ])

    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}", flush=True)
        epoch_start = time.perf_counter()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_f1 = evaluate_with_loss(model, val_loader, criterion, device)
        epoch_seconds = time.perf_counter() - epoch_start

        with history_path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                f"{train_loss:.6f}",
                f"{train_acc:.6f}",
                f"{val_loss:.6f}",
                f"{val_acc:.6f}",
                f"{val_f1:.6f}",
                f"{epoch_seconds:.6f}",
            ])

        if val_acc > best_val:
            best_val = val_acc
            torch.save({"model": model.state_dict()}, best_path)

        print(
            format_epoch_metrics(
                steps=len(train_loader),
                elapsed_seconds=epoch_seconds,
                train_acc=train_acc,
                train_loss=train_loss,
                val_acc=val_acc,
                val_loss=val_loss,
            )
            + f" - val_f1: {val_f1:.4f}",
            flush=True,
        )

    if best_path.exists():
        checkpoint = torch.load(best_path, map_location=device)
        model.load_state_dict(checkpoint["model"])

    test_acc, test_f1 = evaluate(model, test_loader, device)
    params = sum(p.numel() for p in model.parameters())
    gflops = profile_gflops(model, input_shape=(1, 201, 6))
    latency_ms = inference_time_ms(model, test_loader, device)

    summary_path = results_dir / "summary.csv"
    write_header = not summary_path.exists()
    with summary_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                "run_id",
                "model",
                "params",
                "gflops",
                "latency_ms_per_sample",
                "test_acc",
                "test_f1",
            ])
        writer.writerow([
            run_id,
            model_name,
            params,
            f"{gflops:.4f}" if gflops is not None else "",
            f"{latency_ms:.4f}",
            f"{test_acc:.4f}",
            f"{test_f1:.4f}",
        ])

    write_summary_xlsx(summary_path, results_dir / "summary.xlsx")

    print("Done.")
    print(f"Summary: {summary_path}")
    print(f"Excel: {results_dir / 'summary.xlsx'}")
    print(f"History: {history_path}")
    return {
        "run_id": run_id,
        "model": model_name,
        "params": params,
        "gflops": f"{gflops:.4f}" if gflops is not None else "",
        "latency_ms_per_sample": f"{latency_ms:.4f}",
        "test_acc": f"{test_acc:.4f}",
        "test_f1": f"{test_f1:.4f}",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=[*MODEL_NAMES, "all"], default="cnn")
    parser.add_argument(
        "--manifest",
        default=r"G:\DOAN2\data_collection\data\processed_50hz\manifest_50hz.csv",
    )
    parser.add_argument("--root", default=r"G:\DOAN2\data_collection")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--train-subjects", default=None)
    parser.add_argument("--val-subjects", default=None)
    parser.add_argument("--test-subjects", default=None)
    parser.add_argument("--results-dir", default=r"G:\DOAN2\training\results")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    model_names = MODEL_NAMES if args.model == "all" else [args.model]
    results = [train_model(args, model_name) for model_name in model_names]
    if len(results) > 1:
        print("\nFinal summary")
        for row in results:
            print(
                f"{row['model']}: acc={row['test_acc']} f1={row['test_f1']} "
                f"latency_ms={row['latency_ms_per_sample']} run_id={row['run_id']}"
            )


if __name__ == "__main__":
    main()
