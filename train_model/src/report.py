from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.dataset import split_name_for_subject
from src.utils import config_path_default, ensure_output_dirs, load_config, output_root, write_summary_xlsx


SPLIT_ORDER = ["train", "val", "test", "unused"]


def percent(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce") * 100.0


def save_bar(
    df: pd.DataFrame,
    x: str,
    y: str,
    output_path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
    color: str = "#2f6f9f",
    rotate: int = 0,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(max(8, len(df) * 0.45), 5))
    ax.bar(df[x].astype(str), df[y], color=color)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=rotate)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def save_grouped_bar(
    df: pd.DataFrame,
    category_col: str,
    value_col: str,
    group_col: str,
    output_path: Path,
    title: str,
    ylabel: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pivot = df.pivot_table(index=category_col, columns=group_col, values=value_col, aggfunc="mean").fillna(0.0)
    fig_width = max(10, len(pivot.index) * 0.55)
    fig, ax = plt.subplots(figsize=(fig_width, 5.8))
    x = np.arange(len(pivot.index))
    groups = list(pivot.columns)
    width = min(0.8 / max(len(groups), 1), 0.22)
    for idx, group in enumerate(groups):
        offset = (idx - (len(groups) - 1) / 2.0) * width
        ax.bar(x + offset, pivot[group], width=width, label=str(group))
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index.astype(str), rotation=45, ha="right")
    ax.set_ylim(0, max(100.0, float(pivot.max().max()) * 1.08 if not pivot.empty else 100.0))
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=min(len(groups), 4))
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def build_dataset_tables(cfg: dict) -> None:
    dataset_cfg = cfg["dataset"]
    out_root = output_root(cfg)
    table_dir = out_root / "tables"
    figure_dir = out_root / "figures"
    manifest = pd.read_csv(dataset_cfg["manifest"])

    train = set(dataset_cfg["train_subjects"])
    val = set(dataset_cfg["val_subjects"])
    test = set(dataset_cfg["test_subjects"])
    manifest["split"] = manifest["subject_id"].astype(str).map(lambda s: split_name_for_subject(s, train, val, test))
    manifest["samples"] = pd.to_numeric(manifest["samples"], errors="coerce").fillna(0).astype(int)

    split_counts = (
        manifest.groupby("split", as_index=False)
        .agg(trials=("path", "count"), imu_samples=("samples", "sum"))
        .sort_values("split", key=lambda col: col.map({name: idx for idx, name in enumerate(SPLIT_ORDER)}))
    )
    split_counts.to_csv(table_dir / "split_counts.csv", index=False)
    save_bar(split_counts, "split", "trials", figure_dir / "split_trial_counts.png", "Trials by split", "Split", "Trials")

    label_split = manifest.groupby(["label_id", "split"], as_index=False).agg(trials=("path", "count"))
    label_split.to_csv(table_dir / "label_split_counts.csv", index=False)
    save_grouped_bar(
        label_split,
        "label_id",
        "trials",
        "split",
        figure_dir / "label_split_counts.png",
        "Trials by gesture and split",
        "Trials",
    )

    subject_samples = (
        manifest.groupby("subject_id", as_index=False)
        .agg(trials=("path", "count"), imu_samples=("samples", "sum"))
        .sort_values("subject_id")
    )
    subject_samples.to_csv(table_dir / "subject_imu_samples.csv", index=False)
    save_bar(
        subject_samples,
        "subject_id",
        "imu_samples",
        figure_dir / "subject_imu_samples.png",
        "Total IMU samples by subject",
        "Subject",
        "IMU samples",
        color="#477f52",
        rotate=45,
    )


def latest_model_summary(summary_path: Path) -> pd.DataFrame:
    if not summary_path.exists():
        return pd.DataFrame()
    summary = pd.read_csv(summary_path)
    if summary.empty:
        return summary
    return summary.drop_duplicates(subset=["model"], keep="last").reset_index(drop=True)


def build_model_tables(cfg: dict) -> pd.DataFrame:
    out_root = output_root(cfg)
    table_dir = out_root / "tables"
    figure_dir = out_root / "figures"
    latest = latest_model_summary(table_dir / "model_summary.csv")
    if latest.empty:
        return latest

    comparison = latest.copy()
    comparison["accuracy_percent"] = percent(comparison["test_acc"])
    comparison["macro_f1_percent"] = percent(comparison["test_f1"])
    comparison["latency_ms_per_sample"] = pd.to_numeric(comparison["latency_ms_per_sample"], errors="coerce")
    comparison["params"] = pd.to_numeric(comparison["params"], errors="coerce")
    comparison["gflops"] = pd.to_numeric(comparison["gflops"], errors="coerce")
    comparison = comparison[
        [
            "model",
            "params",
            "gflops",
            "latency_ms_per_sample",
            "accuracy_percent",
            "macro_f1_percent",
            "epochs_completed",
            "stop_reason",
            "run_id",
            "run_dir",
        ]
    ]
    comparison.to_csv(table_dir / "model_comparison.csv", index=False)
    write_summary_xlsx(table_dir / "model_comparison.csv", table_dir / "model_comparison.xlsx")

    save_grouped_bar(
        comparison.melt(id_vars=["model"], value_vars=["accuracy_percent", "macro_f1_percent"], var_name="metric", value_name="value"),
        "model",
        "value",
        "metric",
        figure_dir / "model_performance_comparison.png",
        "Accuracy and macro F1 by model",
        "Percent",
    )
    save_bar(
        comparison.sort_values("accuracy_percent", ascending=False),
        "model",
        "accuracy_percent",
        figure_dir / "model_accuracy_comparison.png",
        "Accuracy by method",
        "Model",
        "Accuracy (%)",
        color="#2f6f9f",
        rotate=20,
    )
    save_bar(
        comparison.sort_values("latency_ms_per_sample"),
        "model",
        "latency_ms_per_sample",
        figure_dir / "model_inference_latency.png",
        "Inference latency by method",
        "Model",
        "ms/sample",
        color="#8a5a44",
        rotate=20,
    )
    return comparison


def build_per_activity_tables(cfg: dict, comparison: pd.DataFrame) -> None:
    if comparison.empty:
        return

    out_root = output_root(cfg)
    table_dir = out_root / "tables"
    figure_dir = out_root / "figures"
    frames = []
    for _, row in comparison.iterrows():
        metrics_path = Path(str(row["run_dir"])) / "artifacts" / "per_label_metrics.csv"
        if not metrics_path.exists():
            continue
        df = pd.read_csv(metrics_path)
        df["model"] = row["model"]
        frames.append(df)

    if not frames:
        return

    per_activity = pd.concat(frames, ignore_index=True)
    per_activity["accuracy_percent"] = percent(per_activity["accuracy"])
    per_activity["f1_percent"] = percent(per_activity["f1"])
    per_activity["recall_percent"] = percent(per_activity["recall"])
    per_activity.to_csv(table_dir / "per_activity_metrics.csv", index=False)
    write_summary_xlsx(table_dir / "per_activity_metrics.csv", table_dir / "per_activity_metrics.xlsx")

    save_grouped_bar(
        per_activity,
        "label",
        "accuracy_percent",
        "model",
        figure_dir / "activity_accuracy_by_model.png",
        "Recognition accuracy by gesture",
        "Accuracy (%)",
    )
    save_grouped_bar(
        per_activity,
        "label",
        "f1_percent",
        "model",
        figure_dir / "activity_f1_by_model.png",
        "F1-score by gesture",
        "F1-score (%)",
    )


def build_report_index(cfg: dict, comparison: pd.DataFrame) -> None:
    out_root = output_root(cfg)
    report_path = out_root / "reports" / "report_index.md"
    figures = sorted((out_root / "figures").glob("*.png"))
    tables = sorted((out_root / "tables").glob("*.csv"))
    runs = sorted((out_root / "runs").glob("*"))

    lines = [
        "# Training Report Index",
        "",
        "## Dataset",
        f"- Manifest: `{cfg['dataset']['manifest']}`",
        f"- Sampling: {cfg['dataset']['target_hz']}Hz",
        f"- Window seconds: {cfg['dataset']['window_seconds']}",
        "",
        "## Latest Model Comparison",
    ]
    if comparison.empty:
        lines.append("- No model summary yet. Train models first.")
    else:
        for _, row in comparison.iterrows():
            lines.append(
                f"- {row['model']}: accuracy={row['accuracy_percent']:.2f}%, "
                f"f1={row['macro_f1_percent']:.2f}%, latency={row['latency_ms_per_sample']:.4f}ms"
            )

    lines.extend(["", "## Tables"])
    lines.extend([f"- `{path}`" for path in tables])
    lines.extend(["", "## Figures"])
    lines.extend([f"- `{path}`" for path in figures])
    lines.extend(["", "## Runs"])
    lines.extend([f"- `{path}`" for path in runs if path.is_dir()])

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_report(cfg: dict) -> None:
    ensure_output_dirs(cfg)
    build_dataset_tables(cfg)
    comparison = build_model_tables(cfg)
    build_per_activity_tables(cfg, comparison)
    build_report_index(cfg, comparison)
    print(f"Report saved under: {output_root(cfg)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(config_path_default()))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cfg = load_config(args.config)
    generate_report(cfg)


if __name__ == "__main__":
    main()
