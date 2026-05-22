from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
MANIFEST_PATH = PROJECT_ROOT / "data_collection" / "data" / "processed_50hz" / "manifest_50hz.csv"
PLOTS_DIR = ROOT / "results" / "plots"
TABLES_DIR = ROOT / "results" / "tables"
REPORT_ASSETS = TABLES_DIR / "report_assets.md"

TRAIN_SUBJECTS = {f"S{idx:02d}" for idx in range(1, 12)}
VAL_SUBJECTS = {"S12", "S13"}
TEST_SUBJECTS = {"S14", "S15"}
SPLIT_ORDER = {"train": 0, "val": 1, "test": 2, "other": 3}


def split_for_subject(subject_id: str) -> str:
    if subject_id in TRAIN_SUBJECTS:
        return "train"
    if subject_id in VAL_SUBJECTS:
        return "val"
    if subject_id in TEST_SUBJECTS:
        return "test"
    return "other"


def save_bar(df: pd.DataFrame, x: str, y: str, title: str, ylabel: str, output: Path, color: str) -> None:
    fig, ax = plt.subplots(figsize=(max(9, len(df) * 0.5), 5.5))
    ax.bar(df[x].astype(str), df[y], color=color)
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def save_accuracy_f1_plot(model_df: pd.DataFrame) -> None:
    plot_df = model_df.copy()
    plot_df["accuracy_percent"] = pd.to_numeric(plot_df["accuracy_percent"], errors="coerce")
    plot_df["macro_f1_percent"] = pd.to_numeric(plot_df["macro_f1_percent"], errors="coerce")

    x = np.arange(len(plot_df))
    width = 0.36
    fig, ax = plt.subplots(figsize=(11, 5.8))
    ax.bar(x - width / 2, plot_df["accuracy_percent"], width, label="Accuracy", color="#2f6f9f")
    ax.bar(x + width / 2, plot_df["macro_f1_percent"], width, label="Macro F1", color="#477f52")
    ax.set_title("Accuracy va Macro F1 theo model")
    ax.set_ylabel("Ty le (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["model"], rotation=25, ha="right")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "model_accuracy_f1_comparison.png", dpi=220)
    plt.close(fig)


def build_model_comparison() -> None:
    source = TABLES_DIR / "model_metrics_best_vi.csv"
    if not source.exists():
        return
    df = pd.read_csv(source)
    report_df = df.rename(
        columns={
            "accuracy_percent": "accuracy_percent",
            "macro_f1_percent": "f1_percent",
            "inference_time_ms": "inference_ms",
        }
    )[
        [
            "model",
            "params",
            "gflops",
            "inference_ms",
            "accuracy_percent",
            "f1_percent",
            "run_id",
            "source",
        ]
    ]
    report_df.to_csv(TABLES_DIR / "model_comparison_report_vi.csv", index=False)
    save_accuracy_f1_plot(df)


def build_dataset_assets() -> None:
    manifest = pd.read_csv(MANIFEST_PATH)
    manifest["split"] = manifest["subject_id"].astype(str).map(split_for_subject)
    manifest["samples"] = pd.to_numeric(manifest["samples"], errors="coerce").fillna(0).astype(int)

    split_counts = (
        manifest.groupby("split", as_index=False)
        .agg(count=("path", "count"), imu_samples=("samples", "sum"))
        .sort_values("split", key=lambda col: col.map(SPLIT_ORDER))
    )
    split_counts.to_csv(TABLES_DIR / "dataset_split_samples_vi.csv", index=False)

    subject_counts = (
        manifest.groupby(["subject_id", "split"], as_index=False)
        .agg(trials=("path", "count"), imu_samples=("samples", "sum"))
        .sort_values(["split", "subject_id"], key=lambda col: col.map(SPLIT_ORDER).fillna(99) if col.name == "split" else col)
    )
    subject_counts.to_csv(TABLES_DIR / "subject_imu_samples_vi.csv", index=False)
    save_bar(
        subject_counts,
        "subject_id",
        "imu_samples",
        "Tong so diem lay mau IMU theo nguoi thuc hien",
        "So diem lay mau IMU",
        PLOTS_DIR / "subject_imu_samples.png",
        "#477f52",
    )
    save_bar(
        subject_counts,
        "subject_id",
        "trials",
        "So luong trial theo nguoi thuc hien",
        "So trial",
        PLOTS_DIR / "subject_trial_counts.png",
        "#2f6f9f",
    )


def write_report_assets() -> None:
    content = """# Tai san bao cao 50Hz

## Dataset su dung

- Manifest 50Hz: `G:/DOAN2/data_collection/data/processed_50hz/manifest_50hz.csv`
- Moi trial dai 4 giay, 201 diem thoi gian.
- Chia tap theo subject: train S01-S11, val S12-S13, test S14-S15.

## Bang nen dua vao bao cao

- Bang so sanh model theo Params | GFLOPs | Inference | Accuracy | F1: `training_50hz_clean/results/tables/model_comparison_report_vi.csv`
- Bang tong hop model goc: `training_50hz_clean/results/tables/model_metrics_best_vi.csv`
- Bang so luong mau train/val/test: `training_50hz_clean/results/tables/dataset_split_vi.csv`
- Bang so luong mau va diem IMU train/val/test: `training_50hz_clean/results/tables/dataset_split_samples_vi.csv`
- Bang phan bo mau theo cu chi: `training_50hz_clean/results/tables/gesture_counts_vi.csv`
- Bang so mau IMU theo nguoi thuc hien: `training_50hz_clean/results/tables/subject_imu_samples_vi.csv`

## Hinh nen dua vao bao cao

- Bieu do Accuracy/F1 theo model: `training_50hz_clean/results/plots/model_accuracy_f1_comparison.png`
- Bieu do Accuracy theo model: `training_50hz_clean/results/plots/accuracy_by_model.png`
- Bieu do Macro F1 theo model: `training_50hz_clean/results/plots/f1_by_model.png`
- Bieu do so luong mau theo cu chi: `training_50hz_clean/results/plots/gesture_counts.png`
- Bieu do so luong mau theo tap train/val/test: `training_50hz_clean/results/plots/split_counts.png`
- Bieu do so mau IMU theo nguoi thuc hien: `training_50hz_clean/results/plots/subject_imu_samples.png`
- Bieu do so trial theo nguoi thuc hien: `training_50hz_clean/results/plots/subject_trial_counts.png`

## Confusion matrix cho model chinh

- CNN: `training_50hz_clean/results/confusion/confusion_cnn.png`
- LSTM: `training_50hz_clean/results/confusion/confusion_lstm.png`
- Transformer: `training_50hz_clean/results/confusion/confusion_transformer.png`

## t-SNE cho model hoc sau

- CNN: `training_50hz_clean/results/tsne/tsne_cnn.png`
- LSTM: `training_50hz_clean/results/tsne/tsne_lstm.png`
- Transformer: `training_50hz_clean/results/tsne/tsne_transformer.png`

## Goi y su dung trong bao cao

- Dung bang so sanh model de trinh bay do chinh xac, F1, do tre suy dien va do phuc tap.
- Dung confusion matrix de phan tich cac lop/cu chi de nham lan.
- Dung t-SNE de minh hoa kha nang tach lop cua dac trung hoc duoc.
- Dung cac bieu do phan bo du lieu de chung minh tap du lieu can bang va cach chia train/val/test.
"""
    REPORT_ASSETS.write_text(content, encoding="utf-8")


def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    build_model_comparison()
    build_dataset_assets()
    write_report_assets()
    print(f"Report assets updated under: {ROOT / 'results'}")


if __name__ == "__main__":
    main()
