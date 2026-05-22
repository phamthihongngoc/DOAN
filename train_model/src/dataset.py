from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


DEFAULT_COLS = ["ax_g", "ay_g", "az_g", "gx_dps", "gy_dps", "gz_dps"]
_MANIFEST_CACHE: dict[tuple[str, str, tuple[str, ...]], tuple[pd.DataFrame, np.ndarray]] = {}


def label_sort_key(label_id: str) -> tuple[int, int]:
    label_id = str(label_id).upper()
    prefix = label_id[:1]
    try:
        number = int(label_id[1:])
    except ValueError:
        number = 999
    prefix_order = {"G": 0, "N": 1}.get(prefix, 2)
    return prefix_order, number


def build_label_list(manifest_df: pd.DataFrame) -> list[str]:
    return sorted(manifest_df["label_id"].astype(str).unique(), key=label_sort_key)


def load_manifest_data(
    manifest_path: Path | str,
    root_dir: Path | str,
    cols: list[str] | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    cols = cols or DEFAULT_COLS
    manifest_path = Path(manifest_path)
    root_dir = Path(root_dir)
    key = (str(manifest_path.resolve()), str(root_dir.resolve()), tuple(cols))

    cached = _MANIFEST_CACHE.get(key)
    if cached is not None:
        return cached

    manifest = pd.read_csv(manifest_path).reset_index(drop=True)
    disk_cache = manifest_path.with_name(f"{manifest_path.stem}_cache.npz")
    if disk_cache.exists() and disk_cache.stat().st_mtime >= manifest_path.stat().st_mtime:
        try:
            with np.load(disk_cache) as cached_file:
                cached_cols = tuple(str(item) for item in cached_file["cols"].tolist())
                data = cached_file["data"]
                if cached_cols == tuple(cols) and data.shape[0] == len(manifest):
                    cached = (manifest, data)
                    _MANIFEST_CACHE[key] = cached
                    return cached
        except Exception:
            pass

    samples = []
    for _, row in manifest.iterrows():
        rel_path = Path(str(row["path"]))
        frame = pd.read_csv(root_dir / rel_path, usecols=cols)
        samples.append(frame[cols].to_numpy(dtype=np.float32))

    data = np.stack(samples, axis=0)
    try:
        np.savez(disk_cache, data=data, cols=np.asarray(cols))
    except PermissionError:
        # Training can continue without the disk cache if Windows keeps the file locked.
        pass

    cached = (manifest, data)
    _MANIFEST_CACHE[key] = cached
    return cached


@dataclass
class DatasetConfig:
    manifest_path: Path
    root_dir: Path
    subjects: list[str]
    label_to_index: dict[str, int]
    mean: np.ndarray | None = None
    std: np.ndarray | None = None
    cols: list[str] | None = None


class GestureDataset(Dataset):
    def __init__(self, cfg: DatasetConfig) -> None:
        self.cols = cfg.cols or DEFAULT_COLS
        full_manifest, data = load_manifest_data(cfg.manifest_path, cfg.root_dir, self.cols)
        subject_ids = full_manifest["subject_id"].astype(str)
        if cfg.subjects:
            self.indices = np.flatnonzero(subject_ids.isin(set(cfg.subjects)).to_numpy())
        else:
            self.indices = np.arange(len(full_manifest))

        self.manifest = full_manifest.iloc[self.indices].reset_index(drop=True)
        self.label_ids = full_manifest["label_id"].astype(str).to_numpy()
        self.label_to_index = cfg.label_to_index
        self.mean = None if cfg.mean is None else np.asarray(cfg.mean, dtype=np.float32)
        self.std = None if cfg.std is None else np.asarray(cfg.std, dtype=np.float32)

        samples = data[self.indices].astype(np.float32, copy=True)
        if self.mean is not None and self.std is not None:
            samples = (samples - self.mean) / self.std

        labels = [self.label_to_index[self.label_ids[int(data_idx)]] for data_idx in self.indices]
        self.samples = torch.from_numpy(np.asarray(samples, dtype=np.float32))
        self.targets = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        return self.samples[idx], self.targets[idx]


def parse_subjects(subjects: str | None, fallback: Iterable[str]) -> list[str]:
    if subjects is None:
        return list(fallback)
    subjects = subjects.strip()
    if not subjects:
        return []
    return [item.strip() for item in subjects.split(",") if item.strip()]


def split_name_for_subject(subject_id: str, train: set[str], val: set[str], test: set[str]) -> str:
    if subject_id in train:
        return "train"
    if subject_id in val:
        return "val"
    if subject_id in test:
        return "test"
    return "unused"
