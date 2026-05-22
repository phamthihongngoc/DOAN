from __future__ import annotations

import json
import csv
import time
import zipfile
from pathlib import Path
from typing import Iterable
from xml.sax.saxutils import escape

import numpy as np
import torch
from sklearn.metrics import f1_score

from dataset import load_manifest_data


def seed_all(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_mean_std(
    manifest_path: Path,
    root_dir: Path,
    subjects: Iterable[str],
    cols: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    manifest, data = load_manifest_data(manifest_path, root_dir, cols)
    if subjects:
        mask = manifest["subject_id"].astype(str).isin(set(subjects)).to_numpy()
        data = data[mask]

    if data.size == 0:
        return np.zeros(len(cols), dtype=np.float32), np.ones(len(cols), dtype=np.float32)

    flattened = data.reshape(-1, data.shape[-1]).astype(np.float64, copy=False)
    mean = flattened.mean(axis=0)
    std = flattened.std(axis=0)
    std = np.maximum(std, 1e-6)
    return mean.astype(np.float32), std.astype(np.float32)


def evaluate(model: torch.nn.Module, loader, device: torch.device) -> tuple[float, float]:
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            logits = model(inputs)
            preds = torch.argmax(logits, dim=1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    acc = float((y_true == y_pred).mean())
    f1 = float(f1_score(y_true, y_pred, average="macro"))
    return acc, f1


def inference_time_ms(model: torch.nn.Module, loader, device: torch.device, warmup: int = 3) -> float:
    model.eval()
    total_time = 0.0
    total_samples = 0

    with torch.no_grad():
        for step, (inputs, _) in enumerate(loader):
            inputs = inputs.to(device)
            if step < warmup:
                _ = model(inputs)
                continue
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(inputs)
            if device.type == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()
            total_time += (end - start)
            total_samples += inputs.size(0)

    if total_samples == 0:
        return 0.0
    return (total_time / total_samples) * 1000.0


def profile_gflops(model: torch.nn.Module, input_shape: tuple[int, int, int]) -> float | None:
    try:
        from thop import profile
        device = next(model.parameters()).device
        dummy = torch.randn(*input_shape, device=device)
        macs, _ = profile(model, inputs=(dummy,), verbose=False)
        gflops = (macs * 2.0) / 1e9
        return float(gflops)
    except Exception:
        return None


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def write_summary_xlsx(csv_path: Path, xlsx_path: Path) -> None:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))

    def col_name(index: int) -> str:
        name = ""
        index += 1
        while index:
            index, remainder = divmod(index - 1, 26)
            name = chr(65 + remainder) + name
        return name

    sheet_rows = []
    for row_idx, row in enumerate(rows, start=1):
        cells = []
        for col_idx, value in enumerate(row):
            ref = f"{col_name(col_idx)}{row_idx}"
            text = escape(str(value))
            cells.append(f'<c r="{ref}" t="inlineStr"><is><t>{text}</t></is></c>')
        sheet_rows.append(f'<row r="{row_idx}">{"".join(cells)}</row>')

    worksheet = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f'<sheetData>{"".join(sheet_rows)}</sheetData>'
        "</worksheet>"
    )
    workbook = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        '<sheets><sheet name="summary" sheetId="1" r:id="rId1"/></sheets>'
        "</workbook>"
    )
    content_types = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        '<Override PartName="/xl/worksheets/sheet1.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        "</Types>"
    )
    rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="xl/workbook.xml"/>'
        "</Relationships>"
    )
    workbook_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
        'Target="worksheets/sheet1.xml"/>'
        "</Relationships>"
    )

    xlsx_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(xlsx_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types)
        zf.writestr("_rels/.rels", rels)
        zf.writestr("xl/workbook.xml", workbook)
        zf.writestr("xl/_rels/workbook.xml.rels", workbook_rels)
        zf.writestr("xl/worksheets/sheet1.xml", worksheet)
