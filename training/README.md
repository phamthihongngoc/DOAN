# Training Pipeline (50Hz)

This folder trains 3 models (1D-CNN, LSTM, Transformer) on the 50Hz dataset and produces:
- Accuracy + macro F1
- Params count
- GFLOPs (if thop supports the model)
- Inference time per sample
- t-SNE visualization

## 1) Install dependencies

Use the same Python environment as data_collection or a new one.

```powershell
cd G:\DOAN2\training
..\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 2) Train models

Each run writes a checkpoint and logs under `G:\DOAN2\training\results`.

```powershell
cd G:\DOAN2\training
..\.venv\Scripts\python.exe train.py --model cnn
..\.venv\Scripts\python.exe train.py --model lstm
..\.venv\Scripts\python.exe train.py --model transformer
```

Summary is appended to:
- `G:\DOAN2\training\results\summary.csv`

## 3) t-SNE

Use the best checkpoint from the run folder:

```powershell
..\.venv\Scripts\python.exe tsne.py --model cnn --checkpoint G:\DOAN2\training\results\<run_id>\best.pt
```

Outputs:
- `results\tsne\tsne_<model>.csv`
- `results\tsne\tsne_<model>.png`

## Notes

- Default split is subject-based: train S01-S11, val S12-S13, test S14-S15.
- You can override with `--train-subjects`, `--val-subjects`, `--test-subjects`.
- GFLOPs uses MACs * 2 (FLOPs) from thop. If unsupported, the field is blank.
