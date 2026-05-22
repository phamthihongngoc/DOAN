# Train Model Pipeline

Pipeline rieng cho bo du lieu IMU da chuan hoa 100Hz:

`G:\DOAN2\data_collection\data\processed_dataset_100hz\manifest_100hz.csv`

## Cau truc

- `configs/default.json`: cau hinh dataset, split subject, epoch, early stopping.
- `src/`: ma nguon train, danh gia, ve bieu do, t-SNE.
- `scripts/`: lenh PowerShell/BAT de chay pipeline.
- `outputs/runs/`: checkpoint, history, confusion matrix, t-SNE cua tung lan train.
- `outputs/tables/`: bang tong hop CSV/XLSX.
- `outputs/figures/`: bieu do tong hop cho bao cao.
- `logs/`: log khi chay script.

## Train tat ca 4 model

```powershell
cd G:\DOAN2\train_model
..\.venv\Scripts\python.exe -m src.run_all --config configs\default.json
```

Hoac chay script:

```powershell
G:\DOAN2\train_model\scripts\run_all.ps1
```

Mac dinh pipeline train:

- RandomForest
- 1D-CNN: toi da 100 epochs
- LSTM: toi da 50 epochs
- Transformer: toi da 50 epochs

So epoch toi da duoc cau hinh theo tung model trong `configs/default.json`.
Early stopping chi kich hoat sau `min_epochs` khi validation accuracy/F1 dat nguong
tot hoac khong con cai thien theo `patience`.
Moi epoch duoc ghi vao `history.csv`; neu `save_every_epoch=true`, checkpoint tung
epoch nam trong `checkpoints/`.

Mac dinh man hinh chi in ket qua sau moi epoch de log gon hon. Neu muon hien
`tqdm` theo tung batch ben trong moi epoch, them `--progress`.

## Train tung model

```powershell
cd G:\DOAN2\train_model
..\.venv\Scripts\python.exe -m src.train_random_forest --config configs\default.json
..\.venv\Scripts\python.exe -m src.train_deep --model cnn --config configs\default.json
..\.venv\Scripts\python.exe -m src.train_deep --model lstm --config configs\default.json
..\.venv\Scripts\python.exe -m src.train_deep --model transformer --config configs\default.json
```

Hoac:

```powershell
G:\DOAN2\train_model\scripts\train_one.ps1 -Model cnn
G:\DOAN2\train_model\scripts\train_one.ps1 -Model random_forest
```

## Tao lai report tu ket qua da train

```powershell
cd G:\DOAN2\train_model
..\.venv\Scripts\python.exe -m src.report --config configs\default.json
```

Hoac:

```powershell
G:\DOAN2\train_model\scripts\report.ps1
```

Report gom:

- So luong mau train/val/test.
- Phan bo mau va tong samples IMU theo subject.
- Bang so sanh params, inference, GFLOPs, accuracy, F1.
- So sanh hieu suat cac phuong phap.
- Accuracy/F1 theo tung cu chi/hoat dong.
- Confusion matrix cho moi model.
- t-SNE cho moi model.
