# Training 50Hz Clean

Thu muc nay la ban sap xep lai cua `G:\DOAN2\training` de phuc vu bao cao 50Hz.
Thu muc goc `G:\DOAN2\training` khong bi chinh sua khi tao ban clean nay.

## Dataset

- Manifest: `G:\DOAN2\data_collection\data\processed_50hz\manifest_50hz.csv`
- Tan so lay mau: 50Hz
- Moi trial: 4 giay, 201 diem thoi gian
- Split mac dinh: train S01-S11, val S12-S13, test S14-S15

## Cau truc

```text
training_50hz_clean/
├── src/             # Ma nguon train, baseline, confusion matrix, t-SNE, report assets
├── scripts/         # Script PowerShell de chay lai cac tac vu chinh
├── results/         # Hinh, bang, confusion matrix, t-SNE dung cho bao cao
├── checkpoints/     # Checkpoint va summary cua CNN/LSTM/Transformer
├── baselines/       # Ket qua RandomForest va cac ML baseline
└── archive/         # Run cu/test/misc de doi chieu neu can
```

## File quan trong cho bao cao

- Danh sach bang/hinh nen dua vao bao cao: `results\tables\report_assets.md`
- Bang so sanh model: `results\tables\model_comparison_report_vi.csv`
- Bieu do Accuracy/F1: `results\plots\model_accuracy_f1_comparison.png`
- Confusion matrix: `results\confusion\`
- t-SNE: `results\tsne\`

## Chay lai model

```powershell
G:\DOAN2\training_50hz_clean\scripts\train_cnn_100epoch.ps1
G:\DOAN2\training_50hz_clean\scripts\train_lstm_100epoch.ps1
G:\DOAN2\training_50hz_clean\scripts\train_transformer_100epoch.ps1
G:\DOAN2\training_50hz_clean\scripts\train_random_forest.ps1
```

## Tao lai bang/hinh bao cao

```powershell
G:\DOAN2\training_50hz_clean\scripts\generate_report_assets.ps1
G:\DOAN2\training_50hz_clean\scripts\generate_per_activity_f1.ps1
G:\DOAN2\training_50hz_clean\scripts\plot_training_history.ps1
```

## Ghi chu

- Thu muc `archive/` giu cac run cu/test de xem lai, khong nen dua truc tiep vao bao cao.
- Cac script moi trong `scripts/` ghi ket qua vao chinh `training_50hz_clean`, khong ghi vao `training` goc.
