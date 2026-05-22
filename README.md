# DOAN - Smart Home Gesture Recognition

Du an nhan dang cu chi tay bang cam bien IMU ESP32 + MPU6050/GY-521 va ung
dung vao demo dieu khien nha thong minh. Project gom day du cac phan: thu du
lieu, tien xu ly, huan luyen mo hinh hoc may/hoc sau, tao bang bieu bao cao va
demo web dieu khien thiet bi gia lap theo thoi gian thuc.

## Noi dung chinh

- Thu du lieu IMU 6 truc: gia toc `ax, ay, az` va con quay `gx, gy, gz`.
- Bo nhan gom 15 cu chi dieu khien `G1-G15` va 5 hanh dong nhieu `N1-N5`.
- Dataset goc 100Hz, ban sach 50Hz phuc vu huan luyen va bao cao.
- Huan luyen va danh gia cac mo hinh: Random Forest, CNN, LSTM, Transformer.
- Xuat confusion matrix, t-SNE, bang so sanh Accuracy/F1 va tai san bao cao.
- Demo web FastAPI + WebSocket doc Serial truc tiep tu ESP32 va dieu khien nha
  thong minh gia lap.

## Cau truc thu muc

```text
DOAN2/
  data_collection/       # Thu du lieu ESP32 + MPU6050, Streamlit, firmware, dataset
  train_model/           # Pipeline huan luyen rieng cho dataset 100Hz
  training/              # Ma nguon va ket qua huan luyen ban goc
  training_50hz_clean/   # Ban sap xep sach cho dataset/model/bao cao 50Hz
  demo/                  # Backend FastAPI, web demo, predict CLI
  tools/                 # Script ho tro tao/tong hop bao cao
```

## Yeu cau

- Windows + PowerShell.
- Python 3.10 tro len.
- Git LFS de lay cac file lon nhu dataset, checkpoint va model:

```powershell
git lfs install
git lfs pull
```

Neu clone moi tu GitHub:

```powershell
git clone https://github.com/phamthihongngoc/DOAN.git
cd DOAN
git lfs pull
```

## Cai dat moi truong

Tao moi truong ao tai thu muc goc:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Cai cac goi theo tung phan can chay:

```powershell
python -m pip install -r data_collection\requirements.txt
python -m pip install -r training_50hz_clean\requirements.txt
python -m pip install -r demo\requirements.txt
```

## Thu du lieu

Thu muc chinh: `data_collection/`

Chay giao dien Streamlit:

```powershell
cd data_collection
..\.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

Hoac chay script co san:

```powershell
data_collection\run_streamlit.ps1
```

Quy trinh co ban:

1. Nap firmware `data_collection\firmware\esp32_mpu6050_logger\esp32_mpu6050_logger.ino`
   vao ESP32.
2. Ket noi ESP32 qua cong COM.
3. Chon `Subject ID`, nhan `G1-G15` hoac `N1-N5`.
4. Thu moi trial trong 3-5 giay.
5. File CSV duoc luu vao `data_collection\data\raw\<Subject>\<Label>\`.

Dinh dang CSV:

```text
time,ax_g,ay_g,az_g,gx_dps,gy_dps,gz_dps,label
```

## Bo nhan cu chi

| Nhan | Y nghia |
| --- | --- |
| G1 | Khoi dong he thong |
| G2 | Chon thiet bi/tac vu tiep theo |
| G3 | Kenh TV yeu thich |
| G4 | Chuyen nguon TV |
| G5 | Tang kenh TV |
| G6 | Giam kenh TV |
| G7 | Tim kiem bang giong noi |
| G8 | Tang am luong loa |
| G9 | Giam am luong loa |
| G10 | Bat den |
| G11 | Tat den |
| G12 | Dong rem |
| G13 | Mo rem |
| G14 | Tat he thong |
| G15 | Reset khan cap |
| N1-N5 | Hanh dong nhieu, khong dieu khien thiet bi |

Chi tiet label nam trong `data_collection\labels.json`.

## Huan luyen model 50Hz

Thu muc chinh: `training_50hz_clean/`

Dataset 50Hz:

```text
data_collection\data\processed_50hz\manifest_50hz.csv
```

Chia tap mac dinh:

```text
Train: S01-S11
Validation: S12-S13
Test: S14-S15
```

Chay lai cac model chinh:

```powershell
training_50hz_clean\scripts\train_cnn_100epoch.ps1
training_50hz_clean\scripts\train_lstm_100epoch.ps1
training_50hz_clean\scripts\train_transformer_100epoch.ps1
training_50hz_clean\scripts\train_random_forest.ps1
```

Tao lai bang/hinh bao cao:

```powershell
training_50hz_clean\scripts\generate_report_assets.ps1
training_50hz_clean\scripts\generate_per_activity_f1.ps1
training_50hz_clean\scripts\plot_training_history.ps1
```

Cac ket qua quan trong:

```text
training_50hz_clean\results\tables\model_comparison_report_vi.csv
training_50hz_clean\results\plots\model_accuracy_f1_comparison.png
training_50hz_clean\results\confusion\
training_50hz_clean\results\tsne\
```

## Huan luyen pipeline 100Hz

Thu muc `train_model/` dung cho pipeline rieng voi dataset 100Hz:

```powershell
cd train_model
..\.venv\Scripts\python.exe -m src.run_all --config configs\default.json
```

Hoac:

```powershell
train_model\scripts\run_all.ps1
```

Pipeline nay train Random Forest, CNN, LSTM va Transformer, sau do tao report
tong hop trong `train_model\outputs\`.

## Chay demo smart home

Thu muc chinh: `demo/`

Chay backend va web:

```powershell
demo\run_demo.ps1
```

Hoac chay truc tiep:

```powershell
.\.venv\Scripts\python.exe -m uvicorn demo.backend.app:app --host 0.0.0.0 --port 8000
```

Mo trinh duyet tai:

```text
http://127.0.0.1:8000/
```

Demo ho tro:

- Doc IMU realtime tu ESP32 qua Serial.
- Ve bieu do accelerometer va gyroscope.
- Cat cua so 4 giay, resample ve 201 diem 50Hz.
- Du doan bang model da train.
- Cap nhat trang thai TV, loa, den, rem tren giao dien web.
- Doc thong bao bang text-to-speech khi lenh duoc chap nhan.

## Ghi chu ve du lieu lon

Repo co dataset, checkpoint va model dung Git LFS. Sau khi clone, can chay
`git lfs pull` de tai day du file lon. Cac thu muc moi truong ao, cache va log
da duoc bo qua trong `.gitignore`.

## Tai lieu chi tiet

- `data_collection\README.md`: huong dan thu du lieu.
- `training_50hz_clean\README.md`: huong dan huan luyen va tao bang/hinh 50Hz.
- `train_model\README.md`: pipeline huan luyen 100Hz.
- `demo\README.md`: huong dan chay demo web va API.
