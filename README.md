# Smart Home Gesture Recognition via IMU Sensor

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Framework-FastAPI-009688?logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/Hardware-ESP32%20%2B%20MPU6050-orange" />
  <img src="https://img.shields.io/badge/Models-CNN%20%7C%20LSTM%20%7C%20Transformer%20%7C%20RF-blueviolet" />
  <img src="https://img.shields.io/badge/Platform-Windows%20%2F%20PowerShell-0078D4?logo=windows&logoColor=white" />
  <img src="https://img.shields.io/badge/License-Academic-lightgrey" />
</p>

> **Graduation Thesis** — A real-time hand gesture recognition system using a 6-axis IMU sensor (ESP32 + MPU6050). Gestures are classified by machine learning / deep learning models and mapped to smart home device commands delivered through a live web interface.

---

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Gesture Set](#gesture-set)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Workflow](#workflow)
  - [1. Data Collection](#1-data-collection)
  - [2. Model Training (50 Hz)](#2-model-training-50-hz)
  - [3. Model Training (100 Hz Pipeline)](#3-model-training-100-hz-pipeline)
  - [4. Smart Home Demo](#4-smart-home-demo)
- [Model Performance](#model-performance)
- [Large-File Handling (Git LFS)](#large-file-handling-git-lfs)
- [Detailed Documentation](#detailed-documentation)

---

## Overview

This project explores wearable gesture-based control for smart home environments. An ESP32 microcontroller samples a MPU6050 IMU at **100 Hz**, streaming raw 6-DOF inertial data (`ax, ay, az, gx, gy, gz`) over Serial USB. A Python pipeline:

1. **Collects** labelled trial data via a Streamlit GUI.
2. **Preprocesses** and resamples signals to a clean 50 Hz dataset.
3. **Trains** four classifier architectures — Random Forest, CNN, LSTM, and Transformer — and generates evaluation assets (confusion matrices, t-SNE plots, per-class F1 scores).
4. **Deploys** a FastAPI + WebSocket backend that reads the sensor in real time, classifies each 4-second window, and updates a simulated smart home dashboard in the browser with optional text-to-speech feedback.

---

## System Architecture

```
┌─────────────┐   Serial/USB    ┌──────────────────────────────┐
│  ESP32 +    │ ─────────────▶  │  FastAPI Backend (Python)    │
│  MPU6050    │  100 Hz stream  │  • Serial reader              │
└─────────────┘                 │  • Sliding-window buffer      │
                                │  • Resample → 50 Hz (201 pts) │
                                │  • Trained model inference    │
                                │  • WebSocket broadcast        │
                                └──────────────┬───────────────┘
                                               │ WebSocket
                                ┌──────────────▼───────────────┐
                                │  Web Dashboard (HTML/JS)     │
                                │  • Live IMU charts            │
                                │  • Device state (TV, lights,  │
                                │    speaker, blinds)           │
                                │  • Text-to-speech feedback    │
                                └──────────────────────────────┘
```

---

## Gesture Set

| Label | Command              | Label   | Command                  |
|-------|----------------------|---------|--------------------------|
| G1    | System wake-up       | G9      | Speaker volume down      |
| G2    | Next device / task   | G10     | Turn light on            |
| G3    | Favourite TV channel | G11     | Turn light off           |
| G4    | TV power toggle      | G12     | Close blinds             |
| G5    | Channel up           | G13     | Open blinds              |
| G6    | Channel down         | G14     | System shutdown          |
| G7    | Voice search         | G15     | Emergency reset          |
| G8    | Speaker volume up    | N1–N5   | Noise / non-gesture      |

Full label definitions: [`data_collection/labels.json`](data_collection/labels.json)

---

## Project Structure

```
DOAN2/
├── data_collection/          # Data acquisition — ESP32 firmware, Streamlit GUI, raw CSV dataset
│   ├── firmware/             #   Arduino sketch for ESP32 + MPU6050
│   ├── streamlit_app.py      #   GUI for recording labelled trials
│   ├── data/
│   │   ├── raw/              #   Per-subject per-label CSV files (100 Hz)
│   │   └── processed_50hz/   #   Resampled dataset + manifest
│   └── labels.json           #   Gesture label definitions
│
├── training_50hz_clean/      # Primary training pipeline (50 Hz clean dataset)
│   ├── scripts/              #   PowerShell train / report scripts
│   ├── models/               #   Saved checkpoints (.pt, .joblib)
│   └── results/
│       ├── plots/            #   Accuracy/F1 comparison, training curves, t-SNE
│       ├── confusion/        #   Per-model confusion matrices
│       ├── tsne/             #   t-SNE feature visualisations
│       └── tables/           #   CSV comparison reports
│
├── train_model/              # Alternative 100 Hz training pipeline
│   ├── configs/              #   JSON experiment configs
│   ├── src/                  #   Training source code
│   └── outputs/              #   Model outputs and reports
│
├── training/                 # Legacy training artefacts (reference only)
│
├── demo/                     # Smart home demo
│   ├── backend/              #   FastAPI app, Serial reader, predictor
│   ├── static/               #   Web dashboard (HTML / CSS / JS)
│   ├── run_demo.ps1          #   One-click launcher
│   └── predict_cli.py        #   Offline prediction CLI
│
└── tools/                    # Helper scripts for report generation
```

---

## Requirements

| Dependency | Version |
|---|---|
| Python | 3.10+ |
| OS | Windows 10 / 11 (PowerShell 5.1+) |
| Hardware | ESP32 DevKit + MPU6050 (GY-521) |
| Git LFS | Required for large binary assets |

Install Git LFS once:

```powershell
git lfs install
```

---

## Installation

### 1. Clone the repository

```powershell
git clone https://github.com/phamthihongngoc/DOAN.git
cd DOAN
git lfs pull          # download datasets, checkpoints, and model files
```

### 2. Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### 3. Install dependencies per module

```powershell
# Data collection
python -m pip install -r data_collection\requirements.txt

# Model training (50 Hz)
python -m pip install -r training_50hz_clean\requirements.txt

# Demo backend
python -m pip install -r demo\requirements.txt
```

---

## Workflow

### 1. Data Collection

Flash the firmware and record labelled gesture trials.

**Flash firmware:**

Upload `data_collection\firmware\esp32_mpu6050_logger\esp32_mpu6050_logger.ino` to your ESP32 using the Arduino IDE.

**Launch the Streamlit recording GUI:**

```powershell
cd data_collection
..\.venv\Scripts\python.exe -m streamlit run streamlit_app.py
# or use the convenience script:
.\run_streamlit.ps1
```

**Recording steps:**

1. Connect the ESP32 via USB and select the correct COM port.
2. Enter a **Subject ID** (e.g., `S01`).
3. Click a gesture button (`G1`–`G15`) or noise button (`N1`–`N5`).
4. Hold the gesture for 3–5 seconds.
5. Release — the trial is saved as a CSV file at:
   `data_collection\data\raw\<Subject>\<Label>\<timestamp>.csv`

**CSV schema:**

```
time, ax_g, ay_g, az_g, gx_dps, gy_dps, gz_dps, label
```

**Dataset split (default):**

| Split      | Subjects  |
|------------|-----------|
| Train      | S01–S11   |
| Validation | S12–S13   |
| Test       | S14–S15   |

---

### 2. Model Training (50 Hz)

Working directory: `training_50hz_clean/`

**Train all models (100 epochs each):**

```powershell
training_50hz_clean\scripts\train_cnn_100epoch.ps1
training_50hz_clean\scripts\train_lstm_100epoch.ps1
training_50hz_clean\scripts\train_transformer_100epoch.ps1
training_50hz_clean\scripts\train_random_forest.ps1
```

**Generate report assets:**

```powershell
training_50hz_clean\scripts\generate_report_assets.ps1
training_50hz_clean\scripts\generate_per_activity_f1.ps1
training_50hz_clean\scripts\plot_training_history.ps1
```

**Key outputs:**

| Path | Description |
|---|---|
| `results\tables\model_comparison_report_vi.csv` | Accuracy / F1 comparison table |
| `results\plots\model_accuracy_f1_comparison.png` | Bar chart comparison |
| `results\confusion\` | Per-model confusion matrices |
| `results\tsne\` | t-SNE feature visualisations |

---

### 3. Model Training (100 Hz Pipeline)

Working directory: `train_model/`

```powershell
cd train_model
..\.venv\Scripts\python.exe -m src.run_all --config configs\default.json
# or:
.\scripts\run_all.ps1
```

Trains all four model architectures on the raw 100 Hz dataset and saves a consolidated report under `train_model\outputs\`.

---

### 4. Smart Home Demo

Working directory: `demo/`

**One-click launch (backend + browser):**

```powershell
demo\run_demo.ps1
```

**Manual launch:**

```powershell
.\.venv\Scripts\python.exe -m uvicorn demo.backend.app:app --host 0.0.0.0 --port 8000
```

Open [http://127.0.0.1:8000/](http://127.0.0.1:8000/) in your browser.

**Demo features:**

- Streams live accelerometer and gyroscope charts from the ESP32.
- Buffers a 4-second sliding window and resamples to 201 points at 50 Hz.
- Runs inference with the selected trained model.
- Updates the simulated smart home panel (TV, speaker, lights, blinds) in real time.
- Announces accepted commands via browser text-to-speech.

---

## Model Performance

> Evaluated on the held-out test set (S14–S15), 20 gesture classes (G1–G15 + N1–N5), 50 Hz dataset.

| Model          | Test Accuracy | Macro F1 |
|----------------|:-------------:|:--------:|
| Random Forest  | —             | —        |
| CNN            | —             | —        |
| LSTM           | —             | —        |
| Transformer    | —             | —        |

*Fill in the values from `training_50hz_clean\results\tables\model_comparison_report_vi.csv` after training.*

Detailed confusion matrices and per-gesture F1 plots are in `training_50hz_clean\results\`.

---

## Large-File Handling (Git LFS)

The following file types are tracked by [Git LFS](https://git-lfs.com/):

| Extension | Content |
|-----------|---------|
| `*.pt`     | PyTorch model checkpoints |
| `*.joblib` | Scikit-learn model files |
| `*.npz`    | NumPy compressed arrays (dataset splits) |
| `*.zip`    | Compressed archives |
| `*.xlsx`   | Excel report exports |

After a fresh clone, run `git lfs pull` to download all tracked assets.

Virtual-environment folders, `__pycache__`, log files, and editor settings are excluded via `.gitignore`.

---

## Detailed Documentation

| Module | README |
|---|---|
| Data collection | [`data_collection/README.md`](data_collection/README.md) |
| Training (50 Hz) | [`training_50hz_clean/README.md`](training_50hz_clean/README.md) |
| Training (100 Hz) | [`train_model/README.md`](train_model/README.md) |
| Demo backend & web | [`demo/README.md`](demo/README.md) |

---

<p align="center">
  Made with Python · FastAPI · PyTorch · Scikit-learn · ESP32
</p>
