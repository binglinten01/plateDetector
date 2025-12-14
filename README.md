# plateDetector

Training/inference pipeline for a **license plate detector** (YOLO/Ultralytics):
- **Phase0**: download → prepare → analyze → clean → augment → synthetic → pseudo-label
- **Phase1**: HPO (Optuna) → train teacher → distillation → evaluation → export/inference

> Note: This repo has **spaces in folder names** (e.g. `license plate/phase folder/...`). On Windows, always **wrap paths in quotes** when running commands.

## What files does this repo need?

### `requirements.txt` (Python dependencies)
This project uses many libraries (CV + ML + EDA). Everything is combined into **one file**:
- **`requirements.txt`**: core + export + dev tools.

Most important: **PyTorch** is intentionally **not pinned** inside `requirements.txt` because it depends on your CUDA/CPU environment. Install torch separately for your machine.

### `configs/` (default YAML configs)
Many scripts load `configs/data_license_plate.yaml` by default, so the repo includes a **`configs/`** folder. The YAML files were copied from `license plate/config folder/` into `configs/` so it works out-of-the-box.

## Quick setup (Windows / PowerShell)

### 1) Create a Python environment
Recommended Python: **3.10–3.12**.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### 2) Install PyTorch (GPU or CPU)
- **GPU (CUDA)**: install using the official PyTorch instructions (pick the correct CUDA build)
- **CPU**: install the CPU build if you don’t have a supported GPU

> PyTorch install commands change over time. Get the latest command from `pytorch.org`.

### 3) Install project dependencies

```powershell
pip install -r requirements.txt
```

> ONNX/ONNXRuntime dependencies are included in `requirements.txt`. TensorRT is environment-specific and usually installed separately on Windows.

## Environment variables (.env)
The dataset downloader uses `python-dotenv` and reads:
- `ROBOFLOW_API_KEY`

Create a `.env` file in the repo root:

```env
ROBOFLOW_API_KEY=YOUR_KEY_HERE
```

## How to run (examples)

### Phase0
Run from the repo root. Remember to quote paths because of spaces:

```powershell
python "license plate/phase folder/phase0/01_download_datasets.py"
python "license plate/phase folder/phase0/02_prepare_dataset.py"
python "license plate/phase folder/phase0/03_analyze_dataset.py" --config "configs/data_license_plate.yaml"
python "license plate/phase folder/phase0/04_clean_dataset.py" --config "configs/data_license_plate.yaml"
python "license plate/phase folder/phase0/05_augment_dataset.py"
```

### Phase1 (teacher training)

```powershell
python "license plate/phase folder/phase1/01_train_teacher_ultimate_optimized.py" --data "configs/data_license_plate.yaml"
```

> Note: Ultralytics `--data` usually expects a **YOLO dataset `data.yaml`** (train/val paths). `configs/data_license_plate.yaml` is a pipeline config; depending on the script you may need to point to `datasets/.../data.yaml` instead.

## Quick troubleshooting
- **Paths with spaces**: always run like `python "license plate/phase folder/..."`.
- **Missing Kaggle CLI**: `pip install kaggle`, then put `kaggle.json` in `%USERPROFILE%\.kaggle\`.
- **CUDA/torch issues**: uninstall torch and reinstall the correct CUDA/CPU build for your machine.
