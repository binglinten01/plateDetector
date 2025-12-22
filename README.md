# plateDetector — Quick Guide (Windows)

License plate detection pipeline built on YOLO/Ultralytics. This repo is prepared for Windows, uses default configs from the repo root, and includes example commands to run each phase.

## Quick Start Checklist
- Install Python 3.10–3.12.
- Create and activate a virtual environment.
- Install PyTorch (GPU or CPU) from `pytorch.org`.
- Install project deps: `pip install -r requirements.txt`.
- Add `.env` with `ROBOFLOW_API_KEY` if using downloads.
- Run Phase0 scripts to prepare the dataset.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
# Install PyTorch per your CUDA/CPU from pytorch.org
# Example (CPU): pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Important: folder names contain spaces (e.g., `license plate/phase folder/...`). Always wrap paths in quotes on Windows.

## Pipeline Overview
- Phase0: download → prepare → analyze → clean → augment → synthetic → pseudo-label
- Phase1: HPO (Optuna) → train teacher → distillation → evaluation → export/inference
- Phase2: notebook utilities (read plate)

## Repo Structure

```
gitignore
README.md
requirements.txt
configs/
  data_license_plate.yaml
  inference_optimized.yaml
  search_space_rtx4060.yaml
  train_student_distilled.yaml
  train_teacher_ultimate.yaml
  yolo11s_ultimate.yaml
license plate/
  config folder/ (same YAMLs mirrored in `configs/`)
  log folder/
  phase folder/
    phase0/ (data preparation)
    phase1/ (training, evaluation, export, inference)
    phase2/ (notebook)
report folder/
  ultimate_analysis_report.json
```

The `configs/` directory lets you run scripts directly from the repo root.

## Requirements
- Windows + PowerShell
- Python 3.10–3.12 (recommended)
- PyTorch: not pinned in `requirements.txt`; install separately for your CUDA/CPU.

## Environment Variables (.env)
Used for dataset downloads:
- `ROBOFLOW_API_KEY`

Create `.env` in the repo root:

```env
ROBOFLOW_API_KEY=YOUR_KEY_HERE
```

Kaggle CLI: `pip install kaggle` and place `kaggle.json` in `%USERPROFILE%\.kaggle\`.

## Configs (YAML)
- Many scripts accept `--config` like `--config configs/data_license_plate.yaml`.
- Ultralytics YOLO’s `--data` expects a dataset `data.yaml` with `train/val` paths. In this pipeline, `configs/data_license_plate.yaml` is a pipeline config; for pure YOLO commands, point to your actual dataset `data.yaml`.

Example `data.yaml` template (adjust paths and classes):

```yaml
train: datasets/license_plate/train/images
val: datasets/license_plate/val/images
test: datasets/license_plate/test/images  # optional
nc: 1
names: ["plate"]
```

## How to Run

Run commands from the repo root and quote paths with spaces.

### Phase0 — Data Preparation

```powershell
python "license plate/phase folder/phase0/01_download_datasets.py"
python "license plate/phase folder/phase0/02_prepare_dataset.py"
python "license plate/phase folder/phase0/03_analyze_dataset.py" --config "configs/data_license_plate.yaml"
python "license plate/phase folder/phase0/04_clean_dataset.py" --config "configs/data_license_plate.yaml"
python "license plate/phase folder/phase0/05_augment_dataset.py"
python "license plate/phase folder/phase0/06_generate_synthetic.py"
python "license plate/phase folder/phase0/07_pseudo_label.py"
python "license plate/phase folder/phase0/end_p0_create_optimal_dataset.py"
```

Outputs: a normalized/optimized dataset ready for training.

### Phase1 — Training, Evaluation, Export

Train teacher model:

```powershell
python "license plate/phase folder/phase1/01_train_teacher_ultimate_optimized.py" --data "configs/data_license_plate.yaml"
```

Optional steps:

```powershell
# Hyperparameter optimization (Optuna)
python "license plate/phase folder/phase1/00_hyperparameter_optuna_ultimate.py"

# Knowledge distillation (student)
python "license plate/phase folder/phase1/02_knowledge_distillation_full.py"

# Advanced evaluation
python "license plate/phase folder/phase1/03_advanced_evaluation_suite.py"

# Production export (ONNX, etc.)
python "license plate/phase folder/phase1/04_production_export_ultimate.py"

# Ensemble + TTA inference
python "license plate/phase folder/phase1/05_ensemble_tta_inference.py"

# Pseudo-labeling loop
python "license plate/phase folder/phase1/08_pseudo_labeling.py"
python "license plate/phase folder/phase1/09_build_distill_dataset.py"
```

### Phase2 — Notebook Utilities

```powershell
code "license plate/phase folder/phase2/readPlate.ipynb"
```

## Troubleshooting
- Paths with spaces: always use quoted paths like `"license plate/phase folder/..."`.
- Kaggle CLI: install `kaggle` and place `kaggle.json` in `%USERPROFILE%\.kaggle\`.
- CUDA/PyTorch: if issues arise, uninstall `torch` and reinstall the correct GPU/CPU build.
- ONNXRuntime GPU: repo defaults to `onnxruntime` (CPU). Use `onnxruntime-gpu` if your CUDA environment supports it.

## Notes
- `configs/` is prepared for quick runs from the repo root.
- For vanilla YOLO scripts, `--data` should point to your dataset `data.yaml`; the provided `configs/data_license_plate.yaml` is pipeline-level.
- Example commands above should run after installing `requirements.txt` and the appropriate PyTorch build.
