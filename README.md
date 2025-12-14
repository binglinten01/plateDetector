# plateDetector

Pipeline training/inference cho **detector biển số xe** (YOLO/Ultralytics) gồm:
- **Phase0**: download → prepare → analyze → clean → augment → synthetic → pseudo-label
- **Phase1**: HPO (Optuna) → train teacher → distillation → evaluation → export/inference

> Lưu ý: Các đường dẫn trong repo có **dấu cách** (ví dụ `license plate/phase folder/...`) nên khi chạy trên Windows cần **đặt trong dấu ngoặc kép**.

## Cần những file gì trong repo?

### `requirements.txt` (Python dependencies)
Repo này dùng khá nhiều thư viện (CV + ML + EDA). Mình đã gộp vào **1 file**:
- **`requirements.txt`**: core + export + dev tools.

Quan trọng nhất là **PyTorch**: không nên pin cứng trong `requirements.txt` vì phụ thuộc CUDA/CPU. Bạn cài torch riêng theo máy.

### `configs/` (YAML config mặc định)
Nhiều script mặc định đọc `configs/data_license_plate.yaml`… nên repo cần có thư mục **`configs/`**. Mình đã copy các YAML từ `license plate/config folder/` sang `configs/` để chạy “out-of-the-box”.

## Setup nhanh (Windows / PowerShell)

### 1) Tạo môi trường Python
Khuyến nghị Python **3.10–3.12**.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### 2) Cài PyTorch (GPU hoặc CPU)
- **GPU (CUDA)**: cài theo hướng dẫn chính thức của PyTorch (chọn đúng CUDA).
- **CPU**: nếu không có GPU, cài bản CPU.

> Vì link cài torch thay đổi theo thời gian, bạn nên lấy lệnh cài đặt tại trang chính thức `pytorch.org`.

### 3) Cài dependencies của project

```powershell
pip install -r requirements.txt
```

> Export ONNX/ONNXRuntime đã nằm trong `requirements.txt`. TensorRT vẫn là phần cài riêng tuỳ môi trường.

## Biến môi trường (.env)
Script download dataset dùng `python-dotenv` và đọc biến:
- `ROBOFLOW_API_KEY`

Tạo file `.env` ở root repo:

```env
ROBOFLOW_API_KEY=YOUR_KEY_HERE
```

## Cách chạy (ví dụ)

### Phase0
Chạy từ root repo, nhớ quote path vì có dấu cách:

```powershell
python "license plate/phase folder/phase0/01_download_datasets.py"
python "license plate/phase folder/phase0/02_prepare_dataset.py"
python "license plate/phase folder/phase0/03_analyze_dataset.py" --config "configs/data_license_plate.yaml"
python "license plate/phase folder/phase0/04_clean_dataset.py" --config "configs/data_license_plate.yaml"
python "license plate/phase folder/phase0/05_augment_dataset.py"
```

### Phase1 (training teacher)

```powershell
python "license plate/phase folder/phase1/01_train_teacher_ultimate_optimized.py" --data "configs/data_license_plate.yaml"
```

> Ghi chú: `--data` của Ultralytics thường là **dataset data.yaml** (train/val path). File `configs/data_license_plate.yaml` trong repo là config pipeline; tùy script bạn có thể cần trỏ vào `datasets/.../data.yaml`.

## Troubleshooting nhanh
- **Path có dấu cách**: luôn chạy kiểu `python "license plate/phase folder/..."`.
- **Thiếu `kaggle` CLI**: `pip install kaggle` rồi cấu hình `kaggle.json` trong `%USERPROFILE%\.kaggle\`.
- **CUDA/torch lỗi**: gỡ torch và cài lại đúng build CUDA/CPU cho máy.
