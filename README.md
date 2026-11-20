# Generative Models Comparative Study — Fashion-MNIST

## 1. Environment

Recommended: Python 3.14 with CUDA 13.2.

```bash 
python -m venv .venv
```

Windows PowerShell:
```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Install dependencies:
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Install the PyTorch build appropriate for your CPU/CUDA setup from https://pytorch.org/get-started/locally/ if needed.

## 2. Verify setup and download dataset

```bash
python -m src.data
```

This downloads Fashion-MNIST into `data/` and prints the split sizes and tensor shapes.

## 3. Project conventions

- Dataset: Fashion-MNIST (70,000 images).
- Split: 80% train, 10% validation, 10% test.
- Seed: 42.
- Images: grayscale, 1 x 28 x 28, values in [0, 1].
- Batch size: 128.
- Training data is shuffled; validation/test data are not.
- All reported test metrics must be computed on the held-out test split.

## 4. Planned modules

- `src/data.py`: dataset and reproducible data loaders.
- `src/utils.py`: reproducibility and device helpers.
- Later modules: `models/`, `trainers/`, `evaluation/`, and `experiments/`.

Do not report placeholder metrics. Populate the report tables only after running the experiments.
