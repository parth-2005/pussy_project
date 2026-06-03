# Brain MRI Segmentation with U-Net

This project trains a CUDA-only PyTorch segmentation pipeline for brain tumor masks from UPENN-GBM MRI data. It supports standard U-Net and Attention U-Net models, raw NIfTI input, and a faster preprocessed 2D `.npy` cache for training.

## Features

- Binary segmentation with Dice + BCE loss.
- Standard U-Net and Attention U-Net architectures.
- Raw `.nii.gz` dataset support and cached 2D `.npy` training support.
- CUDA-only training with an explicit GPU smoke test at startup.
- Multi-worker data loading with pinned memory for faster host-to-GPU transfer.

## Requirements

- Windows 10/11 or a compatible Linux system.
- Python 3.11 or newer.
- An NVIDIA GPU with a CUDA-capable PyTorch build.
- The UPENN-GBM dataset placed under `data/UPENN-GBM`.

## Project Layout

```text
.
├── configs/               # Training and dataset configuration
├── data/                  # Raw UPENN-GBM data and processed slice cache
├── datasets/              # Dataset loader for raw and preprocessed data
├── losses/                # Dice + BCE loss
├── metrics/               # Segmentation metrics
├── models/                # UNet and Attention U-Net models
├── preprocess.py          # Converts raw NIfTI volumes into 2D .npy slices
├── trainers/              # Training loop
├── utils/                 # Preprocessing helpers
├── visualization/         # Plotting helpers
├── main.py                # Training entry point
└── pyproject.toml         # Project dependencies for uv
```

## Install uv

`uv` is the recommended way to create the environment and install dependencies.

### Windows PowerShell

With winget:

```powershell
winget install --id Astral.uv -e
```

With PowerShell install script:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

After installation, verify it:

```powershell
uv --version
```

## Install Dependencies

From the project root:

```powershell
uv sync
```

This creates the virtual environment and installs the packages defined in `pyproject.toml`, including the CUDA-enabled PyTorch wheels configured for this project.

If you want to run commands inside the environment manually:

```powershell
uv run python --version
```

## Dataset Setup

### Raw data layout

Place the raw UPENN-GBM patient folders here:

```text
data/UPENN-GBM/
```

Each patient folder should contain the MRI modalities and segmentation mask. The loader supports common naming variations, including `T1GD` for the contrast-enhanced scan and `segm` for the mask.

Example:

```text
data/UPENN-GBM/UPENN-GBM-00002_11/
├── UPENN-GBM-00002_11_FLAIR.nii.gz
├── UPENN-GBM-00002_11_T1.nii.gz
├── UPENN-GBM-00002_11_T1GD.nii.gz
├── UPENN-GBM-00002_11_T2.nii.gz
└── UPENN-GBM-00002_11_segm.nii.gz
```

### Preprocessed data layout

For faster training, run `preprocess.py` once to generate cached 2D slices here:

```text
data/UPENN_GBM_2D_Processed/
```

The training entry point automatically prefers this folder when it exists.

## Preprocess the Dataset

Run the one-time preprocessing step from the project root:

```powershell
uv run python preprocess.py
```

This converts the 3D NIfTI volumes into paired 2D `.npy` files:

```text
slice_000001_img.npy
slice_000001_mask.npy
```

The preprocessing step also binarizes the tumor mask so all non-zero labels become foreground.

## Configure Training

Edit `configs/config.py` to adjust the main settings:

```python
data_root = "data/UPENN-GBM"
processed_data_root = "data/UPENN_GBM_2D_Processed"
model_name = "unet"  # or "attention_unet"
batch_size = 16
learning_rate = 1e-4
epochs = 20
num_workers = 8
```

Important notes:

- Training is CUDA-only. If CUDA is unavailable, `main.py` raises an error instead of falling back to CPU.
- If `data/UPENN_GBM_2D_Processed` exists, training uses it automatically.
- If the processed directory does not exist, training falls back to raw NIfTI loading.

## Run Training

Using uv:

```powershell
uv run python main.py
```

Or, if your shell is already inside the virtual environment, you can run:

```powershell
python main.py
```

On startup, the script prints a CUDA smoke test, the selected device, and the data root being used.

## What the Training Script Does

`main.py` performs the following steps:

1. Confirms CUDA is available.
2. Selects the GPU device and runs a small GPU smoke test.
3. Loads the dataset from either the processed cache or the raw NIfTI folder.
4. Builds the requested model.
5. Trains and saves checkpoints in `checkpoints/`.

## Models

- `UNet`: baseline encoder-decoder segmentation model.
- `AttentionUNet`: U-Net variant with attention gates.

## Loss and Metrics

- Loss: Dice + Binary Cross Entropy.
- Metric: Dice score.

## Output

During training, the project writes:

- checkpoints to `checkpoints/`
- logs to `logs/`

## Troubleshooting

### CUDA is not detected

If `main.py` reports that CUDA is unavailable, check that:

- You installed a CUDA-enabled PyTorch build.
- `nvidia-smi` detects your GPU.
- Your NVIDIA driver is installed and current.

### Dataset has zero slices

If preprocessing saves `0` slices, verify that each patient folder contains a mask file such as `*_segm.nii.gz` or `*_mask.nii.gz`.

### `uv` is not recognized

Install `uv` first using the instructions above, then reopen PowerShell so the PATH change is available.

## Quick Start

```powershell
winget install --id Astral.uv -e
uv sync
uv run python preprocess.py
uv run python main.py
```

## Notes

- The dataset loader binarizes tumor masks so labels like 1, 2, and 4 are treated as foreground.
- The preprocessing pipeline is recommended for faster GPU utilization.
