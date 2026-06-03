# Brain MRI Segmentation with U-Net

This project implements a deep learning pipeline for binary segmentation of brain tumors from multi-modal MRI scans, specifically targeting the UPENN-GBM dataset. It provides a flexible framework for experimenting with different U-Net architectures and loss functions.

## 🚀 Features

- **Multiple Architectures**: Implementation of standard `UNet` and `AttentionUNet`.
- **Multi-Modal Input**: Supports combining multiple MRI modalities (FLAIR, T1, T2, T1CE) as input channels.
- **Custom Loss Function**: Uses a combination of Dice Loss and Binary Cross Entropy (BCE) to handle class imbalance.
- **Comprehensive Pipeline**: Includes data loading, preprocessing, training, validation, and visualization tools.
- **Flexible Configuration**: Centralized configuration via `configs/config.py`.

## 📂 Project Structure

```text
.
├── configs/            # Training and model configurations
├── data/               # Raw MRI data (UPENN-GBM)
├── datasets/           # PyTorch Dataset implementation for Brain MRI
├── losses/             # Implementation of Dice-BCE loss
├── metrics/            # Segmentation metrics (Dice score, etc.)
├── models/             # Model architectures (UNet, AttentionUNet, Factory)
├── trainers/           # Training and validation loop logic
├── utils/              # Preprocessing and synthetic data generation
├── visualization/     # Tools for visualizing segmentation results
├── main.py             # Project entry point
└── pyproject.toml      # Project dependencies
```

## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd unet_implementation
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   # or if using uv:
   uv sync
   ```

## 📖 Usage

### 1. Data Preparation
Place your UPENN-GBM dataset in the `data/UPENN-GBM` directory. The dataset should follow the structure:
`UPENN-GBM/<patient_id>/{FLAIR, T1, T2, T1CE, mask}.nii.gz`

### 2. Configuration
Modify `configs/config.py` to adjust hyperparameters, image size, modalities, or the model choice:
```python
model_name = "unet" # or "attention_unet"
modalities = ["FLAIR", "T1", "T2", "T1CE"]
batch_size = 16
learning_rate = 1e-4
```

### 3. Training
Start the training process by running:
```bash
python main.py
```
The model will be trained and checkpoints will be saved in the `checkpoints/` directory.

## 🧪 Model Architectures

- **UNet**: A classic encoder-decoder architecture with skip connections.
- **Attention UNet**: Enhances the standard U-Net with attention gates to focus on relevant regions of the image and reduce false positives in the segmentation.

## 📊 Evaluation
The project uses the Dice coefficient as the primary metric for evaluating the overlap between the predicted segmentation mask and the ground truth.

## 🖼️ Visualization
You can use the `visualization/visualizer.py` module to compare the original MRI slices, ground truth masks, and the model's predictions.
