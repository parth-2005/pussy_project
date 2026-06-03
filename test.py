import os
import time
import torch
import nibabel as nib
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from configs.config import get_config
from datasets.brain_dataset import BrainMRIDataset

def run_diagnostics():
    print("=== UPENN-GBM DATASET DIAGNOSTICS ===")
    config = get_config()
    
    # 1. RAW DATA CHECK
    print("\n--- 1. Raw NIfTI Inspection ---")
    if not os.path.exists(config.data_root):
        print(f"❌ ERROR: Data root '{config.data_root}' not found!")
        return
        
    patients = sorted(os.listdir(config.data_root))
    print(f"Total Patient Folders Found: {len(patients)}")
    
    if len(patients) > 0:
        first_patient_dir = os.path.join(config.data_root, patients[0])
        files = os.listdir(first_patient_dir)
        print(f"Files in first patient folder ({patients[0]}):")
        for f in files:
            if f.endswith('.nii.gz'):
                vol = nib.load(os.path.join(first_patient_dir, f))
                print(f"  - {f} | Shape: {vol.shape}")

    # 2. DATASET EXTRACTION CHECK
    print("\n--- 2. Dataset Slicing Insights ---")
    start_time = time.time()
    try:
        dataset = BrainMRIDataset(
            root_dir=config.data_root,
            modalities=config.modalities,
            image_size=config.image_size,
            remove_empty_slices=config.remove_empty_slices
        )
        print(f"Dataset Initialization took: {time.time() - start_time:.2f} seconds")
        print(f"Total Valid 2D Slices Extracted: {len(dataset)}")
    except Exception as e:
        print(f"❌ ERROR initializing dataset: {e}")
        return

    # 3. DATALOADER SPEED & TENSOR CHECK
    print("\n--- 3. DataLoader Performance & Shapes ---")
    loader = DataLoader(dataset, batch_size=config.batch_size, num_workers=config.num_workers, shuffle=True)
    
    start_load = time.time()
    # Grab just one batch
    images, masks = next(iter(loader))
    load_time = time.time() - start_load
    
    print(f"Time to load 1 batch of {config.batch_size}: {load_time:.4f} seconds")
    print(f"Images Tensor Shape: {images.shape}  | Dtype: {images.dtype} | Min/Max: {images.min():.2f}/{images.max():.2f}")
    print(f"Masks Tensor Shape:  {masks.shape}   | Dtype: {masks.dtype} | Min/Max: {masks.min():.2f}/{masks.max():.2f}")
    
    # 4. VISUALIZATION
    print("\n--- 4. Visual Verification ---")
    # Take the first slice from the batch
    img = images[0].numpy()  # Shape: (4, 256, 256)
    msk = masks[0].numpy()   # Shape: (1, 256, 256)
    
    fig, axes = plt.subplots(1, len(config.modalities) + 1, figsize=(15, 3))
    
    for i, mod in enumerate(config.modalities):
        axes[i].imshow(img[i], cmap='gray')
        axes[i].set_title(mod)
        axes[i].axis('off')
        
    axes[-1].imshow(msk[0], cmap='gray')
    axes[-1].set_title("Ground Truth Mask")
    axes[-1].axis('off')
    
    out_img = "diagnostic_sample.png"
    plt.tight_layout()
    plt.savefig(out_img)
    print(f"✅ Saved diagnostic image to: {out_img}")
    print("Done!")

if __name__ == "__main__":
    run_diagnostics()