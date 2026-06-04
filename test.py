import os
import torch
import random
import matplotlib.pyplot as plt
import numpy as np

from configs.config import get_config
from datasets.brain_dataset import BrainMRIDataset
from models.factory import ModelFactory

# A slightly tweaked visualization function to ensure it displays nicely 
# when running from a terminal script
def visualize_and_save(image: torch.Tensor, target: torch.Tensor, pred: torch.Tensor, save_name: str):
    img_np = image.cpu().numpy()
    if img_np.ndim == 3:
        img_np = img_np[0] # Just show the FLAIR channel for clarity

    target_np = target.cpu().numpy().squeeze()
    
    # Sigmoid and threshold
    pred_np = torch.sigmoid(pred).cpu().numpy().squeeze()
    pred_np = (pred_np > 0.5).astype(np.float32)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img_np, cmap='gray')
    axes[0].set_title("Input MRI (FLAIR)")
    axes[0].axis('off')

    axes[1].imshow(target_np, cmap='magma') # Colored map to see it better
    axes[1].set_title("Ground Truth Mask")
    axes[1].axis('off')

    axes[2].imshow(pred_np, cmap='magma')
    axes[2].set_title("Model Prediction")
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(save_name)
    print(f"✅ Saved visualization to {save_name}")
    plt.show() # This will pop open a window on your Windows machine!

def run_inference():
    config = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running inference on: {device}")

    # 1. Load Validation Data (The unseen brains!)
    val_dir = os.path.join(config.processed_data_root, "val")
    val_dataset = BrainMRIDataset(
        root_dir=val_dir,
        modalities=config.modalities,
        image_size=config.image_size,
        remove_empty_slices=config.remove_empty_slices
    )
    print(f"Loaded {len(val_dataset)} validation slices.")

    # 2. Load the Model
    model_path = os.path.join(config.checkpoint_dir, "best_model.pth")
    if not os.path.exists(model_path):
        print(f"❌ Could not find model at {model_path}")
        return

    model = ModelFactory.create(
        model_name=config.model_name, 
        in_channels=config.in_channels, 
        out_channels=config.out_channels
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval() # Put model in evaluation mode

    # 3. Pick 3 random slices to visualize
    num_samples = 3
    indices = random.sample(range(len(val_dataset)), num_samples)

    with torch.no_grad(): # No gradients needed for inference
        for i, idx in enumerate(indices):
            image, mask = val_dataset[idx]
            
            # Add fake batch dimension for the model: [1, 4, 256, 256]
            image_batch = image.unsqueeze(0).to(device)
            
            # Run the prediction
            pred_batch = model(image_batch)
            
            # Visualize
            print(f"\nVisualizing Sample {i+1}/{num_samples} (Dataset Index: {idx})")
            visualize_and_save(
                image=image, 
                target=mask, 
                pred=pred_batch.squeeze(0), # Remove batch dimension
                save_name=f"inference_result_{i+1}.png"
            )

if __name__ == "__main__":
    run_inference()