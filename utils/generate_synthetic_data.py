import os
import numpy as np
import nibabel as nib

def create_synthetic_data(root_dir="data/UPENN-GBM", num_patients=5):
    """
    Creates a synthetic dataset with NIfTI files for testing the pipeline.
    """
    os.makedirs(root_dir, exist_ok=True)
    modalities = ["FLAIR", "T1", "T2", "T1CE"]

    for i in range(num_patients):
        patient_dir = os.path.join(root_dir, f"UPENN-GBM-synthetic-{i:05d}_11")
        os.makedirs(patient_dir, exist_ok=True)

        # Create dummy volumes (H, W, D)
        shape = (128, 128, 30)

        # Create modalities
        for mod in modalities:
            data = np.random.rand(*shape).astype(np.float32)
            img = nib.Nifti1Image(data, np.eye(4))
            nib.save(img, os.path.join(patient_dir, f"{mod}.nii.gz"))

        # Create mask (binary)
        mask_data = np.zeros(shape, dtype=np.float32)
        # Add some "tumor" slices
        mask_data[40:80, 40:80, 10:20] = 1.0
        mask_img = nib.Nifti1Image(mask_data, np.eye(4))
        nib.save(mask_img, os.path.join(patient_dir, "mask.nii.gz"))

    print(f"Synthetic data created at {root_dir}")

if __name__ == "__main__":
    create_synthetic_data()
