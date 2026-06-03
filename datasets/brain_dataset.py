import os
import glob
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Dict, Optional
from utils.preprocessing import normalize_intensity, resize_image, resize_mask

class BrainMRIDataset(Dataset):
    """
    Dataset for loading Brain MRI NIfTI volumes and converting them to 2D slices.
    """
    def __init__(
        self,
        root_dir: str,
        modalities: List[str],
        image_size: Tuple[int, int],
        remove_empty_slices: bool = True,
        transform=None
    ):
        self.root_dir = root_dir
        self.modalities = modalities
        self.image_size = image_size
        self.remove_empty_slices = remove_empty_slices
        self.transform = transform

        self.patient_dirs = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.slices = self._prepare_slices()

        if not self.patient_dirs:
            raise RuntimeError(f"No patient directories found in dataset root: {root_dir}")

        if not self.slices:
            preview_dirs = ", ".join(self.patient_dirs[:5])
            raise RuntimeError(
                "No usable training slices were found. "
                "The dataset loader expects each patient folder to contain a mask file ending in .nii.gz "
                "with a name matching one of: mask, seg, segmentation. "
                f"Root: {root_dir}. "
                f"Patient folders found: {len(self.patient_dirs)} ({preview_dirs})."
            )

    def _detect_modality_file(self, patient_dir: str, modality: str) -> Optional[str]:
        """
        Tries to find the file matching the modality name in the patient directory.
        """
        path = os.path.join(self.root_dir, patient_dir)
        files = os.listdir(path)

        # Mapping for common variations
        mapping = {
            "T1CE": ["T1GD", "T1CE", "T1_contrast"],
            "T1": ["T1"],
            "T2": ["T2"],
            "FLAIR": ["FLAIR"],
            "mask": ["mask", "seg", "segmentation"]
        }

        search_terms = mapping.get(modality, [modality])
        for term in search_terms:
            for f in files:
                if term.lower() in f.lower() and f.endswith(".nii.gz"):
                    return os.path.join(path, f)
        return None

    def _prepare_slices(self) -> List[Tuple[str, int]]:
        """
        Iterate through all patients and find all valid slices.
        Returns a list of (patient_dir, slice_index).
        """
        all_slices = []
        for p_dir in self.patient_dirs:
            mask_path = self._detect_modality_file(p_dir, "mask")
            if mask_path is None:
                continue

            mask_vol = nib.load(mask_path).get_fdata()

            # We assume the last dimension is the slicing dimension for simplicity
            # In a real scenario, we should check the affine/header
            num_slices = mask_vol.shape[-1]

            for i in range(num_slices):
                slice_mask = mask_vol[..., i]
                if self.remove_empty_slices and np.sum(slice_mask) == 0:
                    continue
                all_slices.append((p_dir, i))

        return all_slices

    def __len__(self) -> int:
        return len(self.slices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        p_dir, slice_idx = self.slices[idx]

        # Load modalities
        images = []
        for mod in self.modalities:
            mod_path = self._detect_modality_file(p_dir, mod)
            if mod_path:
                vol = nib.load(mod_path).get_fdata()
                slice_img = vol[..., slice_idx]
                slice_img = normalize_intensity(slice_img)
                slice_img = resize_image(slice_img, self.image_size)
                images.append(slice_img)
            else:
                # If modality is missing, provide a zero array
                images.append(np.zeros(self.image_size, dtype=np.float32))

        # Load mask
        mask_path = self._detect_modality_file(p_dir, "mask")
        mask_vol = nib.load(mask_path).get_fdata()
        slice_mask = mask_vol[..., slice_idx]
        slice_mask = resize_mask(slice_mask, self.image_size)

        # Convert to tensors
        # image shape: [channels, height, width]
        image_tensor = torch.from_numpy(np.stack(images, axis=0)).float()
        # mask shape: [1, height, width]
        mask_tensor = torch.from_numpy(slice_mask).float().unsqueeze(0)

        if self.transform:
            image_tensor = self.transform(image_tensor)

        return image_tensor, mask_tensor
