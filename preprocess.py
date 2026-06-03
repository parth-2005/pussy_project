import os
import glob

import cv2
import nibabel as nib
import numpy as np
from tqdm import tqdm

from configs.config import get_config


def _find_first_match(directory: str, patterns: list[str]) -> str | None:
    for pattern in patterns:
        matches = sorted(glob.glob(os.path.join(directory, pattern)))
        if matches:
            return matches[0]
    return None


def _find_modality_file(directory: str, modality: str) -> str | None:
    """
    Resolve the on-disk file for a modality, including common aliases.
    """
    mapping = {
        "T1CE": ["T1GD", "T1CE", "T1_contrast"],
        "T1": ["T1"],
        "T2": ["T2"],
        "FLAIR": ["FLAIR"],
    }

    search_terms = mapping.get(modality, [modality])
    for term in search_terms:
        match = _find_first_match(directory, [f"*{term}*.nii.gz", f"*{term}*"])
        if match:
            return match
    return None


def preprocess_to_2d() -> None:
    config = get_config()
    raw_data_dir = config.data_root
    output_dir = os.path.join(os.path.dirname(raw_data_dir), "UPENN_GBM_2D_Processed")
    os.makedirs(output_dir, exist_ok=True)

    patients = sorted(
        [name for name in os.listdir(raw_data_dir) if os.path.isdir(os.path.join(raw_data_dir, name))]
    )
    print(f"Preprocessing {len(patients)} patients to 2D .npy files...")

    slice_counter = 0

    for patient in tqdm(patients):
        patient_path = os.path.join(raw_data_dir, patient)

        mod_files = {}
        for modality in config.modalities:
            match = _find_modality_file(patient_path, modality)
            if match:
                mod_files[modality] = match

        mask_match = _find_first_match(
            patient_path,
            ["*segm*.nii.gz", "*seg*.nii.gz", "*mask*.nii.gz", "*mask*"],
        )

        if not mask_match or len(mod_files) < len(config.modalities):
            continue

        mask_vol = nib.load(mask_match).get_fdata()
        mod_vols = {mod: nib.load(path).get_fdata() for mod, path in mod_files.items()}

        num_slices = mask_vol.shape[-1]

        for z in range(num_slices):
            mask_slice = mask_vol[..., z]

            if config.remove_empty_slices and np.sum(mask_slice) == 0:
                continue

            binary_mask = np.where(mask_slice > 0, 1.0, 0.0).astype(np.float32)
            binary_mask = cv2.resize(binary_mask, config.image_size, interpolation=cv2.INTER_NEAREST)

            processed_channels = []
            for modality in config.modalities:
                img_slice = mod_vols[modality][..., z]
                img_min = float(np.min(img_slice))
                img_max = float(np.max(img_slice))

                if img_max > img_min:
                    img_slice = (img_slice - img_min) / (img_max - img_min)
                else:
                    img_slice = np.zeros_like(img_slice, dtype=np.float32)

                img_slice = cv2.resize(img_slice.astype(np.float32), config.image_size, interpolation=cv2.INTER_LINEAR)
                processed_channels.append(img_slice)

            stacked_image = np.stack(processed_channels, axis=0).astype(np.float32)
            expanded_mask = np.expand_dims(binary_mask.astype(np.float32), axis=0)

            np.save(os.path.join(output_dir, f"slice_{slice_counter:06d}_img.npy"), stacked_image)
            np.save(os.path.join(output_dir, f"slice_{slice_counter:06d}_mask.npy"), expanded_mask)
            slice_counter += 1

    print(f"\nSuccessfully saved {slice_counter} 2D slices to {output_dir}")


if __name__ == "__main__":
    preprocess_to_2d()