import os
from typing import Callable, Optional, Tuple, List

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm
import glob

from .utils import get_default_transforms


class SegmentationDataset(Dataset):
    """Custom Dataset for image segmentation tasks."""

    def __init__(
        self,
        root_dir: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None
    ):
        """
        Args:
            root_dir (str): Directory with all the data (e.g., 'data/training').
            transform (Callable, optional): Transform to apply to images.
            target_transform (Callable, optional): Transform to apply to masks.
        """
        self.root_dir = root_dir

        if (transform is None) or (target_transform is None):
            image_transforms, mask_transforms = get_default_transforms()

        self.transform = transform if transform is not None else image_transforms
        self.target_transform = target_transform if target_transform is not None else mask_transforms

        self.image_mask_pairs = self._load_image_mask_pairs()

    def _load_image_mask_pairs(self) -> List[Tuple[str, str]]:
        """Load image and mask file paths more efficiently."""

        # Find all image files under root_dir
        image_files = glob.glob(os.path.join(self.root_dir, '**', 'images', '*'), recursive=True)
        # Build a mapping from image filenames to their paths
        image_files_map = {os.path.basename(f): f for f in image_files}

        # Find all mask files under root_dir
        mask_files = glob.glob(os.path.join(self.root_dir, '**', 'masks', '*'), recursive=True)
        # Build a mapping from mask filenames to their paths
        mask_files_map = {os.path.basename(f): f for f in mask_files}

        image_mask_pairs = []

        for img_filename, img_path in tqdm(image_files_map.items(), desc="Getting image mask pairs"):
            if img_filename in mask_files_map:
                mask_path = mask_files_map[img_filename]
                image_mask_pairs.append((img_path, mask_path))
            else:
                print(f"Warning: Mask file for {img_filename} not found.")

        return image_mask_pairs


    def __len__(self) -> int:
        return len(self.image_mask_pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_path, mask_path = self.image_mask_pairs[idx]

        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')  # Grayscale mask

        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)

        if self.target_transform:
            mask = self.target_transform(mask)
        else:
            mask = torch.from_numpy(np.array(mask, dtype=np.int64))

        return image, mask
