import os
from typing import Tuple, List
from PIL import Image
import numpy as np
import glob
from tqdm import tqdm
from torch.utils.data import Dataset

class SegmentationDataset(Dataset):
    """Custom Dataset for image segmentation tasks that returns raw PIL images and binary masks."""
    
    def __init__(self, root_dir: str, threshold: int = 0):
        """
        Args:
            root_dir (str): Directory with all the data (e.g., 'data/training').
            threshold (int): Pixel intensity threshold for converting to binary mask.
        """
        self.root_dir = root_dir
        self.threshold = threshold
        self.image_mask_pairs = self._load_image_mask_pairs()

    def _load_image_mask_pairs(self) -> List[Tuple[str, str]]:
        """Load image and mask file paths more efficiently."""
        image_files = glob.glob(os.path.join(self.root_dir, '**', 'images', '*'), recursive=True)
        image_files_map = {os.path.basename(f): f for f in image_files}

        mask_files = glob.glob(os.path.join(self.root_dir, '**', 'masks', '*'), recursive=True)
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

    def __getitem__(self, idx: int) -> Tuple[Image.Image, Image.Image]:
        image_path, mask_path = self.image_mask_pairs[idx]
        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')  # Grayscale mask
        
        # Convert mask to a numpy array and threshold to obtain binary values (0 or 1)
        mask_np = np.array(mask)
        binary_mask = (mask_np > self.threshold).astype('uint8')
        
        # Optionally, you could convert back to PIL Image if needed
        binary_mask = Image.fromarray(binary_mask)
        
        return image, binary_mask
