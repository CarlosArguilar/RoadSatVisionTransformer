from typing import Callable, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import matplotlib.pyplot as plt


# Define a custom collate function to batch process using the feature extractor
def create_collate_fn(feature_extractor):
    def collate_fn(batch):
        images, masks = zip(*batch)
        # The feature extractor will resize, normalize, and convert the list into tensors.
        inputs = feature_extractor(images=list(images), segmentation_maps=list(masks), return_tensors="pt")
        return inputs["pixel_values"], inputs["labels"]
    return collate_fn

###############################################################
##################

MEAN_IMGNET = [0.485, 0.456, 0.406] # ImageNet mean
STD_IMGNET = [0.229, 0.224, 0.225]  # ImageNet std

def get_default_transforms() -> Tuple[Callable, Callable]:
    """Return default image and mask transformations."""
    image_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=MEAN_IMGNET,  
            std=STD_IMGNET    
        ),
    ])

    mask_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda t: t.squeeze().long()),
    ])

    return image_transforms, mask_transforms




def unnormalize_image(image_tensor: torch.Tensor) -> np.ndarray:
    """Unnormalize a tensor image for visualization.

    Args:
        image_tensor (torch.Tensor): Normalized image tensor.

    Returns:
        np.ndarray: Unnormalized image as a NumPy array.
    """
    image_tensor = image_tensor.clone().cpu()
    image_np = image_tensor.numpy().transpose(1, 2, 0)
    mean = np.array(MEAN_IMGNET)
    std = np.array(STD_IMGNET)
    image_np = std * image_np + mean
    image_np = np.clip(image_np, 0, 1)
    return image_np

def visualize_sample(
    dataset: Dataset,
    index: int = 0,
    unnormalize: Optional[Callable] = None
) -> None:
    """Visualize an image and its corresponding mask from the dataset.

    Args:
        dataset (Dataset): The dataset to visualize from.
        index (int, optional): Index of the sample to visualize. Defaults to 0.
        unnormalize (Callable, optional): Function to unnormalize the image tensor.
    """
    image, mask = dataset[index]

    if unnormalize:
        image_np = unnormalize(image)
    else:
        image_np = image.numpy().transpose(1, 2, 0)

    mask_np = mask.numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(image_np)
    axes[0].set_title('Image')
    axes[0].axis('off')

    axes[1].imshow(mask_np, cmap='gray')
    axes[1].set_title('Mask')
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()
