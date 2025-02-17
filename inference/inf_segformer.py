import logging
from typing import List, Union, Generator, Optional, Tuple

import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import imageio
from PIL import Image

from models import get_segformer

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Constants for normalization (ImageNet)
MEAN_IMGNET = [0.485, 0.456, 0.406]
STD_IMGNET = [0.229, 0.224, 0.225]


class SegmentationInferencePipeline:
    """
    Inference pipeline for segmentation models.
    
    This class instantiates the model, handles image pre-processing, performs batch inference,
    and optionally visualizes results.
    """

    def __init__(
        self,
        model_name: str = 'nvidia/segformer-b1-finetuned-ade-512-512',
        num_labels: int = 2,
        ignore_mismatched_sizes: bool = True,
        transform: Optional[transforms.Compose] = None
    ):
        """
        Initializes the inference pipeline and instantiates the segmentation model.

        Args:
            model_name (str): Name or identifier of the model to load.
            num_labels (int): Number of labels for the segmentation task.
            ignore_mismatched_sizes (bool): Whether to ignore mismatched sizes during model instantiation.
            transform (transforms.Compose, optional): Image transformation pipeline.
                If not provided, defaults to converting to tensor and ImageNet normalization.
        """
        self.model = get_segformer(
            model_name,
            num_labels=num_labels,
            ignore_mismatched_sizes=ignore_mismatched_sizes
        )
        # Determine device from model parameters.
        self.device = next(self.model.parameters()).device if next(self.model.parameters(), None) is not None else torch.device("cpu")
        self.model.eval()  # Set the model to evaluation mode

        # Use provided transform or default ones.
        self.transform = transform if transform is not None else self.get_default_transforms()

    @staticmethod
    def get_default_transforms() -> transforms.Compose:
        """
        Returns the default image transformation pipeline.

        The default transformation converts a PIL Image to a tensor and applies ImageNet normalization.
        """
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN_IMGNET, std=STD_IMGNET),
        ])

    def prepare_image(self, image_input: Union[str, Image.Image]) -> Tuple[Image.Image, torch.Tensor, Tuple[int, int]]:
        """
        Loads and preprocesses a single image.

        Args:
            image_input (str or PIL.Image.Image): File path to an image or a PIL Image.

        Returns:
            tuple:
                - original_image (PIL.Image.Image): The original image (converted to RGB).
                - image_tensor (torch.Tensor): The preprocessed image tensor.
                - original_size (tuple): Original image size as (height, width).
        """
        if isinstance(image_input, str):
            original_image = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            original_image = image_input.convert("RGB")
        else:
            raise ValueError("image_input must be a file path or a PIL.Image.Image")

        # PIL.Image.size returns (width, height); we need (height, width) for F.interpolate.
        original_size = (original_image.height, original_image.width)
        image_tensor = self.transform(original_image)
        return original_image, image_tensor, original_size

    def run_batch_inference(
        self,
        images: Union[List[Union[str, Image.Image]], Generator[Union[str, Image.Image], None, None]],
        visualize: bool = False
    ) -> List[np.ndarray]:
        """
        Runs segmentation inference on a batch of images.

        If all images have the same size after transformation, batch inference is used.
        Otherwise, the images are processed individually.

        Args:
            images: A list or generator of image file paths or PIL images.
            visualize (bool, optional): If True, displays the original image alongside the predicted mask.
                Defaults to False.

        Returns:
            List[np.ndarray]: A list of predicted masks (as numpy arrays).
        """
        # Convert generator to list if needed.
        image_inputs = list(images)
        preprocessed_tensors = []
        original_sizes = []
        original_images = []

        # Pre-process all images.
        for img in image_inputs:
            orig_img, tensor, orig_size = self.prepare_image(img)
            preprocessed_tensors.append(tensor)
            original_sizes.append(orig_size)
            original_images.append(orig_img)

        # Determine if all images share the same dimensions
        shapes = [tensor.shape for tensor in preprocessed_tensors]
        can_batch = all(s == shapes[0] for s in shapes)

        predictions = []

        if can_batch:
            logger.info("Running batch inference on %d images.", len(preprocessed_tensors))
            batch_tensor = torch.stack(preprocessed_tensors).to(self.device)
            with torch.no_grad():
                outputs = self.model(batch_tensor)
                logits = outputs.logits  # Expected shape: (N, num_labels, H, W)
            # Process each output separately: upsample logits to the original image size and extract mask.
            for i in range(len(preprocessed_tensors)):
                target_size = original_sizes[i]  # (height, width)
                # Interpolate the logits to match the original image size.
                resized_logits = F.interpolate(logits[i:i+1], size=target_size, mode='bilinear', align_corners=False)
                pred_mask = torch.argmax(resized_logits, dim=1).squeeze(0).cpu().numpy()
                predictions.append(pred_mask)
                if visualize:
                    self.visualize_result(original_images[i], pred_mask)
        else:
            logger.info("Images have varying shapes. Running inference individually.")
            for i, tensor in enumerate(preprocessed_tensors):
                tensor = tensor.unsqueeze(0).to(self.device)
                with torch.no_grad():
                    outputs = self.model(tensor)
                    logits = outputs.logits
                target_size = original_sizes[i]
                resized_logits = F.interpolate(logits, size=target_size, mode='bilinear', align_corners=False)
                pred_mask = torch.argmax(resized_logits, dim=1).squeeze(0).cpu().numpy()
                predictions.append(pred_mask)
                if visualize:
                    self.visualize_result(original_images[i], pred_mask)

        return predictions

    @staticmethod
    def visualize_result(original_image: Image.Image, mask: np.ndarray) -> None:
        """
        Displays the original image alongside its predicted segmentation mask.

        Args:
            original_image (PIL.Image.Image): The original image.
            mask (np.ndarray): The predicted mask.
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(original_image)
        axes[0].set_title("Original Image")
        axes[0].axis("off")

        axes[1].imshow(mask, cmap="gray")
        axes[1].set_title("Predicted Mask")
        axes[1].axis("off")

        plt.tight_layout()
        plt.show()


# Example usage:
if __name__ == "__main__":
    # Create an instance of the inference pipeline.
    pipeline = SegmentationInferencePipeline()

    # Define a list (or generator) of images (file paths or PIL images)
    image_paths = [
        "/content/2021-01-26-00_00_2021-07-26-23_59_Sentinel-2_L2A_True_color.jpg",
        # Add more image paths or PIL.Image.Image objects here.
    ]

    # Run batch inference (visualization is disabled by default)
    masks = pipeline.run_batch_inference(image_paths, visualize=False)

    # Optionally, save the masks to disk.
    for idx, mask in enumerate(masks):
        output_path = f"predicted_mask_{idx}.png"
        # Multiply mask by 255 and cast to uint8 if needed.
        imageio.imwrite(output_path, (mask.astype(np.uint8) * 255))
        logger.info("Saved predicted mask to %s", output_path)
