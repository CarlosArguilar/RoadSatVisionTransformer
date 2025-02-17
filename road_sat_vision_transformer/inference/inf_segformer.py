import logging
from typing import List, Union, Generator, Optional, Tuple

import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import imageio
from PIL import Image

# Import your custom model loading function
from road_sat_vision_transformer.models import get_segformer

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Constants for normalization (ImageNet)
MEAN_IMGNET = [0.485, 0.456, 0.406]
STD_IMGNET = [0.229, 0.224, 0.225]

TARGET_SIZE = (512, 512)  # Force images to 512x512 for batch processing

class SegmentationInferencePipeline:
    """
    Inference pipeline for segmentation models.
    
    This class instantiates the model, resizes images to a fixed size for batch inference,
    and optionally visualizes results.
    """

    def __init__(
        self,
        model_name: str = 'road_sat_vision_transformer/models/checkpoints/fine-tuned-segformer-b1-27epoch-lre-4-slr-09',
        num_labels: int = 2,
        ignore_mismatched_sizes: bool = True,
        transform: Optional[transforms.Compose] = None
    ):
        """
        Initializes the inference pipeline and instantiates the segmentation model.

        Args:
            model_name (str): Name or path to the model checkpoint to load.
            num_labels (int): Number of labels for the segmentation task.
            ignore_mismatched_sizes (bool): Whether to ignore mismatched sizes during model instantiation.
            transform (transforms.Compose, optional): Image transformation pipeline.
                If not provided, defaults to resizing to 512x512, converting to tensor, and ImageNet normalization.
        """
        self.model = get_segformer(
            model_name,
            num_labels=num_labels,
            ignore_mismatched_sizes=ignore_mismatched_sizes
        )

        first_param = next(iter(self.model.parameters()), None)
        if first_param is not None:
            self.device = first_param.device
        else:
            self.device = torch.device("cpu")
            
        self.model.eval()  # Set the model to evaluation mode

        # Use provided transform or default inference transforms (resizes to 512x512).
        self.transform = transform if transform is not None else self.get_inference_transforms()

    @staticmethod
    def get_inference_transforms() -> transforms.Compose:
        """
        Returns the default inference transform pipeline.

        The default transformation resizes to (512, 512), converts to tensor, 
        and applies ImageNet normalization.
        """
        return transforms.Compose([
            transforms.Resize(TARGET_SIZE),
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
                - image_tensor (torch.Tensor): The preprocessed image tensor (resized to 512x512).
                - original_size (tuple): Original image size as (height, width).
        """
        if isinstance(image_input, str):
            original_image = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            original_image = image_input.convert("RGB")
        else:
            raise ValueError("image_input must be a file path or a PIL.Image.Image")

        original_size = (original_image.height, original_image.width)
        image_tensor = self.transform(original_image)
        return original_image, image_tensor, original_size

    def run_batch_inference(
        self,
        images: Union[List[Union[str, Image.Image]], Generator[Union[str, Image.Image], None, None]],
        visualize: bool = False
    ) -> List[np.ndarray]:
        """
        Runs segmentation inference on a batch of images, forcing all inputs to (512, 512).

        The predicted masks are then upsampled back to each image's original size.

        Args:
            images: A list or generator of image file paths or PIL images.
            visualize (bool, optional): If True, displays the original image alongside the predicted mask.
                Defaults to False.

        Returns:
            List[np.ndarray]: A list of predicted masks (as numpy arrays).
        """
        # Convert generator to list if needed
        image_inputs = list(images)
        preprocessed_tensors = []
        original_sizes = []
        original_images = []

        # Pre-process all images (force them to 512x512)
        for img in image_inputs:
            orig_img, tensor, orig_size = self.prepare_image(img)
            preprocessed_tensors.append(tensor)
            original_sizes.append(orig_size)
            original_images.append(orig_img)

        # Stack into a single batch (B, C, 512, 512)
        batch_tensor = torch.stack(preprocessed_tensors).to(self.device)

        logger.info("Running batch inference on %d images at size %s.", len(preprocessed_tensors), TARGET_SIZE)

        # Forward pass
        with torch.no_grad():
            outputs = self.model(batch_tensor)
            logits = outputs.logits  # (B, num_labels, 512, 512)

        # For each image, upsample logits to the original size, then argmax
        predictions = []
        for i in range(len(preprocessed_tensors)):
            height, width = original_sizes[i]
            # Interpolate the logits back to the original size
            resized_logits = F.interpolate(
                logits[i:i+1],
                size=(height, width),
                mode='bilinear',
                align_corners=False
            )
            pred_mask = torch.argmax(resized_logits, dim=1).squeeze(0).cpu().numpy()
            predictions.append(pred_mask)

            # Optionally visualize
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


# Example usage (for direct script execution):
if __name__ == "__main__":
    # Create an instance of the inference pipeline (resizes inputs to 512x512).
    pipeline = SegmentationInferencePipeline()

    # Define a list (or generator) of images (file paths or PIL images).
    image_paths = [
        "/content/2021-01-26-00_00_2021-07-26-23_59_Sentinel-2_L2A_True_color.jpg",
        # Add more image paths or PIL.Image objects here...
    ]

    # Run batch inference (visualization is disabled by default)
    masks = pipeline.run_batch_inference(image_paths, visualize=False)

    # Optionally, save the masks to disk.
    for idx, mask in enumerate(masks):
        output_path = f"predicted_mask_{idx}.png"
        # Convert mask to [0, 255] range if needed
        imageio.imwrite(output_path, (mask.astype(np.uint8) * 255))
        logger.info("Saved predicted mask to %s", output_path)
