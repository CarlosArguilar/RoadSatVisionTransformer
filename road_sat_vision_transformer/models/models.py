import torch
from typing import Tuple
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

def get_segformer(*args, **kwargs) -> Tuple[SegformerForSemanticSegmentation, SegformerImageProcessor]:
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the pretrained SegFormer model
    segformer = SegformerForSemanticSegmentation.from_pretrained(*args, **kwargs)
    segformer.to(device)

    # Load the corresponding image processor
    image_processor = SegformerImageProcessor.from_pretrained(*args, **kwargs)

    return segformer, image_processor
