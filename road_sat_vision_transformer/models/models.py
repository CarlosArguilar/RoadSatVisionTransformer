import torch
from typing import Tuple
from transformers import SegformerForSemanticSegmentation, SegformerFeatureExtractor

def get_segformer(*args, **kwargs) -> Tuple[SegformerForSemanticSegmentation, SegformerFeatureExtractor]:
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the pretrained SegFormer model
    segformer = SegformerForSemanticSegmentation.from_pretrained(*args, **kwargs)
    segformer.to(device)

    # Load the corresponding feature extractor
    feature_extractor = SegformerFeatureExtractor.from_pretrained(*args, **kwargs)

    return segformer, feature_extractor
