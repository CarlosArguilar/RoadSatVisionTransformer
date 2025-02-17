import torch
from transformers import SegformerForSemanticSegmentation

def get_segformer(*args, **kwargs) -> SegformerForSemanticSegmentation:
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load pretrained SegFormer model
    segformer = SegformerForSemanticSegmentation.from_pretrained(
        *args,
        **kwargs
    )

    segformer.to(device)

    return segformer