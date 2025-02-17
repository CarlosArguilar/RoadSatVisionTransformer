import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

def visualize_predictions(model, dataset, index=0):
    model.eval()
    image, mask = dataset[index]
    image_input = image.unsqueeze(0).to(model.device)

    with torch.no_grad():
        outputs = model(pixel_values=image_input)
        logits = outputs.logits
        if logits.shape[-2:] != mask.shape[-2:]:
            logits = nn.functional.interpolate(
                logits, size=mask.shape[-2:], mode='bilinear', align_corners=False
            )
        preds = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()

    # Unnormalize image for visualization
    image_np = image.cpu().numpy().transpose(1, 2, 0)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image_np = std * image_np + mean
    image_np = np.clip(image_np, 0, 1)

    mask_np = mask.cpu().numpy()

    # Plot the image, ground truth mask, and predicted mask
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(image_np)
    axes[0].set_title('Image')
    axes[0].axis('off')

    axes[1].imshow(mask_np, cmap='gray')
    axes[1].set_title('Ground Truth Mask')
    axes[1].axis('off')

    axes[2].imshow(preds, cmap='gray')
    axes[2].set_title('Predicted Mask')
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()


def visualize_predictions2(model, dataset, index=0):
    model.eval()
    image, mask = dataset[index]
    image_input = image.unsqueeze(0).to(model.device)

    with torch.no_grad():
        enh, outputs = model(image_input,1)
        logits = outputs
        if logits.shape[-2:] != mask.shape[-2:]:
            logits = nn.functional.interpolate(
                logits, size=mask.shape[-2:], mode='bilinear', align_corners=False
            )
        preds = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()

    # Unnormalize image for visualization
    image_np = image.cpu().numpy().transpose(1, 2, 0)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image_np = std * image_np + mean
    image_np = np.clip(image_np, 0, 1)

    mask_np = mask.cpu().numpy()

    # Plot the image, ground truth mask, and predicted mask
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(image_np)
    axes[0].set_title('Image')
    axes[0].axis('off')

    axes[1].imshow(mask_np, cmap='gray')
    axes[1].set_title('Ground Truth Mask')
    axes[1].axis('off')

    axes[2].imshow(preds, cmap='gray')
    axes[2].set_title('Predicted Mask')
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()