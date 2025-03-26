# Road Detection in Tropical Forests Using Sentinel Satellite Images

**Table of Contents**
- [Road Detection in Tropical Forests Using Sentinel Satellite Images](#road-detection-in-tropical-forests-using-sentinel-satellite-images)
  - [Project Overview](#project-overview)
  - [Key Features](#key-features)
  - [Repository Structure](#repository-structure)
  - [Data Preparation](#data-preparation)
    - [1. Downloading the Dataset](#1-downloading-the-dataset)
    - [2. Dataset Structure](#2-dataset-structure)
  - [Usage](#usage)
    - [1. Retrieving Sentinel Data](#1-retrieving-sentinel-data)
    - [2. Inference Pipeline](#2-inference-pipeline)
  - [References \& Citation](#references--citation)

---

## Project Overview

In this project, we focus on detecting newly built roads in tropical forests using **satellite images** from the [Sentinel constellation](https://sentinel.esa.int/). The appearance of unreferenced roads often precedes deforestation, making it imperative to monitor these areas for conservation efforts. Specifically, we use segmentation methods based on the **SegFormer** model to identify roads in SAR (Synthetic Aperture Radar) imagery, a modality well-suited to cloud-covered regions like tropical forests.

This was an academical research project motivated by the findings in [Nature (2024)](https://www.nature.com/articles/s41586-024-07303-5) indicating large-scale construction of illicit roads in regions of Southeast Asia and the Amazon.

---

## Key Features

- **Custom Segmentation Dataset**: A `SegmentationDataset` class designed to handle images and binary road masks.
- **Preprocessing Utilities**: Normalization to ImageNet statistics, collate functions for batch transforms, and data visualization routines.
- **Download & Extraction**: Scripts to automatically download and prepare training data for experimentation.
- **SegFormer Inference**: A pipeline that loads a fine-tuned SegFormer model for semantic segmentation of roads.
- **Cloud Masking & Sentinel-2 Retrieval**: Simple Earth Engine–based retrieval for Sentinel-2 images, applying cloud masking via scene classification (SCL).

---

## Repository Structure

```
road_sat_vision_transformer/
│
├── data/
│   ├── __init__.py
│   ├── dataset.py          # Custom PyTorch dataset for segmentation tasks
│   ├── utils.py            # Data transforms, visualization, collate functions
│   └── download_data.py    # Script to download the dataset and unzip it
│
├── inference/
│   ├── __init__.py
│   └── inf_segformer.py    # SegmentationInferencePipeline for running inference
│
├── losses/
│   ├── __init__.py
│   └── dice_loss.py        # DiceLoss implementation (useful in training)
│
├── models/
│   ├── __init__.py
│   ├── models.py           # get_segformer() loads a SegFormer model
│   └── utils.py            # Visualization utilities for predictions
│
├── sentinel_image_retriever/
│   ├── __init__.py
│   └── image_retriever.py  # Retrieves Sentinel-2 images (using Earth Engine + requests)
│
└── __init__.py
```

---

## Data Preparation

### 1. Downloading the Dataset

We provide a script `download_data.py` inside the `data` directory, which will download and extract a relevant dataset of roads for training/testing. This dataset is based on [Sloan et al. (2024)](https://doi.org/10.3390/rs16050839).

- **Usage**:

  ```bash
  cd road_sat_vision_transformer/data
  python download_data.py
  ```
  
  This will:
  1. Download the dataset (a `.zip` file).
  2. Extract nested ZIP archives.
  3. Clean up the folder so that only `training` and `testing` subfolders remain, containing `images/` and `masks/`.

### 2. Dataset Structure

After running the download script, you should end up with something like:

```
data/
   ├── training/
   │    ├── images/
   │    │    ├── image1.png
   │    │    └── ...
   │    └── masks/
   │         ├── image1.png
   │         └── ...
   └── testing/
        ├── images/
        └── masks/
```

Images are typically in formats like PNG or JPG, and masks are grayscale with pixel intensities denoting roads vs. non-roads.

---

## Usage

Below are the core ways to use this repository:

### 1. Retrieving Sentinel Data

The module `sentinel_image_retriever/image_retriever.py` can be used to fetch Sentinel-2 images from Google Earth Engine, applying a scene classification–based cloud mask. For example:

```python
import ee
from road_sat_vision_transformer.sentinel_image_retriever import get_sentinel2_image

# Initialize Earth Engine (after you've authenticated)
ee.Initialize()

# Define your region of interest (coordinates in [longitude, latitude] order)
coords = [
    [
       [-54.0, -3.0],
       [-54.0, -3.1],
       [-53.9, -3.1],
       [-53.9, -3.0],
       [-54.0, -3.0]
    ]
]

# Retrieve the image
pil_img = get_sentinel2_image(
    coords=coords,
    initial_date='2023-01-01',
    end_date='2023-02-01',
    max_cloud_pct=80
)

# Save it locally or display
pil_img.save('sentinel_example.png')
pil_img.show()
```

### 2. Inference Pipeline

To segment roads in new images using our **SegFormer** pipeline, see `inference/inf_segformer.py`. Inference loads a fine-tuned SegFormer model and applies it to one or more images:

```python
from road_sat_vision_transformer.inference import SegmentationInferencePipeline

# Initialize the pipeline
pipeline = SegmentationInferencePipeline(
    model_name='path/to/your/checkpoint-or-huggingface-model',  # Fine-tuned weights
    num_labels=2,  # For road vs. background
    ignore_mismatched_sizes=True
)

# Run inference on a list of image paths
image_paths = ["test_image1.png", "test_image2.png"]
predicted_masks = pipeline.run_batch_inference(image_paths, visualize=True)

# Save each predicted mask
import numpy as np
import imageio
for idx, mask in enumerate(predicted_masks):
    output_path = f"predicted_mask_{idx}.png"
    imageio.imwrite(output_path, (mask.astype(np.uint8) * 255))
    print(f"Saved predicted mask to {output_path}")
```

- **Visualization**: If `visualize=True`, you will see each image displayed side by side with its predicted mask.

---

## References & Citation

1. **Road Detection Motivation & Initial Study**  
   \- [A recent Nature article (2024)](https://www.nature.com/articles/s41586-024-07303-5) describing illicit road expansion in tropical forests.

2. **Remote Roads Dataset**  
   \- [Sloan, Sean et al. (2024)](https://doi.org/10.3390/rs16050839), _Mapping Remote Roads Using Artificial Intelligence and Satellite Imagery_, **Remote Sensing**.
