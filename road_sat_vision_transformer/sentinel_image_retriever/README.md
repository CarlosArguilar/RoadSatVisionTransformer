# Sentinel-2 Image Retrieval

The function **get_sentinel2_image** retrieves a cloud-reducted median composite from Sentinel-2 satellite imagery for a specified region and date range. It uses Google Earth Engine (GEE) for processing and returns the result as a PIL image.

---

## Example Usage

```python
# Import required libraries
import ee
import PIL.Image

# Initialize Earth Engine (requires authentication)
ee.Initialize()

# Define coordinates for a region of interest
coords = [
    [-122.52, 37.70],
    [-122.34, 37.70],
    [-122.34, 37.82],
    [-122.52, 37.82],
    [-122.52, 37.70]
]

# Fetch a cloud-free Sentinel-2 image
img = get_sentinel2_cloudfree_pil(
    coords=coords,
    initial_date="2023-01-01",
    end_date="2023-03-01",
    max_cloud_pct=20,  # Allow up to 20% cloud cover
    vis_max=6000,      Stretch RGB values to 0-6000 reflectance
    dimensions=1024    # Request a 1024x1024 image
)

# Display or save the image
img.show()
img.save("output.png")
```

---

## Function Overview

### `get_sentinel2_image(...)`

Retrieves a median composite of Sentinel-2 Surface Reflectance (SR) data, applies cloud masking using the Scene Classification Layer (SCL), and returns the result as an 8-bit RGB PIL image.

#### Parameters

| Name             | Type            | Default      | Description                                                                 |
|------------------|-----------------|--------------|-----------------------------------------------------------------------------|
| `coords`         | `list`          | **Required** | 2D list of `[lon, lat]` coordinates defining the polygon boundary.         |
| `initial_date`   | `str`           | **Required** | Start date in `YYYY-MM-DD` format.                                          |
| `end_date`       | `str`           | Today        | End date in `YYYY-MM-DD` format. Defaults to the current date.              |
| `max_cloud_pct`  | `int`           | `80`         | Maximum allowed `CLOUDY_PIXEL_PERCENTAGE` (0-100).                         |
| `vis_min`        | `int`           | `0`          | Minimum reflectance value for visualization stretch.                       |
| `vis_max`        | `int`           | `6000`       | Maximum reflectance value for visualization stretch.                       |
| `dimensions`     | `int`           | `512`        | Width/height of the output image in pixels (square aspect ratio).          |

#### Returns
- `PIL.Image`: 8-bit RGB image of the cloud-free mosaic.

---

## How It Works

### Key Steps:
1. **Date Handling**:  
   - If `end_date` is `None`, it defaults to today's date.
   - Filters the Sentinel-2 collection between `initial_date` and `end_date`.

2. **Region Definition**:  
   - Converts the input `coords` into an Earth Engine `Polygon`.

3. **Cloud Masking**:  
   - Uses the Sentinel-2 **SCL band** (Scene Classification Layer) to mask out:  
     - Cloud shadows (class 3)  
     - Medium/high probability clouds (classes 8, 9)  
     - Thin cirrus (class 10)  
     - Snow (class 11)  

4. **Image Collection Filtering**:  
   - Filters Sentinel-2 SR (`COPERNICUS/S2_SR`) images:  
     - Intersecting the region.  
     - Within the date range.  
     - With `CLOUDY_PIXEL_PERCENTAGE < max_cloud_pct`.  

5. **Median Composite**:  
   - Generates a median composite from the filtered images and clips it to the region.

6. **RGB Visualization**:  
   - Selects bands **B4 (Red)**, **B3 (Green)**, and **B2 (Blue)**.  
   - Converts reflectance values (range `vis_min` to `vis_max`) to 8-bit using `visualize()`.  

7. **Thumbnail Generation**:  
   - Requests a PNG thumbnail from GEE with the specified `dimensions`.

---

## Notes

- **Earth Engine Authentication**:  
  You must initialize Earth Engine with `ee.Initialize()` before calling this function.

- **Reflectance Scaling**:  
  The default `vis_max=6000` scales Sentinel-2 reflectance values (0–10,000) to 8-bit (0–255). Adjust `vis_min`/`vis_max` to control brightness/contrast.

- **Cloud Masking**:  
  The SCL-based masking is more reliable than Sentinel-2’s native QA60 band for excluding clouds and shadows.

- **Error Handling**:  
  - Raises `ValueError` if no images match the criteria.  
  - Raises `RuntimeError` if image download fails.
