
import ee
import datetime
import requests
from PIL import Image
from io import BytesIO

def get_sentinel2_image(
    coords,
    initial_date,
    end_date=None,
    max_cloud_pct=80,
    vis_min=0,
    vis_max=6000,
    dimensions=512
):
    """
    Retrieve a median composite of Sentinel-2 SR data with SCL-based cloud masking,
    then return it as a PIL image using visualize() for proper 8-bit RGB.
    """
    if end_date is None:
        end_date = datetime.date.today().isoformat()

    region = ee.Geometry.Polygon(coords)

    def mask_s2_clouds_scl(image):
        scl = image.select('SCL')
        mask = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)) \
                      .And(scl.neq(10)).And(scl.neq(11))
        return image.updateMask(mask)

    # Load and filter the collection, then select the same bands for each image.
    s2_col = (
        ee.ImageCollection('COPERNICUS/S2_SR')
        .filterBounds(region)
        .filterDate(initial_date, end_date)
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', max_cloud_pct))
        .map(mask_s2_clouds_scl)
        # Ensure each image has only the desired bands
        .map(lambda img: img.select(['B2', 'B3', 'B4', 'SCL']))
    )

    if s2_col.size().getInfo() == 0:
        raise ValueError("No images found for the specified date range and region.")

    composite = s2_col.median().clip(region)

    # Select only the RGB bands for visualization
    rgb = composite.select(['B4', 'B3', 'B2'])

    vis_rgb = rgb.visualize(
        bands=['B4', 'B3', 'B2'],
        min=vis_min,
        max=vis_max
    )

    thumb_params = {
        'region': region.toGeoJSONString(),
        'dimensions': dimensions,
        'format': 'png'
    }

    url = vis_rgb.getThumbURL(thumb_params)

    response = requests.get(url)
    if response.status_code != 200:
        raise RuntimeError(f"Error {response.status_code} downloading image from {url}")

    pil_img = Image.open(BytesIO(response.content))
    return pil_img
