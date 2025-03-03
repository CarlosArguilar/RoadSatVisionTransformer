
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

    Args:
        coords (list): 2D list of [lon, lat] forming a polygon boundary.
        initial_date (str): Start date (YYYY-MM-DD).
        end_date (str, optional): End date (YYYY-MM-DD). Defaults to today's date.
        max_cloud_pct (int, optional): Max CLOUDY_PIXEL_PERCENTAGE for filtering. Defaults to 80.
        vis_min (int, optional): Min reflectance value for visualization. Defaults to 0.
        vis_max (int, optional): Max reflectance value for visualization. Defaults to 3000.
        dimensions (int, optional): Requested width or height in pixels. Defaults to 512.

    Returns:
        PIL.Image: PIL image (8-bit RGB) of the cloud-free mosaic.
    """
    # If end_date is None, default to today's date
    if end_date is None:
        end_date = datetime.date.today().isoformat()

    # 2) Create a polygon from the provided coords
    region = ee.Geometry.Polygon(coords)

    # 3) Define cloud masking using the SCL band
    def mask_s2_clouds_scl(image):
        # SCL band = Scene Classification Layer
        scl = image.select('SCL')
        # Exclude classes: 3 = Cloud Shadows, 8 = Cloud medium probability,
        # 9 = Cloud high probability, 10 = Thin cirrus, 11 = Snow
        mask = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)) \
                      .And(scl.neq(10)).And(scl.neq(11))
        return image.updateMask(mask)

    # 4) Load Sentinel-2 SR collection and apply filters
    s2_col = (
        ee.ImageCollection('COPERNICUS/S2_SR')
        .filterBounds(region)
        .filterDate(initial_date, end_date)
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', max_cloud_pct))
        .map(mask_s2_clouds_scl)
    )

    # 5) Check if collection is empty
    if s2_col.size().getInfo() == 0:
        raise ValueError("No images found for the specified date range and region.")

    # 6) Create a median composite and clip
    composite = s2_col.median().clip(region)

    # 7) Select the RGB bands
    rgb = composite.select(['B4', 'B3', 'B2'])

    # 8) Convert float reflectance to 8-bit RGB using visualize()
    #    This step is crucial to avoid the "Must specify visualization parameters" error
    vis_rgb = rgb.visualize(
        bands=['B4', 'B3', 'B2'],
        min=vis_min,
        max=vis_max
    )

    # 9) Define thumbnail parameters
    thumb_params = {
        'region': region.toGeoJSONString(),
        'dimensions': dimensions,
        'format': 'png'
    }

    # 10) Request the thumbnail URL
    url = vis_rgb.getThumbURL(thumb_params)

    # 11) Download the PNG and open as PIL image
    response = requests.get(url)
    if response.status_code != 200:
        raise RuntimeError(f"Error {response.status_code} downloading image from {url}")

    pil_img = Image.open(BytesIO(response.content))
    return pil_img
