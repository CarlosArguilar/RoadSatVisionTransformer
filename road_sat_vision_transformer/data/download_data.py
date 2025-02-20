import os
import io
import zipfile
import requests
import shutil
from tqdm import tqdm

__all__ = ['download_data']

def _download_and_unzip(url, extract_to):
    """
    Downloads a zip file from a URL with a Firefox User-Agent and extracts its contents,
    while displaying download progress.

    Args:
        url (str): The URL of the zip file.
        extract_to (str): The directory to extract the files to.
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:106.0) Gecko/20100101 Firefox/106.0'
    }
    
    try:
        with requests.get(url, headers=headers, stream=True) as response:
            response.raise_for_status()
            total_size = int(response.headers.get('content-length', 0))
            chunk_size = 1024  # 1 KB chunks
            
            # Set up the progress bar
            with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as progress_bar:
                content = io.BytesIO()
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        content.write(chunk)
                        progress_bar.update(len(chunk))
            
            # Extract the downloaded zip file
            content.seek(0)
            with zipfile.ZipFile(content) as zf:
                zf.extractall(extract_to)
            print(f"Successfully downloaded and extracted files to '{extract_to}'")
    
    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e}")
    except zipfile.BadZipFile as e:
        print(f"Error extracting zip file: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def _extract_zip_files(source_folder, extract_to):
    """
    Extracts all ZIP files in a specified folder to the target directory.

    Args:
        source_folder (str): The folder containing the zip files.
        extract_to (str): The directory to extract the files to.
    """
    try:
        files = os.listdir(source_folder)
        zip_files = [file for file in files if file.endswith('.zip')]
        
        if not zip_files:
            print("No zip files found in the specified folder.")
            return
        
        for zip_file in zip_files:
            zip_path = os.path.join(source_folder, zip_file)
            print(f"Extracting '{zip_path}'...")
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(extract_to)
            print(f"Extracted '{zip_file}' to '{extract_to}'")
    
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except zipfile.BadZipFile as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def _cleanup_data_folder(data_folder):
    """
    Removes all files and folders from the data folder except for 'training' and 'testing'.

    Args:
        data_folder (str): The folder to clean up.
    """
    keep_folders = {'training', 'testing'}
    for item in os.listdir(data_folder):
        if item not in keep_folders:
            item_path = os.path.join(data_folder, item)
            try:
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                else:
                    os.remove(item_path)
                print(f"Removed '{item_path}'")
            except Exception as e:
                print(f"Error removing '{item_path}': {e}")

def download_data(data_folder):
    """
    Downloads the main dataset zip file, extracts any nested zip files, and cleans up the data folder
    so that only the 'training' and 'testing' folders remain.

    Args:
        data_folder (str): The directory where the data should be downloaded and extracted.
    """
    # Ensure the target folder exists
    os.makedirs(data_folder, exist_ok=True)
    
    # URL for the main dataset zip file
    url = "https://datadryad.org/stash/downloads/file_stream/3046117"
    
    # Download and extract the main zip file
    _download_and_unzip(url=url, extract_to=data_folder)
    
    # Extract any nested zip files (if present) from the "Training and Testing Tiles" folder
    nested_zip_folder = os.path.join(data_folder, "Training and Testing Tiles")
    if os.path.exists(nested_zip_folder):
        _extract_zip_files(source_folder=nested_zip_folder, extract_to=data_folder)
    
    # Cleanup: remove all files/folders except 'training' and 'testing'
    _cleanup_data_folder(data_folder)
    print("Data preparation complete. Data is in 'training' and 'testing' folders.")
