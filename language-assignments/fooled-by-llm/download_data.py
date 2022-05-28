"""
Downloads the image data from a zip file in google drive
"""

from google_drive_downloader import GoogleDriveDownloader as gdd
import src.util as util

FILE_ID = "1IoTRrJNDJqvaG3hnUpnHQyGvPAJbO8y3"
DATA_DIR = util.create_dir("data/raw")
DESTINATION = DATA_DIR / "data.zip"
if __name__ == "__main__":
    gdd.download_file_from_google_drive(
        file_id=FILE_ID, dest_path=DESTINATION, unzip=True
    )

