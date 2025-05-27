'''
Author: chasey && melancholycy@gmail.com
Date: 2025-04-09 01:34:44
LastEditTime: 2025-05-18 14:03:35
FilePath: /POAM/data/download.py
Description: 
Reference: 
Copyright (c) 2025 by chasey && melancholycy@gmail.com, All Rights Reserved. 
'''
from pathlib import Path
from urllib import request
import numpy as np
from PIL import Image


def download_environment(name):
    path = Path(f"./images/{name.lower()}.jpg")
    if not path.is_file():
        print(f"Downloading to {path}...this step might take some time.")
        request.urlretrieve(
            url="https://e4ftl01.cr.usgs.gov/MEASURES/SRTMGL1.003/2000.02.11/"
            + f"{name}.SRTMGL1.2.jpg",
            filename=path,
        )
        print("Done")


def preprocess_environment(image_path, array_path, resize=(360, 360)):
    print(f"Preprocessing {image_path}...")
    image = Image.open(image_path).convert("L") # open image and convert to GrayImage
    image = image.resize(size=resize, resample=Image.BICUBIC) # resize and resample through 双三次插值
    array = np.array(image).astype(np.float64)
    np.savez_compressed(array_path, array)
    print(f"Saved to {array_path}.")


if __name__ == "__main__":
    # name = "N44W111"
    name = "N17E073"
    # name = "N47W124"
    # name = "N35W107"
    
    Path(f"./images").mkdir(parents=True, exist_ok=True)
    Path(f"./arrays").mkdir(parents=True, exist_ok=True)
    image_path = Path(f"./images/{name.lower()}.jpg")
    array_path = Path(f"./arrays/{name.lower()}")
    download_environment(name)
    preprocess_environment(image_path, array_path)
