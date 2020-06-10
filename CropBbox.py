from PIL import Image
import cv2
import os

# Crops the empty alpha surrounding the actual images inside a folder
def CropContents(dir):
    for dirName, subdirList, fileList in os.walk(dir):
        for filename in fileList:
            file = os.path.join(dirName, filename)
            if filename.endswith(".jpg") or filename.endswith(".png"):
                image = Image.open(file)
                bbox = image.convert("RGBa").getbbox()
                image2 = image.crop(bbox)
                image2.save(file)
