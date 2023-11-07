from PIL import Image
import pandas as pd

image = Image.open("C:\\Users\\wildlife_guest\\Documents\\nullremove\\ogawa.2020.12\\21\\10030020.jpg")
print(image)
exif_data = image._getexif()
print(type(exif_data[36867]))
date, time = exif_data[36867].split(' ')
print("Date:", date)
print("Time:", time)