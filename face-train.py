import os
import numpy as np
from PIL import Image

# Searching for and adding png/jpeg files into a training dataset
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# The image directory
image_dir = os.path.join(BASE_DIR, "images")

y_labels = []
x_train = []

# See the images in there
for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):
            # Path of the image file (within directory)
            path = os.path.join(root, file)
            # Name of the folder that contains the images
            label = os.path.basename(os.path.dirname(path)).replace(" ", "-").lower()
            print(label, path)
            # Some number for labels
            # y_labels.append(label)
            # Verify image and convert it into a NUMPY array (greyscale image)
            # x_train.append(path)
            # Retrieves an image from the path and converts it to grayscale
            pil_image = Image.open(path).convert("L")
            # Contains the numbers within the image
            image_array = np.array(pil_image, "uint8")
            print(image_array)