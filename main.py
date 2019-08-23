from LBPH import LBPH
import os
import cv2
from PIL import Image
import numpy as np

# Initialising LBPH object
testing_lbph = LBPH()

# Points to the entire project directory
base_directory = os.path.dirname(os.path.abspath(__file__))

# The directory containing all test images (that originate from LFW DB)
LFW_images = os.path.join(base_directory, "LFW-images")

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')

img_array = None

# Training Phase
# ==============
for root, directories, files in os.walk(LFW_images):
    for file in files:
        if (file.endswith("png") or file.endswith("jpg")) and "001" in file:
            path = os.path.join(root, file)
            label = file[:3]
            img = cv2.imread(path)

            # Converting the image into greyscale
            pil_image = Image.open(path).convert("L")
            img_array = np.array(pil_image, "uint8")
            faces = face_cascade.detectMultiScale(img_array, scaleFactor=1.5, minNeighbors=5)
            for (x, y, w, h) in faces:
                region_of_interest = img_array[y:y + h, x:x + w]
                # Training data
                testing_lbph.add_to_x_train(region_of_interest, label)

# Testing & Evaluation Phase
# ==========================
correct = 0
count = 0
for root, directories, files in os.walk(LFW_images):
    for file in files:
        if (file.endswith("png") or file.endswith("jpg")) and "002" in file:

            path = os.path.join(root, file)
            img = cv2.imread(path)

            # Converting the image into greyscale
            pil_image = Image.open(path).convert("L")
            img_array = np.array(pil_image, "uint8")
            faces = face_cascade.detectMultiScale(img_array, scaleFactor=1.5, minNeighbors=5)

            count += 1
            for (x, y, w, h) in faces:
                region_of_interest = img_array[y:y + h, x:x + w]
                label = testing_lbph.histogram_matching(region_of_interest)
                if label == file[:3]:
                    # Displays the match
                    print(file, label)
                    # Calculating the amount of times a match has been found
                    correct += 1

accuracy = correct / float(count) * 100.0
print(accuracy)