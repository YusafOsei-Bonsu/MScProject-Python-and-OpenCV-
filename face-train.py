import os
import cv2
import numpy as np
from PIL import Image
import pickle

# Searching for and adding png/jpeg files from the training dataset
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# The image directory
image_dir = os.path.join(BASE_DIR, "images")

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')

# LBPH Face Recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

current_id = 0
# List of ids
label_ids = {}
y_labels = []
x_train = []

# See the images in there
image_array = None

for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):
            # Path of the image file (within directory)
            path = os.path.join(root, file)
            # Name of the folder that contains the images
            label = os.path.basename(os.path.dirname(path)).replace(" ", "-").lower()
            # print(label, path)

            # Assigning ids to labels
            if not label in label_ids:
                label_ids[label] = current_id
                current_id += 1
            id_ = label_ids[label]
            # print(label_ids)

            # Some number for labels
            # y_labels.append(label)

            # Verify image and convert it into a NUMPY array (greyscale image)
            # x_train.append(path)

            # Retrieves an image from the path and converts it to grayscale
            pil_image = Image.open(path).convert("L")

            # Image size
            size = (16, 16)
            final_image = pil_image.resize(size, Image.ANTIALIAS)

            # Contains the numbers within the image
            image_array = np.array(pil_image, "uint8")
            # print(image_array)
            faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)

            for (x, y, w, h) in faces:
                # Region of interest
                roi = image_array[y:y+h, x:x+w]
                # Training data
                x_train.append(roi)
                y_labels.append(id_)

'''print(image_array)

for i in range(16):
    for j in range(16):
        print(image_array[i][j], end=" ")
    print('\n')'''

# print(y_labels)
# print(x_train)

# Save labels so they can be used by main.py
with open("labels.pickle", 'wb') as f:
    pickle.dump(label_ids, f)

recognizer.train(x_train, np.array(y_labels))
recognizer.save("trainer.yml")
