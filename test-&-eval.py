from modified_LBPH import ModifiedLBPH
import os
import cv2
from PIL import Image
import numpy as np

testing_lbph = ModifiedLBPH()

# Points to the entire project directory
base_directory = os.path.dirname(os.path.abspath(__file__))
# The directory containing all test images (that originate from LFW DB)
LFW_images = os.path.join(base_directory, "LFW-images")

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_default.xml')

img_array = None

for root, directories, files in os.walk(LFW_images):
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):
            path = os.path.join(root, file)
            # print(path)
            label = os.path.basename(os.path.dirname(path))
            img = cv2.imread(path)

            # Converting the image into greyscale
            pil_image = Image.open(path).convert("L")

            img_array = np.array(pil_image, "uint8")
            faces = face_cascade.detectMultiScale(img_array, scaleFactor=1.5, minNeighbors=5)

            for (x, y, w, h) in faces:
                # image
                region_of_interest = img_array[y:y + h, x:x + w]
                # Training data
                testing_lbph.add_to_x_train(region_of_interest, label)
                # result = testing_lbph.histogram_matching(region_of_interest)

            # Prints the label that possesses a histogram that's closest to the"
            # histogram of the target img
            # print(result)

# This is the test
# 100 should be returned
# img_100001_path = "C:\\Users\\Yusaf\\MScProject-Python-and-OpenCV-\\LFW-images\\100\\100001.jpg"
# img_100001 = cv2.imread(img_100001_path)
# print(testing_lbph.histogram_matching(img_100001))

