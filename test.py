import numpy as np
import cv2
import pickle
from modified_LBPH import ModifiedLBPH
import os
from PIL import Image


def train():
    lbph = ModifiedLBPH()

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    image_dir = os.path.join(BASE_DIR, "images2")

    face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')

    recognizer = cv2.face.LBPHFaceRecognizer_create()

    current_id = 0
    label_ids = {}
    y_labels = []
    x_train = []

    image_array = None

    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if file.endswith("png") or file.endswith("jpg"):
                path = os.path.join(root, file)
                label = os.path.basename(os.path.dirname(path)).replace(" ", "-").lower()

                if not label in label_ids:
                    label_ids[label] = current_id
                    current_id += 1
                id_ = label_ids[label]

                pil_image = Image.open(path).convert("L")

                size = (16, 16)
                final_image = pil_image.resize(size, Image.ANTIALIAS)

                image_array = np.array(pil_image, "uint8")
                faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)

                for (x, y, w, h) in faces:
                    region_of_interest = image_array[y:y+h, x:x+w]
                    # print(region_of_interest)
                    lbph.addImage(region_of_interest)
                    y_labels.append(id_)

    return lbph, y_labels


def main():

    fr, labels = train()


main()