import numpy as np
from PIL import Image
import cv2
import os
from modified_LBPH import ModifiedLBPH

def train():
    lbph = ModifiedLBPH()

    base_directory = os.path.dirname(os.path.abspath(__file__))
    images2 = os.path.join(base_directory, "images2")

    face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')

    image_array = None

    for root, dirs, files in os.walk(images2):
        for file in files:
            if file.endswith("png") or file.endswith("jpg"):
                # Converting each image to grayscale then
                # computing the LBP representation of each image
                # and then the histogram of each LBP representation is generated.
                # Each generated histogram is stored
                path = os.path.join(root, file)
                label = os.path.basename(os.path.dirname(path)).replace(" ", "-").lower()
                image = cv2.imread(path)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                # lbph.LBP_process(gray)

                # Contains the name of each person into the label storage
                # lbph.add_label(label)

                # id_ = lbph.get_id(label)

                # Retrieve and convert an image from the path and converts it to grayscale
                pil_image = Image.open(path).convert("L")

                # Image size
                # size = (16, 16)
                # final_image = pil_image.resize(size, Image.ANTIALIAS)

                # Contains the numbers within the image
                image_array = np.array(pil_image, "uint8")
                faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)

                for (x, y, w, h) in faces:
                    # image
                    region_of_interest = image_array[y:y+h, x:x+w]
                    # Training data
                    lbph.add_to_x_train(region_of_interest, label)
                    result = lbph.histogram_matching(region_of_interest)

                # Prints the label that possesses a histogram that's closest to the
                # histogram of the target img
                print(result)

    return lbph

def main():
    # train()
    fr = train()

main()