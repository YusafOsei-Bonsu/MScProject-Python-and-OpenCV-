import numpy as np
from PIL import Image
import cv2
import os
from modified_LBPH import ModifiedLBPH

'''# Construct argument parse and parse the arguments
ap = argparse.ArgumentParser()

# Divided the images into two sets:
# A training set of 6 images per person
# A testing set of one image per person
ap.add_argument("-t", "--training", required=True, help="path to the training images")
ap.add_argument("-e", "--testing", required=True, help="path to the testing images")
args = vars(ap.parse_args())

# Initializing the LBPs descriptor along with the data and label lists
desc = ModifiedLBPH()

# Stores the feature vectors
data = []

# Stores the names of each person
labels = []

# Traverse through the training set
for imagePath in paths.list_images(["training"]):
    # Load image, convert it to grayscale and describe it
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    histogram = desc.describe(gray)

    # Extract label from image path, update the label and data lists
    labels.append(imagePath.split(os.path.sep)[-2])
    data.append(histogram)'''

def train():
    lbph = ModifiedLBPH()

    base_directory = os.path.dirname(os.path.abspath(__file__))
    images2 = os.path.join(base_directory, "images2")

    face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')

    recognizer = cv2.face.LBPHFaceRecognizer_create()

    current_id = 0
    label_ids = {}
    y_labels = []
    x_train = []

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
                lbph.describe(gray)

                # Contains the name of each person into the label storage
                lbph.add_label(label)

                id_ = lbph.get_id(label)

                # Retrieve and convert an image from the path and converts it to grayscale
                pil_image = Image.open(path).convert("L")

                # Image size
                size = (16, 16)
                final_image = pil_image.resize(size, Image.ANTIALIAS)

                # Contains the numbers within the image
                image_array = np.array(pil_image, "uint8")
                faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)

                for (x, y, w, h) in faces:
                    # Region of interest
                    region_of_interest = image_array[y:y+h, x:x+w]
                    # Training data
                    lbph.add_to_x_train(region_of_interest)
                    # x_train.append(region_of_interest)
                    lbph.add_to_y_labels(id_)
                    # y_labels.append(id_)

    # return lbph, y_labels


def main():
    train()
    # fr, labels = train()

main()