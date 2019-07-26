import numpy as np

# 'feature' contains the implementation of the LBP descriptor
from skimage import feature

class ModifiedLBPH:
    current_id = 0

    # Stores the feature vectors
    data = []

    # Stores the name of each person
    labels = {}

    # List of IDs
    y_labels = []
    x_train = []

    def __init__(self, radius=1, neighbors=8):
        '''
        :param radius: Builds the circular local binary pattern and represents the radius around the central pixel (usually set to 1).
        :param neighbors: Number of sample points to build the circular local binary pattern (usually set to 8).
        '''
        self.radius = radius
        self.neighbors = neighbors

    def LBP_process(self, image, eps=1e-7):
        # Computes the LBP representation of the image
        # Afterwards, use the LBP representation to generate the histogram of patterns
        lbp = feature.local_binary_pattern(image, self.neighbors, self.radius, method="uniform")
        (histogram, _) = np.histogram(lbp.ravel(), bins=np.arange(0, self.neighbors + 3), range=(0, self.neighbors + 2))

        # Normalize histogram
        histogram = histogram.astype("float")
        histogram /= (histogram.sum() + eps)

        # Add the histogram (of the image) to the list
        self.data.append(histogram)

    def add_label(self, label):

        if label not in self.labels:
            self.labels[label] = self.current_id
            self.current_id += 1

    def add_to_x_train(self, roi):
        self.x_train.append(roi)

    def add_to_y_labels(self, _id):
        self.y_labels.append(_id)

    def get_y_labels(self):
        return self.y_labels

    def get_id(self, label):
        return self.labels[label]
