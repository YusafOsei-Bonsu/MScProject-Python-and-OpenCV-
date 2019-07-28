import numpy as np
import math

# 'feature' contains the implementation of the LBP descriptor
from skimage import feature

class ModifiedLBPH:

    def __init__(self, radius=1, neighbors=8):
        '''
        :param radius: Builds the circular local binary pattern and represents the radius around the central pixel (usually set to 1).
        :param neighbors: Number of sample points to build the circular local binary pattern (usually set to 8).
        '''
        self.histograms = None
        self.radius = radius
        self.neighbors = neighbors
        self.labels = {}

    def lbp_process(self, image, eps=1e-7):
        # Computes the LBP representation of the image
        # Afterwards, use the LBP representation to generate the histogram of patterns
        lbp = feature.local_binary_pattern(image, self.neighbors, self.radius, method="uniform")
        (histogram, _) = np.histogram(lbp.ravel(), bins=np.arange(0, self.neighbors + 3), range=(0, self.neighbors + 2))

        # Normalize histogram
        histogram = histogram.astype("float")
        histogram /= (histogram.sum() + eps)

        # Add the histogram (of the image) to the list
        return histogram

    def add_label(self, label):

        if label not in self.labels:
            self.labels[label] = []

    def add_to_x_train(self, roi, label):
        self.add_label(label)
        histogram = self.lbp_process(roi)
        histograms = self.labels[label]
        # print(self.labels['yusaf'])
        histograms.append(histogram)

    # The histogram closest to the target img is the match.
    def histogram_matching(self, image):
        # Computing histogram from given image
        histogram = self.lbp_process(image)
        l = 0
        m = 999

        for j in len(self.labels):
            for k in len(j):
                d = self.distance(histogram, self.labels[j][k])
                if d < m:
                    l = j
                    m = d
        return l

    # Calculating the distance between two histograms
    def distance(self, histogram_1, histogram_2):
        sum = 0

        for i in range(1, 1000):
            sum += ((histogram_1[i] - histogram_2[i]) ** 2)

        return math.sqrt(sum)
