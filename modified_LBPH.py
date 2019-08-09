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
        self.radius = radius
        self.neighbors = neighbors
        self.labels = {} # Histograms

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
        histograms.append(histogram)
        # print(self.labels['yusaf'])

    # The histogram closest to the target img is the match.
    def histogram_matching(self, image):
        # Computing the histogram of the image
        img_histogram = self.lbp_process(image)
        m = 999

        for label, histograms in self.labels.items():
            for histogram in histograms:
                # Calculating the distance between the target image's histogram
                # and the histogram of 'label'
                dist = self.distance(img_histogram, histogram)
                if dist < m:
                    l = label
                    m = dist

        return l

    # Calculating the distance between two histograms
    def distance(self, histogram_1, histogram_2):
        ith = 0
        sum = 0

        while ith < len(histogram_1) and ith < len(histogram_2):
            sum += (histogram_1[ith] - histogram_2[ith])**2
            ith += 1

        # distance between histogram 1 and 2
        return math.sqrt(sum)
