import numpy as np
import math

# 'feature' contains the implementation of the LBP descriptor
from skimage import feature

class LBPH:

    def __init__(self, radius=1, neighbors=8):
        '''
        :param radius: Builds the circular local binary pattern and represents the radius around the central pixel (usually set to 1).
        :param neighbors: Number of sample points to build the circular local binary pattern (usually set to 8).
        '''
        self.radius = radius
        self.neighbors = neighbors
        self.labels = {} # Histograms

    # This performs the entire LBPH process
    def lbph_process(self, image, eps=1e-7):
        # Computes the LBP representation of the image
        # Afterwards, use the LBP representation to generate the histogram of patterns
        local_binary_pattern = feature.local_binary_pattern(image, self.neighbors, self.radius, method="uniform")
        (histogram, _) = np.histogram(local_binary_pattern.ravel(), bins=np.arange(0, self.neighbors + 3), range=(0, self.neighbors + 2))

        # Normalize histogram
        histogram = histogram.astype("float")
        histogram /= (histogram.sum() + eps)

        # Add the histogram (of the image) to the list
        return histogram

    # Unrecorded labels are added into the list
    # after coming across it
    def add_label(self, label):

        if label not in self.labels:
            self.labels[label] = []

    def add_to_x_train(self, region_of_interest, label):
        self.add_label(label)
        histogram = self.lbph_process(region_of_interest)
        histograms = self.labels[label]
        histograms.append(histogram)

    # The histogram closest to the target img is the match.
    def histogram_matching(self, image):
        # Computing the histogram of the image
        img_histogram = self.lbph_process(image)
        m = 999

        for label, histograms in self.labels.items():
            for histogram in histograms:
                # Calculating the distance between the target image's histogram
                # and the histogram of 'label'
                dist = self.chi_square_distance(img_histogram, histogram)
                if dist < m:
                    img_label = label
                    m = dist

        return img_label

    # Euclidean Distance algorithm
    def euclidean_distance(self, histogram_1, histogram_2):
        ith = 0
        sum = 0

        while ith < len(histogram_1) and ith < len(histogram_2):
            sum += (histogram_1[ith] - histogram_2[ith])**2
            ith += 1

        # distance between histogram 1 and 2
        return math.sqrt(sum)

    # Chi Square algorithm
    def chi_square_distance(self, histogram_1, histogram_2):
        ith = 0
        sum = 0

        while ith < len(histogram_1) and ith < len(histogram_2):
            sum += ((histogram_1[ith] - histogram_2[ith])**2) / histogram_1[ith]
            ith += 1

        # distance between histogram 1 and 2
        return sum

    # Normalized euclidean Algorithm
    def normalized_eucl_distance(self, histogram_1, histogram_2):
        ith = 0
        sum = 0

        while ith < len(histogram_1) and ith < len(histogram_2):
            sum += ((histogram_1[ith] - histogram_2[ith])**2) / len(histogram_1)
            ith += 1

        # distance between histogram 1 and 2
        return math.sqrt(sum)
