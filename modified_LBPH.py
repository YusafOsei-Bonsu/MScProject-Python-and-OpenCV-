import numpy as np

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

    def describe(self, image, eps=1e-7):
        # Computes the LBP representation of the image
        # Afterwards, use the LBP representation to generate the histogram of patterns
        lbp = feature.local_binary_pattern(image, self.neighbors, self.radius, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, self.neighbors + 3), range=(0, self.neighbors + 2))

        # Normalize histogram
        hist = hist.astype("float")
        hist /= (hist.sum() + eps)

        # Return the histogram of the Local Binary Patterns
        return hist