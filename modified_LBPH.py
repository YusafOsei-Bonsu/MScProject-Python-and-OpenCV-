import numpy as np


class ModifiedLBPH:
    binary_matrix = []

    def __init__(self, radius=1, neighbors=8, grid_x=8, grid_y=8):
        '''
        :param radius: Builds the circular local binary pattern and represents the radius around the central pixel (usually set to 1).
        :param neighbors: Number of sample points to build the circular local binary pattern (usually set to 8).
        :param grid_x: Number of cells in the horizontal direction (usually set to 8).
        :param grid_y: Number of cells in the vertical direction (usually set to 8).
        '''
        self.radius = radius
        self.neighbors = neighbors
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.images ={}

    # Converts the 3x3 matrix into a binary matrix
    def binary_matrix_conversion(self, three_by_three_matrix, threshold):
        # Turns the 3x3 matrix into a single list
        flattened_3_x_3_matrix = [item for sublist in three_by_three_matrix for item in sublist]
        flattened_3_x_3_matrix.pop(4)

        binary_matrix = []

        # Creates the binary matrix
        for neighbour in flattened_3_x_3_matrix:
            # Add a 1 to the binary matrix if the neighbour is bigger than the threshold
            if neighbour >= threshold:
                binary_matrix.append(1)
            # Add a 0 to the binary matrix if the neighbour is smaller than the threshold
            elif neighbour < threshold:
                binary_matrix.append(0)

        lst = []
        for index in range(0, len(binary_matrix), 4):
            lst.append(binary_matrix[index:index + 4])

        binary_matrix = np.array(lst)

        return binary_matrix

    def add_image(self, image):

        row = 0
        column = 0

        while row + 3 < len(image[0]):
            while column + 3 < len(image[0]):
                grey_sample = image[row:row+3, column:column+3]
                # print(grey_sample)
                # Central value of the matrix
                threshold = image[row+1, column+1]
                # Converting the 3x3 matrix into a binary 3x3 matrix
                b = self.binary_matrix_conversion(grey_sample, threshold)
                print(b)

                # Concatenate the binary values into one string
                binary_values = [str(digit) for sublist in b for digit in sublist]
                binary_values = "".join(binary_values)

                # Converts the binary number into a decimal one
                decimal_num = int(binary_values, 2)
                print(decimal_num)
                column += 3
            print("\n")
            row += 3
