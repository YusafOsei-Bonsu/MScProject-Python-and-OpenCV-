class ModifiedLBPH:
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

    def add_image(self, image):

        row = 0
        col = 0

        while row + 3 < len(image):
            while col + 3 < len(image):
                print("3x3 matrix:")
                grey_sample = image[row:row+3, col:col+3]
                print(grey_sample)
                # Central value of the matrix
                threshold = image[row+1, col+1]
                # Converting the 3x3 matrix into a binary 3x3 matrix
                binary_matrix = binary_matrix_conversion(grey_sample, threshold)
                # Converts the binary number into a decimal one
                decimal_num = binary_to_dec(binary_matrix)
                col += 3
            print("\n")
            row += 3


# Converts the 3x3 matrix into a binary matrix
def binary_matrix_conversion(three_by_three_matrix, threshold):
    # Turns the 3x3 matrix into a single list
    flattened_3_x_3_matrix = [item for sublist in three_by_three_matrix for item in sublist]

    # The central value of the 3x3 matrix
    # threshold = flattened_3_x_3_matrix[int((len(flattened_3_x_3_matrix) - 1) / 2)]

    binary_matrix = []

    # Creates the binary matrix
    for neighbour in flattened_3_x_3_matrix:
        # If the position of the current neighbour is the same as the one in the threshold
        if flattened_3_x_3_matrix.index(neighbour) == flattened_3_x_3_matrix.index(threshold):
            # then make the central value into an empty space
            pass
        elif threshold <= neighbour:
            # Add a 1 to the binary matrix if the threshold is bigger than the current neighbour
            binary_matrix.append(1)
        elif threshold > neighbour:
            # Add a 0 to the binary matrix if the threshold is bigger than the current neighbour
            binary_matrix.append(0)

        '''
        three_by_three_binary_matrix = []

        # Converting the binary list into a 3x3 matrix
        for index in range(0, len(binary_matrix), 3):
        three_by_three_binary_matrix.append(binary_matrix[index:index + 3])
            '''

    return binary_matrix


# Finds the decimal equivalent of the binary number
def binary_to_dec(binary_number):
    binary = ""
    for digit in binary_number:
        binary += str(digit)

    # Converts binary into its decimal equivalent
    return int(binary, 2)
