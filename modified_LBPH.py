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

