import numpy as np

class Field:
    def __init__(self, g):
        # Initialize at zero.
        self.data = np.zeros((g.kcells, g.jcells, g.icells))
    
    def randomize(self, amplitude):
        random_data = amplitude*(np.random.rand(*self.data.shape) - 0.5)
        self.data[:,:,:] += random_data
