import numpy as np

class Grid:
    def init(self, x, y, z, zbot, ztop):
        # Initialize the grid.
        self.igc = 1
        self.jgc = 1
        self.kgc = 1

        self.itot = x.size
        self.jtot = y.size
        self.ktot = z.size

        self.icells = self.itot + 2*gc
        self.jcells = self.jtot + 2*gc
        self.kcells = self.ktot + 2*gc

        self.x = np.empty(self.icells)
        self.y = np.empty(self.jcells)
        self.z = np.empty(self.kcells)

        self.istart = gc
        self.jstart = gc
        self.kstart = gc

        self.iend = self.itot + gc
        self.jend = self.jtot + gc
        self.kend = self.ktot + gc

        self.x[self.istart:self.iend] = x[:]
        self.y[self.jstart:self.jend] = y[:]
        self.z[self.kstart:self.kend] = z[:]

        self.dx = self.x[self.istart+1] - self.x[self.istart]
        self.dy = self.y[self.jstart+1] - self.y[self.jstart]
