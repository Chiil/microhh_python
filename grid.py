import numpy as np

class Grid:
    def __init__(self, itot, jtot, ktot, xsize, ysize, zsize, z):
        # Initialize the grid.
        self.xsize = xsize
        self.ysize = ysize
        self.zsize = zsize

        self.igc = 1
        self.jgc = 1
        self.kgc = 1

        self.itot = itot
        self.jtot = jtot
        self.ktot = ktot

        self.icells = self.itot + 2*self.igc
        self.jcells = self.jtot + 2*self.jgc
        self.kcells = self.ktot + 2*self.kgc

        self.x = np.empty(self.icells)
        self.y = np.empty(self.jcells)
        self.z = np.empty(self.kcells)
        self.xh = np.empty(self.icells)
        self.yh = np.empty(self.jcells)
        self.zh = np.empty(self.kcells)

        self.istart = self.igc
        self.jstart = self.jgc
        self.kstart = self.kgc

        self.iend = self.itot + self.igc
        self.jend = self.jtot + self.jgc
        self.kend = self.ktot + self.kgc

        self.dx = self.xsize / self.itot
        self.dy = self.ysize / self.jtot
        self.dxi = 1./self.dx;
        self.dyi = 1./self.dy;

        self.x [:] = 0.5*self.dx + (np.arange(self.icells)-self.igc)*self.dy
        self.xh[:] = (np.arange(self.icells)-self.igc)*self.dx

        self.y [:] = 0.5*self.dy + (np.arange(self.jcells)-self.jgc)*self.dy
        self.yh[:] = (np.arange(self.jcells)-self.jgc)*self.dy

        self.z[self.kstart:self.kend] = z[:]

        self.zh = np.empty(self.kcells)
        self.dz = np.empty(self.kcells)
        self.dzh = np.empty(self.kcells)
        self.dzi = np.empty(self.kcells)
        self.dzhi = np.empty(self.kcells)

        self.z[self.kstart-1] = -self.z[self.kstart];
        self.z[self.kend]     = 2.*self.zsize - self.z[self.kend-1];

        self.zh[self.kstart+1:self.kend] = 0.5*(self.z[self.kstart:self.kend-1]+self.z[self.kstart+1:self.kend]);
        self.zh[self.kstart] = 0.;
        self.zh[self.kend] = self.zsize;

        self.dzh [1:self.kcells] = self.z[1:self.kcells] - self.z[:self.kcells-1];
        self.dzhi[1:self.kcells] = 1./self.dzh[1:self.kcells];

        self.dzh [self.kstart-1] = self.dzh [self.kstart+1];
        self.dzhi[self.kstart-1] = self.dzhi[self.kstart+1];

        self.dz [self.kstart:self.kend] = self.zh[self.kstart+1:self.kend+1] - self.zh[self.kstart:self.kend];
        self.dzi[self.kstart:self.kend] = 1./self.dz[1:self.kcells-1];

        self.dz [self.kstart-1] = self.dz [self.kstart];
        self.dzi[self.kstart-1] = self.dzi[self.kstart];
        self.dz [self.kend]     = self.dz [self.kend-1];
        self.dzi[self.kend]     = self.dzi[self.kend-1];

    def i(self,k,j,i):
        return np.s_[self.kstart+k:self.kend+k, self.jstart+j:self.jend+j, self.istart+i:self.iend+i]

    def ih(self,k,j,i):
        return np.s_[self.kstart+k:self.kend+1+k, self.jstart+j:self.jend+j, self.istart+i:self.iend+i]

    def k(self,k):
        return np.s_[self.kstart+k:self.kend+k]

    def kh(self,k):
        return np.s_[self.kstart+k:self.kend+1+k]

