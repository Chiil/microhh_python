import numpy as np

def no_slip(a, g):
    a.data[g.kstart-1,:,:] = -1.*a.data[g.kstart,:,:]
    a.data[g.kend    ,:,:] = -1.*a.data[g.kend-1,:,:]

def no_penetration(a, g):
    a.data[g.kstart,:,:] = 0.
    a.data[g.kend  ,:,:] = 0.

def cyclic_boundaries(a, g):
    a.data[:,:,g.istart-1] = a.data[:,:,g.iend-1]
    a.data[:,:,g.iend    ] = a.data[:,:,g.istart]
    a.data[:,g.jstart-1,:] = a.data[:,g.jend-1,:]
    a.data[:,g.jend    ,:] = a.data[:,g.jstart,:]

def i2(a, b):
    return 0.5*(a + b)

def advection_uvw(ut, vt, wt, u, v, w, g):
    lc = np.s_[g.kstart:g.kend, g.jstart:g.jend, g.istart:g.iend]
    lw = np.s_[g.kstart:g.kend, g.jstart:g.jend, g.istart-1:g.iend-1]
    le = np.s_[g.kstart:g.kend, g.jstart:g.jend, g.istart+1:g.iend+1]
    ls = np.s_[g.kstart:g.kend, g.jstart-1:g.jend-1, g.istart:g.iend]
    ln = np.s_[g.kstart:g.kend, g.jstart+1:g.jend+1, g.istart:g.iend]
    lb = np.s_[g.kstart-1:g.kend-1, g.jstart:g.jend, g.istart:g.iend]
    lt = np.s_[g.kstart+1:g.kend+1, g.jstart:g.jend, g.istart:g.iend]

    ut[lc] -= ( i2(u[lc], u[le])**2 \
              - i2(u[lw], u[le])**2 ) / g.dx \
            + ( i2(u[lc], u[ln]) * i2(u[lc], u[ln]) \
              - i2(u[ls], u[lc]) * i2(u[ls], u[lc]) ) / g.dy \
            + ( i2(u[lc], u[lt]) * i2(u[lc], u[lt]) \
              - i2(u[lb], u[lc]) * i2(u[lb], u[lc]) ) / g.dy

