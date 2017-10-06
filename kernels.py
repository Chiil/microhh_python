import numpy as np

def no_slip(a, g):
    a.data[g.kstart-1,:,:] = -1.*a.data[g.kstart,:,:]
    a.data[g.kend    ,:,:] = -1.*a.data[g.kend-1,:,:]

def no_penetration(a, g):
    a.data[g.kstart,:,:] = 0.
    a.data[g.kend  ,:,:] = 0.

def cyclic_boundaries(a, g):
    a.data[:,:,0:g.istart] = a.data[:,:,g.iend-g.igc:g.iend]
    a.data[:,0:g.jstart,:] = a.data[:,g.jend-g.jgc:g.jend,:]
    a.data[:,:,g.iend:] = a.data[:,:,g.istart:g.istart+g.igc]
    a.data[:,g.jend:,:] = a.data[:,g.jstart:g.jstart+g.jgc,:]

def i2(a, b):
    return 0.5*(a + b)

def advection_uvw(ut, vt, wt, u, v, w, g):
    ut[g.i(0,0,0)] -= ( i2(u[g.i( 0, 0, 0)], u[g.i( 0, 0,+1)]) * i2(u[g.i( 0, 0, 0)], u[g.i( 0, 0,+1)]) \
                      - i2(u[g.i( 0, 0,-1)], u[g.i( 0, 0, 0)]) * i2(u[g.i( 0, 0,-1)], u[g.i( 0, 0, 0)]) ) * g.dxi \
                    + ( i2(v[g.i( 0,+1,-1)], v[g.i( 0,+1, 0)]) * i2(u[g.i( 0, 0, 0)], u[g.i( 0,+1, 0)]) \
                      - i2(v[g.i( 0, 0,-1)], v[g.i( 0, 0, 0)]) * i2(u[g.i( 0,-1, 0)], u[g.i( 0, 0, 0)]) ) * g.dyi \
                    + ( i2(w[g.i(+1, 0,-1)], w[g.i(+1, 0, 0)]) * i2(u[g.i( 0, 0, 0)], u[g.i(+1, 0, 0)]) \
                      - i2(w[g.i( 0, 0,-1)], w[g.i( 0, 0, 0)]) * i2(u[g.i(-1, 0, 0)], u[g.i( 0, 0, 0)]) ) * g.dzi[g.k(0),np.newaxis,np.newaxis]

    vt[g.i(0,0,0)] -= ( i2(u[g.i( 0, 0, 0)], u[g.i( 0, 0,+1)]) * i2(v[g.i( 0, 0, 0)], v[g.i( 0, 0,+1)]) \
                      - i2(u[g.i( 0, 0,-1)], u[g.i( 0, 0, 0)]) * i2(v[g.i( 0, 0,-1)], v[g.i( 0, 0, 0)]) ) * g.dxi \
                    + ( i2(v[g.i( 0, 0, 0)], v[g.i( 0,+1, 0)]) * i2(v[g.i( 0, 0, 0)], v[g.i( 0,+1, 0)]) \
                      - i2(v[g.i( 0,-1, 0)], v[g.i( 0, 0, 0)]) * i2(v[g.i( 0,-1, 0)], v[g.i( 0, 0, 0)]) ) * g.dyi \
                    + ( i2(w[g.i(+1, 0,-1)], w[g.i(+1, 0, 0)]) * i2(v[g.i( 0, 0, 0)], v[g.i(+1, 0, 0)]) \
                      - i2(w[g.i( 0, 0,-1)], w[g.i( 0, 0, 0)]) * i2(v[g.i(-1, 0, 0)], v[g.i( 0, 0, 0)]) ) * g.dzi[g.k(0),np.newaxis,np.newaxis]

    wt[g.ih(0,0,0)] -= ( i2(u[g.ih( 0, 0, 0)], u[g.ih( 0, 0,+1)]) * i2(w[g.ih( 0, 0, 0)], w[g.ih( 0, 0,+1)]) \
                       - i2(u[g.ih( 0, 0,-1)], u[g.ih( 0, 0, 0)]) * i2(w[g.ih( 0, 0,-1)], w[g.ih( 0, 0, 0)]) ) * g.dxi \
                     + ( i2(v[g.ih( 0,+1,-1)], v[g.ih( 0,+1, 0)]) * i2(w[g.ih( 0, 0, 0)], w[g.ih( 0,+1, 0)]) \
                       - i2(v[g.ih( 0, 0,-1)], v[g.ih( 0, 0, 0)]) * i2(w[g.ih( 0,-1, 0)], w[g.ih( 0, 0, 0)]) ) * g.dyi \
                     + ( i2(w[g.ih( 0, 0, 0)], w[g.ih(+1, 0, 0)]) * i2(w[g.ih( 0, 0, 0)], w[g.ih(+1, 0, 0)]) \
                       - i2(w[g.ih(-1, 0, 0)], w[g.ih( 0, 0, 0)]) * i2(w[g.ih(-1, 0, 0)], w[g.ih( 0, 0, 0)]) ) * g.dzhi[g.kh(0),np.newaxis,np.newaxis]

