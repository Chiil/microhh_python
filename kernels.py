import numpy as np

def no_slip(a, g):
    a[g.kstart-1,:,:] = -1.*a[g.kstart,:,:]
    a[g.kend    ,:,:] = -1.*a[g.kend-1,:,:]

def no_penetration(a, g):
    a[g.kstart,:,:] = 0.
    a[g.kend  ,:,:] = 0.
    a[g.kstart-1,:,:] = -1.*a[g.kstart+1,:,:]
    a[g.kend+1  ,:,:] = -1.*a[g.kend-1,:,:]

def cyclic_boundaries(a, g):
    a[:,:,0:g.istart] = a[:,:,g.iend-g.igc:g.iend]
    a[:,0:g.jstart,:] = a[:,g.jend-g.jgc:g.jend,:]
    a[:,:,g.iend:] = a[:,:,g.istart:g.istart+g.igc]
    a[:,g.jend:,:] = a[:,g.jstart:g.jstart+g.jgc,:]

def i2(a, b):
    return 0.5*(a + b)

def advection_u(ut, u, v, w, g):
    ut[g.i(0,0,0)] -= ( i2(u[g.i( 0, 0, 0)], u[g.i( 0, 0,+1)]) * i2(u[g.i( 0, 0, 0)], u[g.i( 0, 0,+1)]) \
                      - i2(u[g.i( 0, 0,-1)], u[g.i( 0, 0, 0)]) * i2(u[g.i( 0, 0,-1)], u[g.i( 0, 0, 0)]) ) * g.dxi \
                    + ( i2(v[g.i( 0,+1,-1)], v[g.i( 0,+1, 0)]) * i2(u[g.i( 0, 0, 0)], u[g.i( 0,+1, 0)]) \
                      - i2(v[g.i( 0, 0,-1)], v[g.i( 0, 0, 0)]) * i2(u[g.i( 0,-1, 0)], u[g.i( 0, 0, 0)]) ) * g.dyi \
                    + ( i2(w[g.i(+1, 0,-1)], w[g.i(+1, 0, 0)]) * i2(u[g.i( 0, 0, 0)], u[g.i(+1, 0, 0)]) \
                      - i2(w[g.i( 0, 0,-1)], w[g.i( 0, 0, 0)]) * i2(u[g.i(-1, 0, 0)], u[g.i( 0, 0, 0)]) ) * g.dzi[g.k(0),None,None]

def advection_v(vt, u, v, w, g):
    vt[g.i(0,0,0)] -= ( i2(u[g.i( 0,-1,+1)], u[g.i( 0, 0,+1)]) * i2(v[g.i( 0, 0, 0)], v[g.i( 0, 0,+1)]) \
                      - i2(u[g.i( 0,-1, 0)], u[g.i( 0, 0, 0)]) * i2(v[g.i( 0, 0,-1)], v[g.i( 0, 0, 0)]) ) * g.dxi \
                    + ( i2(v[g.i( 0, 0, 0)], v[g.i( 0,+1, 0)]) * i2(v[g.i( 0, 0, 0)], v[g.i( 0,+1, 0)]) \
                      - i2(v[g.i( 0,-1, 0)], v[g.i( 0, 0, 0)]) * i2(v[g.i( 0,-1, 0)], v[g.i( 0, 0, 0)]) ) * g.dyi \
                    + ( i2(w[g.i(+1, 0,-1)], w[g.i(+1, 0, 0)]) * i2(v[g.i( 0, 0, 0)], v[g.i(+1, 0, 0)]) \
                      - i2(w[g.i( 0, 0,-1)], w[g.i( 0, 0, 0)]) * i2(v[g.i(-1, 0, 0)], v[g.i( 0, 0, 0)]) ) * g.dzi[g.k(0),None,None]

def advection_w(wt, u, v, w, g):
    wt[g.ih(0,0,0)] -= ( i2(u[g.ih( 0, 0, 0)], u[g.ih( 0, 0,+1)]) * i2(w[g.ih( 0, 0, 0)], w[g.ih( 0, 0,+1)]) \
                       - i2(u[g.ih( 0, 0,-1)], u[g.ih( 0, 0, 0)]) * i2(w[g.ih( 0, 0,-1)], w[g.ih( 0, 0, 0)]) ) * g.dxi \
                     + ( i2(v[g.ih( 0,+1,-1)], v[g.ih( 0,+1, 0)]) * i2(w[g.ih( 0, 0, 0)], w[g.ih( 0,+1, 0)]) \
                       - i2(v[g.ih( 0, 0,-1)], v[g.ih( 0, 0, 0)]) * i2(w[g.ih( 0,-1, 0)], w[g.ih( 0, 0, 0)]) ) * g.dyi \
                     + ( i2(w[g.ih( 0, 0, 0)], w[g.ih(+1, 0, 0)]) * i2(w[g.ih( 0, 0, 0)], w[g.ih(+1, 0, 0)]) \
                       - i2(w[g.ih(-1, 0, 0)], w[g.ih( 0, 0, 0)]) * i2(w[g.ih(-1, 0, 0)], w[g.ih( 0, 0, 0)]) ) * g.dzhi[g.kh(0),None,None]

def diffusion(at, a, visc, g):
    dxidxi = g.dxi**2
    dyidyi = g.dyi**2
    at[g.i(0,0,0)] += visc * ( \
                    + ( (a[g.i( 0, 0,+1)] - a[g.i( 0, 0, 0)]) \
                      - (a[g.i( 0, 0, 0)] - a[g.i( 0, 0,-1)]) ) * dxidxi \
                    + ( (a[g.i( 0,+1, 0)] - a[g.i( 0, 0, 0)]) \
                      - (a[g.i( 0, 0, 0)] - a[g.i( 0,-1, 0)]) ) * dyidyi \
                    + ( (a[g.i(+1, 0, 0)] - a[g.i( 0, 0, 0)]) * g.dzhi[g.k(+1),None,None] \
                      - (a[g.i( 0, 0, 0)] - a[g.i(-1, 0, 0)]) * g.dzhi[g.k( 0),None,None] ) * g.dzi[g.k(0),None,None] )

def diffusion_w(at, a, visc, g):
    dxidxi = g.dxi**2
    dyidyi = g.dyi**2
    at[g.ih(0,0,0)] += visc * ( \
                     + ( (a[g.ih( 0, 0,+1)] - a[g.ih( 0, 0, 0)]) \
                       - (a[g.ih( 0, 0, 0)] - a[g.ih( 0, 0,-1)]) ) * dxidxi \
                     + ( (a[g.ih( 0,+1, 0)] - a[g.ih( 0, 0, 0)]) \
                       - (a[g.ih( 0, 0, 0)] - a[g.ih( 0,-1, 0)]) ) * dyidyi \
                     + ( (a[g.ih(+1, 0, 0)] - a[g.ih( 0, 0, 0)]) * g.dzi[g.kh( 0),None,None] \
                       - (a[g.ih( 0, 0, 0)] - a[g.ih(-1, 0, 0)]) * g.dzi[g.kh(-1),None,None] ) * g.dzhi[g.kh(0),None,None] )

def pressure_solve(p, u, v, w, ut, vt, wt, g, dt):
    cyclic_boundaries(ut, g)
    cyclic_boundaries(vt, g)

    tmp = ( (ut[g.i( 0, 0,+1)] + u[g.i( 0, 0,+1)]/dt) - (ut[g.i( 0, 0, 0)] + u[g.i( 0, 0, 0)]/dt) ) * g.dxi \
        + ( (vt[g.i( 0,+1, 0)] + v[g.i( 0,+1, 0)]/dt) - (vt[g.i( 0, 0, 0)] + v[g.i( 0, 0, 0)]/dt) ) * g.dyi \
        + ( (wt[g.i(+1, 0, 0)] + w[g.i(+1, 0, 0)]/dt) - (wt[g.i( 0, 0, 0)] + w[g.i( 0, 0, 0)]/dt) ) * g.dzi[g.k(0), None, None]

    tmp = np.fft.rfft2(tmp)

    tmp = np.fft.irfft2(tmp)

    p[g.i( 0, 0, 0)] = tmp[:,:,:]

    # Set a zero gradient at the wall.
    p[g.kstart-1,:,:] = p[g.kstart,:,:]

def pressure_tendency(ut, vt, wt, p, g):
    ut[g.i ( 0, 0, 0)] -= (p[g.i ( 0, 0,+1)] - p[g.i ( 0, 0, 0)]) * g.dxi
    vt[g.i ( 0, 0, 0)] -= (p[g.i ( 0,+1, 0)] - p[g.i ( 0, 0, 0)]) * g.dyi
    wt[g.ih( 0, 0, 0)] -= (p[g.ih(+1, 0, 0)] - p[g.ih( 0, 0, 0)]) * g.dzhi[g.kh(0), None, None]

