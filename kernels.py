import numpy as np

def no_slip(a, g):
    a.data[g.kstart-1,:,:] = -1.*a.data[g.kstart,:,:]
    a.data[g.kend    ,:,:] = -1.*a.data[g.kend-1,:,:]

def no_penetration(a, g):
    a.data[g.kstart,:,:] = 0.
    a.data[g.kend  ,:,:] = 0.
    a.data[g.kstart-1,:,:] = -1.*a.data[g.kstart+1,:,:]
    a.data[g.kend+1  ,:,:] = -1.*a.data[g.kend-1,:,:]

def cyclic_boundaries(a, g):
    a.data[:,:,0:g.istart] = a.data[:,:,g.iend-g.igc:g.iend]
    a.data[:,0:g.jstart,:] = a.data[:,g.jend-g.jgc:g.jend,:]
    a.data[:,:,g.iend:] = a.data[:,:,g.istart:g.istart+g.igc]
    a.data[:,g.jend:,:] = a.data[:,g.jstart:g.jstart+g.jgc,:]

def i2(a, b):
    return 0.5*(a + b)

def advection_u(ut, u, v, w, g):
    ut[g.i(0,0,0)] -= ( i2(u[g.i( 0, 0, 0)], u[g.i( 0, 0,+1)]) * i2(u[g.i( 0, 0, 0)], u[g.i( 0, 0,+1)]) \
                      - i2(u[g.i( 0, 0,-1)], u[g.i( 0, 0, 0)]) * i2(u[g.i( 0, 0,-1)], u[g.i( 0, 0, 0)]) ) * g.dxi \
                    + ( i2(v[g.i( 0,+1,-1)], v[g.i( 0,+1, 0)]) * i2(u[g.i( 0, 0, 0)], u[g.i( 0,+1, 0)]) \
                      - i2(v[g.i( 0, 0,-1)], v[g.i( 0, 0, 0)]) * i2(u[g.i( 0,-1, 0)], u[g.i( 0, 0, 0)]) ) * g.dyi \
                    + ( i2(w[g.i(+1, 0,-1)], w[g.i(+1, 0, 0)]) * i2(u[g.i( 0, 0, 0)], u[g.i(+1, 0, 0)]) \
                      - i2(w[g.i( 0, 0,-1)], w[g.i( 0, 0, 0)]) * i2(u[g.i(-1, 0, 0)], u[g.i( 0, 0, 0)]) ) * g.dzi[g.k(0),np.newaxis,np.newaxis]

def advection_v(vt, u, v, w, g):
    vt[g.i(0,0,0)] -= ( i2(u[g.i( 0,-1,+1)], u[g.i( 0, 0,+1)]) * i2(v[g.i( 0, 0, 0)], v[g.i( 0, 0,+1)]) \
                      - i2(u[g.i( 0,-1, 0)], u[g.i( 0, 0, 0)]) * i2(v[g.i( 0, 0,-1)], v[g.i( 0, 0, 0)]) ) * g.dxi \
                    + ( i2(v[g.i( 0, 0, 0)], v[g.i( 0,+1, 0)]) * i2(v[g.i( 0, 0, 0)], v[g.i( 0,+1, 0)]) \
                      - i2(v[g.i( 0,-1, 0)], v[g.i( 0, 0, 0)]) * i2(v[g.i( 0,-1, 0)], v[g.i( 0, 0, 0)]) ) * g.dyi \
                    + ( i2(w[g.i(+1, 0,-1)], w[g.i(+1, 0, 0)]) * i2(v[g.i( 0, 0, 0)], v[g.i(+1, 0, 0)]) \
                      - i2(w[g.i( 0, 0,-1)], w[g.i( 0, 0, 0)]) * i2(v[g.i(-1, 0, 0)], v[g.i( 0, 0, 0)]) ) * g.dzi[g.k(0),np.newaxis,np.newaxis]

def advection_w(wt, u, v, w, g):
    wt[g.ih(0,0,0)] -= ( i2(u[g.ih( 0, 0, 0)], u[g.ih( 0, 0,+1)]) * i2(w[g.ih( 0, 0, 0)], w[g.ih( 0, 0,+1)]) \
                       - i2(u[g.ih( 0, 0,-1)], u[g.ih( 0, 0, 0)]) * i2(w[g.ih( 0, 0,-1)], w[g.ih( 0, 0, 0)]) ) * g.dxi \
                     + ( i2(v[g.ih( 0,+1,-1)], v[g.ih( 0,+1, 0)]) * i2(w[g.ih( 0, 0, 0)], w[g.ih( 0,+1, 0)]) \
                       - i2(v[g.ih( 0, 0,-1)], v[g.ih( 0, 0, 0)]) * i2(w[g.ih( 0,-1, 0)], w[g.ih( 0, 0, 0)]) ) * g.dyi \
                     + ( i2(w[g.ih( 0, 0, 0)], w[g.ih(+1, 0, 0)]) * i2(w[g.ih( 0, 0, 0)], w[g.ih(+1, 0, 0)]) \
                       - i2(w[g.ih(-1, 0, 0)], w[g.ih( 0, 0, 0)]) * i2(w[g.ih(-1, 0, 0)], w[g.ih( 0, 0, 0)]) ) * g.dzhi[g.kh(0),np.newaxis,np.newaxis]

def diffusion(at, a, visc, g):
    dxidxi = g.dxi**2
    dyidyi = g.dyi**2
    at[g.i(0,0,0)] += visc * ( \
                    + ( (a[g.i( 0, 0,+1)] - a[g.i( 0, 0, 0)]) \
                      - (a[g.i( 0, 0, 0)] - a[g.i( 0, 0,-1)]) ) * dxidxi \
                    + ( (a[g.i( 0,+1, 0)] - a[g.i( 0, 0, 0)]) \
                      - (a[g.i( 0, 0, 0)] - a[g.i( 0,-1, 0)]) ) * dyidyi \
                    + ( (a[g.i(+1, 0, 0)] - a[g.i( 0, 0, 0)]) * g.dzhi[g.k(+1),np.newaxis,np.newaxis] \
                      - (a[g.i( 0, 0, 0)] - a[g.i(-1, 0, 0)]) * g.dzhi[g.k( 0),np.newaxis,np.newaxis] ) * g.dzi[g.k(0),np.newaxis,np.newaxis] )

def diffusion_w(at, a, visc, g):
    dxidxi = g.dxi**2
    dyidyi = g.dyi**2
    at[g.ih(0,0,0)] += visc * ( \
                     + ( (a[g.ih( 0, 0,+1)] - a[g.ih( 0, 0, 0)]) \
                       - (a[g.ih( 0, 0, 0)] - a[g.ih( 0, 0,-1)]) ) * dxidxi \
                     + ( (a[g.ih( 0,+1, 0)] - a[g.ih( 0, 0, 0)]) \
                       - (a[g.ih( 0, 0, 0)] - a[g.ih( 0,-1, 0)]) ) * dyidyi \
                     + ( (a[g.ih(+1, 0, 0)] - a[g.ih( 0, 0, 0)]) * g.dzi[g.kh( 0),np.newaxis,np.newaxis] \
                       - (a[g.ih( 0, 0, 0)] - a[g.ih(-1, 0, 0)]) * g.dzi[g.kh(-1),np.newaxis,np.newaxis] ) * g.dzhi[g.kh(0),np.newaxis,np.newaxis] )

#    // set the cyclic boundary conditions for the tendencies
#    grid->boundary_cyclic(ut, East_west_edge  );
#    grid->boundary_cyclic(vt, North_south_edge);
#
#    // write pressure as a 3d array without ghost cells
#    for (int k=0; k<grid->kmax; k++)
#        for (int j=0; j<grid->jmax; j++)
##pragma ivdep
#            for (int i=0; i<grid->imax; i++)
#            {
#                const int ijkp = i + j*jjp + k*kkp;
#                const int ijk  = i+igc + (j+jgc)*jj + (k+kgc)*kk;
#                p[ijkp] = rhoref[k+kgc] * ( (ut[ijk+ii] + u[ijk+ii] * dti) - (ut[ijk] + u[ijk] * dti) ) * dxi
#                        + rhoref[k+kgc] * ( (vt[ijk+jj] + v[ijk+jj] * dti) - (vt[ijk] + v[ijk] * dti) ) * dyi
#                        + ( rhorefh[k+kgc+1] * (wt[ijk+kk] + w[ijk+kk] * dti) 
#                          - rhorefh[k+kgc  ] * (wt[ijk   ] + w[ijk   ] * dti) ) * dzi[k+kgc];
#            }

def pressure_input(p, u, v, w, u_tend, v_tend, w_tend, g):
    cyclic_boundaries(u_tend, g)
    cyclic_boundaries(v_tend, g)

def pressure_solve():
    return

def pressure_tendency():
    return

