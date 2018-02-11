import numpy as np
import matplotlib.pyplot as plt

import grid
import field
import kernels

# Settings for simulation.
itot = 64
jtot = 48
ktot = 32

xsize = 2.*np.pi
ysize = np.pi
zsize = 2.

z = np.zeros(ktot)
alpha = 0.967
for k in range(ktot):
    eta  = -1. + 2.*((k+1)-0.5) / ktot
    z[k] = zsize / (2.*alpha) * np.tanh(eta*0.5*(np.log(1.+alpha) - np.log(1.-alpha))) + 0.5*zsize

visc = 1.e-5
# End of settings.

print('Start')

# Initialization.
g = grid.Grid(itot, jtot, ktot, xsize, ysize, zsize, z)

u = field.Field(g)
v = field.Field(g)
w = field.Field(g)
p = field.Field(g)

u_tend = field.Field(g)
v_tend = field.Field(g)
w_tend = field.Field(g)

u.randomize(1e-3)
v.randomize(1e-3)
w.randomize(1e-3)

for n in range(1):
    # Calculate right-hand side.
    kernels.no_slip(u, g)
    kernels.no_slip(v, g)
    
    kernels.no_penetration(w, g)
    
    kernels.cyclic_boundaries(u, g)
    kernels.cyclic_boundaries(v, g)
    kernels.cyclic_boundaries(w, g)
    
    kernels.advection_u(u_tend.data, u.data, v.data, w.data, g)
    kernels.advection_v(v_tend.data, u.data, v.data, w.data, g)
    kernels.advection_w(w_tend.data, u.data, v.data, w.data, g)
    
    kernels.diffusion  (u_tend.data, u.data, visc, g)
    kernels.diffusion  (v_tend.data, v.data, visc, g)
    kernels.diffusion_w(w_tend.data, w.data, visc, g)
    
    kernels.pressure_input(p, u, v, w, u_tend, v_tend, w_tend, g)
    kernels.pressure_solve()
    kernels.pressure_tendency()
    # End of right-hand side.

print('Finished')
