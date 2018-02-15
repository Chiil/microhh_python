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
    kernels.no_slip(u.data, g)
    kernels.no_slip(v.data, g)
    
    kernels.no_penetration(w.data, g)
    
    kernels.cyclic_boundaries(u.data, g)
    kernels.cyclic_boundaries(v.data, g)
    kernels.cyclic_boundaries(w.data, g)
    
    kernels.advection_u(u_tend.data, u.data, v.data, w.data, g)
    kernels.advection_v(v_tend.data, u.data, v.data, w.data, g)
    kernels.advection_w(w_tend.data, u.data, v.data, w.data, g)
    
    kernels.diffusion  (u_tend.data, u.data, visc, g)
    kernels.diffusion  (v_tend.data, v.data, visc, g)
    kernels.diffusion_w(w_tend.data, w.data, visc, g)
   
    dt = 0.001
    kernels.pressure_solve(p.data, u.data, v.data, w.data, u_tend.data, v_tend.data, w_tend.data, g, dt)
    kernels.pressure_tendency(u_tend.data, v_tend.data, w_tend.data, p.data, g)

    u.data += dt*u_tend.data
    v.data += dt*v_tend.data
    w.data += dt*w_tend.data

    kernels.calc_divergence(u.data, v.data, w.data, g)
    # End of right-hand side.

print('Finished')
