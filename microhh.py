import numpy as np
import matplotlib.pyplot as plt

import grid
import field
import kernels

itot = 8
jtot = 6
ktot = 4

x = np.arange(0, 2.*np.pi, 2*np.pi/itot)
y = np.arange(0, np.pi, np.pi/jtot)
z = np.arange(0, 2., 2./ktot)

g = grid.Grid(x, y, z, 0, 2.)

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

for i in range(100):
    kernels.no_slip(u, g)
    kernels.no_slip(v, g)
    
    kernels.no_penetration(w, g)
    
    kernels.cyclic_boundaries(u, g)
    kernels.cyclic_boundaries(v, g)
    kernels.cyclic_boundaries(w, g)
    
    kernels.advection_uvw(u_tend.data, v_tend.data, w_tend.data,
                          u.data, v.data, w.data, g)

