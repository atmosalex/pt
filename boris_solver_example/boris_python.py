 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 11 2019

@author: vedantvarshney
"""

import scipy as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#Constants

global m1, q1, R_e, gamma, dtp, mu

m1 = 1*1.67262189821e-27/1836.15267389
q1 = -1*1.602176620898e-19 #electron charge
mq = abs(m1/q1)
c = 299792458
R_e = 6.4e6

#INITIALISATION
R_b = 6;
ke = 50e6;
gamma = (ke*abs(q1))/(m1*c*c)+1;
ke = (gamma-1.)*m1*c*c / abs(q1);

gamma_plus = gamma
vmag = sp.sqrt((c**2)-((c**2)/(gamma**2)))

dirx = 0.5
diry = 0.0
dirz = 0.5


norm = sp.sqrt(pow(dirx,2) + pow(diry,2) + pow(dirz,2))  #magnitude - for normalisation

dirx = dirx/norm #normalised directions
diry = diry/norm
dirz = dirz/norm

uxn = dirx*vmag*gamma #x component of u where u is relativistic momentum ie u = gamma * velocity
uyn = diry*vmag*gamma
uzn = dirz*vmag*gamma

vsq = vmag**2

#INITIALISE POSITION
x0 = 0
y0 = -R_b*R_e
z0 = 0

#DIPOLE FIELD - field has to be interpolated to the particle position
def interpf(xh, yh, zh):
    Mdir_x = 0
    Mdir_y = 0
    Mdir_z = -1 #FIELD ONLY IN Z
  
    M = 7.94e22
    r = sp.sqrt( pow(xh,2) + pow(yh,2) + pow(zh,2) )
    C1 = 1e-7*M/(r**3)
    mr = Mdir_x*xh + Mdir_y*yh + Mdir_z*zh
    bx = C1*(3*xh*mr/(r**2) - Mdir_x)
    by = C1*(3*yh*mr/(r**2) - Mdir_y)
    bz = C1*(3*zh*mr/(r**2) - Mdir_z)
    
#    B=[bx,by,bz]
    
    return (bx, by, bz)

#    INITIALISE B CALL
bx0, by0, bz0 = interpf(x0, y0, z0) # B-field values at initial position
Bmag = sp.sqrt(bx0**2 + by0**2 + bz0**2) #B-field magnitude

# TODO - Re-init position with gydroradius offset

#CALCULATE TIME STEP
dtp = 1/(abs(q1)*Bmag/(m1*gamma)) / 10 #time step - will be adaptive as spatially varying B-field
dtp = 3.054432563061632E-004

t=0 #start time
val = 15; Dval_time = val/10

#store lists - list of paramter values at each time

#store the initial values at the zeroth (first) index
x_store = [x0]
y_store = [y0]
z_store = [z0]
t_store = [t]
ke_store = [ke]
gamma_store = [gamma]

vp_store = []
vper_store = []
vpara_store = []
E_store = []
vpara_store2 = []
mu_store = []
mu2_store = []
dtpa_store = []
dtpw_store = []
dtp_store = []

# [ctime1, tf]=clock;val_time=0 NO DIRECT TRANSLATION

#i=0 #Python indexing starts at 0 (MATLAB starts at 1)


#EXTRA
xnh_store = []
ynh_store = []
znh_store = []

while t<val:
    t=t+dtp #increase by time step - dtp will change later
    t_store.append(t)
    
    qdto2m = (q1/m1) * (dtp/2) #to be used later in equation 10 to find u minus (um)
    
#    UPDATE PARTICLE POSITIONS
    xnh = x0 + (uxn/gamma_plus) * 0.5 * dtp #the middle term is simply the average velocity in the x-component as defined for the Boris method
    ynh  = y0 + (uyn/gamma_plus) * 0.5 * dtp
    znh  = z0 + (uzn/gamma_plus) * 0.5 * dtp
    
#    EXTRA
    xnh_store.append(xnh)
    ynh_store.append(ynh)
    znh_store.append(znh)
    
#    B field call
    bx0, by0, bz0 = interpf(xnh, ynh, znh) #interpolate the B-field at the new positions

#    print(bx0)

    Bmag = sp.sqrt(pow(bx0,2) + pow(by0,2) + pow(bz0,2)) #magnitude of B-field at new positions
    
#     E field set to zero
    qEx = 0; qEy = 0; qEz = 0 #Boris method only conserves energy for pure magnetic fields (i.e. no electric field)
    
# Equation 10 - First half of the electric field acceleration. m means minus
    uxm = uxn + qdto2m*qEx
    uym = uyn + qdto2m*qEy 
    uzm = uzn + qdto2m*qEz 
    
    um_mag = pow((uxm*uxm + uym*uym + uzm*uzm),0.5)

    gamma_minus = sp.sqrt(1 + (um_mag/c)**2) #gamma_minus as per definition
    
#    First half of the rotation, v'  = v- + v- x t, v- = vn in absence of E-field.
    tx = (qdto2m * bx0) / gamma_minus   #!bxc(ix,iy,iz) calculating the t components as defined
    ty = (qdto2m * by0) / gamma_minus   #!byc(ix,iy,iz)
    tz = (qdto2m * bz0) / gamma_minus   #!bzc(ix,iy,iz)
    tmag  = sp.sqrt(tx**2 + ty**2 + tz**2) 
    sx = 2 * tx / (1 + tmag**2) #calculating s components as defined
    sy = 2 * ty / (1 + tmag**2) 
    sz = 2 * tz / (1 + tmag**2) 
    
#     Eq. 11 middle terms
    utx = uxm + (uym * tz - uzm * ty) 
    uty = uym + (uzm * tx - uxm * tz) 
    utz = uzm + (uxm * ty - uym * tx) 
    
#    !- Second half of the rotation, v+ = v- + v' x s, v+ = vn+1 in absence of E-field
#    !-  Therefore vn+1 = vn + [(vn + vn x t) x s]
    
#     Eq. 11 end term x s    
    upx = uxm + (uty * sz - utz * sy) 
    upy = uym + (utz * sx - utx * sz) 
    upz = uzm + (utx * sy - uty * sx) 

#     Eq. 12 - second half of the electric field acceleration
    uxn = upx + qdto2m*qEx 
    uyn = upy + qdto2m*qEy 
    uzn = upz + qdto2m*qEz 
    up_mag = (uxn*uxn + uyn*uyn + uzn*uzn)**0.5
  
    gamma_plus = sp.sqrt(1+(up_mag/c)**2); # gamma_minus. Calculating the new gamma plus
    vmag = sp.sqrt((uxn/gamma_plus)**2 + (uyn/gamma_plus)**2 + (uzn/gamma_plus)**2)

#    !- Update particle position - new particle positions to be used in the next iteration.
    x0  = xnh + (uxn/gamma_plus) * 0.5 * dtp 
    y0  = ynh + (uyn/gamma_plus) * 0.5 * dtp 
    z0  = znh + (uzn/gamma_plus) * 0.5 * dtp 
    

#originally /100
    dtp = 1/(abs(q1)*Bmag/(m1*gamma)) / 100
    
#    Store the new parameters
    x_store.append(x0)
    y_store.append(y0)
    z_store.append(z0)
    ke_store.append(ke)
    gamma_store.append(gamma)
    
#    Print statement to check progress through iteration.
  #  print('t: ',t)
    
#PLOT TRAJECTORIES

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot(sp.asarray(x_store)/R_e, sp.asarray(y_store)/R_e, sp.asarray(z_store)/R_e)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

plt.savefig('dipole_analyt_plot.png')
plt.show()

