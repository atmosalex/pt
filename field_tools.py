#https://www.researchgate.net/publication/329004945_On_the_Boris_solver_in_particle-in-cell_simulation
import field_h5
field_h5.verbose = False
from os.path import exists
import numpy as np
class constants:
    c = 299792458
    MeV2J = 1.60218e-13
    G2T = 1e-4
    nT2T = 1e-9
    mu0 = 1.25663706e-6
    RE = 6.3712e6
    mass0_proton = 1.67262189821e-27
    charge_proton = 1.602176620898e-19
pi = np.pi
import sys
from math import cos, sin, sqrt, atan2, tan


class Epulse: #method of Li et al, 1993
    def __init__(self, E0=240e-3, c1=0.8, c2=0.8, c3=8., v0=2.e6, ti=80, phi0=pi/4, d=30.e6):
        self.E0 = E0 #V/m
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.v0 = v0 #m/s
        self.ti = ti
        self.phi0 = phi0 #rad
        self.d = d #m
        self.td = 2 * 1.03 * constants.RE / v0 #reflection occurs at 1.03RE

    def Ephi_dEphidr(self, t, r, phi): #r in m
        #r and phi must be in the GSE frame
        # phi increases eastward

        #calculate Ephi and dEphidr:
        tph = self.ti + (self.c3*constants.RE/self.v0) * (1 - cos(phi - self.phi0))
        xi2 = ((r + self.v0 * (t - tph)) / self.d)**2
        eta2 = ((r - self.v0 * (t - tph + self.td)) / self.d)**2

        dxi2dr = (2 * r + 2 * self.v0 * (t - tph))/(self.d**2)
        deta2dr = (2 * r - 2 * self.v0 * (t - tph + self.td))/(self.d**2)

        a = -self.E0 * (1 + self.c1 * cos(phi - self.phi0))
        Ephi = a * (np.exp(-1*xi2) - self.c2 * np.exp(-1*eta2))

        dEphidr = a * (-1 * dxi2dr * np.exp(-1*xi2) - self.c2 * -1 * deta2dr * np.exp(-1*eta2))

        return Ephi, dEphidr

    # def Ephi_max(self): #future update: implement this function to find the maximum pulse amplitude
    #     Ephi1 = 0
    #     return Ephi1

#
#
# COORDINATE TRANSFORMATIONS/ PROJECTIONS
#
#
# def coord_car2sph_np(xyz): #fast cartesian to spherical, from https://stackoverflow.com/questions/4116658/faster-numpy-cartesian-to-spherical-coordinate-conversion
#     ptsnew = np.hstack((xyz, np.zeros(xyz.shape)))
#     xy = xyz[:,0]**2 + xyz[:,1]**2
#     ptsnew[:,3] = np.sqrt(xy + xyz[:,2]**2)
#     ptsnew[:,4] = np.arctan2(np.sqrt(xy), xyz[:,2]) # for elevation angle defined from Z-axis down
#     #ptsnew[:,4] = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
#     ptsnew[:,5] = np.arctan2(xyz[:,1], xyz[:,0])
#     return ptsnew
def coord_car2sph(x,y,z): #cartesian to spherical
    xy = x**2 + y**2
    r = sqrt(xy + z**2)
    th = atan2(sqrt(xy), z) # for elevation angle defined from Z-axis down
    phi = atan2(y, x)
    return r, th, phi
def coord_sph2car(r_, theta, phi):
    x = r_*cos(phi)*sin(theta)
    y = r_*sin(phi)*sin(theta)
    z = r_*cos(theta)
    return x, y, z

def project_car2sph(r, th, phi, vx, vy, vz):
    #r, th, phi = coord_car2sph(x,y,z)
    A = np.array([
        [np.sin(th) * np.cos(phi), np.sin(th) * np.sin(phi), np.cos(th)],
        [np.cos(th) * np.cos(phi), np.cos(th) * np.sin(phi), -np.sin(th)],
        [-np.sin(phi), np.cos(phi), 0]
    ])
    b = np.array([vx, vy, vz]).T
    return np.matmul(A, b)
def project_sph2car_np(r, th, phi, vr, vth, vphi):
    A = np.array([
        [np.sin(th) * np.cos(phi), np.sin(th) * np.sin(phi), np.cos(th)],
        [np.cos(th) * np.cos(phi), np.cos(th) * np.sin(phi), -np.sin(th)],
        [-np.sin(phi), np.cos(phi), 0]
    ])
    b = np.array([vr, vth, vphi]).T
    return np.matmul(A.T, b)
def project_sph2car(r, th, phi, vr, vth, vphi):
    return (sin(th) * cos(phi) * vr + cos(th) * cos(phi) * vth - sin(phi) * vphi,
        sin(th) * sin(phi) * vr + cos(th) * sin(phi) * vth + cos(phi) * vphi,
        cos(th) * vr - sin(th) * vth)

def coord_car_rz(x, y, z, dphi):
     return (cos(dphi) * x - sin(dphi) * y,
        sin(dphi) * x + cos(dphi) * y,
        z)


def surroundidx(array, x0):
    xi = np.abs(array - x0).argmin()
    if x0 < array[xi]:
        return (xi-1, xi)
    else:
        return (xi, xi+1)

def nearestidx_sph_np(field_r, field_theta, field_phi, r0, theta0, phi0):
    #ti = np.abs(field_time - t0).argmin()
    # a = abs(field_r-r0).argmin()
    # b = abs(field_theta-theta0).argmin()
    # c = abs(field_phi - phi0).argmin()
    # return a,b,c
    ii = np.abs(field_r-r0).argmin()
    ji = np.abs(field_theta-theta0).argmin()
    ki = np.abs(field_phi - phi0).argmin()
    return ii, ji, ki

def nearestidx_sph(field_r, field_theta, field_phi, r0, theta0, phi0):
    ii = min(range(len(field_r)), key=lambda x: abs(field_r[x]-r0))# np.abs(field_r-r0).argmin()
    ji = min(range(len(field_theta)), key=lambda x: abs(field_theta[x]-theta0))# np.abs(field_theta-theta0).argmin()
    ki = min(range(len(field_phi)), key=lambda x: abs(field_phi[x]-phi0))#np.abs(field_phi - phi0).argmin()
    return ii, ji, ki

def solvefield_pulse(pulse, fpath_sol, t0_ts, dur, resolution):
    import IRBEM as ib
    import datetime
    
    #mf_MAG = ib.MagFields(options=[0,0,0,0,0], verbose=False, kext='T89', sysaxes=6, alpha=[90])
    #mf_GSE = ib.MagFields(options=[0,0,0,0,0], verbose=False, kext='T89', sysaxes=3, alpha=[90])
    coords = ib.Coords()

    
    #create a grid in the GSE frame:
    #coordinate resolution:
    nx, ny, nz, nt = resolution
    #coordinate axes:
    xlim = 8
    x = np.linspace(-xlim, xlim, nx)
    y = np.linspace(-xlim, xlim, ny)
    z = np.linspace(-xlim, xlim, nz)
    time = np.linspace(0, dur, nt)

    #coordinate grids:
    xx, yy, zz = np.meshgrid(x, y, z, sparse=False, indexing='ij')
    assert np.all(xx[:,0,0] == x)
    assert np.all(yy[0,:,0] == y)
    assert np.all(zz[0,0,:] == z)
    #finite difference elements:
    dt = time[1] - time[0]
    # dxx = xx[1, 0, 0] - xx[0, 0, 0]
    # dyy = yy[0, 1, 0] - yy[0, 0, 0]
    # dzz = zz[0, 0, 1] - zz[0, 0, 0]
    #solution grids:
    # electric field:
    sol_Ex = np.zeros((nt, nx, ny, nz))
    sol_Ey = np.zeros((nt, nx, ny, nz))
    sol_Ez = np.zeros((nt, nx, ny, nz))
    # background perturbation in B:
    sol_Bwx = np.zeros((nt, nx, ny, nz))
    sol_Bwy = np.zeros((nt, nx, ny, nz))
    sol_Bwz = np.zeros((nt, nx, ny, nz))


    print("memory/storeage required for field (mb) > 6 x {:.2f}mb".format(sol_Bwx.nbytes / 1024 / 1024))
    print("E0 = ", pulse.E0, "V/m")
    #solution storage on disk:
    file_exists = exists(fpath_sol)
    disk = field_h5.HDF5_field(fpath_sol, existing = file_exists, delete = False)
    disk.add_dataset(disk.group_name_data, "t0", t0_ts)

    #solve time evolution:
    #bfield = pt_tools.dipolefield(pt_tools.constants.RE, 2015)
    for t in range(0, nt):
        tn = time[t]
        t_datetime = datetime.datetime.utcfromtimestamp(tn + t0_ts)
        print("", "solving", t_datetime)

        #calculate the rotation matrix from GSE to MAG at this time:
        rot_GSE_to_MAG = coords.transform([t_datetime, t_datetime, t_datetime], [[1,0,0], [0,1,0], [0,0,1]], 'GSE', 'MAG').T
        
        for i in range(nx):
            x_MAG = x[i] * constants.RE
            for j in range(ny):
                y_MAG = y[j] * constants.RE
                for k in range(nz):
                    z_MAG = z[k] * constants.RE

                    #convert from MAG to GSE:
                    x_, y_, z_ = coords.transform([t_datetime], [x_MAG, y_MAG, z_MAG], 'MAG', 'GSE')[0]

                    r_, th, phi = coord_car2sph(x_, y_, z_) #GSE frame
                    #print(x[i], y[j], z[k], r_/constants.RE, th*180/pi, phi*180/pi)

                    #solve pulse equation for electric field components Ephi
                    Ephi_, dEphidr = pulse.Ephi_dEphidr(tn, r_, phi)

                    #convert e field to cartesian frame, GSE:
                    Ex, Ey, Ez = project_sph2car(r_, th, phi, 0, 0, Ephi_) #+ve for r

                    #rotate this vector back into MAG frame:
                    Ex_MAG, Ey_MAG, Ez_MAG = np.matmul(rot_GSE_to_MAG, np.array([Ex, Ey, Ez]))

                    sol_Ex[t][i][j][k] = Ex_MAG
                    sol_Ey[t][i][j][k] = Ey_MAG
                    sol_Ez[t][i][j][k] = Ez_MAG



                    #solving Faraday's law for the magnetic field components Br and Btheta
                    dbwr = -dt*(Ephi_/(r_ * tan(th)))
                    dbwt =  dt*(Ephi_/r_ + dEphidr)
                    dbwp = 0.

                    #convert field perturbation to cartesian frame:
                    dBwx, dBwy, dBwz = project_sph2car(r_, th, phi, dbwr, dbwt, dbwp)

                    #rotate this vector back into MAG frame:
                    dBwx_MAG, dBwy_MAG, dBwz_MAG = np.matmul(rot_GSE_to_MAG, np.array([dBwx, dBwy, dBwz]))

                    if t == nt -1: continue
                    sol_Bwx[t+1][i][j][k] = sol_Bwx[t][i][j][k] + dBwx_MAG
                    sol_Bwy[t+1][i][j][k] = sol_Bwy[t][i][j][k] + dBwy_MAG
                    sol_Bwz[t+1][i][j][k] = sol_Bwz[t][i][j][k] + dBwz_MAG

    print("storing fields...")
    #store axes:
    disk.add_dataset(disk.group_name_data, "x", x*constants.RE)
    disk.add_dataset(disk.group_name_data, "y", y*constants.RE)
    disk.add_dataset(disk.group_name_data, "z", z*constants.RE)
    disk.add_dataset(disk.group_name_data, "time", time)
    #store solutions:
    disk.add_dataset(disk.group_name_data, "Ex", sol_Ex)
    disk.add_dataset(disk.group_name_data, "Ey", sol_Ey)
    disk.add_dataset(disk.group_name_data, "Ez", sol_Ez)
    disk.add_dataset(disk.group_name_data, "Bwx", sol_Bwx)
    disk.add_dataset(disk.group_name_data, "Bwy", sol_Bwy)
    disk.add_dataset(disk.group_name_data, "Bwz", sol_Bwz)


def study_march91(fpath_sol, redo = True):
    #redo = True #restart and overwrite the field solution
    file_exists = exists(fpath_sol)

    # instantiate the pulse:
    march91pulse = Epulse(240e-3, 0.8, 0.8, 8.0, 2000e3, 80, np.pi / 4, 30000e3)
    t0_ts = 669786080.0 # corresponds to beginning of time axis in figure 1, Li et al., 1993
    #Ephimax, *_ = np.abs(march91pulse.Ephi_dEphidr(0, 25 * constants.RE, pi / 8))  # get the maximum amplitude of the pulse


    if (not file_exists) or redo:
        print("solving field...")
        resolution = (100, 100, 100, 50)
        #resolution = (30, 30, 30, 30)
        solvefield_pulse(march91pulse, fpath_sol, t0_ts, 180, resolution)
        print("", "done")





# def solvefield_pulse_sph(pulse, fpath_sol, dur = 160, resolution = [18, 11, 12, 20]):
#     #coordinate resolution:
#     nr, ntheta, nphi, nt = resolution
#     #coordinate axes:
#     r = np.linspace(1, 9, nr)
#     theta = np.linspace(0, pi, ntheta+2)[1:-1] #we don't want 0 or 180 degrees, it's at the poles
#     #phi = np.linspace(0, 2*pi, nphi)
#     phi = np.linspace(0, 2 * pi, nphi + 1)[:-1]
#     time = np.linspace(0, dur, nt)

#     #coordinate grids:
#     rr, tt, pp = np.meshgrid(r, theta, phi, sparse=False, indexing='ij')
#     assert np.all(rr[:,0,0] == r)
#     assert np.all(tt[0,:,0] == theta)
#     assert np.all(pp[0,0,:] == phi)
#     #finite difference elements:
#     dt = time[1] - time[0]
#     drr = rr[1, 0, 0] - rr[0, 0, 0]
#     dtt = tt[0, 1, 0] - tt[0, 0, 0]
#     dpp = pp[0, 0, 1] - pp[0, 0, 0]
#     #solution grids:
#     sol_Ephi = np.zeros((nt, nr, ntheta, nphi)) #2D, no theta dependence
#     sol_dEphidr = np.zeros((nt, nr, ntheta, nphi))
#     sol_dEphidth = np.zeros((nt, nr, ntheta, nphi)) #2D, no theta dependence
#     sol_dEphidphi = np.zeros((nt, nr, ntheta, nphi))
#     sol_Br = np.zeros((nt, nr, ntheta, nphi))
#     sol_Btheta = np.zeros((nt, nr, ntheta, nphi))
#     sol_Bphi = np.zeros((nt, nr, ntheta, nphi))
#     sol_dBrdr = np.zeros((nt, nr, ntheta, nphi)) #d Br / d r
#     sol_dBrdth = np.zeros((nt, nr, ntheta, nphi))  # d Br / d theta
#     sol_dBrdphi = np.zeros((nt, nr, ntheta, nphi))  # d Br / d phi
#     sol_dBthetadr = np.zeros((nt, nr, ntheta, nphi))  # d Btheta / d r
#     sol_dBthetadth = np.zeros((nt, nr, ntheta, nphi)) #d Btheta / d theta
#     sol_dBthetadphi = np.zeros((nt, nr, ntheta, nphi)) #d Btheta / d phi
#     sol_dBphidr = np.zeros((nt, nr, ntheta, nphi))  # d Bphi / d r
#     sol_dBphidth = np.zeros((nt, nr, ntheta, nphi)) #d Bphi / d theta
#     sol_dBphidphi = np.zeros((nt, nr, ntheta, nphi)) #d Bphi / d phi
#     print("memory/storeage required for field (mb) > 16 x {:.2f}mb".format(sol_Br.nbytes / 1024 / 1024))
#     #solution storage on disk:
#     file_exists = exists(fpath_sol)
#     disk = field_h5.HDF5_field(fpath_sol, existing = file_exists, delete = file_exists)

#     #march91pulse = field_tools.Epulse(240e-3, 0.8, 0.8, 8.0, 2000e3, 80, pi / 8, 30000e3)
#     #Ephimax, *_ = np.abs(march91pulse.Ephi_dEphidr(0, 25 * field_tools.constants.RE, pi / 8))  # get the maximum amplitude of the pulse


#     #solve time evolution:
#     #bfield = pt_tools.dipolefield(pt_tools.constants.RE, 2015)
#     for t in range(0, nt):
#         tn = time[t]
#         for i in range(nr):
#             r_ = rr[i, 0, 0] * constants.RE
#             #r_ = r[i] * constants.RE
#             for k in range(nphi):
#                 Ephi_, dEphidr = pulse.Ephi_dEphidr(tn, r_, pp[i,0,k])
#                 sol_Ephi[t, i, :, k] = Ephi_
#                 sol_dEphidr[t, i, :, k] = dEphidr

#                 #solving Faraday's law for the magnetic field components Br and Btheta
#                 # finite difference:
#                 for j in range(ntheta):

#                     #br, btheta = bfield.getBsph(r_, tt[i,j,k])

#                     bwr = sol_Br[t][i][j][k] - dt*(Ephi_/(r_ * tan(tt[0,j,0])))
#                     bwt = sol_Btheta[t][i][j][k] + dt*(Ephi_/r_ + dEphidr)


#                     sol_Br[t][i][j][k] = bwr
#                     sol_Btheta[t][i][j][k] = bwt
#                     sol_Bphi[t][i][j][k] = 0

#         #pre-compute differential elements at this timestep:
#         for i in range(1, nr-1):
#             for j in range(1, ntheta - 1):
#                 for k in range(nphi):
#                     #sol_dEphidr[t][i][j][k] = 0.5*(sol_Ephi[t][i+1][j][k] - sol_Ephi[t][i-1][j][k])/drr   # d Ephi / d r
#                     sol_dEphidth[t][i][j][k] = 0.5*(sol_Ephi[t][i][j+1][k] - sol_Ephi[t][i][j-1][k])/dtt  # d Ephi / d theta
#                     sol_dEphidphi[t][i][j][k] = 0.5*(sol_Ephi[t][i][j][(k+1)%nphi] - sol_Ephi[t][i][j][(k-1)%nphi])/dpp  # d Ephi / d phi
#                     sol_dBrdr[t][i][j][k] = 0.5*(sol_Br[t][i+1][j][k] - sol_Br[t][i-1][j][k])/drr   # d Br / d r
#                     sol_dBrdth[t][i][j][k] = 0.5*(sol_Br[t][i][j+1][k] - sol_Br[t][i][j-1][k])/dtt  # d Br / d theta
#                     sol_dBrdphi[t][i][j][k] = 0.5*(sol_Br[t][i][j][(k+1)%nphi] - sol_Br[t][i][j][(k-1)%nphi])/dpp  # d Br / d phi
#                     sol_dBthetadr[t][i][j][k] = 0.5*(sol_Btheta[t][i+1][j][k] - sol_Btheta[t][i-1][j][k])/drr  # d Btheta / d r
#                     sol_dBthetadth[t][i][j][k] = 0.5*(sol_Btheta[t][i][j+1][k] - sol_Btheta[t][i][j-1][k])/dtt   # d Btheta / d theta
#                     sol_dBthetadphi[t][i][j][k] = 0.5*(sol_Btheta[t][i][j][(k+1)%nphi] - sol_Btheta[t][i][j][(k-1)%nphi])/dpp  # d Btheta / d phi
#                     sol_dBphidr[t][i][j][k] = 0.5*(sol_Bphi[t][i+1][j][k] - sol_Bphi[t][i-1][j][k])/drr  # d Bphi / d r
#                     sol_dBphidth[t][i][j][k] = 0.5*(sol_Bphi[t][i][j+1][k] - sol_Bphi[t][i][j-1][k])/dtt   # d Bphi / d theta
#                     sol_dBphidphi[t][i][j][k] = 0.5*(sol_Bphi[t][i][j][(k+1)%nphi] - sol_Bphi[t][i][j][(k-1)%nphi])/dpp  # d Bphi / d phi
#         #remaining differences in r direction:
#         for j in range(ntheta):
#             for k in range(nphi):
#                 i = 0
#                 #sol_dEphidr[t][i][j][k] = 0.5*(sol_Ephi[t][i + 1][j][k] - sol_Ephi[t][i][j][k])/drr   # d Ephi / d r
#                 sol_dBrdr[t][i][j][k] = (sol_Br[t][i + 1][j][k] - sol_Br[t][i][j][k]) / drr  # d Br / d r
#                 sol_dBthetadr[t][i][j][k] = (sol_Btheta[t][i + 1][j][k] - sol_Btheta[t][i][j][k]) / drr  # d Btheta / d r
#                 sol_dBphidr[t][i][j][k] = (sol_Bphi[t][i + 1][j][k] - sol_Bphi[t][i][j][k]) / drr  # d Bphi / d r
#                 i = nr - 1
#                 #sol_dEphidr[t][i][j][k] = 0.5*(sol_Ephi[t][i][j][k] - sol_Ephi[t][i - 1][j][k])/drr   # d Ephi / d r
#                 sol_dBrdr[t][i][j][k] = (sol_Br[t][i][j][k] - sol_Br[t][i - 1][j][k]) / drr  # d Br / d r
#                 sol_dBthetadr[t][i][j][k] = (sol_Btheta[t][i][j][k] - sol_Btheta[t][i - 1][j][k]) / drr  # d Btheta / d r
#                 sol_dBphidr[t][i][j][k] = (sol_Bphi[t][i][j][k] - sol_Bphi[t][i - 1][j][k]) / drr  # d Bphi / d r
#         #remaining differences in theta direction:
#         for i in range(nr):
#             for k in range(nphi):
#                 j = 0
#                 sol_dEphidth[t][i][j][k] = (sol_Ephi[t][i][j + 1][k] - sol_Ephi[t][i][j][k]) / dtt  # d Ephi / d theta
#                 sol_dBrdth[t][i][j][k] = (sol_Br[t][i][j + 1][k] - sol_Br[t][i][j][k]) / dtt  # d Br / d theta
#                 sol_dBthetadth[t][i][j][k] = (sol_Btheta[t][i][j + 1][k] - sol_Btheta[t][i][j][k]) / dtt  # d Btheta / d theta
#                 sol_dBphidth[t][i][j][k] = (sol_Bphi[t][i][j + 1][k] - sol_Bphi[t][i][j][k]) / dpp  # d Btheta / d phi
#                 j = ntheta - 1
#                 sol_dEphidth[t][i][j][k] = (sol_Ephi[t][i][j][k] - sol_Ephi[t][i][j][k]) / dtt  # d Ephi / d theta
#                 sol_dBrdth[t][i][j][k] = (sol_Br[t][i][j][k] - sol_Br[t][i][j - 1][k]) / dtt  # d Br / d theta
#                 sol_dBthetadth[t][i][j][k] = (sol_Btheta[t][i][j][k] - sol_Btheta[t][i][j - 1][k]) / dtt  # d Btheta / d theta
#                 sol_dBphidth[t][i][j][k] = (sol_Bphi[t][i][j][k] - sol_Bphi[t][i][j - 1][k]) / dpp  # d Btheta / d phi

#     print("storing fields...")
#     #store axes:
#     disk.add_dataset(disk.group_name_data, "r", r)
#     disk.add_dataset(disk.group_name_data, "theta", theta)
#     disk.add_dataset(disk.group_name_data, "phi", phi)
#     disk.add_dataset(disk.group_name_data, "time", time)
#     #store solutions:
#     disk.add_dataset(disk.group_name_data, "Ephi", sol_Ephi)
#     disk.add_dataset(disk.group_name_data, "dEphidr", sol_dEphidr)
#     disk.add_dataset(disk.group_name_data, "dEphidth", sol_dEphidth)
#     disk.add_dataset(disk.group_name_data, "dEphidphi", sol_dEphidphi)
#     disk.add_dataset(disk.group_name_data, "Br", sol_Br)
#     disk.add_dataset(disk.group_name_data, "Btheta", sol_Btheta)
#     disk.add_dataset(disk.group_name_data, "Bphi", sol_Bphi)
#     disk.add_dataset(disk.group_name_data, "dBrdr", sol_dBrdr)
#     disk.add_dataset(disk.group_name_data, "dBrdth", sol_dBrdth)
#     disk.add_dataset(disk.group_name_data, "dBrdphi", sol_dBrdphi)
#     disk.add_dataset(disk.group_name_data, "dBthetadr", sol_dBthetadr)
#     disk.add_dataset(disk.group_name_data, "dBthetadth", sol_dBthetadth)
#     disk.add_dataset(disk.group_name_data, "dBthetadphi", sol_dBthetadphi)
#     disk.add_dataset(disk.group_name_data, "dBphidr", sol_dBphidr)
#     disk.add_dataset(disk.group_name_data, "dBphidth", sol_dBphidth)
#     disk.add_dataset(disk.group_name_data, "dBphidphi", sol_dBphidphi)
#     #disk.print_file_tree()


# fpath_sol = "./simulation.h5"
# file_exists = exists(fpath_sol)
# if not file_exists:
#     # instantiate the pulse:
#     march91pulse = perturbB.Epulse(240e-3, 0.8, 0.8, 8.0, 2000e3, 80, pi/8, 30000e3)
#     Ephimax = np.abs(march91pulse.Ephi(0, 25, pi/8)) #get the maximum amplitude of the pulse
#     perturbB.solvefield_pulse(march91pulse, Ephimax, fpath_sol)


# disk = storeh5.HDF5_file(fpath_sol, existing = True)
# field_r = disk.read_dataset(disk.group_name_data, "r")
# field_theta = disk.read_dataset(disk.group_name_data, "theta")
# field_phi = disk.read_dataset(disk.group_name_data, "phi")
# field_time = disk.read_dataset(disk.group_name_data, "time")
# field_Ephi = disk.read_dataset(disk.group_name_data, "Ephi")
# field_Br = disk.read_dataset(disk.group_name_data, "Br")
# field_Btheta = disk.read_dataset(disk.group_name_data, "Btheta")
# field_dBrdr = disk.read_dataset(disk.group_name_data, "dBrdr")
# field_dBrdth = disk.read_dataset(disk.group_name_data, "dBrdth")
# field_dBrdphi = disk.read_dataset(disk.group_name_data, "dBrdphi")
# field_dBthetadr = disk.read_dataset(disk.group_name_data, "dBthetadr")
# field_dBthetadth = disk.read_dataset(disk.group_name_data, "dBthetadth")
# field_dBthetadphi = disk.read_dataset(disk.group_name_data, "dBthetadphi")