import field_h5
field_h5.verbose = False
from os.path import exists
import numpy as np
import IGRF_tools
from datetime import datetime, timezone
from pt_tools import constants, dt_to_dec

# we work in the MAG frame because:
#  we can transform between GEO and MAG using only three IGRF parameters (no IRBEM dependence, etc.)
#  the convenient transformation between MAG and GEO allows us to include models of Earth's surface
#  the transformation between the MAG frame and an offset, eccentric dipole frame is a simple translation, to which vectors are invariant

verbose = True
def v_print(*a, **b):
    """Thread safe print function"""

    if not verbose:
        return
    print(*a, **b)

pi = np.pi
import sys
from math import cos, sin, tan, acos, asin, atan, atan2, sqrt, pi, floor


def get_rotation_GEO_to_MAG(year_dec):
    """
    year_dec : date as a decimal of year, i.e. 2015.25

    returns rotation matrix from GEO to MAG frame

    from the Spenvis help page (https://www.spenvis.oma.be/help/background/coortran/coortran.html),
    the equation to solve is:
        T_5 = <phi - 90d, Y> * <lambda, Z>

    <lambda, Z> is a rotation in the plane of the Earth's equator form the Greenwich meridian to the meridian containing the dipole pole
    <phi - 90d, Y> is a rotation in that meridian from the geographic pole to the dipole pole
    """
    g, h = IGRF_tools.arrange_IGRF_coeffs(year_dec)
    #B0_2 = g[1][0] ** 2 + g[1][1] ** 2 + h[1][1] ** 2

    lmbda = atan(h[1][1]/g[1][1])
    phi = np.pi/2 - atan((g[1][1]*cos(lmbda) + h[1][1]*sin(lmbda)) / g[1][0])

    R_mer = np.array([[cos(lmbda), -1*sin(lmbda), 0],[sin(lmbda), cos(lmbda), 0], [0,0,1]])
    R_pole = np.array([[cos(phi-np.pi/2), 0, sin(phi-np.pi/2)],
                       [0, 1, 0],
                       [-1 * sin(phi-np.pi/2), 0, cos(phi-np.pi/2)]])
    T5 = R_pole @ (R_mer @ np.identity(3)).T
    return T5 #validated using IRBEM

    ###validation using IRBEM for year_dec = 2015.0:
    # import datetime
    # from datetime import timezone
    # import IRBEM as ib
    # t_datetime = datetime.datetime(year=2015, month=1, day=1, tzinfo=timezone.utc)
    # coords = ib.Coords()
    # rot_GEO_to_MAG = coords.transform([t_datetime, t_datetime, t_datetime], [[1, 0, 0], [0, 1, 0], [0, 0, 1]], 'GEO', 'MAG').T
    # print(rot_GEO_to_MAG)

def get_eccentric_centre_GEO(year_dec):
    """return vector from origin to eccentric dipole centre in GEO frame [m] """
    g, h = IGRF_tools.arrange_IGRF_coeffs(year_dec)

    B0_2 = g[1][0] ** 2 + g[1][1] ** 2 + h[1][1] ** 2
    B0_nT = sqrt(B0_2)
    #B0_ = B0_ * constants.nT2T

    L0 = 2*g[1][0]*g[2][0] + sqrt(3)*(g[1][1]*g[2][1] + h[1][1]*h[2][1])
    L1 = -g[1][1]*g[2][0] + sqrt(3)*(g[1][0]*g[2][1] + g[1][1]*g[2][2] + h[1][1]*h[2][2])
    L2 = -h[1][1]*g[2][0] + sqrt(3)*(g[1][0]*h[2][1] - h[1][1]*g[2][2] + g[1][1]*h[2][2])
    E = (L0 * g[1][0] + L1*g[1][1] + L2*h[1][1]) / (4*((B0_nT)**2))
    xi =  (L0 - g[1][0]*E)/(3*((B0_nT)**2))
    eta = (L1 - g[1][1]*E)/(3*((B0_nT)**2))
    zeta =(L2 - h[1][1]*E)/(3*((B0_nT)**2))

    # print(L0)
    # print(L1)
    # print(L2)
    # print(E)
    # print(eta)
    # print(zeta)
    # print(xi)
    #validated against Spenvis values for IGRF2000: https://www.spenvis.oma.be/help/background/magfield/cd.html
    return constants.RE * np.array([eta, zeta, xi]) #meters

def get_eccentric_centre_MAG(year_dec):
    x_ed_GEO = get_eccentric_centre_GEO(year_dec)
    #MAG frame is rotated from GEO:
    M_GEO_to_MAG = get_rotation_GEO_to_MAG(year_dec)
    x_ed_MAG = M_GEO_to_MAG @ x_ed_GEO
    return x_ed_MAG

class Dipolefield:
    """
    this class describes an eccentric, tilted dipole
    an eccentric, tilted dipole is offset from the MAG frame, so we must translate the input coordinates of some functions that use dipole equations
    """
    def __init__(self, year_dec):
        self.year_dec = year_dec
        self.B0, self.M = self.get_B0_m(year_dec)
        self.field_time = [0]
        self.origin_MAG = get_eccentric_centre_MAG(year_dec)
        self.B_grid = False

    def get_dipolelc(self, Lb, atm_height):
        RE = constants.RE
        ra = (RE + atm_height) / RE  # ~Earth's surface + atm_height_dipolelc m

        if ra >= Lb:
            return np.nan
        else:
            Ba = (self.B0 / (ra ** 3)) * (4 - 3 * ra / Lb) ** (0.5)
            dipole_lc = asin(sqrt((self.B0 / Lb ** 3) / Ba)) * 180 / pi
            return dipole_lc

    def getB_dipole(self, xh_MAG, yh_MAG, zh_MAG):
        """
        input: coordinates in m
        """
        xh = xh_MAG - self.origin_MAG[0]
        yh = yh_MAG - self.origin_MAG[1]
        zh = zh_MAG - self.origin_MAG[2]

        Mdir_x = 0
        Mdir_y = 0
        Mdir_z = -1

        r = sqrt(pow(xh, 2) + pow(yh, 2) + pow(zh, 2))
        C1 = 1e-7 * self.M / (r ** 3)
        mr = Mdir_x * xh + Mdir_y * yh + Mdir_z * zh
        bx = C1 * (3 * xh * mr / (r ** 2) - Mdir_x)
        by = C1 * (3 * yh * mr / (r ** 2) - Mdir_y)
        bz = C1 * (3 * zh * mr / (r ** 2) - Mdir_z)

        return bx, by, bz

    def getBE(self, xh_MAG, yh_MAG, zh_MAG, t=0):
        """
        input: coordinates in m
        """
        bx, by, bz = self.getB_dipole(xh_MAG, yh_MAG, zh_MAG)
        return bx, by, bz, 0, 0, 0

    def get_L(self, x1_MAG):
        """
        takes distance r (Earth radii) and magnetic latitude (radians)
        returns dipole L
        """
        x1 = x1_MAG - self.origin_MAG
        r_ = np.linalg.norm(x1) / constants.RE
        mag_lat = atan2(x1[2], sqrt(x1[0] ** 2 + x1[1] ** 2))
        return r_ / (cos(mag_lat) ** 2)

    def get_B0_m(self, year):
        """Get the average dipole field strength around Earth's equator and dipole moment. Use like so: B0,m = get_B0_m(2000.0)"""
        g, h = IGRF_tools.arrange_IGRF_coeffs(year)

        B0_2 = g[1][0] ** 2 + g[1][1] ** 2 + h[1][1] ** 2
        B0_ = sqrt(B0_2)
        B0_ = B0_ * constants.nT2T
        M_ = B0_ * (constants.RE ** 3) * 4 * pi / constants.mu0

        return B0_, M_

    def find_magequator_z(self, xs, ys, zs, ti):
        #return z component of offset, eccentric dipole offset vector in the MAG frame
        return self.origin_MAG[2]

    def get_aclockw_angle_around_dipole_z(self, x1_MAG):
        """
        get anticlockwise angle of x1 around z axis from [1, 0]
        """
        x1 = x1_MAG - self.origin_MAG
        return (np.angle(x1[0] + x1[1] * 1j, deg=True) + 360) % 360

    def calculate_initial_GC(self, init_L, iphase_drift):
        """
        get initial position of the GC in the MAG frame
        """
        R_gc_RE = init_L
        R_gc = R_gc_RE * constants.RE

        xgc, ygc, zgc = coord_car_rz(R_gc, 0, 0, np.radians(iphase_drift))

        x0_GC = np.array([xgc, ygc, zgc])
        x0_GC_MAG = x0_GC + self.origin_MAG
        #what should be done here in the case of an arbitrary field?
        # the particle GC should be at the point of minimum B because we can prescribe equatorial pitch angle here
        return x0_GC_MAG

class Dipolefield_With_Perturbation(Dipolefield):
    def __init__(self, fileload, reversetime=-1):
        # load the HDF5 file
        v_print("Loading E, B field perturbations from", fileload)

        disk = field_h5.HDF5_field(fileload, existing=True)

        t0_ts = disk.read_dataset(disk.group_name_data, "t0")
        t0 = datetime.fromtimestamp(t0_ts, tz=timezone.utc)
        year_dec = dt_to_dec(t0)
        super().__init__(year_dec)
        self.t0 = t0

        self.pert_time = disk.read_dataset(disk.group_name_data, "time")
        self.pert_dt = self.pert_time[1] - self.pert_time[0]
        self.pert_t_min = self.pert_time[0]
        # self.pert_t_max = self.pert_time[-1]

        self.pert_x = disk.read_dataset(disk.group_name_data, "x")
        self.pert_dx = self.pert_x[1] - self.pert_x[0]
        self.pert_x_min = self.pert_x[0]
        # self.pert_x_max = self.pert_x[-1]

        self.pert_y = disk.read_dataset(disk.group_name_data, "y")
        self.pert_dy = self.pert_y[1] - self.pert_y[0]
        self.pert_y_min = self.pert_y[0]
        # self.pert_y_max = self.pert_y[-1]

        self.pert_z = disk.read_dataset(disk.group_name_data, "z")
        self.pert_dz = self.pert_z[1] - self.pert_z[0]
        self.pert_z_min = self.pert_z[0]
        # self.pert_z_max = self.pert_z[-1]

        # store solutions:
        nt = np.size(self.pert_time)
        nx = np.size(self.pert_x)
        ny = np.size(self.pert_y)
        nz = np.size(self.pert_z)
        self.pert_BE = np.zeros((6, nt, nx, ny, nz))
        self.pert_BE[0, :, :, :, :] = disk.read_dataset(disk.group_name_data, "Bwx")
        self.pert_BE[1, :, :, :, :] = disk.read_dataset(disk.group_name_data, "Bwy")
        self.pert_BE[2, :, :, :, :] = disk.read_dataset(disk.group_name_data, "Bwz")
        self.pert_BE[3, :, :, :, :] = disk.read_dataset(disk.group_name_data, "Ex")
        self.pert_BE[4, :, :, :, :] = disk.read_dataset(disk.group_name_data, "Ey")
        self.pert_BE[5, :, :, :, :] = disk.read_dataset(disk.group_name_data, "Ez")

        if reversetime > 0:
            # modify calls to int_field so that time becomes reversetime - ti:
            self.reversetime = reversetime
            self.tmult = -1
        else:
            self.reversetime = 0
            self.tmult = 1

        v_print("", "done\n")

        self.range_adequate = True

    def int_field(self, xi, yi, zi, ti):
        ti = self.reversetime + self.tmult * ti
        # if reversed, time evolution goes backwards from self.reversetime

        # global R_e dg dx dy dz xmint ymint zmint
        dx = self.pert_dx
        dy = self.pert_dy
        dz = self.pert_dz
        dt = self.pert_dt

        pxe0 = floor((xi - self.pert_x_min) / dx)
        pye0 = floor((yi - self.pert_y_min) / dy)
        pze0 = floor((zi - self.pert_z_min) / dz)
        pte0 = floor((ti - self.pert_t_min) / dt)

        if pxe0 < 0:
            self.range_adequate = False
            return [0, 0, 0, 0, 0, 0]
        if pye0 < 0:
            self.range_adequate = False
            return [0, 0, 0, 0, 0, 0]
        if pze0 < 0:
            self.range_adequate = False
            return [0, 0, 0, 0, 0, 0]
        if pxe0 > len(self.pert_x) - 2:
            self.range_adequate = False
            return [0, 0, 0, 0, 0, 0]
        if pye0 > len(self.pert_y) - 2:
            self.range_adequate = False
            return [0, 0, 0, 0, 0, 0]
        if pze0 > len(self.pert_z) - 2:
            self.range_adequate = False
            return [0, 0, 0, 0, 0, 0]

        xfac = (xi - self.pert_x_min - (pxe0) * dx) / dx;
        yfac = (yi - self.pert_y_min - (pye0) * dy) / dy;
        zfac = (zi - self.pert_z_min - (pze0) * dz) / dz;
        tfac = (ti - self.pert_t_min - (pte0) * dt) / dt;

        # check:
        # print(self.pert_x[pxe0]/constants.RE, xi/constants.RE, self.pert_x[pxe0+1]/constants.RE, xfac)
        # print(self.pert_y[pye0]/constants.RE, yi/constants.RE, self.pert_y[pye0+1]/constants.RE, yfac)
        # print(self.pert_z[pze0]/constants.RE, zi/constants.RE, self.pert_z[pze0+1]/constants.RE, zfac)
        # print(self.pert_time[pte0], ti, self.pert_time[pte0+1], tfac)
        # print()

        ns = [0, 0, 0, 0, 0, 0, 0, 0]
        interp_vals = [0, 0, 0, 0, 0, 0]
        t_idxs = [pte0, pte0 + 1]
        t_facs = [1 - tfac, tfac]

        for idxt in range(2):
            pte = t_idxs[idxt]
            time_fac = t_facs[idxt]
            for idx in range(6):
                ns[0] = self.pert_BE[idx, pte, pxe0, pye0, pze0]
                ns[1] = self.pert_BE[idx, pte, pxe0 + 1, pye0, pze0]
                ns[2] = self.pert_BE[idx, pte, pxe0, pye0 + 1, pze0]
                ns[3] = self.pert_BE[idx, pte, pxe0 + 1, pye0 + 1, pze0]
                ns[4] = self.pert_BE[idx, pte, pxe0, pye0, pze0 + 1]
                ns[5] = self.pert_BE[idx, pte, pxe0 + 1, pye0, pze0 + 1]
                ns[6] = self.pert_BE[idx, pte, pxe0, pye0 + 1, pze0 + 1]
                ns[7] = self.pert_BE[idx, pte, pxe0 + 1, pye0 + 1, pze0 + 1]

                nsa = ns[0] + (ns[1] - ns[0]) * xfac;
                nsb = ns[2] + (ns[3] - ns[2]) * xfac;
                nsc = ns[4] + (ns[5] - ns[4]) * xfac;
                nsd = ns[6] + (ns[7] - ns[6]) * xfac;

                nsp = nsa + (nsb - nsa) * yfac;
                nsq = nsc + (nsd - nsc) * yfac;

                interp_val = nsp + (nsq - nsp) * zfac;

                interp_vals[idx] += interp_val * time_fac

        return interp_vals

    def getBE(self, xh_MAG, yh_MAG, zh_MAG, t=0):
        """
        input: coordinates in m
        """
        bx, by, bz = self.getB_dipole(xh_MAG, yh_MAG, zh_MAG)

        bwx0, bwy0, bwz0, qEx, qEy, qEz = self.int_field(xh_MAG, yh_MAG, zh_MAG, t)

        return bx + bwx0, by + bwy0, bz + bwz0, qEx, qEy, qEz

    # def getBsph_dipole(self, rh, thetah, t=0):
    #     """
    #     input: coordinates r [m], theta
    #     """
    #
    #     br = -2 * self.B0 * ((self.RE / rh) ** 3) * cos(thetah)
    #     btheta = -self.B0 * ((self.RE / rh) ** 3) * sin(thetah)
    #
    #     return br, btheta

class Customfield(Dipolefield):
    def __init__(self, fileload, reversetime=-1):
        # load the HDF5 file
        v_print("Loading B field from", fileload)

        disk = field_h5.HDF5_field(fileload, existing=True)

        t0_ts = disk.read_dataset(disk.group_name_data, "t0")
        t0 = datetime.fromtimestamp(t0_ts, tz=timezone.utc)
        year_dec = dt_to_dec(t0)
        super().__init__(year_dec)  # defines B0, M
        self.B_grid = True
        #self.origin_MAG = np.array([0, 0, 0])
        self.t0 = t0

        self.field_time = disk.read_dataset(disk.group_name_data, "time")
        self.field_dt = self.field_time[1] - self.field_time[0]
        self.field_t_min = self.field_time[0]

        self.field_x = disk.read_dataset(disk.group_name_data, "x")
        self.field_dx = self.field_x[1] - self.field_x[0]
        self.field_x_min = self.field_x[0]

        self.field_y = disk.read_dataset(disk.group_name_data, "y")
        self.field_dy = self.field_y[1] - self.field_y[0]
        self.field_y_min = self.field_y[0]

        self.field_z = disk.read_dataset(disk.group_name_data, "z")
        self.field_dz = self.field_z[1] - self.field_z[0]
        self.field_z_min = self.field_z[0]

        # store solutions:
        nt = np.size(self.field_time)
        nx = np.size(self.field_x)
        ny = np.size(self.field_y)
        nz = np.size(self.field_z)
        self.field_B = np.zeros((3, nt, nx, ny, nz))
        self.field_B[0, :, :, :, :] = disk.read_dataset(disk.group_name_data, "Bx")
        self.field_B[1, :, :, :, :] = disk.read_dataset(disk.group_name_data, "By")
        self.field_B[2, :, :, :, :] = disk.read_dataset(disk.group_name_data, "Bz")

        if reversetime > 0:
            # modify calls to int_field so that time becomes reversetime - ti:
            self.reversetime = reversetime
            self.tmult = -1
        else:
            self.reversetime = 0
            self.tmult = 1

        v_print("", "done\n")

        self.range_adequate = True

    def int_field(self, xi, yi, zi, ti):
        ti = self.reversetime + self.tmult * ti
        # if reversed, time evolution goes backwards from self.reversetime

        # global R_e dg dx dy dz xmint ymint zmint
        dx = self.field_dx
        dy = self.field_dy
        dz = self.field_dz
        dt = self.field_dt

        pxe0 = floor((xi - self.field_x_min) / dx)
        pye0 = floor((yi - self.field_y_min) / dy)
        pze0 = floor((zi - self.field_z_min) / dz)
        pte0 = floor((ti - self.field_t_min) / dt)

        if pxe0 < 0:
            self.range_adequate = False
            return [0, 0, 0, 0, 0, 0]
        if pye0 < 0:
            self.range_adequate = False
            return [0, 0, 0, 0, 0, 0]
        if pze0 < 0:
            self.range_adequate = False
            return [0, 0, 0, 0, 0, 0]
        if pte0 < 0:
            self.range_adequate = False
            return [0, 0, 0, 0, 0, 0]
        if pxe0 > len(self.field_x) - 2:
            self.range_adequate = False
            return [0, 0, 0, 0, 0, 0]
        if pye0 > len(self.field_y) - 2:
            self.range_adequate = False
            return [0, 0, 0, 0, 0, 0]
        if pze0 > len(self.field_z) - 2:
            self.range_adequate = False
            return [0, 0, 0, 0, 0, 0]
        if pte0 > len(self.field_time) - 2:
            self.range_adequate = False
            return [0, 0, 0, 0, 0, 0]

        xfac = (xi - self.field_x_min - (pxe0) * dx) / dx;
        yfac = (yi - self.field_y_min - (pye0) * dy) / dy;
        zfac = (zi - self.field_z_min - (pze0) * dz) / dz;
        tfac = (ti - self.field_t_min - (pte0) * dt) / dt;

        ns = [0, 0, 0, 0, 0, 0, 0, 0]
        interp_vals = [0, 0, 0]
        t_idxs = [pte0, pte0 + 1]
        t_facs = [1 - tfac, tfac]

        for idxt in range(2):
            pte = t_idxs[idxt]
            time_fac = t_facs[idxt]
            for idx in range(3):
                ns[0] = self.field_B[idx, pte, pxe0, pye0, pze0]
                ns[1] = self.field_B[idx, pte, pxe0 + 1, pye0, pze0]
                ns[2] = self.field_B[idx, pte, pxe0, pye0 + 1, pze0]
                ns[3] = self.field_B[idx, pte, pxe0 + 1, pye0 + 1, pze0]
                ns[4] = self.field_B[idx, pte, pxe0, pye0, pze0 + 1]
                ns[5] = self.field_B[idx, pte, pxe0 + 1, pye0, pze0 + 1]
                ns[6] = self.field_B[idx, pte, pxe0, pye0 + 1, pze0 + 1]
                ns[7] = self.field_B[idx, pte, pxe0 + 1, pye0 + 1, pze0 + 1]

                nsa = ns[0] + (ns[1] - ns[0]) * xfac;
                nsb = ns[2] + (ns[3] - ns[2]) * xfac;
                nsc = ns[4] + (ns[5] - ns[4]) * xfac;
                nsd = ns[6] + (ns[7] - ns[6]) * xfac;

                nsp = nsa + (nsb - nsa) * yfac;
                nsq = nsc + (nsd - nsc) * yfac;

                interp_val = nsp + (nsq - nsp) * zfac;

                interp_vals[idx] += interp_val * time_fac

        return interp_vals

    def getBE(self, xh_MAG, yh_MAG, zh_MAG, t=0):
        """
        input: coordinates in m
        """
        bx, by, bz = self.int_field(xh_MAG, yh_MAG, zh_MAG, t)
        return bx, by, bz, 0, 0, 0

    def find_magequator(self, xs, ys, zs, ti, trace_ds=0.75e-4 * constants.RE, level=0, direction=1):
        """
        xs, ys, zs is the starting point of the trace
        """
        max_level = 5
        max_R = 10 * sqrt(xs ** 2 + ys ** 2 + zs ** 2)
        xi = xs
        yi = ys
        zi = zs
        tried_reverse = False
        found_equator = False

        Bvec = self.int_field(xi, yi, zi, ti)
        absB = np.linalg.norm(Bvec)
        absB_min = absB
        while sqrt(xi ** 2 + yi ** 2 + zi ** 2) < max_R:
            xi += direction * trace_ds * Bvec[0] / absB
            yi += direction * trace_ds * Bvec[1] / absB
            zi += direction * trace_ds * Bvec[2] / absB

            Bvec = self.int_field(xi, yi, zi, ti)
            absB = np.linalg.norm(Bvec)
            if absB < absB_min:
                absB_min = absB
                tried_reverse = True #no need to try reversing if we are finding that field strength decreases in this direction
            elif not tried_reverse:
                # reset with negative step
                xi = xs
                yi = ys
                zi = zs
                direction = direction * -1
                tried_reverse = True
                #print("reversing...")
            else:
                xe = xi
                ye = yi
                ze = zi
                found_equator = True
                break

        if found_equator:
            return xe, ye, ze
        elif not found_equator and level < max_level:
            return self.find_magequator(xs, ys, zs, ti, trace_ds=trace_ds / 2, level=level + 1)
        elif not found_equator:
            print("error: could not find the magnetic equator via field line tracing")
            sys.exit()

    def find_magequator_z(self, xs, ys, zs, ti, trace_ds=0.75e-4 * constants.RE):
        _, _, ze = self.find_magequator(xs, ys, zs, ti, trace_ds=0.75e-4 * constants.RE)
        return ze


class Customfield_With_Perturbation(Dipolefield):
    def __init__(self, bgload, pertload, reversetime=-1):
        print("not implemented yet!")
        sys.exit()


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
    from datetime import timezone
    
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
    disk = field_h5.HDF5_field(fpath_sol, existing = file_exists, delete = True)
    disk.add_dataset(disk.group_name_data, "t0", t0_ts)

    #solve time evolution:
    for t in range(0, nt):
        tn = time[t]
        #t_datetime = datetime.datetime.utcfromtimestamp(tn + t0_ts)
        t_datetime = datetime.datetime.fromtimestamp(tn + t0_ts, tz=timezone.utc)
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

def produce_dipolefield_for_validation_of_customfield(dur=10000, redo = True):
    import IRBEM as ib
    import datetime
    from datetime import timezone
    output = "configs/dipolefield_verification.h5"
    file_exists = exists(output)

    #t0_ts = 669786080.0 # corresponds to beginning of time axis in figure 1, Li et al., 1993
    t0_ts = 1420070400.0 # corresponds to 2015

    if file_exists and (not redo):
        print("field is already solved, exiting...")
        sys.exit(1)

    resolution = (201, 201, 201, 2)

    #mf_MAG_cdip = ib.MagFields(options=[0,0,0,0,5], verbose=False, kext='None', sysaxes=6, alpha=[90])
    mf_MAG_offdip = ib.MagFields(options=[0,0,0,0,1], verbose=False, kext='None', sysaxes=6, alpha=[90])
    #mf_analytical = Dipolefield(pt_tools.dt_to_dec(datetime.datetime.fromtimestamp(t0_ts, tz=timezone.utc)))

    coords = ib.Coords()

    # create a grid in the GSE frame:
    # coordinate resolution:
    nx, ny, nz, nt = resolution
    # coordinate axes:
    xlim = 6
    x = np.linspace(-xlim, xlim, nx)
    y = np.linspace(-xlim, xlim, ny)
    z = np.linspace(-xlim, xlim, nz)

    time = np.linspace(0, dur, nt)

    # coordinate grids:
    xx, yy, zz = np.meshgrid(x, y, z, sparse=False, indexing='ij')
    assert np.all(xx[:, 0, 0] == x)
    assert np.all(yy[0, :, 0] == y)
    assert np.all(zz[0, 0, :] == z)
    # finite difference elements:
    dt = time[1] - time[0]
    # background perturbation in B:
    sol_Bx = np.zeros((nt, nx, ny, nz))
    sol_By = np.zeros((nt, nx, ny, nz))
    sol_Bz = np.zeros((nt, nx, ny, nz))

    print("memory/storeage required for field (mb) > 3 x {:.2f}mb".format(sol_Bx.nbytes / 1024 / 1024))

    # solution storage on disk:
    disk = field_h5.HDF5_field(output, existing=file_exists, delete=True)
    disk.add_dataset(disk.group_name_data, "t0", t0_ts)
    # solve time evolution:
    # bfield = dipolefield(constants.RE, 2015)
    for t in range(0, nt):
        tn = time[t]
        #t_datetime = datetime.datetime.utcfromtimestamp(tn + t0_ts)
        t_datetime = datetime.datetime.fromtimestamp(tn + t0_ts, tz=timezone.utc)
        print("", "solving", t_datetime)

        #calculate the rotation matrix from GEO to MAG at this time:
        rot_GEO_to_MAG = coords.transform([t_datetime, t_datetime, t_datetime], [[1, 0, 0], [0, 1, 0], [0, 0, 1]], 'GEO', 'MAG').T
        XYZ = {}
        for i in range(nx):
            #x_MAG = x[i] * constants.RE
            for j in range(ny):
                #y_MAG = y[j] * constants.RE
                for k in range(nz):
                    #z_MAG = z[k] * constants.RE
                    XYZ['x1'] = x[i]#x_MAG
                    XYZ['x2'] = y[j]#y_MAG
                    XYZ['x3'] = z[k]#z_MAG
                    XYZ['dateTime'] = t_datetime
                    maginput = {}
                    B_ = mf_MAG_offdip.get_field_multi(XYZ, maginput)
                    Bvec = [B_['BxGEO'][0], B_['ByGEO'][0], B_['BzGEO'][0]]
                    # rotate this vector back into MAG frame:
                    Bvec_MAG = np.matmul(rot_GEO_to_MAG, np.array(Bvec))/1e9

                    #bx_val, by_val, bz_val, _, _, _ = mf_analytical.getBE(x_MAG, y_MAG, z_MAG)
                    #Bvec_MAG_val = [bx_val, by_val, bz_val]
                    sol_Bx[t][i][j][k] = Bvec_MAG[0]
                    sol_By[t][i][j][k] = Bvec_MAG[1]
                    sol_Bz[t][i][j][k] = Bvec_MAG[2]

    print("storing fields...")
    # store axes:
    disk.add_dataset(disk.group_name_data, "x", x * constants.RE)
    disk.add_dataset(disk.group_name_data, "y", y * constants.RE)
    disk.add_dataset(disk.group_name_data, "z", z * constants.RE)
    disk.add_dataset(disk.group_name_data, "time", time)
    # store solutions:
    disk.add_dataset(disk.group_name_data, "Bx", sol_Bx)
    disk.add_dataset(disk.group_name_data, "By", sol_By)
    disk.add_dataset(disk.group_name_data, "Bz", sol_Bz)
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
#     #bfield = dipolefield(constants.RE, 2015)
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