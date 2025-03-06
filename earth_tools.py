from math import cos, sin, sqrt, atan2, tan, atan
import numpy as np
import IGRF_tools
from pt_tools import constants
import sys

class WGS84:
    """General parameters defined by the WGS84 system"""
    #WGS84 is aligned closely with GEO (but not quite...)
    # https://en.wikipedia.org/wiki/World_Geodetic_System
    #Semimajor axis length (m)
    a = 6378137.0
    #Semiminor axis length (m)
    b = 6356752.3142
    #Ellipsoid flatness (unitless)
    f = (a - b) / a
    #Eccentricity (unitless)
    e = sqrt(f * (2 - f))
    #as a matrix
    M = np.array([[1/(a**2),0,0],[0,1/(a**2),0],[0,0,1/(b**2)]])

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
    return T5

    ###validation using IRBEM for year_dec = 2015.0:
    # import datetime
    # from datetime import timezone
    # import IRBEM as ib
    # t_datetime = datetime.datetime(year=2015, month=1, day=1, tzinfo=timezone.utc)
    # coords = ib.Coords()
    # rot_GEO_to_MAG = coords.transform([t_datetime, t_datetime, t_datetime], [[1, 0, 0], [0, 1, 0], [0, 0, 1]], 'GEO', 'MAG').T
    # print(rot_GEO_to_MAG)

def get_eccentric_centre_GEO(year_dec):
    """return vector from 0 to eccentric dipole centre in GEO frame [m] """
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
    M_GEO_to_MAG = get_rotation_GEO_to_MAG(year_dec)
    x_ed_MAG = M_GEO_to_MAG @ x_ed_GEO
    return x_ed_MAG

# class WGS84_atm:
#     """General parameters defined by the WGS84 system...
#     plus an approximate atmospheric height added to each axis"""
#     height_atm = 100000.
#     #Semimajor axis length (m)
#     a = WGS84.a + height_atm
#     #Semiminor axis length (m)
#     b = WGS84.b + height_atm
#     #Ellipsoid flatness (unitless)
#     f = (a - b) / a
#     #Eccentricity (unitless)
#     e = sqrt(f * (2 - f))
#     #as a matrix
#     M = np.array([[1/(a**2),0,0],[0,1/(a**2),0],[0,0,1/(b**2)]])

class Earth:
    def __init__(self):
        a = 6378137.0 #m
        b = 6356752.314245 #m
        inv_flat = 298.257223563

        self.ellipsoid_M_MAG
        self.ellipsoid_centre_MAG

    def intersection_line_ellipsoid(Q, o, l, c):
        """
        calculates lambda for an intersecting line, ellipsoid given by:
            x = o + lambda * l,
            (x - c).T * Q * (x - c) = 1
        respectively, where:
            o is the origin of the line;
            c is centre of the ellipsoid;
            l is direction vector of the line;
            lambda is parametric distance along the line from o;
            Q is the ellipsoid matrix.

        all parameters must be defined in the same coordinate system

        only real values are returned as we are dealing with physical space
        """
        v = (o - c)
        # l_col = np.swapaxes(np.array([l]),0,1)
        # v_col = np.swapaxes(np.array([v]),0,1)

        # quadratic equation in terms of lambda with the following coefficients:
        # a = l.T * Q * l
        a = np.matmul(Q, l)
        a = np.matmul(l, a)

        # b1 = l.T * Q * v
        b1 = np.matmul(Q, v)
        b1 = np.matmul(l, b1)

        # b2 = l.T * Q * v
        b2 = np.matmul(Q, l)
        b2 = np.matmul(v, b2)

        b = b1 + b2  # actually, b1 and b2 are equal because Q is symmetric in practise (property: a.T Q b == b.T Q a)

        # c = v.T * Q * v
        c = np.matmul(Q, v)
        c = np.matmul(v, c)

        p = [a, b, c - 1]

        # disc = b**2 - 4 * a * c

        sols = np.roots(p)
        return sols[np.isreal(sols)]

    def shoot_Earth(self, p0, p1):
        # negative tangent to trajectory:
        l = - (p1 - p0)  # s.t. p1 + l = p0
        l = l / np.linalg.norm(l)

        # check the particle is above the surface of the atmosphere at 100km:
        tospace = self.intersection_line_ellipsoid(self.ellipsoid_M_MAG, self.ellipsoid_centre_MAG + p0, p0 / np.linalg.norm(p0), ellipsoid_centre_MAG)
        if len(tospace):  # take the smallest POSITIVE root (collision with closest face in the direction of Earth)
            for solution in tospace:
                if solution > 0:
                    belowsurface = True
                    print("error 1"); sys.exit()

        # calculate the intersection point (if any) between the negative tangent and ellipsoid:
        # intersection_line_ellipsoid(...) returns the parameter lambda, distance along a vector with unit l until collision
        lam = self.intersection_line_ellipsoid(self.ellipsoid_M_MAG, p0, l, self.ellipsoid_centre_MAG)
        positive_solutions = []
        if len(lam):  # take the smallest POSITIVE root (collision with closest face IN THE DIRECTION OF EARTH, NOT AWAY)
            for solution in lam:
                if solution > 0:
                    positive_solutions.append(solution)
        else:  # no intersection
            print("error 2"); sys.exit()
        if len(positive_solutions):
            lam = min(positive_solutions)
        else:  # no intersection in the correct direction
            print("error 3"); sys.exit()

        atm_hit_point = p0 + lam * l  # mag coordinates
        atm_hit_point_GEO = np.matmul(R_mag_to_GEO, atm_hit_point)