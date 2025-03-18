from math import cos, sin, sqrt, atan2, tan, atan
import numpy as np
import sys
import field_tools

class WGS84:
    """General parameters defined by the WGS84 system"""
    #WGS84 is aligned closely with GEO (but not quite...)
    # https://en.wikipedia.org/wiki/World_Geodetic_System
    #Semimajor axis length (m)
    a = 6378137.0
    #Semiminor axis length (m)
    b = 6356752.314245
    #Ellipsoid flatness (unitless)
    f = (a - b) / a
    #Eccentricity (unitless)
    e = sqrt(f * (2 - f))

class Earth:
    def __init__(self, year_dec):
        #WGS84 ellipsoid:
        M_GEO = np.array([[1/(WGS84.a**2),0,0],[0,1/(WGS84.a**2),0],[0,0,1/(WGS84.b**2)]])
        #rotate the WGS84 ellipsoid from GEO to MAG:
        R_G2M = field_tools.get_rotation_GEO_to_MAG(year_dec)
        self.M_MAG = R_G2M @ M_GEO @ R_G2M.T
        # this is Earth's surface in the MAG frame
        self.c_MAG = np.zeros(3)

    def solve_lambda_intersection(self, o, l):
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
        Q = self.M_MAG
        c = self.c_MAG
        v = (o - c)

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

    def get_height_above_surface(self, px):
        #check if px is above the surface:
        lmbda_max = -1 * np.inf
        sol = self.solve_lambda_intersection(px, px / np.linalg.norm(px)) #from origin in direction of p0 unit vector
        for lmbda in sol:
            if lmbda > 0:
                return None #px is below the surface
            else: #lmbda <= 0:
                if lmbda > lmbda_max:
                    lmbda_max = lmbda
        return -1 * lmbda_max

    # def land(self, p0, p1):
    #     """
    #     calculate the intersection point pX between p0 and p1 at Earth's surface, if there is one
    #     warning: this function assumes that p0 is above Earth's surface!
    #
    #      .p0
    #       \
    #        \
    #     ____x___ Earth
    #          \
    #           \
    #            .p1
    #     """
    #     # check if p1 is below the surface:
    #     sol = self.solve_lambda_intersection(p1, p1 / np.linalg.norm(p1))  # from origin in direction of p0 unit vector
    #     for lmbda in sol:
    #         if lmbda > 0:
    #             p1_below_surface = True
    #             break
    #
    #     if not p1_below_surface:
    #         pX = None  # miss
    #     else:
    #         # tangent to particle trajectory:
    #         l = (p1 - p0)
    #         l = l / np.linalg.norm(l)
    #
    #         # calculate the intersection point between the tangent and ellipsoid:
    #         # if p0 is above the surface, and p1 at or below the surface, there are one or two positive solutions for lambda
    #         # one solution in the case that p1 is at the surface and l direction skims the surface without passing through
    #         sol = self.solve_lambda_intersection(p0, l)
    #         pX = p0 + np.min(sol) * l
    #     return pX
    def land(self, p0, p1):
        """
        p0 and p1 must be numpy arrays
        calculate the intersection point pX between p0 and p1 at Earth's surface, if there is one
        warning: this function assumes that p0 is above Earth's surface!

         .p0
          \
           \
        ____x___ Earth
             \
              \
               .p1
        """

        #tangent to particle trajectory:
        l = (p1 - p0)
        lmag = np.linalg.norm(l)
        lnorm = l / lmag

        #calculate the intersection point of the tangent and ellipsoid:
        # if p0 is above the surface, and p1 at or below the surface, there are one or two positive solutions for lambda
        # one solution in the case that p1 is at the surface and l direction skims the surface without passing through
        pX = None
        sol = self.solve_lambda_intersection(p0, lnorm)
        if len(sol):
            sol_nearest = min(sol)
            if sol_nearest > 0 and sol_nearest < lmag:
                #must be positive aiming from p0 to p1, must be smaller than lmag to be between p0 and p1
                pX = p0 + sol_nearest * lnorm
        return pX
