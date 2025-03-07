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
        self.M_MAG = np.matmul(field_tools.get_rotation_GEO_to_MAG(year_dec), M_GEO)
        # this is Earth's surface in the MAG frame

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