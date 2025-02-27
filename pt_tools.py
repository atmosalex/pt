import numpy as np
#from pyIGRF.loadCoeffs import get_coeffs
import pyIGRF
from datetime import datetime, timezone
import math
from math import cos, sin, tan, acos, asin, atan, atan2, sqrt, pi, floor
import sys
import h5py
import field_h5
import field_tools
from scipy import interpolate
#from scipy.interpolate import RegularGridInterpolator 

verbose = True
def v_print(*a, **b):
    """Thread safe print function"""

    if not verbose:
        return
    print(*a, **b)

class constants:
    c = 299792458
    MeV2J = 1.60218e-13
    G2T = 1e-4
    nT2T = 1e-9
    mu0 = 1.25663706e-6
    RE = 6.3712e6
    mass0_proton = 1.67262189821e-27
    mass0_electron = 9.10938356e-31
    charge_proton = 1.602176620898e-19
    charge_electron = -1* charge_proton

class WGS84:
    """General parameters defined by the WGS84 system"""
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

def arrange_IGRF_coeffs(coeffs):
    N = 13
    g = np.ones((N+1, N+1)) * np.nan
    h = np.ones((N+1, N+1)) * np.nan
    idx = 0
    for n in range(1, N + 1):
        # n, m=0
        m = 0
        g[n, m] = coeffs[idx]
        # print("g,{},{},{}".format(n,m,coeffs[idx]))
        idx += 1
        for m in range(1, n + 1):
            # n, m=1 to n-1
            g[n, m] = coeffs[idx]
            # print("g,{},{},{}".format(n,m,coeffs[idx]))
            idx += 1
            h[n, m] = coeffs[idx]
            # print("h,{},{},{}".format(n,m,coeffs[idx]))
            idx += 1

    return g, h

class WGS84_atm:
    """General parameters defined by the WGS84 system...
    plus an approximate atmospheric height added to each axis"""
    height_atm = 100000.
    #Semimajor axis length (m)
    a = WGS84.a + height_atm
    #Semiminor axis length (m)
    b = WGS84.b + height_atm
    #Ellipsoid flatness (unitless)
    f = (a - b) / a
    #Eccentricity (unitless)
    e = sqrt(f * (2 - f))
    #as a matrix
    M = np.array([[1/(a**2),0,0],[0,1/(a**2),0],[0,0,1/(b**2)]])


class Dipolefield:
    def __init__(self, year_dec):
        self.year_dec = year_dec
        self.B0, self.M = self.get_B0_m(year_dec)
        self.field_time = [0]

    def get_dipolelc(self, Lb, atm_height):
        Ba = (self.B0) * (4 - 3/Lb)**(0.5)
        RE = constants.RE
        ra = (RE + atm_height)/RE #~Earth's surface + atm_height_dipolelc m
        
        if ra >= Lb:
            return np.nan
        else:
            Ba = (self.B0/(ra**3)) * (4 - 3*ra/Lb)**(0.5)
            dipole_lc = asin(sqrt((self.B0 / Lb**3)/Ba)) * 180 / pi
            return dipole_lc

    def getBE(self, xh, yh, zh, t = 0):
        """
        input: coordinates in m
        """
        Mdir_x = 0
        Mdir_y = 0
        Mdir_z = -1

        r = sqrt( pow(xh,2) + pow(yh,2) + pow(zh,2) )
        C1 = 1e-7*self.M/(r**3)
        mr = Mdir_x*xh + Mdir_y*yh + Mdir_z*zh
        bx = C1*(3*xh*mr/(r**2) - Mdir_x)
        by = C1*(3*yh*mr/(r**2) - Mdir_y)
        bz = C1*(3*zh*mr/(r**2) - Mdir_z)

        return bx, by, bz, 0, 0, 0

    def get_L(self, r, mag_lat):
        """
        takes distance r (Earth radii) and magnetic latitude (radians)
        returns dipole L
        """
        return r/(cos(mag_lat)**2)

    def get_B0_m(self, year):
        """Get the average dipole field strength around Earth's equator and dipole moment. Use like so: B0,m = get_B0_m(2000.0)"""
        #g, h = get_coeffs(year)
        f = interpolate.interp1d(pyIGRF.igrf.time, pyIGRF.igrf.coeffs, fill_value='extrapolate')
        g, h = arrange_IGRF_coeffs(f(year))

        B0_2 = g[1][0]**2 + g[1][1]**2 + h[1][1]**2
        B0_ = sqrt(B0_2)
        B0_ = B0_*constants.nT2T
        M_ = B0_ * (constants.RE ** 3) * 4 * pi / constants.mu0

        return B0_, M_

    def get_eccentric_centre_GEO(self):
        """return vector from 0 to eccentric dipole centre in GEO frame [m] """
        #g, h = get_coeffs(self.year_dec)
        f = interpolate.interp1d(pyIGRF.igrf.time, pyIGRF.igrf.coeffs, fill_value='extrapolate')
        g, h = arrange_IGRF_coeffs(f(self.year_dec))

        L0 = 2*g[1][0]*g[2][0] + sqrt(3)*(g[1][1]*g[2][1] + h[1][1]*h[2][1])
        L1 = -g[1][1]*g[2][0] + sqrt(3)*(g[1][0]*g[2][1] + g[1][1]*g[2][2] + h[1][1]*h[2][2])
        L2 = -h[1][1]*g[2][0] + sqrt(3)*(g[1][0]*h[2][1] - h[1][1]*g[2][2] + g[1][1]*h[2][2])
        E = (L0 * g[1][0] + L1*g[1][1] + L2*h[1][1]) / (4*((1e9*self.B0)**2))
        xi =  (L0 - g[1][0]*E)/(3*((1e9*self.B0)**2))
        eta = (L1 - g[1][1]*E)/(3*((1e9*self.B0)**2))
        zeta =(L2 - h[1][1]*E)/(3*((1e9*self.B0)**2))

        # print(L0)
        # print(L1)
        # print(L2)
        # print(E)
        # print(eta)
        # print(zeta)
        # print(xi)
        #validated against Spenvis values for IGRF2000: https://www.spenvis.oma.be/help/background/magfield/cd.html
        return constants.RE * np.array([eta, zeta, xi])

class Dipolefield_With_Perturbation(Dipolefield):
    def __init__(self, fileload, reversetime = -1):
        #load the HDF5 file
        v_print("Loading E, B field perturbations from", fileload)

        disk = field_h5.HDF5_field(fileload, existing = True)

        t0_ts = disk.read_dataset(disk.group_name_data, "t0")
        t0 = datetime.fromtimestamp(t0_ts, tz=timezone.utc)
        year_dec = dt_to_dec(t0)
        super().__init__(year_dec)
        self.t0 = t0

        self.field_time = disk.read_dataset(disk.group_name_data, "time")
        self.field_dt = self.field_time[1] - self.field_time[0]
        self.field_t_min = self.field_time[0]
        #self.field_t_max = self.field_time[-1]

        self.field_x = disk.read_dataset(disk.group_name_data, "x")
        self.field_dx = self.field_x[1] - self.field_x[0]
        self.field_x_min = self.field_x[0]
        #self.field_x_max = self.field_x[-1]

        self.field_y = disk.read_dataset(disk.group_name_data, "y")
        self.field_dy = self.field_y[1] - self.field_y[0]
        self.field_y_min = self.field_y[0]
        #self.field_y_max = self.field_y[-1]

        self.field_z = disk.read_dataset(disk.group_name_data, "z")
        self.field_dz = self.field_z[1] - self.field_z[0]
        self.field_z_min = self.field_z[0]
        #self.field_z_max = self.field_z[-1]

        #store solutions:
        nt = np.size(self.field_time)
        nx = np.size(self.field_x)
        ny = np.size(self.field_y)
        nz = np.size(self.field_z)
        self.field_BE = np.zeros((6, nt, nx, ny, nz))
        self.field_BE[0, :, :, :, :] = disk.read_dataset(disk.group_name_data, "Bwx")
        self.field_BE[1, :, :, :, :] = disk.read_dataset(disk.group_name_data, "Bwy")
        self.field_BE[2, :, :, :, :] = disk.read_dataset(disk.group_name_data, "Bwz")
        self.field_BE[3, :, :, :, :] = disk.read_dataset(disk.group_name_data, "Ex" )
        self.field_BE[4, :, :, :, :] = disk.read_dataset(disk.group_name_data, "Ey" )
        self.field_BE[5, :, :, :, :] = disk.read_dataset(disk.group_name_data, "Ez" )

        if reversetime > 0:
            #modify calls to int_field so that time becomes reversetime - ti:
            self.reversetime = reversetime
            self.tmult = -1
        else:
            self.reversetime = 0
            self.tmult = 1

        v_print("","done\n")

        self.range_adequate = True

    def int_field(self, xi, yi, zi, ti):
        ti = self.reversetime + self.tmult * ti
        # if reversed, time evolution goes backwards from self.reversetime

        #global R_e dg dx dy dz xmint ymint zmint
        dx = self.field_dx
        dy = self.field_dy
        dz = self.field_dz
        dt = self.field_dt

        pxe0 = floor( (xi - self.field_x_min) / dx )
        pye0 = floor( (yi - self.field_y_min) / dy )
        pze0 = floor( (zi - self.field_z_min) / dz )
        pte0 = floor( (ti - self.field_t_min) / dt )

        if pxe0 < 0:
            self.range_adequate = False
            return [0, 0, 0, 0, 0, 0]
        if pye0 < 0:
            self.range_adequate = False
            return [0, 0, 0, 0, 0, 0]
        if pze0 < 0:
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

        xfac = (xi - self.field_x_min-(pxe0)*dx)/dx;
        yfac = (yi - self.field_y_min-(pye0)*dy)/dy;
        zfac = (zi - self.field_z_min-(pze0)*dz)/dz;
        tfac = (ti - self.field_t_min-(pte0)*dt)/dt;

        # check:
        # print(self.field_x[pxe0]/constants.RE, xi/constants.RE, self.field_x[pxe0+1]/constants.RE, xfac)
        # print(self.field_y[pye0]/constants.RE, yi/constants.RE, self.field_y[pye0+1]/constants.RE, yfac)
        # print(self.field_z[pze0]/constants.RE, zi/constants.RE, self.field_z[pze0+1]/constants.RE, zfac)
        # print(self.field_time[pte0], ti, self.field_time[pte0+1], tfac)
        # print()

        ns = [0, 0, 0, 0, 0, 0, 0, 0]
        interp_vals = [0, 0, 0, 0, 0, 0]
        t_idxs = [pte0, pte0+1]
        t_facs = [1-tfac, tfac]
        
        for idxt in range(2):
            pte = t_idxs[idxt]
            time_fac = t_facs[idxt]
            for idx in range(6):
                ns[0] = self.field_BE[idx, pte, pxe0,  pye0,  pze0  ]
                ns[1] = self.field_BE[idx, pte, pxe0+1,pye0,  pze0  ]
                ns[2] = self.field_BE[idx, pte, pxe0,  pye0+1,pze0  ]
                ns[3] = self.field_BE[idx, pte, pxe0+1,pye0+1,pze0  ]
                ns[4] = self.field_BE[idx, pte, pxe0,  pye0,  pze0+1]
                ns[5] = self.field_BE[idx, pte, pxe0+1,pye0,  pze0+1]
                ns[6] = self.field_BE[idx, pte, pxe0,  pye0+1,pze0+1]
                ns[7] = self.field_BE[idx, pte, pxe0+1,pye0+1,pze0+1]

                nsa = ns[0] + (ns[1]-ns[0])*xfac ;
                nsb = ns[2] + (ns[3]-ns[2])*xfac ;
                nsc = ns[4] + (ns[5]-ns[4])*xfac ;
                nsd = ns[6] + (ns[7]-ns[6])*xfac ;

                nsp = nsa+(nsb-nsa)*yfac ;
                nsq = nsc+(nsd-nsc)*yfac ;

                interp_val = nsp+(nsq-nsp)*zfac ;

                interp_vals[idx] += interp_val * time_fac

        return interp_vals


    def getB_dipole(self, xh, yh, zh):
        """
        input: coordinates in m
        """
        Mdir_x = 0
        Mdir_y = 0
        Mdir_z = -1

        r = sqrt( pow(xh,2) + pow(yh,2) + pow(zh,2) )
        C1 = 1e-7*self.M/(r**3)
        mr = Mdir_x*xh + Mdir_y*yh + Mdir_z*zh
        bx = C1*(3*xh*mr/(r**2) - Mdir_x)
        by = C1*(3*yh*mr/(r**2) - Mdir_y)
        bz = C1*(3*zh*mr/(r**2) - Mdir_z)

        return bx, by, bz

    def getBsph_dipole(self, rh, thetah, t = 0):
        """
        input: coordinates r [m], theta
        """

        br = -2*self.B0 * ((self.RE/ rh)**3) * cos(thetah)
        btheta = -self.B0 * ((self.RE/ rh)**3) * sin(thetah)

        return br, btheta

    def getBE(self, xh, yh, zh, t = 0):
        """
        input: coordinates in m
        """
        bx, by, bz = self.getB_dipole(xh, yh, zh)

        bwx0, bwy0, bwz0, qEx, qEy, qEz = self.int_field(xh, yh, zh, t)
        
        return bx + bwx0, by + bwy0, bz + bwz0, qEx, qEy, qEz

    # def getE(self, xh, yh, zh, t = -1):
    #     return 0, 0, 0



class HDF5_pt:
    def __init__(self, filepath, existing = False):
        """create a HDF5 file"""
        self.informed = False
        self.tracklist_ID = None
        self.filepath = filepath
        self.writeprotectroot = existing
        self.group_name_tracks = 'tracks'
        self.group_name_extra = 'extra'
        self.dataset_name_muenKalphaL0 = 'muenKalphaL0'
        self.dataset_name_muenKalphaL1 = 'muenKalphaL1'

    def setup(self, config_dic, tracklist_dict): #call from pt_handler.py
        """save metadata about the simulation"""
        if (self.writeprotectroot):
            print("Error: could not set up", self.filepath, "using a new configuration - it already exists!")
            sys.exit(1)
        fo = h5py.File(self.filepath, 'w')

        #fo is the root group, we will add our attributes here:
        for attr_name in config_dic.keys():
            fo.create_dataset(attr_name, data=config_dic[attr_name])

        info_keys = list(tracklist_dict.keys())
        info_keys.sort()

        #create datasets of particle properties for each ID:
        tracklist_ID = info_keys
        tracklist_mu = []
        tracklist_pa = []
        tracklist_L = []
        tracklist_pg = []
        tracklist_pb = []
        tracklist_pd = []
        for key in info_keys:
            tracklist_mu.append(tracklist_dict[key][0])
            tracklist_pa.append(tracklist_dict[key][1])
            tracklist_L.append(tracklist_dict[key][2])
            tracklist_pg.append(tracklist_dict[key][3])
            tracklist_pb.append(tracklist_dict[key][4])
            tracklist_pd.append(tracklist_dict[key][5])

        fo.create_dataset('tracklist_ID', data=np.array(tracklist_ID))
        fo.create_dataset('tracklist_mu', data=np.array(tracklist_mu))
        fo.create_dataset('tracklist_pa', data=np.array(tracklist_pa))
        fo.create_dataset('tracklist_L', data=np.array(tracklist_L))
        fo.create_dataset('tracklist_pg', data=np.array(tracklist_pg))
        fo.create_dataset('tracklist_pb', data=np.array(tracklist_pb))
        fo.create_dataset('tracklist_pd', data=np.array(tracklist_pd))
        fo.create_dataset('tracklist_check', data=np.zeros(len(tracklist_ID)), dtype='i')

        fo.create_group(self.group_name_tracks)

        self.informed = True
        self.tracklist_ID = tracklist_ID
        fo.close()

    def read_root(self): #call from pt_fp.py, etc.
        """copy all data in the root group and return it"""
        loadeditems = {}

        with h5py.File(self.filepath, 'r', swmr=True) as fo:
            keylist = list(fo.keys())
            for key in keylist:
                if (key != self.group_name_tracks) and (key != self.group_name_extra):
                    loadeditems[key] = fo[key][()]

        return loadeditems

    def update_dataset(self, qname, quantity, compressmethod=None, quiet=False):
        if not quiet:
            v_print("replacing", qname, "in", self.filepath, ", of length", len(quantity))

        fo = h5py.File(self.filepath, 'a')

        qexists = qname in fo
        if not (qexists):
            print("Error: quantity to update does not exist")
            fo.close()
            sys.exit(1)
        else:
            # group = fo.get(gname_full)
            quantity_ow = fo[qname]  # load the data
            quantity_ow[...] = quantity
        fo.close()

    def add_track(self, id, particle, compressmethod = None, skipeveryn = 1, checkcode = 1, quiet=False):
        """add new data corresponding to a particle ID"""

        #checkcode = 0 is used to indicate that a solution has not been attempted yet
        #checkcode = 1 is used to indicate a successful solution
        #checkcode = 2 is used to indicate an error - i.e. invalid drift orbit


        times = particle.gettimes()
        pt = particle.getpt()

        if len(pt) == 0:
            times = np.array([np.nan])
            pt = np.array([[np.nan,np.nan,np.nan]])
        else:
            times = times[::skipeveryn]
            pt = pt[:, :3][::skipeveryn]

        if not quiet:
            v_print("","adding track", id, "to", self.filepath,", length ",len(times))

        fo = h5py.File(self.filepath, 'a')
        checklist = fo['tracklist_check']
        checkcode_existing = int(checklist[id])

        newgroupname = self.group_name_tracks + "/" + str(id)
        if (checkcode_existing != 0):
            #print("Error: trying to append a new track over an existing solution with ID", id, ", check code", checkcode_existing)
            #fo.close()
            #sys.exit(1)
            print("Warning: overwriting an existing solution with ID", id, ", check code", checkcode_existing)
            checklist[id] = checkcode
            newgroup = fo[newgroupname]

            if not newgroupname in fo:
                newgroup = fo.create_group(newgroupname)
            else:
                if 'time' in fo[newgroupname]:
                    del fo[newgroupname+'/time']
                if 'position' in fo[newgroupname]:
                    del fo[newgroupname+'/position']
                if self.dataset_name_muenKalphaL0 in fo[newgroupname]:
                    del fo[newgroupname+'/'+self.dataset_name_muenKalphaL0]
                if self.dataset_name_muenKalphaL1 in fo[newgroupname]:
                    del fo[newgroupname+'/'+self.dataset_name_muenKalphaL1]
        else:
            #new data:
            checklist[id] = checkcode
            newgroup = fo.create_group(newgroupname)

        #newgroup.attrs['compressed'] = np.string_(compressmethod)
        newgroup.create_dataset('time', data=times, compression = compressmethod)
        newgroup.create_dataset('position', data=pt, compression = compressmethod)

        newgroup.create_dataset(self.dataset_name_muenKalphaL0, data=particle.muenKalphaL[0])
        newgroup.create_dataset(self.dataset_name_muenKalphaL1, data=particle.muenKalphaL[1])
        fo.close()

    def add_extra_group(self, gname, quiet=False):
        """add new group"""

        if not quiet:
            v_print("adding group", gname, "to", self.filepath)

        gname_full = "/"+self.group_name_extra+"/"+gname

        fo = h5py.File(self.filepath, 'a')
        gexists = gname_full in fo

        if (gexists):
            print("Warning: group already exists, continuing...")
        else:
            fo.create_group(gname_full)
        fo.close()

    def add_extra_group_quantity(self, gname, qname, quantity, compressmethod=None, quiet=False):
        """add new data array to a group"""

        if not quiet:
            v_print("adding",qname,"to group", gname, "in", self.filepath, ", of length", len(quantity))

        gname_full = "/" + self.group_name_extra + "/" + gname
        qname_full = gname_full +"/" + qname
        fo = h5py.File(self.filepath, 'a')
        gexists = gname_full in fo
        if not (gexists):
            print("Error: trying to append data to a group that doesn't exist")
            fo.close()
            sys.exit(1)

        qexists = qname_full in fo
        if (qexists):
            print("Error: quantity already exists in this group, leaving as-is and continuing...")
        else:
            group = fo.get(gname_full)

            #newgroup.attrs['compressed'] = np.string_(compressmethod)
            group.create_dataset(qname, data=quantity, compression = compressmethod)
        fo.close()

    def overwrite_extra_group_quantity(self, gname, qname, quantity, compressmethod=None, quiet=False):
        """add new data array to a group"""

        if not quiet:
            v_print("replacing",qname,"in group", gname, "in", self.filepath, ", of length", len(quantity))

        gname_full = "/" + self.group_name_extra + "/" + gname
        qname_full = gname_full +"/" + qname
        fo = h5py.File(self.filepath, 'a')
        gexists = gname_full in fo
        if not (gexists):
            print("Error: trying to append data to a group that doesn't exist")
            fo.close()
            sys.exit(1)

        qexists = qname_full in fo
        if not (qexists):
            print("Error: quantity to update does not exist")
            fo.close()
            sys.exit(1)
        else:
            #group = fo.get(gname_full)
            quantity_ow = fo[qname_full]  # load the data
            if np.shape(quantity_ow[()]) != np.shape(quantity):
                print("Error: overwrite quantity must have the same shape as existing data but it does not")
                sys.exit(1)
            #the data MUST be the same dimensions
            quantity_ow[...] = quantity
        fo.close()

    def rename_extra_group(self, gname_old, gname_new, quiet = False):

        if gname_old == gname_new:
            print("Error: cannot rename a group to the same name")
            sys.exit(1)

        if not quiet:
            v_print("renaming group", gname_old, "to", gname_new, "in", self.filepath)

        fo = h5py.File(self.filepath, 'a')

        gname_full_old = "/" + self.group_name_extra + "/" + gname_old
        gname_full_new = "/" + self.group_name_extra + "/" + gname_new

        gexists = gname_full_old in fo
        if not (gexists):
            print("Error: trying to rename a group that doesn't exist")
            fo.close()
            sys.exit(1)

        fo[gname_full_new] = fo[gname_full_old]
        del fo[gname_full_old]
        #del fo["/" + self.group_name_extra + "/" + gname]
        fo.close()

    def delete_extra_group(self, gname, quiet = False):

        if not quiet:
            v_print("removing group", gname, "from", self.filepath)

        fo = h5py.File(self.filepath, 'a')

        gname_full = "/" + self.group_name_extra + "/" + gname

        gexists = gname_full in fo
        if not (gexists):
            print("Error: trying to delete a group that doesn't exist")
            fo.close()
            sys.exit(1)

        del fo[gname_full]
        #del fo["/" + self.group_name_extra + "/" + gname]
        fo.close()

    def delete_all_extra_groups(self, quiet = False):

        if not quiet:
            v_print("removing all extra groups from", self.filepath)

        fo = h5py.File(self.filepath, 'a')

        gname_full = "/" + self.group_name_extra + "/"

        gexists = gname_full in fo
        if not (gexists):
            fo.create_group(gname_full) #restore the 'extra' group, but keep it empty
            print("Error: no extra groups exist - nothing to delete, continuing...")
            fo.close()
        else:
            del fo[self.group_name_extra]
            fo.create_group(gname_full) #restore the 'extra' group, but keep it empty
            fo.close()


    def read_extra_group_quantity(self, gname, qname):

        qname_full = "/" + self.group_name_extra + "/" + gname + "/" + qname

        fo = h5py.File(self.filepath, 'r', swmr=True)

        qexists = qname_full in fo

        if not (qexists):
            print("Error:",qname_full, "does not exist in",self.filepath)
            fo.close()
            return None
        else:
            q = fo.get(qname_full)[()]
            fo.close()
            return q

    def read_track(self, id, verbose=True, skipeveryn = 1):
        fo = h5py.File(self.filepath, 'r', swmr=True)
        checklist = fo['tracklist_check']
        checkcode = checklist[id][()]

        if verbose: print("Reading track of particle ID", id)
        if checkcode != 1:
            if verbose: print(" warning: checkcode is ",checkcode)

            #fo.close()
            #return np.array([]), np.array([])
        
        times = fo.get(self.group_name_tracks + "/" + str(id) + '/time')[::skipeveryn][()]
        pos = fo.get(self.group_name_tracks + "/" + str(id) + '/position')[::skipeveryn][()]

        fo.close()
        return times, pos

    def read_invariants(self, id):
        fo = h5py.File(self.filepath, 'r', swmr=True)
        checklist = fo['tracklist_check']
        checkcode = checklist[id][()]
        # if checkcode != 1:
        #     print(" error: no track is stored, checkcode is ",checkcode)

        #     fo.close()
        #     return np.array([]), np.array([])
        # else:
        muenKalphaL0 = fo.get(self.group_name_tracks + "/" + str(id) + '/' + self.dataset_name_muenKalphaL0)
        if not muenKalphaL0 is None:
            muenKalphaL0 = muenKalphaL0[()]
        muenKalphaL1 = fo.get(self.group_name_tracks + "/" + str(id) + '/' + self.dataset_name_muenKalphaL1)
        if not muenKalphaL1 is None:
            muenKalphaL1 = muenKalphaL1[()]
        fo.close()
        
        return muenKalphaL0, muenKalphaL1

    def get_existing_tracklist(self):
        fo = h5py.File(self.filepath, 'r', swmr=True)
        ids = fo.get('tracklist_ID')[()]
        tracklist_L = fo.get('tracklist_L')[()]
        tracklist_mu = fo.get('tracklist_mu')[()]
        tracklist_pa = fo.get('tracklist_pa')[()]
        tracklist_pg = fo.get('tracklist_pg')[()]
        tracklist_pb = fo.get('tracklist_pb')[()]
        tracklist_pd = fo.get('tracklist_pd')[()]

        tracklist = {}
        for idx in ids:
            L = tracklist_L[idx]
            mu = tracklist_mu[idx]
            pa = tracklist_pa[idx]
            pg = tracklist_pg[idx]
            pb = tracklist_pb[idx]
            pd = tracklist_pd[idx]
            tracklist[idx] = [L, mu, pa, pg, pb, pd]

        fo.close()

        self.tracklist_ID = ids
        return tracklist

    def get_solved_ids(self):
        fo = h5py.File(self.filepath, 'r', swmr=True)
        ids = fo.get('tracklist_ID')[()]
        checklist = fo.get('tracklist_check')[()]

        id_checkcodes = {}
        for id in ids:
            id_checkcodes[id] = checklist[id]

        fo.close()

        self.tracklist_ID = ids
        return id_checkcodes

    def get_extra_group_names(self):
        group_names = []
        def visitor_func(name, node):
            if isinstance(node, h5py.Dataset):
                # node is a dataset
                pass
            else:
                # node is a group
                group_names.append(name)


        fo = h5py.File(self.filepath, 'r', swmr=True)
        #extra_grouops = fo.get('tracklist_ID')
        extra_groups = fo[self.group_name_extra]
        extra_groups.visititems(visitor_func)
        fo.close()
        return group_names
        
    # def print_file_tree(self):
    #     import nexusformat.nexus as nx
    #     f = nx.nxload(self.filepath)
    #     print(f.tree)


def coord_car_rz(x, y, z, dphi):
    """
    rotation of a Cartesian position around z axis ANTI-CLOCKWISE
    rotate vector x, y, z by dphi
    """
    return (cos(dphi) * x - sin(dphi) * y, sin(dphi) * x + cos(dphi) * y, z)

def coord_car_get_anticlockwise_angle(x1):
    """
    get anticlockwise angle of x1 around z axis from [1, 0]
    """
    return (np.angle(x1[0] + x1[1] * 1j, deg=True) + 360) % 360

class Proton_trace:
    def __init__(self, mu, alpha, L, iphase_gyro=0, iphase_bounce=0, iphase_drift=0, storetrack = True):
        #proton properties:
        self.name = "proton"
        self.m0 = constants.mass0_proton
        self.q = constants.charge_proton
        self.recommended_tsperorbit = 620 # "1/100 of the particle gyroperiod" - see 10.1002/2014JA020899

        self.init_mu = mu
        self.init_alpha = alpha
        self.init_L = L
        self.iphase_gyro = 360 * iphase_gyro #degrees
        self.iphase_bounce = iphase_bounce #fraction along bounce between 0 and 1
        self.iphase_drift = 360 * iphase_drift #degrees

        self.times = []
        self.pt = []
        self.gc_times = []
        self.gc_pos = []

        self.muenKalphaL = np.array([[mu, -1, -1, alpha, L, iphase_gyro, iphase_bounce, iphase_drift],
                                     [-1, -1, -1, -1, -1, -1, -1, -1]])

        self.storetrack = storetrack
        if storetrack:
            self.update = self.update_keep #function pointer
        else:
            self.times = [0]
            self.pt = [[np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]]
            self.update = self.update_discard
       
    def getpt(self, tlimit = -1):
        idx = len(self.times)
        if tlimit > 0:
            if self.times[0] > tlimit:
                return np.array([])

            while self.times[idx-1] > tlimit:
                idx -= 1
        return np.array(self.pt[:idx])

    def gettimes(self, tlimit = -1):
        idx = len(self.times)
        if tlimit > 0:
            if self.times[0] > tlimit:
                return np.array([])

            while self.times[idx-1] > tlimit:
                idx -= 1
        return np.array(self.times[:idx])
        
    def update_keep(self, time, state):
        """
        time: float
        state: array of floats, length 6
        """
        self.times.append(time)
        self.pt.append(state)

    def update_discard(self, time, state):
        """
        time: float
        state: array of floats, length 6
        """
        self.times[-1] = time
        self.pt[-1] = state

    # def cap_to_bounce(self, z_equator = 0, shift0 = 0):
    #     """
    #     this function finds the index after the last full bounce, using recursion
    #     """
    #
    #     #positions = self.getpt()[:,:3]
    #     zco = self.getpt()[shift0:,2]
    #     shift1 = 0
    #
    #     #our process for integrating right up until z = 0 to get a full bounce may have caused a z < 0
    #     while zco[shift1] < z_equator:
    #         shift1 += 1
    #         if shift1 == np.size(zco):
    #             #v_print("Error: no bounce detected at all")
    #             return shift0
    #     #find out the index of the first z < 0:
    #     idx_bounce_half = shift1 + ((zco[shift1:] < z_equator).argmax(axis=0))
    #
    #
    #     if idx_bounce_half <= shift1:
    #         #the first (shifted) position is definitely not below zero, therefore...
    #         # we didn't even get half a bounce
    #         #v_print("Error: didn't half bounce after", self.bounces, "bounces")
    #         return shift0
    #
    #     idx_bounce_nexthalf = ((zco[idx_bounce_half:] > z_equator).argmax(axis=0))
    #
    #     if idx_bounce_nexthalf == 0:
    #         #the first position is definitely not above zero, therefore...
    #         # we didn't get another half bounce
    #         #v_print("Error: didn't half bounce after", self.bounces, "bounces")
    #         return shift0
    #
    #     shift1 = idx_bounce_half + idx_bounce_nexthalf + 1
    #
    #     return self.cap_to_bounce(z_equator=z_equator, shift0 = shift0 + shift1)


    def pop_track(self, requiredlen = 0):
        if len(self.pt) > requiredlen and len(self.times) > requiredlen:
            self.pt.pop()
            self.times.pop()
            return 1
        else:
            return 0

    def calculate_initial_GC(self):
        #get initial position of the GC based on particle L:
        R_gc_RE = self.init_L
        R_gc = R_gc_RE * constants.RE

        xgc, ygc, zgc = coord_car_rz(R_gc, 0, 0, np.radians(self.iphase_drift))

        x0_GC = np.array([xgc, ygc, zgc])
        return x0_GC

    def calculate_initial_momentum(self, B_GC):
        """
        calculates a possible momentum vector on the equator given:
            the equatorial field
            properties of the particle
        """
        #bz is positive (+z direction)

        R_gc_RE = self.init_L
        R_gc = R_gc_RE * constants.RE

        p0_perp = sqrt(self.init_mu*2*self.m0*np.linalg.norm(B_GC))
        p0_par = 1./tan(self.init_alpha) * p0_perp
        p0mag = sqrt(p0_perp**2 + p0_par**2)
        ga = sqrt(1 + (p0mag/(self.m0 * constants.c))**2)
        massr = ga*self.m0

        #derive velocity:
        v0_perp = p0_perp/massr #relativistic velocity

        #0 degree gyrophase velocity:
        v0x = 0
        v0y = -v0_perp
        #rotate the vector to the correct gyrophase:
        v0r = coord_car_rz(v0x, v0y, 0, np.radians(self.iphase_gyro))
        v0x = v0r[0]
        v0y = v0r[1]

        v0z = p0_par/massr

        #set the initial relativistic velocity:
        v0x, v0y, v0z = coord_car_rz(v0x, v0y, v0z, np.radians(self.iphase_drift))
        v0 = np.array([v0x, v0y, v0z])
        p0 = massr * v0
        #E0_J = self.calculateKE(bfield) #it is useful to have a separate funciton, so we can calculate energy from different scripts
        return p0
        
    def calculate_initial_position(self, x0_GC, rg):
        x0 = [rg, 0, 0] #0 degrees

        #rotate the vector to the correct gyrophase:
        x0r = coord_car_rz(x0[0], x0[1], x0[2], np.radians(self.iphase_gyro))
        x0 = x0r

        #rotate the vector to the correct drift phase:
        x0r= coord_car_rz(x0[0], x0[1], x0[2], np.radians(self.iphase_drift))
        x0 = x0r

        x0 = x0 + x0_GC

        return x0

    def derive_KE0(self, bfield, t):
        x0_GC = self.calculate_initial_GC()
        B_GC = bfield.getBE(*x0_GC, t)[:3]

        p0 = self.calculate_initial_momentum(B_GC)
        p0mag = np.linalg.norm(p0)

        gamma = sqrt(1 + (p0mag/(self.m0 * constants.c))**2)
        E0_J = (gamma - 1)*self.m0*(constants.c**2) #J
        #E0 = E0_J / constants.MeV2J #KE energy in MeV

        return E0_J

class Electron_trace(Proton_trace):
    def __init__(self, mu, alpha, L, iphase_gyro, iphase_bounce, iphase_drift,  storetrack = True):
        super(Electron_trace, self).__init__(mu, alpha, L, iphase_gyro, iphase_bounce, iphase_drift, storetrack)
        #proton properties:
        self.name = "electron"
        self.m0 = constants.mass0_electron
        self.q = constants.charge_electron
        self.recommended_tsperorbit = 100



def approx_Yb(BeBmratio):
    Y_ = 2.760346 + 2.357194 * sqrt(BeBmratio) - 5.117540 * (BeBmratio**(3./8))
    return Y_
def approx_Ya(aeq):
    y = sin(aeq)
    Y_ = 2.760346 + 2.357194 * y - 5.117540 * (y**(3./4))
    return Y_


def dt_to_dec(dt):
    """Convert a datetime to decimal year. Use like so: dt = datetime(2020, 1, 1); year = dt_to_dec(dt)"""
    year_start = datetime(dt.year, 1, 1, tzinfo=timezone.utc)
    year_end = year_start.replace(year=dt.year+1)

    return dt.year + ((dt - year_start).total_seconds() /  # seconds so far
        float((year_end - year_start).total_seconds()))  # seconds in year


#for reading fortran output files:
class r8mat_reader:
    def __init__(self, filename, separator = " ", filler = None):
        self.filename = filename
        self.separator = separator
        self.data = []
        self.filler = filler

    def read_nofiller(self, quiet):
        if not quiet: v_print("Reading", self.filename)
        with open(self.filename,'r') as rf:
            line = rf.readline().strip('\n')
            while len(line) != 0:
                self.data.append([float(x) for x in line.split(self.separator)])
                line = rf.readline()
            if not self.data:
                return 0
            return 1

    def read_filler(self, quiet):
        if not quiet: v_print("Reading", self.filename)
        with open(self.filename,'r') as rf:
            line = rf.readline().strip('\n')
            while not self.filler in line and len(line) != 0:
                self.data.append([float(x) for x in line.split(self.separator)])
                line = rf.readline()
            if not self.data:
                return 0
            return 1

    def read(self, quiet=False):
        if self.filler == None:
            return self.read_nofiller(quiet)
        else:
            return self.read_filler(quiet)

# class r8vec_reader:
#     def __init__(self, filename, separator = " ", filler = None):
#         self.filename = filename
#         self.separator = separator
#         self.data = []
#         self.filler = filler
        
#     def read(self, quiet=False):
#         if not quiet: v_print("Reading", self.filename)
#         with open(self.filename,'r') as rf:
#             line = rf.readline().strip('\n')
#             while not self.filler in line and len(line) != 0:
#                 self.data.append(float(line))
#                 line = rf.readline()
#             if not self.data:
#                 return 0
#             return 1     

# def numpy1dout(filename, ar):
#     with open(filename,"wo") as fo:
#         for element in ar:
#             fo.writeline(element)

def array1dout(filename, data, isnumeric=False, overwrite=True, quiet=False):
    prec = 8
    fmtdec = '{:.' + str(prec) + 'E}'

    if not quiet:
        if overwrite:
            v_print("writing new file to", filename)
        else:
            v_print("appending to", filename)
    
    if overwrite:
        openflag = "w"
    else:
        openflag = "a"
    
    with open(filename, openflag) as fo:
        for datum in data:
            if isnumeric: datumfmt = str(fmtdec.format(datum))
            fo.write(str(datumfmt)+"\n")

#for reading follow particle trajectories:
class config_rw:
    def __init__(self,filename):
        self.filename = filename
        self.datadic = {}

        self.species_kw = "species"
        self.orbit_kw = "orbit"
        self.storetrack_kw = "store trajectory"
        self.storegc_kw = "store GC"
        self.findK0_kw = "find initial K"
        self.reeval_kw = "re-calculate invariants"
        self.duration_kw = "duration to solve"
        self.reverse_kw = "reverse"
        self.year_kw = "year"
        self.month_kw = "month"
        self.day_kw = "day"
        self.lmin_kw = "Lmin"
        self.lmax_kw = "Lmax"
        self.nl_kw = "nL"
        self.amin_kw = "amin"
        self.amax_kw = "amax"
        self.na_kw = "na"
        self.logmumin_kw = "logmumin"
        self.logmumax_kw = "logmumax"
        self.nmu_kw = "nmu"
        self.nphase_gyro_kw = "nphase_gyro"
        self.nphase_bounce_kw = "nphase_bounce"
        self.nphase_drift_kw = "nphase_drift"
        self.perturbation_grid_kw = "perturbation_grid"
        self.skipeveryn_kw = "skipeveryn"
        self.emin_kw = "emin"
        self.emax_kw = "emax"
        self.iphase_gyro_kw = "iphase_gyro"
        self.iphase_bounce_kw = "iphase_bounce"
        self.iphase_drift_kw = "iphase_drift"
        self.continuefrom_kw = "continuefrom"
        self.override_energy_axis_kw = "override energy axis"


    def convert_types(self):
        try:
            self.datadic[self.species_kw] = str(self.datadic[self.species_kw][0])
            self.datadic[self.orbit_kw] = str(self.datadic[self.orbit_kw][0])
            
            self.datadic[self.duration_kw] = float(self.datadic[self.duration_kw][0])

            self.datadic[self.reverse_kw] = str(self.datadic[self.reverse_kw][0])
            self.datadic[self.storetrack_kw] = str(self.datadic[self.storetrack_kw][0])
            self.datadic[self.storegc_kw] = str(self.datadic[self.storegc_kw][0])
            self.datadic[self.findK0_kw] = str(self.datadic[self.findK0_kw][0])
            self.datadic[self.reeval_kw] = str(self.datadic[self.reeval_kw][0])

            self.datadic[self.year_kw] = int(self.datadic[self.year_kw][0])
            self.datadic[self.month_kw] = int(self.datadic[self.month_kw][0])
            self.datadic[self.day_kw] = int(self.datadic[self.day_kw][0])

            self.datadic[self.lmin_kw] = float(self.datadic[self.lmin_kw][0])
            self.datadic[self.lmax_kw] = float(self.datadic[self.lmax_kw][0])
            self.datadic[self.nl_kw] = int(self.datadic[self.nl_kw][0])

            self.datadic[self.amin_kw] = float(self.datadic[self.amin_kw][0])
            self.datadic[self.amax_kw] = float(self.datadic[self.amax_kw][0])
            self.datadic[self.na_kw] = int(self.datadic[self.na_kw][0])

            self.datadic[self.logmumin_kw] = float(self.datadic[self.logmumin_kw][0])
            self.datadic[self.logmumax_kw] = float(self.datadic[self.logmumax_kw][0])
            self.datadic[self.nmu_kw] = int(self.datadic[self.nmu_kw][0])

            self.datadic[self.nphase_gyro_kw] = int(self.datadic[self.nphase_gyro_kw][0]) 
            self.datadic[self.nphase_bounce_kw] = int(self.datadic[self.nphase_bounce_kw][0])
            self.datadic[self.nphase_drift_kw] = int(self.datadic[self.nphase_drift_kw][0])

            self.datadic[self.perturbation_grid_kw] = str(self.datadic[self.perturbation_grid_kw][0])

            self.datadic[self.skipeveryn_kw] = int(self.datadic[self.skipeveryn_kw][0])

            self.datadic[self.emin_kw] = float(self.datadic[self.emin_kw][0])
            self.datadic[self.emax_kw] = float(self.datadic[self.emax_kw][0])

            self.datadic[self.iphase_gyro_kw] = float(self.datadic[self.iphase_gyro_kw][0]) 
            self.datadic[self.iphase_bounce_kw] = float(self.datadic[self.iphase_bounce_kw][0])
            self.datadic[self.iphase_drift_kw] = float(self.datadic[self.iphase_drift_kw][0])

            self.datadic[self.continuefrom_kw] = str(self.datadic[self.continuefrom_kw][0])
            if str(self.datadic[self.override_energy_axis_kw][0]):
                self.datadic[self.override_energy_axis_kw] = np.array([float(x) for x in self.datadic[self.override_energy_axis_kw]])
            else:
                self.datadic[self.override_energy_axis_kw] = np.array([])
            return 1
        except Exception as e:
            print(e)
            return 0


    def read(self, quiet=False):
        if not quiet: v_print("Reading", self.filename)
        count = 1
        with open(self.filename,'r') as rf:
            for line in rf:
                line = line.strip('\n')
                if len(line.strip('#')) != 0:
                    #cut off comments:
                    if '#' in line:
                        line = line[:line.find('#')]
                    splitline = [x for x in line.split(',')]
                    lineempty = True
                    for item in splitline:
                        if len(item.strip(' ')):
                            lineempty = False

                    line = splitline
                    if not lineempty:
                        self.datadic[line[0]] = [x.strip(' ') for x in line[1:]]

                count += 1
            if not self.datadic:
                return 0

        #convert each data type:
        return self.convert_types()

    def saveas(self, filename, towrite=None, quiet=False, topcomments=[]):
        """write out a dictionary in configuration file format: key, [value list]"""
        if not towrite: towrite = self.datadic
        if not quiet: v_print("Writing", filename)
        if not towrite:
            v_print("Error, can't write with no data!")
            return 0
        if not isinstance(towrite, dict):
            v_print("Error, need data as type dictionary to write a configuration file!")
            return 0
        count = 1
        with open(filename,'w') as wf:
            for comment in topcomments:
                wf.write('#' + comment + '\n')
            for keyname in towrite:
                wf.write(str(keyname))
                wf.write(",")
                if isinstance(towrite[keyname], list) or isinstance(towrite[keyname], np.ndarray):
                    listastext = str(", ".join(["{}".format(x) for x in towrite[keyname]]))
                    wf.write(listastext)
                else:
                    wf.write(str(towrite[keyname]))
                wf.write("\n")
        return 1


#for reading follow particle trajectories:
class fp_reader:
    def __init__(self, filename, separator = ','):
        self.filename = filename
        self.data = []
        self.separator = separator

    def read(self, quiet=False, headerlines=0):
        if not quiet: v_print("Reading", self.filename)
        
        with open(self.filename,'r') as rf:
            while headerlines > 0:
                rf.readline()
                headerlines -= 1
            line = rf.readline().strip('\n')
            while len(line) != 0:
                self.data.append([float(x) for x in filter(None,line.split(self.separator))])
                line = rf.readline()
            if not self.data:
                return 0
            return 1
