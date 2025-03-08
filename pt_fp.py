import sys
import numpy as np
from scipy.integrate import ode
from math import cos, sin, tan, acos, asin, atan, atan2, sqrt, pi
from datetime import datetime
import argparse
import earth
import pt_pushers
import pt_tools
import time
import copy
import pt_pushers as pushers

G2T = pt_tools.constants.G2T
c = pt_tools.constants.c  # 299792458
RE = pt_tools.constants.RE  # 6.3712e6

warned_grid_resolution = False
# critical settings:------------------------------------------------+
aeq_max_for_bounce_detection = 89.5
# above this threshold, particles will be considered equatorial,
# and their trajectories will not be cut down to a bounce period
pusher = pushers.boris_fwd #default particle pusher
# ------------------------------------------------------------------+

def tb_estimate(R_gc, vmag, aeq):
    """
    R_gc is in the frame with the dipole field at the origin!
    this approximate formula is from Walt, equation 4.28
    contains about "0.5%" error
    """
    tb = 0.117 * (R_gc/RE) * (1/(vmag/c)) * (1- 0.4635*((sin(aeq))**0.75)) 

    #uncomment the below to plot tb as a function of eq.alpha:
##    import matplotlib.pyplot as plt
##    L=1.5
##    alpha = np.arange(91)
##    tbs = []
##    for a in alpha:
##        aeq = a*pi/180
##        tbs.append(0.117 * (L) * (1/(vmag/c)) * (1- 0.4635*((sin(aeq))**0.75)))
##    
##    tbs = np.array(tbs)
##    plt.plot(alpha,tbs)
##    plt.xlabel('eq. alpha')
##    plt.ylabel('tb')
##    plt.title("Estimate tb vs. eq. alpha for a proton at L=" + str(L) + ", v/c=" + str(np.round(vmag/c,2)))
##    plt.show()
##    sys.exit(1)
    return tb


def dYdt(t, Y, particle, bfield):
    """Computes the derivative of the state vector y according to the equation of motion:
    Y is the state vector (x1, x2, x3, p1, p2, p3) === (position, momentum).
    returns dY/dt.

    first two arguments must be t, Y as per set_integrator() rules
    """
    #http://kfe.fjfi.cvut.cz/~horny/NME/NME-motionsolver/pohyboverovnice.pdf
    

    x1, x2, x3 = Y[0], Y[1], Y[2]
    p1, p2, p3 = Y[3], Y[4], Y[5]
    
    #pmag = pow((p1*p1 + p2*p2 + p3*p3),0.5)
    pmag = np.linalg.norm(Y[3:])
    gamma = sqrt(1 + (pmag/(particle.m0 * pt_tools.constants.c))**2)

    #E0 = m0 * pt_tools.constants.c * pt_tools.constants.c
    #ga = sqrt(E0 * E0 + pmag * pmag * pt_tools.constants.c * pt_tools.constants.c)/E0
    #print(ga)
    
    #E = ga * m0 * c**2

    #v = Y[3:] * c**2 / E
    #v = Y[3:] / (gamma * particle.m0)
    v1 = Y[3] / (gamma * particle.m0)
    v2 = Y[4] / (gamma * particle.m0)
    v3 = Y[5] / (gamma * particle.m0)
    
    B1, B2, B3, E1, E2, E3 = bfield.getBE(x1, x2, x3, t)

    #Calculate the Lorentz force in the observer frame:
    F1 = particle.q * (v2*B3 - v3*B2) + particle.q * E1
    F2 = particle.q * (v3*B1 - v1*B3) + particle.q * E2
    F3 = particle.q * (v1*B2 - v2*B1) + particle.q * E3

    return np.array([v1, v2, v3, F1, F2, F3]) #[0.0, -87183091.06369066, 103900761.98862153, -1.2361325231523902e-16, 0.0, 0.0]

def angle_between(v1, v2):
    #from https://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def calc_rg(Y0, bfield, particle, t=0):
    """
    takes a state vector
    calculates the Lorentz force
    assumes F = mv2 / r
    result is always positive
    """

    #calculate the lorentz factor given the total momentum:
    p0mag = np.linalg.norm(Y0[3:])
    gamma = sqrt(1 + (p0mag/(particle.m0 * pt_tools.constants.c))**2)

    dY0dt = dYdt(t, Y0, particle, bfield) #first argument doesn't matter
    
    v2 = np.linalg.norm(dY0dt[:2])**2 #magnitude of the x,y velocity
    F1 = np.linalg.norm(dY0dt[3:]) #Lorentz force directed toward GC

    return abs(gamma*particle.m0*(v2)/F1)

def get_GC_from_track(rg0, bfield, particle, tsperorbit, tlimit = -1, freezefield = -1, get_K=True): #freezefield not implemented yet
    """get the moving average gyrocentre position over each gyration"""

    def get_field_time_freeze(time):
        return freezefield
    def get_field_time_particle(time):
        return time
    if freezefield >= 0:
        get_field_time = get_field_time_freeze
    else:
        get_field_time = get_field_time_particle


    pt = particle.getpt(tlimit)
    times = particle.gettimes(tlimit)

    t_ = get_field_time(times[0])
    B_0 = bfield.getBE(*pt[0][:3], t_)[:3]
    Bm_actual = np.linalg.norm(B_0) #not true, but will be updated
    pperp0 = np.linalg.norm(np.cross(pt[0][3:], B_0/np.linalg.norm(B_0)))

    track_gyrocentre = []
    track_gyrocentre_time = []

    #find the direction vector of the force on the particle at each time: 
    pt_up = np.roll(pt[:,3:], -1, axis = 0)
    dp = pt_up - pt[:,3:]
    times_up = np.roll(times, -1)
    dt = times_up - times
    force_ns = dp/ np.linalg.norm(dp, axis = 1)[:,None]

    for idx in range(len(pt)-1):
        p_ = pt[idx][3:]
        t_ = get_field_time(times[idx])

        #calculate the direction vector of the local field (at the midpoint):
        B_ = bfield.getBE(*pt[idx][:3], t_)[:3]
        B_n = B_/np.linalg.norm(B_)
        
        #magnitude of perpendicular momentum is similar at the particle position vs. gyrocentre:
        # ppar = np.abs(np.dot(p_, B_n))
        pperp = np.linalg.norm(np.cross(p_, B_n))

        # the quantity rg * pperp is conserved
        rg = (rg0 * pperp0)/pperp


        #gc = (pt[idx+1][:3] + pt[idx][:3])/2 + force_ns[idx] * rg
        gc = pt[idx][:3] + force_ns[idx] * rg
        track_gyrocentre.append(gc)
        track_gyrocentre_time.append(times[idx])

    #
    # take the moving average of the gyrocentre over the number of timesteps in a gyration:
    #
    if tsperorbit % 2 != 0:
        print("Warning: time steps per orbit is an odd number, which will cause a small error in approximation of K")
    track_gyrocentre = np.array(track_gyrocentre)
    track_gyrocentre_time = np.array(track_gyrocentre_time)
    track_gyrocentre = pt_tools.moving_average(track_gyrocentre, tsperorbit)
    track_gyrocentre_time = pt_tools.moving_average(track_gyrocentre_time, tsperorbit)

    if not get_K:
        return list(track_gyrocentre_time), list(track_gyrocentre), -1, -1

    #find visits to the peak magnetic field along the gyrocentre:
    Bmax = 0
    idx_Bmax = 0
    Bmax_peaks = []
    idx_Bmax_peaks = []
    gradB = 0
    npeaks = 0
    foundnewpeak = False
    Bmags = []
    for idx in range(len(track_gyrocentre)-1):
        t_ = get_field_time(track_gyrocentre_time[idx])

        bx0, by0, bz0 = bfield.getBE(*track_gyrocentre[idx], t_)[:3]
        Bmag = sqrt(pow(bx0,2) + pow(by0,2) + pow(bz0,2))
        Bmags.append(Bmag)
        if Bmag > Bmax:
            Bmax = Bmag
            idx_Bmax = idx
            foundnewpeak = True
        #wait a few gyrations before confirming we found a new peak
        if idx - idx_Bmax >= 3*tsperorbit and foundnewpeak:
            Bmax_peaks.append(Bmax)
            idx_Bmax_peaks.append(idx_Bmax)
            foundnewpeak = False
            Bmax = Bmag #since we are heading away from a mirror point, the next Bmag will be smaller

        if len(Bmax_peaks) > 3: break #don't need any more to calculate K near the beginning of the trajectory

    # import matplotlib.pyplot as plt
    # fig, axs = plt.subplots(3, sharex=False)
    # #axs = [axs]
    # axs[0].plot(times, pt[:,2])
    # axs[0].plot(track_gyrocentre_time, track_gyrocentre[:,2], color='red')
    # axs[1].plot(track_gyrocentre_time[:len(Bmags)], Bmags)
    #
    # axs[2].plot(pt[:,0], pt[:,1])
    # axs[2].plot(track_gyrocentre[:,0], track_gyrocentre[:,1])
    # plt.show()
    # sys.exit()

    if len(idx_Bmax_peaks) <= 3:
        print("Error: could not detect a whole bounce orbit to calculate K")
        print("","if using an interpolated B field and aeq~90, this could be because the field is fuzzy at the scale of the particle's motion along Z, try decreasing aeq")
        print("","if using a field with longitudinal asymmetry, this could be because the estimate for tb became invalid as the particle drifted, and not enough trajectory was calculated")
        return list(track_gyrocentre_time), list(track_gyrocentre), -1, -1

    #isolate one GC bounce:
    track_gyrocentre_bounce = track_gyrocentre[idx_Bmax_peaks[0]:idx_Bmax_peaks[2]+1]
    track_gyrocentre_bounce_time = track_gyrocentre_time[idx_Bmax_peaks[0]:idx_Bmax_peaks[2]+1]
    tb_est = track_gyrocentre_bounce_time[-2] - track_gyrocentre_bounce_time[0]

    #momentum along the isolated bounce track:
    # len(pt)) = len(track_gyrocentre) + tsperorbit)
    p_gyrocentre = pt[tsperorbit//2 + idx_Bmax_peaks[0]:,3:]

    #find K along the isolated bounce:
    J2 = 0
    for idx in range(len(track_gyrocentre_bounce)-1):
        t_ = get_field_time(track_gyrocentre_bounce_time[idx])
        
        Bgc = bfield.getBE(*track_gyrocentre_bounce[idx], t_)[:3]
        Bgc_n = Bgc/np.linalg.norm(Bgc)
        ppar = np.abs(np.dot(p_gyrocentre[idx], Bgc_n))

        dl = np.linalg.norm(track_gyrocentre_bounce[idx+1] - track_gyrocentre_bounce[idx])
        J2 += dl * ppar
    p0 = np.linalg.norm(pt[0,3:])
    I = J2 / (2*p0)
    K = np.power(np.max(Bmax_peaks[:3])/G2T,0.5) * I / RE

    #return as much of the GC as possible, and K:
    return list(track_gyrocentre_time), list(track_gyrocentre), K, tb_est

def get_bouncecapped_trajectory_dipole(bfield, tocap_time, tocap_pt, tsperorbit, idx_left_equator_safe = 0, reverse=False):
    """
    cap the trajectory to the soonest equatorial crossing at 0 bounce phase (negative to positive dZ relative to the magnetic equator)
    if pt begins at dZ=0 and idx_left_equator_safe = 0, this function will return the first point in the trajectory
     therefore set idx_left_equator_safe to a value s.t. the particle will not trip a bounce detection near the very beginning of its trajectory
    input must be numpy arrays
    returns floats, lists

    works only when the magnetic equator is at constant z around the drift orbit (as in the case of a dipole aligned with MAG)
    """

    # find where Z goes from negative to positive, this is where bounce phase gets reset:
    dz = tocap_pt[:, 2] - bfield.origin_MAG[2]
    dz_sign = np.sign(dz)
    # ... or if reverse tracing, find where Z goes from positive to negative:
    if reverse:
        dz_sign = -1 * dz_sign

    idx_newbounce = idx_left_equator_safe

    while dz_sign[idx_newbounce] > 0 and idx_newbounce < len(dz_sign):
        idx_newbounce += 1
    if idx_newbounce >= len(dz_sign):
        print("Error: could not detect an equatorial crossing, particle is not bouncing as expected")
        return -1, -1, [], []
    #print(idx_newbounce, len(dz_sign), idx_newbounce/len(dz_sign))

    while dz_sign[idx_newbounce] <= 0:
        idx_newbounce += 1
    if idx_newbounce >= len(dz_sign) and idx_newbounce < len(dz_sign):
        print("Error: could not detect a return to 0 bounce phase, particle is not bouncing as expected")
        return -1, -1, [], []
    #print(idx_newbounce, len(dz_sign), idx_newbounce/len(dz_sign))

    capped_pt = list(tocap_pt[:idx_newbounce])
    capped_times = list(tocap_time[:idx_newbounce])

    # interpolate the (future) state vector at the equator:
    dpt = tocap_pt[idx_newbounce] - tocap_pt[idx_newbounce - 1]
    dz = dpt[2]
    frac_dz = (bfield.origin_MAG[2] - tocap_pt[idx_newbounce - 1][2]) / dz
    if frac_dz == 0:
        # we are already at the equator at idx_newbounce - 1 of the trajectory
        pt_eq = tocap_pt[idx_newbounce - 1]
        t_eq = tocap_time[idx_newbounce - 1]
    else:
        pt_eq = tocap_pt[idx_newbounce - 1] + dpt * frac_dz

        # get the time between index 0 (final particle position) and the equatorial crossing:
        dt = (tocap_time[idx_newbounce] - tocap_time[idx_newbounce - 1])
        t_eq = tocap_time[idx_newbounce - 1] + dt * frac_dz

        capped_pt.append(pt_eq)
        capped_times.append(t_eq)

    return t_eq, pt_eq, capped_times, capped_pt

def converge_on_first_crossing_idx(xyz, ti, bfield, idx_jump, count_index = 0, time_direction = 1):
    # find where dZ goes from negative to positive across the equator, this is where bounce phase gets reset
    #print("checking from", count_index, ", jumping ", idx_jump)
    zeq = bfield.find_magequator_z(*xyz[0][:3], ti[0])
    dz_last = xyz[0][2] - zeq
    idx = 0 #passed to the second recursive call below if this for loop is not entered because idx_jump >= len(xyz)
    for idx in range(idx_jump, len(xyz), idx_jump):
        zeq = bfield.find_magequator_z(*xyz[idx][:3], ti[idx])
        dz = xyz[idx][2] - zeq
        if dz_last * time_direction < 0 and dz * time_direction >= 0:
            # forward in time: if dz_last is negative, dz is positive
            if idx_jump == 1:
                return count_index + idx
            else:
                #print(" somewhere between", count_index + idx - idx_jump, "and", count_index + idx)
                return converge_on_first_crossing_idx(xyz[idx - idx_jump:], ti[idx - idx_jump:], bfield, idx_jump//2, count_index + idx - idx_jump, time_direction=time_direction)
        else:
            dz_last = dz

    if idx_jump > 1:
        return converge_on_first_crossing_idx(xyz[idx:], ti[idx:], bfield, idx_jump // 2, count_index + idx, time_direction=time_direction)
    else:
        return -1

def get_bouncecapped_trajectory(bfield, tocap_time, tocap_pt, tsperorbit, idx_left_equator_safe = 0, reverse=False):
    """
    cap the trajectory to the soonest equatorial crossing at 0 bounce phase (negative to positive dZ relative to the magnetic equator)
    if pt begins at dZ=0 and idx_left_equator_safe = 0, this function will return the first point in the trajectory
     therefore set idx_left_equator_safe to a value s.t. the particle will not trip a bounce detection near the very beginning of its trajectory
     this is a problem when the resolution of the magnetic equator is poor
    input must be numpy arrays
    returns floats, lists
    """
    global warned_grid_resolution

    if bfield.B_grid:
        dz_bounce = np.max(tocap_pt[:,2]) - np.min(tocap_pt[:,2])
        dz_fieldres = bfield.field_dz
        if dz_bounce < 0.5*dz_fieldres and not warned_grid_resolution:
            print("Warning: bounce orbit scale height comparable to field dz, may not be able to detect a bounce")
            warned_grid_resolution = True

    # #diagnosis code:###########
    # print("Finding magnetic equator numerically...")
    # pt_plot = np.array(tocap_pt)
    # t_plot = np.array(tocap_time)
    # zeq_list = []
    # idx_skip = tsperorbit
    # for idx in range(0, len(pt_plot), idx_skip):
    #     zeq_list.append(bfield.find_magequator_z(*pt_plot[idx][:3], t_plot[idx]))
    # zeq_list = np.array(zeq_list)
    # #
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(1)
    # ax.plot(pt_plot[:,1][::idx_skip], zeq_list, color='black', ls=':') #equator
    # ax.scatter([pt_plot[:,1][0]], [pt_plot[:,2][0]],marker='x', s=10)
    # #ax.scatter(pt_plot[:,1], pt_plot[:,2],marker='.', s=5)
    # #ax.scatter([capped_pt[-1][0]], [capped_pt[-1][2]],marker='x', color='black')
    # #ax.plot(pt_plot[:, 1], pt_plot[:, 2],color='black',lw=0.3)
    # ax.plot(pt_plot[:, 1][idx_left_equator_safe:], pt_plot[:, 2][idx_left_equator_safe:], color='red', lw=0.3)
    # plt.show()
    # ###########################

    if reverse:
        time_direction = -1
    else:
        time_direction = 1
    idx_x = converge_on_first_crossing_idx(tocap_pt[idx_left_equator_safe:], tocap_time[idx_left_equator_safe:], bfield, tsperorbit, time_direction=time_direction)
    if idx_x < 0:
        print("Error: could not detect a whole bounce")
        sys.exit()
    idx_x = idx_x + idx_left_equator_safe
    capped_pt = list(tocap_pt[:idx_x])
    capped_times = list(tocap_time[:idx_x])

    # interpolate the (future) state vector at the equator:
    dpt = tocap_pt[idx_x] - tocap_pt[idx_x - 1]
    dz = dpt[2]
    frac_dz = (bfield.origin_MAG[2] - tocap_pt[idx_x - 1][2]) / dz
    if frac_dz == 0:
        # we are already at the equator at idx_x - 1 of the trajectory
        pt_eq = tocap_pt[idx_x - 1]
        t_eq = tocap_time[idx_x - 1]
    else:
        pt_eq = tocap_pt[idx_x - 1] + dpt * frac_dz

        # get the time between index 0 (final particle position) and the equatorial crossing:
        dt = (tocap_time[idx_x] - tocap_time[idx_x - 1])
        t_eq = tocap_time[idx_x - 1] + dt * frac_dz

        capped_pt.append(pt_eq)
        capped_times.append(t_eq)

    return t_eq, pt_eq, capped_times, capped_pt

def extract_GC_only(particle, bfield, existing_skipeveryn=1):
    if (not particle.storetrack):
        print("Error: cannot calculate the GC trajectory because the particle object has no track stored")
        print("", "skipping...")
        return 2

    t0 = 0.
    tsperorbit = particle.recommended_tsperorbit//existing_skipeveryn
    x0_GC_MAG = bfield.calculate_initial_GC(particle.init_L, particle.iphase_drift)
    B_GC = bfield.getBE(*x0_GC_MAG, t0)[:3]
    p0 = particle.calculate_initial_momentum(B_GC)

    # initial state vector of the GC:
    Y0_GC = np.hstack((x0_GC_MAG, p0))

    # calculate the gyroradius assuming the Lorentz force is constant within a gyroorbit:
    rg0 = calc_rg(Y0_GC, bfield, particle)  # relativistically correct
    particle.gc_times, particle.gc_pos, _, _ = get_GC_from_track(rg0, bfield, particle, tsperorbit, get_K=False)
    return 1

def solve_trajectory(particle, bfield, delta_az_solve, duration_solve = -1, findK0 = False, storegc = False, reverse=False):
    global pusher
    global warned_grid_resolution
    warned_grid_resolution = False  # turn off this warning after one instance
    planet = earth.Earth(bfield.year_dec)

    if (not particle.storetrack) and storegc:
        print("Error: cannot calculate the GC trajectory because the particle object elects not to store its track")
        print("","skipping...")
        return 2


    exect0 = time.perf_counter()
    MeV2J = pt_tools.constants.MeV2J

    #SI unit quantities:
    mu = particle.init_mu
    aeq = particle.init_alpha
    L = particle.init_L

    t0 = 0.

    tsperorbit = particle.recommended_tsperorbit
    
    #get initial position of the GC based on L:
    x0_GC_MAG = bfield.calculate_initial_GC(particle.init_L, particle.iphase_drift)

    #get a possible initial momentum vector on the magnetic equator,
    #   multiple solutions because adiabatic invariants say nothing about phase:
    B_GC = bfield.getBE(*x0_GC_MAG, t0)[:3]

    p0 = particle.calculate_initial_momentum(B_GC)

    #initial state vector of the GC:
    Y0_GC = np.hstack((x0_GC_MAG, p0))

    #calculate initial velocity for the momentum decided on:
    p0mag = np.linalg.norm(p0)
    gamma = sqrt(1 + (p0mag/(particle.m0 * c))**2)
    v0 = p0/(gamma*particle.m0)

    #calculate the gyroradius assuming the Lorentz force is constant within a gyroorbit:
    rg0 = calc_rg(Y0_GC, bfield, particle) #relativistically correct

    #initial condition x0:
    x0 = particle.calculate_initial_position(x0_GC_MAG, rg0)

    #kinetic energy:
    E0_J = (gamma - 1)*particle.m0*(c**2)
    E0 = E0_J /MeV2J #KE energy in MeV
    particle.muenKalphaL[0,1] = E0

    #energy = particle.derive_KE0(bfield)

    #calculate the mirror field strength:
    Be = np.linalg.norm(B_GC)
    Bm_approx = Be / (sin(aeq)**2 )
    I_approx = L * RE* pt_tools.approx_Ya(aeq)
    K_approx = np.power(Bm_approx/G2T,0.5) * I_approx / RE
    particle.muenKalphaL[0,2] = K_approx #includes dipole approximation

    #calculate (a dipole estimate of) the time for one bounce period
    tb_est_dipole = tb_estimate(np.linalg.norm(x0_GC_MAG - bfield.origin_MAG), np.linalg.norm(v0), aeq)

    t0_init = t0
    x0_init = x0
    p0_init = p0

    iphase_gyro = particle.iphase_gyro
    iphase_bounce = particle.iphase_bounce
    iphase_drift = particle.iphase_drift

    if reverse:
        p0_init = -1 * p0_init
        #particle should bounce 1-phase_b backwards in time to the simulation start position
        iphase_bounce = 1 - iphase_bounce
        pusher = pt_pushers.boris_bkwd


    #print some information:
    print("Particle's initial properties:")
    print(" energy          = {:.3f}MeV".format(E0))
    print(" mu              = {:.3f}MeV/G".format(mu*G2T/MeV2J))
    print(" eq. pitch angle = {:.3f}d".format(aeq* 180/pi))
    print(" L               = {:.3f}".format(L))
    print(" gyrophase       = {:.3f}d".format(iphase_gyro))
    print(" bounce phase    = {:.3f}".format(iphase_bounce))
    print(" drift phase     = {:.3f}d".format(iphase_drift))
    print(" lorentz factor  = {:.3f}".format(gamma))
    print(" speed           = {:.3f}c".format(np.linalg.norm(v0)/c))
    print(" x0              = {:.3f}, {:.3f}, {:.3f} RE".format(x0[0]/RE, x0[1]/RE, x0[2]/RE))
    print(" p0              = {:.3e}, {:.3e}, {:.3e} kgm/s".format(p0[0], p0[1], p0[2]))
    print(" bounce time     ~ {:.3f}s".format(tb_est_dipole))
    print("#")

    #
    #   Place the particle on the equator:
    #
    particle.update(t0_init, [*x0_init, *p0_init])

    #
    #   Track the particle to some phase along the bounce in a static field at t0:
    #
    print("Initialising fraction {:.3f} along the first bounce...".format(iphase_bounce))

    #modify the particle to keep the track for our starting calculatings:
    storetrack_setting = particle.storetrack
    particle.update = particle.update_keep

    #use a static field at t0 to do this: we want to find where the particle would be at t0:
    t1, x1, p1 = pusher(particle, bfield, 1.0/0.9 * tb_est_dipole, tsperorbit, t_limit_exact = False)#, freezefield = t0)

    #cut the trajectory down to exactly one bounce:
    # assume the particle has left the equator from a position index of tsperorbit
    t_eq, pt_eq, capped_times, capped_pt = get_bouncecapped_trajectory(bfield, np.array(particle.times), np.array(particle.pt), tsperorbit=tsperorbit, idx_left_equator_safe=tsperorbit, reverse=reverse)
    particle.times = list(capped_times)
    particle.pt = list(capped_pt)
    if t_eq < 0:
        print("Skipping: a bounce could not be numerically detected")
        return 2

    #we now have an accurate, numerically-derived approximation of bounce time:
    tb_est = particle.times[-1]
    dt_solve_increment = tb_est

    #find time at which the particle is iphase_bounce along the bounce
    start_time = tb_est * iphase_bounce
    start_idx = np.argmin(np.abs(particle.times - start_time))
    #
    # #just keep the section of trajectory after this time:
    # t0_shift = particle.times[start_idx]
    # particle.pt = particle.pt[start_idx:]
    # particle.times = particle.times[start_idx:]
    #
    # #shift the particle times back so it started from t0:
    # for idx in range(len(particle.times)):
    #     particle.times[idx] -= t0_shift
    #
    #
    # # this approach seems more efficient, but it is technically incorrect,
    # # in a time-dependent field, there is no guarantee the motion we just solved will be repeated at a different time
    # # therefore, we have to start again...
    #derive the starting position from this partial bounce solution:
    particle.pt = [particle.pt[start_idx]]
    particle.times = [t0_init]


    #restore particle properties if necessary:
    if not storetrack_setting:
        particle.update = particle.update_discard
        # particle.times = [particle.times[-1]]
        # particle.pt = [particle.pt[-1]]

    #
    #
    #   bounce tracking finished

    height = planet.get_height_above_surface(particle.pt[0][:3])
    if not height:
        print("Skipping because the particle was initialized below Earth's surface")
        return 2

    #
    #   Option to solve for a specific duration:
    if duration_solve > 0:
        #if duration_solve < particle.times[-1]: #we already solved for long enough:
        #    while particle.times[-1] > duration_solve:
        #        particle.pop_track()
        dt_solve_increment = duration_solve# - particle.times[-1]


    #
    #   SOLVE THE (REMAINING) PARTICLE TRAJECTORY:
    #
    print("Tracking...")
    delta_az = 0
    while abs(delta_az) <= delta_az_solve and bfield.range_adequate:
        t1, x1, p1 = pusher(particle, bfield, dt_solve_increment, tsperorbit)
        delta_az += angle_between(x0, x1)
        x0 = x1
        nbounces_stored_approx = (t1 - t0) / tb_est
        if delta_az_solve > 0: print("","{:.2f}%".format(100*delta_az/(delta_az_solve))) #only works if dt_solve_increment << drift orbit time

    #if we wanted one drift orbit exactly
    # we now are at one drift orbit +-dt_solve_increment
    # instead, we could cap the orbit to one drift here...

    #
    #   FIND THE VALUE OF THE SECOND INVARIANT NEAR THE BEGINNING OF THE TRAJECTORY:
    #
    if findK0 or storegc:
        bfield_range_ok_prior = bfield.range_adequate

        if particle.storetrack and bfield_range_ok_prior:
            stored_len = len(particle.times)
        elif particle.storetrack:
            #we need to begin tracking again for K, but back up the failed track and restore it after:
            stored_times = copy.deepcopy(particle.times)
            stored_pt = copy.deepcopy(particle.pt)

            #re-initialise the particle from x0, p0
            particle.times = [t0_init]
            particle.pt = [[*x0_init, *p0_init]]
            nbounces_stored_approx = 0
            bfield.range_adequate = True       
        else:
            #re-initialise the particle from x0, p0, but track the trajectory now
            particle.update(t0_init, [*x0_init, *p0_init])
            particle.update = particle.update_keep
            nbounces_stored_approx = 0
            bfield.range_adequate = True

        if nbounces_stored_approx < 2:
            print("Tracking an extra {:.2f} bounces for K calculation...".format(2 - nbounces_stored_approx))
            t1_extra, x1_extra, p1_extra = pusher(particle, bfield, (2 - nbounces_stored_approx)*tb_est, tsperorbit)
        
        if not bfield.range_adequate:
            print("Error: cannot find K near the beginning of the trajectory because the particle goes out of range")
        else:
            print("Calculating initial K...")
            _, _, K_, _ = get_GC_from_track(rg0, bfield, particle, tsperorbit, tlimit = 2*tb_est)
            particle.muenKalphaL[0,2] = K_

        #using the same logic as above, restore the particle track if needed:
        if particle.storetrack and bfield_range_ok_prior:
            while particle.pop_track(stored_len): continue
            if storegc:
                print("Calculating GC...")
                particle.gc_times, particle.gc_pos, _, _ = get_GC_from_track(rg0, bfield, particle, tsperorbit, get_K=False)
        elif particle.storetrack:
            particle.times = stored_times
            particle.pt = stored_pt
            if storegc:
                print("Calculating GC...")
                particle.gc_times, particle.gc_pos, _, _ = get_GC_from_track(rg0, bfield, particle, tsperorbit, get_K=False)
        else:
            particle.update = particle.update_discard
            particle.times = [t1]
            particle.pt = [[*x1, *p1]]

        bfield.range_adequate = bfield_range_ok_prior


    if not bfield.range_adequate:
        print("","tracking failed - particle went outside of field domain in time or space")
        return 2

    exect1 = time.perf_counter()
    print("","tracked with execution time of {:.2f}s".format(exect1 - exect0))
    #azimuth is not tracked properly if dt_solve_increment > time for 1 drift orbit
    #print("","change in azimuth: {:.3f}d".format(delta_az * 180/pi))
    print("#")

    return 1


def solve_trajectory_bounce(particle, bfield, dummy, findK0 = False, storegc = False, reverse = False):
    """solve for one bounce"""
    delta_az_solve = 0
    return solve_trajectory(particle, bfield, delta_az_solve, duration_solve = -1, findK0 = findK0, storegc = storegc, reverse = reverse)

def solve_trajectory_drift(particle, bfield, dummy, findK0 = False, storegc = False, n_solve = 1, reverse = False):
    """solve for n drifts"""
    delta_az_solve = n_solve*2*pi
    return solve_trajectory(particle, bfield, delta_az_solve, duration_solve = -1, findK0 = findK0, storegc = storegc, reverse = reverse)

def solve_trajectory_time(particle, bfield, duration_solve, findK0 = False, storegc = False, reverse = False):
    """solve for a fixed duration"""
    delta_az_solve = 0
    return solve_trajectory(particle, bfield, delta_az_solve, duration_solve, findK0 = findK0, storegc = storegc, reverse = reverse)

def derive_invariants(particle, bfield, reverse=False): #called from pt_run
    """
    method to find K:
    - do another ~2 bounces from t1
    - get a moving average position based on the timestep and extract the GC
    method to find bounce phase:
    - estimate the bounce period from the GC trajectory (time between peaks in field)
    - calculate the time between x1 and the next equatorial crossing
    - 1 minus this ratio vs. bounce period
    method for calculating the other invariants:
    - evaluate at the first equatorial crossing calculated for the above quantities
    - this is in the future trajectory of the particle
    """
    global pusher

    invariants = [-1, -1, -1, -1, -1, -1, -1, -1] #fill values
    if not len(particle.pt):
        print("Error: cannot derive invariants from a particle with no state vector")
        return invariants

    #get some physical quantities:
    t1 = particle.times[-1] #use this to query the field
    pt1_original = np.array(particle.pt[-1])
    pt1_fwd = np.array(pt1_original)
    if reverse:
        pusher = pt_pushers.boris_bkwd
        #reverse momentum to calculate GC, etc.
        pt1_fwd[3:] = -1 * pt1_fwd[3:] #only used to calculate tb_est

    x1 = pt1_fwd[:3]
    p1 = pt1_fwd[3:]

    p1mag = np.linalg.norm(p1)
    gamma = sqrt(1 + (p1mag/(particle.m0 * c))**2)
    v1 = p1/(gamma*particle.m0)

    #get derivative of the state vector (velocity, force) from Lorentz force at t1:
    dY0dt = dYdt(t1, pt1_fwd, particle, bfield) #first argument doesn't matter
    force = dY0dt[3:]
    #calculate gyrocentre:
    # calculate gyroradius rg
    rg = calc_rg(pt1_fwd, bfield, particle, t1)
    # go rg in the direction of the Lorentz force to get to the GC:
    step = rg * force/np.linalg.norm(force)
    x0_GC_MAG = x1 + step

    #local magnetic field at GC:
    bl = bfield.getBE(*x0_GC_MAG, t1)[:3]
    #local pitch angle:
    a_ = angle_between(bl, p1)
    bl = np.linalg.norm(bl)

    #
    # FIND L (guess for estimating tb)
    #
    L_dip = bfield.get_L(x1)

    #
    # FIND EQUATORIAL PITCH ANGLE (approximation for estimating tb)
    #
    be = bfield.getBE(L_dip * pt_tools.constants.RE, 0, 0, t1)[:3]
    # symmetrical for a dipole, HOWEVER, this is an incorrect assumption when B field is interpolated!!
    be = np.linalg.norm(be)
    bebl = min([1., be/bl])
    aeq = asin(sqrt(bebl * (sin(a_)**2)))
    #the above calculations are at t1 and will be discarded


    #estimate tb:
    v1mag = np.linalg.norm(v1)
    tb_est = tb_estimate(L_dip * pt_tools.constants.RE, v1mag, aeq)

    #
    # FIND K
    #
    #make a copy of the particle track:
    stored_times = copy.deepcopy(particle.times)
    stored_pt = copy.deepcopy(particle.pt)

    #delete the entire track and initialise from the last state vector:
    particle.times = [t1]
    particle.pt = [[*pt1_original]]
    #modify the particle to keep the track:
    func_ptr_original = particle.update
    particle.update = particle.update_keep

    #visit conjugate mirror points over two bounce paths with the field frozen at t1:
    tsperorbit = particle.recommended_tsperorbit
    print("Solving for just over two bounce orbits from t1 in a static field at t1...")
    t2, x2, p2 = pusher(particle, bfield, 2.5/0.9 * tb_est, tsperorbit, t_limit_exact = False, freezefield = t1)

    print("Calculating invariants...")
    _, _, K_, tb_est = get_GC_from_track(rg, bfield, particle, tsperorbit, freezefield = t1)

    #store the track we just calculated separately:
    extended_t = np.array(particle.times)
    extended_pt = np.array(particle.pt)

    #restore original particle properties:
    particle.times = copy.deepcopy(stored_times)
    particle.pt = copy.deepcopy(stored_pt)
    particle.update = func_ptr_original


    #return if we could not properly detect a bounce:
    if tb_est < 0:
        return invariants
    #else use future time and xp to calculate the other invariants:

    #
    # FIND BOUNCE PHASE
    #
    #we have an accuarate estimate of the bounce time tb_est (from field peak intervals along the GC)
    # use it to estimate bounce phase at the particle, get next bounce phase = 0 time t_eq:
    t_eq, pt_eq, _, _ = get_bouncecapped_trajectory(bfield, extended_t, extended_pt, tsperorbit=tsperorbit, reverse=reverse)
    if t_eq < 0:
        return invariants

    dt_toeq = t_eq - extended_t[0]

    if reverse:
        phase_b = dt_toeq / tb_est
    else:
        phase_b = 1 - dt_toeq / tb_est

    print("","particle was followed for an extra {:.5f}s in B, E at t={:.5f}s".format(dt_toeq, t1))

    #RE-EVALUATE ALL INVARIANTS:
    # # use x, p of the particle at the future magnetic equatorial crossing if the particle bounces
    # # otherwise, just use x, p at the end of the original trajectory
    # # check if the particle is equatorial or not according to our threshold:
    # if aeq > nonmirror_threshold * pi / 180:
    #     # the particle will only be bounced backward at the start of the reverse simulation if False
    #     # therefore, if this is True, store the invariants/phases at t1, not from the future traj.
    #     x1 = pt_eq[:3]
    #     p1 = pt_eq[3:]
    # else:
    #     x1 = pt_eq[:3]
    #     p1 = pt_eq[3:]
    # actually there is no need for this - if the particle is nonmirroring we will have continue the trajectory just fractionally past the end of the simulation
    # but the drift phase will have barely changed, and since the particle would be initialized at the equator in a reverse sim., it is a better place to evaluate gyrophase, aeq
    if reverse:
        # we traced backward in time to the equator, now consider the forward momentum
        # because we are deriving quantities to help us re-initialize a simulation
        pt_eq[3:] = -1 * pt_eq[3:]
    x1 = pt_eq[:3] #MAG
    p1 = pt_eq[3:] #MAG

    # recalculate physical quantities:
    p1mag = np.linalg.norm(p1)
    gamma = sqrt(1 + (p1mag / (particle.m0 * c)) ** 2)

    # get derivative of the state vector (velocity, force) from Lorentz force at t1:
    dY0dt = dYdt(t1, pt_eq, particle, bfield)  # first argument doesn't matter
    force = dY0dt[3:]
    # calculate gyrocentre:
    # calculate gyroradius rg
    rg = calc_rg(pt_eq, bfield, particle, t1)
    # go rg in the direction of the Lorentz force to get to the GC:
    step = rg * force / np.linalg.norm(force)
    xeq_GC_MAG = x1 + step

    # local magnetic field at GC:
    bl = bfield.getBE(*xeq_GC_MAG, t1)[:3]
    # local pitch angle:
    a_ = angle_between(bl, p1)
    bl = np.linalg.norm(bl)

    #since we followed the particle to the magnetic equator, this is equal to the equatorial pitch angle:
    aeq = a_

    #
    # FIND L
    #
    L_dip = bfield.get_L(xeq_GC_MAG)

    # now we can continue from the previous calculation and find the other invariants:
    #
    # FIND mu:
    #
    mu = ((np.linalg.norm(p1) * sin(a_)) ** 2) / (2 * particle.m0 * bl)
    # kinetic energy:
    E1_J = (gamma - 1) * particle.m0 * (c ** 2)
    E1 = E1_J / pt_tools.constants.MeV2J  # KE energy in MeV

    #
    # FIND drift phase:
    #
    # in calculate_initial_GC(...), the vector R_gc, 0, 0 in the dipole frame has zero rotation, i.e. drift direction vector is [1, 0, 0]
    phase_d = bfield.get_aclockw_angle_around_dipole_z(xeq_GC_MAG)

    #
    # FIND gyrophase:
    #
    # reverse the order of steps in calculate_initial_position(...)
    steprr = x1 - xeq_GC_MAG #step is from GC to particle now
    # rotate steprr back by drift phase:
    stepr = np.array(pt_tools.coord_car_rz(*steprr, -1 * np.radians(phase_d)))
    # find the phase of stepr, this is the gyrophase:
    phase_g = bfield.get_aclockw_angle_around_dipole_z(stepr + bfield.origin_MAG)

    invariants = [mu, E1, K_, aeq, L_dip, phase_g/360, phase_b, phase_d/360]
    return invariants