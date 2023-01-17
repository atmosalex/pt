import sys
import numpy as np
from scipy.integrate import ode
from math import cos, sin, tan, acos, asin, atan, atan2, sqrt, pi
from datetime import datetime
import argparse
import pt_tools
import time
import copy

def tb_estimate(R_gc, vmag, aeq):
    """
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
    
    B1, B2, B3, E1, E2, E3 = bfield.getBE(x1, x2, x3, t) #B in the observer frame

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
    """

    #calculate the lorentz factor given the total momentum:
    p0mag = np.linalg.norm(Y0[3:])
    gamma = sqrt(1 + (p0mag/(particle.m0 * pt_tools.constants.c))**2)

    dY0dt = dYdt(t, Y0, particle, bfield) #first argument doesn't matter
    
    v2 = np.linalg.norm(dY0dt[:2])**2 #magnitude of the x,y velocity
    F1 = np.linalg.norm(dY0dt[3:]) #Lorentz force directed toward GC

    return abs(gamma*particle.m0*(v2)/F1)


def get_GC_from_track(rg0, bfield, particle, tsperorbit, tlimit = -1, freezefield = -1): #freezefield not implemented yet
    """get the moving average gyrocentre position over each gyration"""
    def moving_average(a, n) :
        ret = np.cumsum(a, dtype=float, axis = 0)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n

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
    track_gyrocentre = moving_average(track_gyrocentre, tsperorbit)
    track_gyrocentre_time = moving_average(track_gyrocentre_time, tsperorbit)

    #find visits to the peak magnetic field along the gyrocentre:
    Bmax = 0
    idx_Bmax = 0
    Bmax_peaks = []
    idx_Bmax_peaks = []
    gradB = 0
    npeaks = 0
    foundnewpeak = False
    for idx in range(len(track_gyrocentre)-1):
        t_ = get_field_time(track_gyrocentre_time[idx])

        bx0, by0, bz0 = bfield.getBE(*track_gyrocentre[idx], t_)[:3]
        Bmag = sqrt(pow(bx0,2) + pow(by0,2) + pow(bz0,2))
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

    if len(idx_Bmax_peaks) <= 3:
        print("Error: could not extract a whole bounce orbit, try passing a longer trajectory")
        return list(track_gyrocentre_time), list(track_gyrocentre), -1

    #isolate one GC bounce:
    track_gyrocentre_bounce = track_gyrocentre[idx_Bmax_peaks[0]:idx_Bmax_peaks[2]+1]
    track_gyrocentre_bounce_time = track_gyrocentre_time[idx_Bmax_peaks[0]:idx_Bmax_peaks[2]+1]

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
    return list(track_gyrocentre_time), list(track_gyrocentre), K


def cut_exact_bounce(particle, bfield, tsperorbit):
    #for bouncing particles, reduce the amount of saved trajectory to a complete number of bounces by:
    # a) delete the last part of the trajcetory until just before the last equatorial crossing
    # b) solve forward in time until the equator is just reached
    
    z_equator = 0
    track_cap_idx = particle.cap_to_bounce(z_equator = z_equator)
    if track_cap_idx == 0:
        return 0
    particle.pt = particle.pt[:track_cap_idx] #remove after double timestep
    particle.times = particle.times[:track_cap_idx] #remove after double timestep


    #calculate the momentum of the particle after one bounce period:
    z1_aftercross = particle.pt[-1][2]
    z1_beforecross = particle.pt[-2][2]
    z1_delta = z1_aftercross - z1_beforecross

    #in terms of z, the fraction of the step towards the equator we need to get exactly one bounce from pt[-2]:
    frac_step = abs((z_equator - z1_beforecross)/z1_delta) #very approximate

    #integrate forward in time until the equator is reached (approximately):
    time_beforexing = particle.times[-2]
    pt_beforexing = particle.pt[-2][:]

    #r = ode(rel).set_integrator('dop853', nsteps=9999999)
    #r.set_initial_value(pt_beforexing, time_beforexing).set_f_params(particle.q, particle.m0, bfield.getB)

    #calculate the extra timestep required to reach the equator for exactly one bounce:
    dt = (particle.times[-1] - particle.times[-2])*frac_step
    #r.integrate(r.t+dt)

    #remove last element from particle trajectory and times:
    if not particle.pop_track():
        print("Error: track is empty")
        return 0

    solver_boris(particle, bfield, dt, tsperorbit)

    return 1

def solver_boris(particle, bfield, dt_solve, tsperorbit, t_limit_exact = True, freezefield = -1):
    """
    Uses boris algorithm to solve the trajectory of the particle
    note: the initial state vector must be stored in the particle object at particle.pt
    """

    def get_field_time_freeze(time):
        return freezefield
    def get_field_time_particle(time):
        return time
    if freezefield >= 0:
        get_field_time = get_field_time_freeze
    else:
        get_field_time = get_field_time_particle

    t0 = particle.times[-1]
    x0 = particle.pt[-1][:3]
    p0 = particle.pt[-1][3:]
    t1_aim = t0 + dt_solve

    #just in case dt_solve = 0:
    t1 = t0
    x1 = x0
    p1 = p0

    p0mag = np.linalg.norm(p0)
    gamma = sqrt(1 + (p0mag/(particle.m0 * c))**2)
    gamma_plus = gamma

    #CALCULATE TIME STEP ZERO
    bx0, by0, bz0 = bfield.getBE(*x0, t0)[:3]
    Bmag = sqrt(pow(bx0,2) + pow(by0,2) + pow(bz0,2))
    dtp = 2 * np.pi * particle.m0*gamma / (tsperorbit * abs(particle.q)*np.linalg.norm(Bmag))

    if dt_solve < dtp:
        dtp = dt_solve
        t_limit_exact = False #won't enter another recursive level

    while t1 < t1_aim * bfield.range_adequate:
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        #
        #  the code inside this while loop is based on Ravindra Desai's Boris solver
        #
        #
        #ensure t0, x0, p0 are one timestep behind:
        t0 = t1
        x0 = x1
        p0 = p1
        #
        #t_field = get_field_time(t1 + 0.5*dtp) #time to query the fields at
        t_field = get_field_time(t1 + 0.5*dtp) #time to query the fields at
        #
        t1 = t1 + dtp #increase by time step
        #
        qdto2m = (particle.q/particle.m0) * (dtp/2) #to be used later in equation 10 to find u minus (um)
        #
        # update particle positions
        uxn = particle.pt[-1][3]/particle.m0
        uyn = particle.pt[-1][4]/particle.m0
        uzn = particle.pt[-1][5]/particle.m0
        xnh = particle.pt[-1][0] + (uxn/gamma_plus) * 0.5 * dtp #the middle term is simply the average velocity in the x-component as defined for the Boris method
        ynh  = particle.pt[-1][1] + (uyn/gamma_plus) * 0.5 * dtp
        znh  = particle.pt[-1][2] + (uzn/gamma_plus) * 0.5 * dtp
        #
        #
        # B, E field call
        bx0, by0, bz0, qEx, qEy, qEz = bfield.getBE(xnh, ynh, znh, t_field) #interpolate the B-field at the new positions
        Bmag = sqrt(pow(bx0,2) + pow(by0,2) + pow(bz0,2)) #magnitude of B-field at new positions
        #print(xnh, ynh, znh, t_field, bx0, by0, bz0, qEx, qEy, qEz)
        #
        # Equation 10 - First half of the electric field acceleration. m means minus
        uxm = uxn + qdto2m*qEx
        uym = uyn + qdto2m*qEy 
        uzm = uzn + qdto2m*qEz
        #
        um_mag = pow((uxm*uxm + uym*uym + uzm*uzm),0.5)
        #
        gamma_minus = sqrt(1 + (um_mag/c)**2) #gamma_minus as per definition
        #
        # first half of the rotation, v'  = v- + v- x t, v- = vn in absence of E-field.
        tx = (qdto2m * bx0) / gamma_minus   #!bxc(ix,iy,iz) calculating the t components as defined
        ty = (qdto2m * by0) / gamma_minus   #!byc(ix,iy,iz)
        tz = (qdto2m * bz0) / gamma_minus   #!bzc(ix,iy,iz)
        tmag  = sqrt(tx**2 + ty**2 + tz**2) 
        sx = 2 * tx / (1 + tmag**2) #calculating s components as defined
        sy = 2 * ty / (1 + tmag**2) 
        sz = 2 * tz / (1 + tmag**2) 
        #
        # Eq. 11 middle terms
        utx = uxm + (uym * tz - uzm * ty) 
        uty = uym + (uzm * tx - uxm * tz) 
        utz = uzm + (uxm * ty - uym * tx) 
        #
        # second half of the rotation, v+ = v- + v' x s, v+ = vn+1 in absence of E-field
        # therefore vn+1 = vn + [(vn + vn x t) x s]
        #
        # Eq. 11 end term x s    
        upx = uxm + (uty * sz - utz * sy) 
        upy = uym + (utz * sx - utx * sz) 
        upz = uzm + (utx * sy - uty * sx) 
        #
        # Eq. 12 - second half of the electric field acceleration
        uxn = upx + qdto2m*qEx 
        uyn = upy + qdto2m*qEy 
        uzn = upz + qdto2m*qEz 
        up_mag = (uxn*uxn + uyn*uyn + uzn*uzn)**0.5
        #
        gamma_plus = sqrt(1+(up_mag/c)**2); # gamma_minus. Calculating the new gamma plus
        vmag = sqrt((uxn/gamma_plus)**2 + (uyn/gamma_plus)**2 + (uzn/gamma_plus)**2)
        #
        # new particle positions to be used in the next iteration.
        x1 = [xnh + (uxn/gamma_plus) * 0.5 * dtp, 
            ynh + (uyn/gamma_plus) * 0.5 * dtp,
            znh + (uzn/gamma_plus) * 0.5 * dtp]
        p1 = [uxn * particle.m0, uyn * particle.m0, uzn * particle.m0]
        particle.update(t1, [*x1, *p1])
        #
        #timestep
        #dtp = 1/(abs(particle.q)*Bmag/(particle.m0*gamma)) / tsperorbit
        dtp = 2 * np.pi * particle.m0*gamma / (tsperorbit * abs(particle.q)*np.linalg.norm(Bmag))
        #
        #
        #
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    if t_limit_exact and bfield.range_adequate: #go back one timestep, then solve for the remaining fraction of a timestep
        particle.update(t0, [*x0, *p0]) #if storetrack = False, track length is still 1, if storetrack = True, track length is +1
        particle.pop_track(1) #if storetrack = False, track length is still 1, if storetrack = True, track length is +0
        particle.pop_track(1) #if storetrack = False, track length is still 1, if storetrack = True, track length is -1
        dt_remaining = t1_aim - t1
        t1, x1, p1 = solver_boris(particle, bfield, dt_remaining, tsperorbit, t_limit_exact = False)

    return t1, x1, p1


def derive_invariants(particle, bfield):
    if not len(particle.pt):
        print("Error: cannot derive invariants from a particle with no state vector")
        return [-1, -1, -1, -1]

    #get some physical quantities:
    t1 = particle.times[-1] #use this to query the field
    x1 = np.array(particle.pt[-1][:3])
    p1 = np.array(particle.pt[-1][3:])
    p1mag = np.linalg.norm(p1)
    gamma = sqrt(1 + (p1mag/(particle.m0 * c))**2)
    v1 = p1/(gamma*particle.m0)

    #get derivative of the state vector (velocity, force) from Lorentz force at t1:
    dY0dt = dYdt(t1, particle.pt[-1], particle, bfield) #first argument doesn't matter
    force = dY0dt[3:]
    #calculate gyrocentre:
    # calculate gyroradius rg
    rg = calc_rg(particle.pt[-1], bfield, particle, t1)
    # go rg in the direction of the Lorentz force to get to the GC:
    step = rg * force/np.linalg.norm(force)
    x0_GC = x1 + step

    #local magnetic field at GC:
    bl = bfield.getBE(*x0_GC, t1)[:3]
    #local pitch angle:
    a_ = angle_between(bl, p1)
    bl = np.linalg.norm(bl)

    #
    # FIND L
    #
    r_ = np.linalg.norm(x1)
    MLAT = atan2(x1[2], sqrt(x1[0]**2 + x1[1]**2 ))
    L_dip = bfield.get_L(r_/pt_tools.constants.RE, MLAT)

    #
    # FIND EQUATORIAL PITCH ANGLE
    #
    be = bfield.getBE(L_dip * pt_tools.constants.RE, 0, 0, t1)[:3] #symmetrical in the MAG frame
    be = np.linalg.norm(be)
    bebl = min([1., be/bl])
    aeq = asin(sqrt(bebl * (sin(a_)**2)))


    #
    # FIND K
    #
    #make a copy of the particle track:
    stored_times = copy.deepcopy(particle.times)
    stored_pt = copy.deepcopy(particle.pt)

    #delete the entire track amd initialise from the last state vector:
    particle.times = [t1]
    particle.pt = [[*x1, *p1]]
    #modify the particle to keep the track:
    func_ptr_original = particle.update
    particle.update = particle.update_keep

    #estimate tb:
    v1mag = np.linalg.norm(v1)
    tb_est = tb_estimate(L_dip * pt_tools.constants.RE, v1mag, aeq)

    #visit conjugate mirror points over two bounce paths with the field frozen at t1:
    tsperorbit = particle.recommended_tsperorbit
    print("Solving for just over two bounce orbits from t1 in a static field at t1...")
    t2, x2, p2 = solver_boris(particle, bfield, 2.5/0.9 * tb_est, tsperorbit, t_limit_exact = False, freezefield = t1)
    print("Calculating invariant K...")
    K_ = get_GC_from_track(rg, bfield, particle, tsperorbit, freezefield = t1)[2]

    #restore particle properties:
    particle.times = copy.deepcopy(stored_times)
    particle.pt = copy.deepcopy(stored_pt)
    particle.update = func_ptr_original

    #
    # FIND mu:
    #
    mu = ((np.linalg.norm(p1) * sin(a_))**2)/(2 * particle.m0 * bl)
    #kinetic energy:
    E1_J = (gamma - 1)*particle.m0*(c**2)
    E1 = E1_J /pt_tools.constants.MeV2J #KE energy in MeV

    return [mu, E1, K_, aeq, L_dip]


def solve_trajectory(particle, bfield, delta_az_solve, duration_solve = -1, findK0 = False, storegc = False):

    if (not particle.storetrack) and storegc:
        print("Error: cannot calculate the GC trajectory because the particle object elects not to store its track")
        print("","skipping...")
        return 2

    exect0 = time.perf_counter()
    global RE, c, G2T
    MeV2J = pt_tools.constants.MeV2J
    G2T = pt_tools.constants.G2T
    c = pt_tools.constants.c #299792458
    RE = pt_tools.constants.RE #6.3712e6

    #SI unit quantities:
    mu = particle.init_mu
    aeq = particle.init_alpha
    L = particle.init_L
    iphase_gyro = particle.iphase_gyro
    iphase_bounce = particle.iphase_bounce
    iphase_drift = particle.iphase_drift
    

    t0 = 0.


    #critical settings:------------------------------------------------+
    nonmirror_threshold = 89
    #above this threshold, particles will be considered equatorial,
    # and their trajectories will not be cut down to one bounce period
    #------------------------------------------------------------------+

    #check if the particle is equatorial or not according to our threshold:
    if aeq > nonmirror_threshold * pi / 180:
        nonmirroring = True
    else:
        nonmirroring = False


    tsperorbit = particle.recommended_tsperorbit
    
    #get initial position of the GC based on particle L:
    x0_GC = particle.calculate_initial_GC()

    #get a possible initial momentum vector on the magnetic equator,
    #   multiple solutions because adiabatic invariants say nothing about phase:
    B_GC = bfield.getBE(*x0_GC, t0)[:3]
    p0 = particle.calculate_initial_momentum(B_GC)

    #initial state vector of the GC:
    Y0_GC = np.hstack((x0_GC,p0)) 

    #calculate initial velocity for the momentum decided on:
    p0mag = np.linalg.norm(p0)
    gamma = sqrt(1 + (p0mag/(particle.m0 * c))**2)
    v0 = p0/(gamma*particle.m0)

    #calculate the gyroradius assuming the Lorentz force is constant within a gyroorbit:
    rg0 = calc_rg(Y0_GC, bfield, particle) #relativistically correct

    #initial condition x0:
    x0 = particle.calculate_initial_position(x0_GC, rg0)
    
    #kinetic energy:
    E0_J = (gamma - 1)*particle.m0*(c**2)
    E0 = E0_J /MeV2J #KE energy in MeV
    particle.muenKalphaL[0,1] = E0

    #energy = particle.derive_KE0(bfield)

    # #calculate the mirror field strength:
    Be = np.linalg.norm(B_GC)
    Bm_approx = Be / (sin(aeq)**2 )
    I_approx = L * RE* pt_tools.approx_Ya(aeq)
    K_approx = np.power(Bm_approx/G2T,0.5) * I_approx / RE
    particle.muenKalphaL[0,2] = K_approx #includes dipole approximation

    #calculate (an estimate of) the time for one bounce period
    tb_est = tb_estimate(np.linalg.norm(x0_GC), np.linalg.norm(v0), aeq)

    t0_init = t0
    x0_init = x0
    p0_init = p0

    #print some information:
    print("Particle's initial properties:")
    print(" energy          = {:.3f}MeV".format(E0))
    print(" mu              = {:.3f}MeV/G".format(mu*G2T/MeV2J))
    print(" eq. pitch angle = {:.3f}d".format(aeq* 180/pi))
    print(" L               = {:.3f}".format(L))
    print(" gyrophase       = {:.3f}d".format(iphase_gyro))
    print(" bounce phase    = {:.3f}d".format(iphase_bounce))
    print(" drift phase     = {:.3f}d".format(iphase_drift))
    print(" lorentz factor  = {:.3f}".format(gamma))
    print(" speed           = {:.3f}c".format(np.linalg.norm(v0)/c))
    print(" x0              = {:.3f}, {:.3f}, {:.3f} RE".format(x0[0]/RE, x0[1]/RE, x0[2]/RE))
    print(" p0              = {:.3e}, {:.3e}, {:.3e} kgm/s".format(p0[0], p0[1], p0[2]))
    print(" bounce time     ~ {:.3f}s".format(tb_est))
    print("#")

    #
    #   Place the particle:
    #
    particle.update(t0_init, [*x0_init, *p0_init])
    #
    # #test derive_invariants(...):
    # mu_guess, K_guess, aeq_guess, L_guess = derive_invariants(particle, bfield)
    # print(mu_guess*G2T/MeV2J)
    # print(mu*G2T/MeV2J)
    # print()
    # print(K_guess)
    # print(K_approx)
    # print()
    # print(L_guess)
    # print(L)
    # print()
    # sys.exit(1)

    if nonmirroring:
        dt_solve_increment = tb_est #tb_estimate(np.linalg.norm(x0_GC), np.linalg.norm(v0), nonmirror_threshold * pi / 180)
        t1 = t0_init
        x1 = x0_init
        p1 = p0_init
    else:
        #
        #   Track the particle to some phase along the bounce in a static field at t0:
        #
        print("Initialising fraction {:.3f} along the first bounce...".format(iphase_bounce))
        if iphase_bounce > 1:
            print("Skipping because bounce phase has been specified incorrectly (please specify 0 <= x <= 1)")
            return 2


        #modify the particle to keep the track:
        storetrack_setting = particle.storetrack
        particle.update = particle.update_keep


        #use a static field at t0 to do this: we want to find where the particle would be at t0:
        t1, x1, p1 = solver_boris(particle, bfield, 1.0/0.9 * tb_est, tsperorbit, t_limit_exact = False)#, freezefield = t0)
        #cut the trajectory down to exactly one bounce:
        capok = cut_exact_bounce(particle, bfield, tsperorbit)

        if not capok:
            print("Skipping because particle didn't bounce properly")
            return 2

        #we now have an accurate, numerically-derived approximation of bounce time:
        tb_est = particle.times[-1]

        #find time at which the particle is iphase_bounce along the bounce
        start_time = tb_est * iphase_bounce
        start_idx = np.argmin(np.abs(particle.times-start_time))

        #just keep the section of trajectory after this time:
        t0_shift = particle.times[start_idx]
        particle.pt = particle.pt[start_idx:]
        particle.times = particle.times[start_idx:]
        #shift the particle times back so it started from t0:
        for idx in range(len(particle.times)):
            particle.times[idx] -= t0_shift

        dt_solve_increment = tb_est - particle.times[-1] #remaining fraction of the first bounce to solve
        # this will be updated to tb_est in the first solver iteration below


        #restore particle properties if necessary:
        if not storetrack_setting:
            particle.update = particle.update_discard
            particle.times = [particle.times[-1]]
            particle.pt = [particle.pt[-1]]

    #
    #   Option to solve for a specific duration:
    if duration_solve > 0:
        if duration_solve < particle.times[-1]:#we already solved for long enough:
            while particle.times[-1] > duration_solve:
                particle.pop_track()
        dt_solve_increment = duration_solve - particle.times[-1]


    #
    #   SOLVE THE (REMAINING) PARTICLE TRAJECTORY:
    #
    print("Tracking...")
    nbounces_stored_approx = (t1 - t0) / tb_est
    delta_az = 0
    while abs(delta_az) <= delta_az_solve and bfield.range_adequate:
        t1, x1, p1 = solver_boris(particle, bfield, dt_solve_increment, tsperorbit)
        delta_az += angle_between(x0, x1)
        x0 = x1
        dt_solve_increment = tb_est #reset the dt incrememnt to bounce time, in case it started as a fraction of a bounce
        nbounces_stored_approx = (t1 - t0) / tb_est
        
        if delta_az_solve > 0: print("","{:.2f}%".format(100*delta_az/(delta_az_solve))) #only works if dt_solve_increment << drift orbit time
    

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
            print("Tracking an extra {} bounces for K calculation...".format(2 - nbounces_stored_approx))
            t1_extra, x1_extra, p1_extra = solver_boris(particle, bfield, (2 - nbounces_stored_approx)*tb_est, tsperorbit)
        
        if not bfield.range_adequate:
            print("Error: cannot find K near the beginning of the trajectory because the particle goes out of range")
        else:
            print("Calculating initial K...")
            gc_bounce_time, gc_bounce_pos, K_ = get_GC_from_track(rg0, bfield, particle, tsperorbit, tlimit = 2*tb_est)
            particle.muenKalphaL[0,2] = K_

        #using the same logic as above, restore the particle track if needed:
        if particle.storetrack and bfield_range_ok_prior:
            while particle.pop_track(stored_len): continue
            if storegc:
                print("Calculating GC...")
                particle.gc_times, particle.gc_pos = get_GC_from_track(rg0, bfield, particle, tsperorbit)[:2]
        elif particle.storetrack:
            particle.times = stored_times
            particle.pt = stored_pt
            if storegc:
                print("Calculating GC...")
                particle.gc_times, particle.gc_pos = get_GC_from_track(rg0, bfield, particle, tsperorbit)[:2]
        else:
            particle.update = particle.update_discard
            particle.times = [t1]
            particle.pt = [[*x1, *p1]]

        bfield.range_adequate = bfield_range_ok_prior

 

    if not bfield.range_adequate:
        print("","tracking failed - particle went out of EM field range")
        return 2


    exect1 = time.perf_counter()
    print("","tracked with execution time of {:.2f}s".format(exect1 - exect0))
    #azimuth is not tracked properly if dt_solve_increment > time for 1 drift orbit
    #print("","change in azimuth: {:.3f}d".format(delta_az * 180/pi))
    print("#")

    return 1


def solve_trajectory_bounce(particle, bfield, dummy, findK0 = False, storegc = False):
    """solve for one bounce"""
    delta_az_solve = 0
    return solve_trajectory(particle, bfield, delta_az_solve, duration_solve = -1, findK0 = findK0, storegc = storegc)

def solve_trajectory_drift(particle, bfield, dummy, findK0 = False, storegc = False, n_solve = 1):
    """solve for n drifts"""
    delta_az_solve = n_solve*2*pi
    return solve_trajectory(particle, bfield, delta_az_solve, duration_solve = -1, findK0 = findK0, storegc = storegc)

def solve_trajectory_time(particle, bfield, duration_solve, findK0 = False, storegc = False):
    """solve for a fixed duration"""
    delta_az_solve = 0
    return solve_trajectory(particle, bfield, delta_az_solve, duration_solve, findK0 = findK0, storegc = storegc)


# def plot1_againsttime(y, t, ylab = None, label=None, filename=None, constax1 = None):
#     import matplotlib.pyplot as plt
#     logy = False

#     a = t[:len(y)]
    
#     #set up figure and axes:
#     fig, ax1 = plt.subplots()

#     color = 'tab:red'
#     ax1.set_xlabel('time (s)')
#     ax1.set_ylabel(ylab, color=color)
#     ax1.plot(a, y, color=color)
#     ax1.tick_params(axis='y', labelcolor=color)
#     ax1.ticklabel_format(useOffset=False)
#     if logy: ax1.set_yscale('log')

#     fig.tight_layout()  # otherwise the right y-label is slightly clipped
    
#     #add label
#     if label:
#         #xmin, xmax = ax1.get_xlim()
#         #ymin, ymax = ax1.get_ylim()
#         #plt.text(xmin + 0.05*(xmax-xmin), ymin + 0.5*(ymax-ymin), label)
#         plt.text(0.05, 0.2,text,
#          horizontalalignment='left',
#          verticalalignment='center',
#          transform = ax1.transAxes)

#     if constax1:
#         ax1.plot([t[0], t[-1]],[constax1, constax1], linestyle="--", color="black")
        
    
#     if filename:
#         plt.savefig(filename, dpi=300, facecolor='w', edgecolor='w', orientation='portrait')
#     else:
#         plt.show()
#     plt.close()

# def plot2_againsttime(y, z, t, ylab = None, zlab = None, label=None, filename=None, constax1 = None):
#     import matplotlib.pyplot as plt
#     logy = False
#     logz = False

#     a = t[:len(y)]
#     b = t[:len(z)]
    
#     #set up figure and axes:
#     fig, ax1 = plt.subplots()

#     color = 'tab:red'
#     ax1.set_xlabel('time (s)')
#     ax1.set_ylabel(ylab, color=color)
#     ax1.plot(a, y, color=color)
#     ax1.tick_params(axis='y', labelcolor=color)
#     ax1.ticklabel_format(useOffset=False)
#     if logy: ax1.set_yscale('log')

#     ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

#     color = 'tab:blue'
#     ax2.set_ylabel(zlab, color=color)  # we already handled the x-label with ax1
#     ax2.plot(b, z, color=color)
#     ax2.tick_params(axis='y', labelcolor=color)
#     ax2.ticklabel_format(useOffset=False)
#     if logz: ax2.set_yscale('log')

#     fig.tight_layout()  # otherwise the right y-label is slightly clipped
    
#     #add label
#     if label:
#         #xmin, xmax = ax1.get_xlim()
#         #ymin, ymax = ax1.get_ylim()
#         #plt.text(xmin + 0.05*(xmax-xmin), ymin + 0.5*(ymax-ymin), label)
#         plt.text(0.05, 0.2,text,
#          horizontalalignment='left',
#          verticalalignment='center',
#          transform = ax1.transAxes)

#     if constax1:
#         ax1.plot([t[0], t[-1]],[constax1, constax1], linestyle="--", color="black")
        
    
#     if filename:
#         plt.savefig(filename, dpi=300, facecolor='w', edgecolor='w', orientation='portrait')
#     else:
#         plt.show()
#     plt.close()
    

# def mag(X):
#     return np.linalg.norm(X)

# def gamma_v(v):
#     return 1./sqrt(1-(v**2)/c**2)

# def gamma_p(pmag, particle):
#     return sqrt(1 + (pmag/(particle.m0 * c))**2)

# def invar_lat(L):
#     """
#     returns lambda which, on the surface of the earth, is equal to the magnetic latitude 
#     """
#     return acos(1/L**0.5)

##def Brth(R, theta):
##    """
##    dipole field model
##    """
##    Br = -2*B0*((RE/R)**3)*cos(theta)
##    Bth = -B0*((RE/R)**3)*sin(theta)
##    return (Br, Bth)
##
##def Bmag(R, theta):
##    return B0*((RE/R)**3)*((1+3*(cos(theta))**2)**0.5)


# def unit_vector(vector):
#     """ Returns the unit vector of the vector.  """
#     return vector / np.linalg.norm(vector)


# def getB_ndipole(x,y,z, cosys="MAG",extMag='0'): #NON-DIPOLE
#     """
#     takes position (metres) (vector from origin to particle)
#     returns the magnetic field vector (T) (at particle) in cartesian coordinates
#     """
#     t = spt.Ticktock(['2000-01-01T00:00:00'], 'UTC')
#     #print([x/RE,y/RE,z/RE])
#     pos = spc.Coords([[x/RE,y/RE,z/RE]], cosys, 'car')
#     #print(ib.get_Bfield(t,y,extMag='OPQUIET',options=[0, 0, 0, 1, 0])["Blocal"])
#     Bvec = ib.get_Bfield(t,pos,extMag,options=[0, 0, 0, 1, 0])["Bvec"][0]
#
#     return np.array([Bvec[0]*1e-9, Bvec[1]*1e-9, Bvec[2]*1e-9])


##def test_magsync(cosys='GEO'):
##    t = spt.Ticktock(['2015-01-01T00:00:00'], 'UTC')
##    y = spc.Coords([[0,2.2,0]], cosys, 'car')
##    print(y)
##    print(ib.get_Bfield(t,y,extMag='OPQUIET',options=[0, 0, 0, 1, 0])["Blocal"])
##    print(ib.get_Bfield(t,y,extMag='OPQUIET',options=[0, 0, 0, 1, 0])["Bvec"][0])
##    sys.exit(1)
##
##def test_whereisthemagneticequator(x,y,z, cosys="MAG",extMag='OPQUIET'):
##    """
##    takes position (metres) (vector from origin to particle)
##    returns the magnetic field vector (T) (at particle) in cartesian coordinates
##    """
##    t = spt.Ticktock(['2015-01-01T00:00:00'], 'UTC')
##    print([x/RE,y/RE,z/RE])
##    pos = spc.Coords([[x/RE,y/RE,z/RE]], cosys, 'car')
##    #print(ib.get_Bfield(t,y,extMag='OPQUIET',options=[0, 0, 0, 1, 0])["Blocal"])
##    Bvec = ib.get_Bfield(t,pos,extMag,options=[0, 0, 0, 1, 0])["Bvec"][0]
##
##    #testing:
##    #enter GEO coords:
##    y = spc.Coords([[1.5,0,0]], 'GEO', 'car')
##    #find magnetic equator in GEO:
##    y = ib.find_magequator(t,y)["loci"]
##    #find Bvec at magnetic equator:
##    Bvec = ib.get_Bfield(t,y,extMag,options=[0, 0, 0, 1, 0])["Bvec"][0]
##
##    print()
##    print(y) #magnetic equator in GEO
##    y.ticks = t
##    newcoord = y.convert('GSM', 'car')
##    print("magnetic equatori in GSM:", newcoord) #magnetic equator in GSM
##    print("magnetic equatori Bvec:",Bvec) #Bvec
##
##    Bvec = ib.get_Bfield(t,newcoord,extMag,options=[0, 0, 0, 1, 0])["Bvec"][0]    
##    print("magnetic equatori Bvec:",Bvec) #Bvec
##    sys.exit(1)

# def getlosscone(bfield,particleL):

#     #calculate enclosed flux for a given L, from eq. 4, Selesnick et al. 2016
#     enclosed_flux = 2*pi*bfield.M/ (RE*particleL)

#     #calculate the equatorial position in RE that encloses this much flux according to a dipole:
#     # from eq. 4.51, Walt
#     R_gc_RE = (2*pi*bfield.B0*RE**3)/enclosed_flux

#     #some correction (check where this comes from, maybe the definition of dipole moment):
#     R_gc_RE = R_gc_RE *pi/2
    
#     R_gc = R_gc_RE *RE

    
#     #estimate the loss cone angle for a particle at zero azmiuth:
#     mag_lat = invar_lat(particleL) #magnetic latitude of intersection with Earth's surface

#     surface_L = [RE*cos(mag_lat), 0, RE*sin(mag_lat)]

#     Ba = bfield.getB(surface_L[0], surface_L[1], surface_L[2])
#     Ba = mag(Ba)

#     Beq = bfield.getB(R_gc, 0, 0) 
#     Beq = mag(Beq)

#     return asin(sqrt(Beq/Ba))

