import pt_tools
import sys
import numpy as np
import pt_fp
from math import log10
from datetime import datetime, timezone
import argparse
import os
from pathlib import Path
# Groups work like dictionaries, and datasets work like NumPy arrays
#from pyIGRF import calculate
from math import pi
MeV2J = pt_tools.constants.MeV2J
G2T = pt_tools.constants.G2T
"""
Organise the generation of particle tracks
"""

outdir = "./pt_solutions/"
Path(outdir).mkdir(parents=True, exist_ok=True) #make the directory if it doesn't exist

#set up parser and arguments: ------------------------------------------------+
parser = argparse.ArgumentParser(description='Get configuration file')

parser.add_argument("--config",type=str, required=True)
parser.add_argument("--runname",type=str, required=False)
parser.add_argument("--extractgc", required=False, action='store_true')
parser.add_argument("--repair", type=int, required=False, default=-1)

args = parser.parse_args()

configfile = args.config
runname = args.runname
extractgc = args.extractgc
repair = args.repair
#-----------------------------------------------------------------------------+

#
#   Read a config file:
#
config = pt_tools.config_rw(configfile)
if not (config.read()):
    print("Error reading configuration file - ensure every parameter is present and of the correct type")
    sys.exit(1)
#assign all the values from the config file to internal variables:
particletype = config.datadic[config.species_kw]
orbittype = config.datadic[config.orbit_kw]
storetrack = config.datadic[config.storetrack_kw]
storegc = config.datadic[config.storegc_kw]
findK0 = config.datadic[config.findK0_kw]
reevaluate_invariants = config.datadic[config.reeval_kw]
duration_solve = config.datadic[config.duration_kw]
reverse = config.datadic[config.reverse_kw]
year = config.datadic[config.year_kw]
month = config.datadic[config.month_kw]
day = config.datadic[config.day_kw]
dt = datetime(year, month, day, tzinfo=timezone.utc)
lmin = config.datadic[config.lmin_kw]
lmax = config.datadic[config.lmax_kw]
nl = config.datadic[config.nl_kw]
amin = config.datadic[config.amin_kw]
amax = config.datadic[config.amax_kw]
na = config.datadic[config.na_kw]
logmumin = config.datadic[config.logmumin_kw]
logmumax = config.datadic[config.logmumax_kw]
nmu = config.datadic[config.nmu_kw]
nphase_gyro = config.datadic[config.nphase_gyro_kw]
nphase_bounce = config.datadic[config.nphase_bounce_kw]
nphase_drift = config.datadic[config.nphase_drift_kw]
fieldpath = config.datadic[config.fieldpath_kw]
skipeveryn = config.datadic[config.skipeveryn_kw]
emin = config.datadic[config.emin_kw]
emax = config.datadic[config.emax_kw]
iphase_gyro = config.datadic[config.iphase_gyro_kw]
iphase_bounce = config.datadic[config.iphase_bounce_kw]
iphase_drift = config.datadic[config.iphase_drift_kw]
continuefrom = config.datadic[config.continuefrom_kw]
override_energy_axis = config.datadic[config.override_energy_axis_kw]

# process configuration file options:
if particletype[0].lower() == "p":
    Particle = pt_tools.Proton_trace
elif particletype[0].lower() == "e":
    Particle = pt_tools.Electron_trace
else:
    print("Error: particle species '{}' not recognised".format(particletype))

if orbittype[0].lower() == "b":
    print("Bounce mode")
    call_to_solver = pt_fp.solve_trajectory_bounce
    duration_solve = -1
elif orbittype[0].lower() == "d":
    print("Drift mode")
    call_to_solver = pt_fp.solve_trajectory_drift
    duration_solve = -1
elif orbittype[0].lower() == "t":
    print("Duration mode")
    call_to_solver = pt_fp.solve_trajectory_time
else:
    print("Error: orbit type '{}' not recognised".format(orbittype))

if storetrack[0].lower() == "y" or storetrack[0].lower() == "t":
    storetrack = True
else:
    storetrack = False
if storegc[0].lower() == "y" or storegc[0].lower() == "t":
    storegc = True
else:
    storegc = False
if findK0[0].lower() == "y" or findK0[0].lower() == "t":
    findK0 = True
else:
    findK0 = False
if reevaluate_invariants[0].lower() == "y" or reevaluate_invariants[0].lower() == "t":
    reevaluate_invariants = True
else:
    reevaluate_invariants = False
if reverse.lower() == "y":
    reverse = True
else:
    reverse = False

if nphase_gyro < 1: nphase_gyro = 1
if nphase_bounce < 1: nphase_bounce = 1
if nphase_drift < 1: nphase_drift = 1

if extractgc and storegc == True:
    print("Error: extractgc flag was set but config indicates that GC is already stored/set to be stored")
    sys.exit()
elif extractgc and not len(continuefrom):
    print("Error: to extract guiding centers from a previously-computed solution, the continuefrom option must be set as the path to that solution")
    sys.exit()

#
#   Generate a grid of invariant coordinates:
#
if override_energy_axis.size:
    nmu = override_energy_axis.size
    logmumin = np.nan
    logmumax = np.nan
mur = np.linspace(logmumin, logmumax, nmu)
mur = np.power(10 * np.ones(mur.shape), mur)
mur = mur[::-1]  # reverse to start at higher energy and work down
lr = np.linspace(lmin, lmax, nl)
ar = np.linspace(amin, amax, na)
#phases to initialise between 0 and 1:
phase_gyro = (np.linspace(0, 1, nphase_gyro + 1)[:nphase_gyro] + iphase_gyro) % 1
phase_bounce = (np.linspace(0, 1, nphase_bounce + 1)[:nphase_bounce] + iphase_bounce) % 1
phase_drift = (np.linspace(0, 1, nphase_drift + 1)[:nphase_drift] + iphase_drift) % 1

year_dec = pt_tools.dt_to_dec(dt)

#calcualte the total number of tracks required:
numberoftracks = len(mur) * len(ar) * len(lr) * nphase_drift * nphase_bounce * nphase_gyro
print("Other information:")
print("","date:", dt)
print("","mu axis:", mur)
print("","alpha axis:", ar)
print("","L axis:", lr)
print("","phase gyro:", phase_gyro)
print("","phase bounce:", phase_bounce)
print("","phase drift:", phase_drift)
print("","total combinations:", numberoftracks)

#
#   Load or create a HDF5 file for trajectories:
#
if len(continuefrom):
    #read the existing hdf5 file, check the checklist, and make an array of keys that haven't yet been solved
    filename_hdf5 = continuefrom #includes the directory name
    print("Continuing previous solution", filename_hdf5)

    #recreate the tracklist variable from the file:
    if not os.path.exists(filename_hdf5):
        print("Error: the solutions file does not exist at", filename_hdf5)
        sys.exit(1)
    resultfile = pt_tools.HDF5_pt(filename_hdf5, existing = True)

    if extractgc:
        filename_hdf5_GC = filename_hdf5.replace(".h5", "_GC.h5")
        # write the basic structure and metadata of the HDF5 GC file:
        tracklist_existing = resultfile.get_existing_tracklist()
        resultfile_GC = pt_tools.HDF5_pt(filename_hdf5_GC)
        resultfile_GC.setup(config.datadic, tracklist_existing)
        if skipeveryn > 1:
            print("Warning: skipeveryn was set to {} in original trajectory calculation...".format(skipeveryn))
            print("","this will be accounted for, but may limit the accuracy of the guiding center reanalysis")
else:
    print("Starting new pt solution:")
    #launch = input("Ready to launch? (press enter)")
    if type(runname) == type(None):
        runname = "pt_" + datetime.now().strftime("%Y%m%d-%H%M%S_")
        print("Using generated run name:",runname)
    else:
        runname += "_"
        print("Using supplied run name:",runname)
    filename_hdf5 = outdir + runname + "solutions.h5"

    #copy the configuration file but with the new run name at the end, and the solution name inside to continue from:
    config.datadic[config.continuefrom_kw] = filename_hdf5
    comments = ["Copy and paste this file to the directory containing pt_handler.py to resume this simulation"]
    config.saveas(outdir + runname + "config.txt", topcomments=comments)

    #generate a unique number for each track required to complete this solution:
    tracklist = {}
    count = 0
    for mu in mur:
        for pa in ar:
            for L in lr:
                for pg in phase_gyro:
                    for pb in phase_bounce:
                        for pd in phase_drift:
                            #trackname = runname + str(count).zfill(1+int(log10(numberoftracks))) + ".pt"
                            tracklist[count] = [mu, pa, L, pg, pb, pd]
                            count+=1


    #write the basic structure and metadata of the HDF5 results file:
    resultfile = pt_tools.HDF5_pt(filename_hdf5)
    resultfile.setup(config.datadic, tracklist)

    #obsolete, but we can still write out a tracklist in plain text:
    #config.saveas(outdir + runname + "tracklist.txt", towrite = tracklist)
    print()


#
#   Get configuration information from the HDF5 file:
#
# #load the drift average database file to get information about the simulation:
metadata = resultfile.read_root()
tracklist_ID = metadata['tracklist_ID']
checkcodes = resultfile.get_solved_ids()
tracklist_mu_changes = metadata['tracklist_mu'] #recalculated as we iterate through the particles below
if override_energy_axis.size:
    tracklist_energy = dict(zip(np.arange(tracklist_ID.size), np.repeat(override_energy_axis, ar.size * lr.size * phase_gyro.size * phase_bounce.size * phase_drift.size)))

#
#   Instantiate magnetic field
#
if len(fieldpath): #include non-dipolar field perturbations and time variation from the file
    if reverse:
        bfield = pt_tools.Geofield(fieldpath, reversetime = duration_solve)
    else:
        bfield = pt_tools.Geofield(fieldpath)
    if duration_solve > bfield.field_time[-1]:
        print("Error: cannot solve for longer than the field is specified ({}s)".format(bfield.field_time[-1]))
        sys.exit(1)
else:
    bfield = pt_tools.Dipolefield(year_dec)

#
#   Instantiate a particle for each coordinate and solve the track:
#
print("Calling solver...")
count = 0
for pt_id in tracklist_ID:
    print()
    print("Tracking {} #{} / {}".format(particletype, pt_id+1, numberoftracks))
    print("#")
    bfield.range_adequate = True
    if (checkcodes[pt_id] > 0) and not extractgc and not repair == pt_id:
        print("Skipping already-calculated track ID", pt_id)
        continue
    elif extractgc:
        print("Reanalyzing track ID {} to extract GC".format(pt_id))

    #particle-specific information:
    id = metadata['tracklist_ID'][pt_id]
    
    mu = metadata['tracklist_mu'][pt_id]
    mu_SI = mu * MeV2J / G2T #change units of mu to SI
    pa = metadata['tracklist_pa'][pt_id]
    pa = pa * pi / 180 #change units of pi to radian
    if pa > pt_fp.aeq_max_for_bounce_detection * pi / 180:
        print("Warning: particle with aeq={}d has had pitch angle reduced to {}d".format(pa * 180/np.pi, pt_fp.aeq_max_for_bounce_detection))
        pa = pt_fp.aeq_max_for_bounce_detection * pi / 180
    L = metadata['tracklist_L'][pt_id]
    phase_g = metadata['tracklist_pg'][pt_id]
    phase_b = metadata['tracklist_pb'][pt_id]
    phase_d = metadata['tracklist_pd'][pt_id]

    particle = Particle(mu_SI, pa, L, phase_g, phase_b, phase_d, storetrack=storetrack)

    if mu != mu:  # NaN value for when we are overriding the mu axis with some specific energy
        #use energy to calculate mu:
        # get initial position of the GC based on particle L:
        x0_GC = particle.calculate_initial_GC()
        # get a possible initial momentum vector on the magnetic equator,
        #   multiple solutions because adiabatic invariants say nothing about phase:
        B_GC = np.linalg.norm(bfield.getBE(*x0_GC, 0)[:3])
        E0_J = tracklist_energy[pt_id] * pt_tools.constants.MeV2J
        gamma = 1 + E0_J/ (particle.m0 * (pt_tools.constants.c ** 2))
        p0mag = (particle.m0 * pt_tools.constants.c) * np.sqrt(gamma**2 - 1)
        p0_perp = p0mag * np.sin(pa)
        mu_SI = (p0_perp**2)/(2 * B_GC * particle.m0)
        mu = mu_SI * pt_tools.constants.G2T / pt_tools.constants.MeV2J

        #reinstantiate the particle:
        particle = Particle(mu_SI, pa, L, phase_g, phase_b, phase_d, storetrack=storetrack)

        #update tracklist and H5 file with mu:
        tracklist_mu_changes[pt_id] = mu
        if checkcodes[pt_id] == 0:
            #update if new:
            resultfile.update_dataset('tracklist_mu', tracklist_mu_changes, compressmethod=None, quiet=True)


    #
    #   Extract guiding center from previously-completed simulation:
    #
    if extractgc:
        resultfile_GC.update_dataset('tracklist_mu', tracklist_mu_changes, compressmethod=None, quiet=True)

        if checkcodes[pt_id] == 1: #if the track has been successfully calculated
            #calculate initial momentum and relativistic mass:
            x0_GC = particle.calculate_initial_GC()
            B_GC = np.linalg.norm(bfield.getBE(*x0_GC, 0)[:3])
            p0 = particle.calculate_initial_momentum(B_GC)
            gamma = np.sqrt(1 + (np.linalg.norm(p0)/(particle.m0 * pt_tools.constants.c))**2)
            massr = particle.m0#gamma*particle.m0

            times, position = resultfile.read_track(pt_id, False)
            
            #infer momentum from previously calculated trajectory:
            velocity = (position[1:] - position[:-1])/(times[1:, np.newaxis] - times[:-1, np.newaxis])
            gamma = 1.0/(np.sqrt(1-velocity*velocity/(pt_tools.constants.c**2)))
            momentum = gamma * particle.m0 * velocity

            times = times[:-1]
            position = position[:-1]

            particle.times = times
            particle.pt = list(np.hstack((position, momentum)))
            code_success = pt_fp.extract_GC_only(particle, bfield, existing_skipeveryn=skipeveryn)
            particle.times = particle.gc_times
            particle.pt = particle.gc_pos
        else:
            print("Warning: could not extract GC for particle track ID {}".format(pt_id))
            code_success = checkcodes[pt_id]


        resultfile_GC.add_track(pt_id, particle, checkcode=code_success, compressmethod="gzip", skipeveryn=skipeveryn)
        count += 1
        continue

    #
    #   Check the energy and pitch angle:
    #
    code_success = 1
    #check the energy is within range:
    E0 = particle.derive_KE0(bfield, 0)/MeV2J
    if E0 > emax:
        print("Skipping because T > {:.2f}MeV".format(emax))
        code_success = 2
    elif E0 < emin:
        print("Skipping because T < {:.2f}MeV".format(emin))
        code_success = 2
    #calculate the dipole loss cone:
    pa_lc_approx = bfield.get_dipolelc(L, 0) * pi / 180 #this is a BAD approximation in an eccentric dipole        
    #check if the particle is in the loss cone:
    if pa <= pa_lc_approx:
        print("Skipping because particle within loss cone")
        code_success = 2


    if code_success == 1:       
        #duration_solve = duration_field
        #code_success = pt_fp.solve_trajectory_bounce(particle, bfield, duration_solve, findK0 = findK0, storegc = storegc)
        #code_success = pt_fp.solve_trajectory_drift(particle, bfield, duration_solve, findK0 = findK0, storegc = storegc)
        #code_success = pt_fp.solve_trajectory_time(particle, bfield, duration_solve, findK0 = findK0, storegc = storegc)
        code_success = call_to_solver(particle, bfield, duration_solve, findK0 = findK0, storegc = storegc, reverse = reverse)

        #code_success will not be 1 if the particle goes out of range of the fields
        if code_success == 1 and reevaluate_invariants: #this should work even when particle.storetrack = False
            invariants_post = pt_fp.derive_invariants(particle, bfield, reverse = reverse)
            particle.muenKalphaL[1,:] = invariants_post

        #code_success will not be 1 if storetrack is False and storegc is True
        if storegc:
            particle.times = particle.gc_times
            particle.pt = particle.gc_pos

    #
    #   Store the particle track in the HDF5 file:
    #
    resultfile.add_track(pt_id, particle, checkcode = code_success, compressmethod = "gzip", skipeveryn = skipeveryn)
    print()
    print()
    count += 1

print("...{} simulations performed".format(count))
print()