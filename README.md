# **Trajectory Redistribution In Phase Space**

# Authorship and Citation 

This project was developed by Alexander R. Lozinski with contributions from Ravindra T. Desai. Please cite as follows:

```
@misc{TRIPS25,
  author = {Lozinski, Alexander R and Desai, Ravindra T},
  title = {Trajectory Redistribution In Phase Space Python Code}, 
  year = {2025},
  doi = {10.5281/zenodo.14908242},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/atmosalex/pt}}
}
```

# Usage

This set of scripts can be used to solve electron or proton particle trajectories initialised from a grid of adiabatic coordinates. If using a dipole field, the steps are:

1. Write a configuration file (see Configuration section)
2. Run python pt\_run.py --config configs/example1.txt, where configs/example1.txt is replaced by the local path to the desired configuration file. Particle trajectories will be solved and output to the pt\_solutions/ directory in a HDF5-format file.
3. Optionally, plot the trajectory (if stored) by running python pt\_plot.py --solution pt\_solutions/example1\_solutions.h5, where pt\_solutions/example1.h5 is replaced by the local path to the solution file.

The configuration file allows the user to control the initial distribution of particles in gyro, bounce and drift phase independently. The user can also configure a simulation to calculate the values of adiabatic invariants before and after particle tracing. See the examples below.

If the user wishes to simulate inside a non-dipole field, or include time variation in the field, the field must be solved beforehand and stored on a grid in the MAG frame, where Z=0 will be taken as the magnetic equator. To demonstrate the processing and format required, code is included to calculate the electromagnetic perturbation [modeled by Li et al. (1993)](https://doi.org/10.1029/93GL02701). To run this code, execute the following Python from the top level directory of the repository, which will produce the file `configs/output_filename.h5`:

```
import field_tools
field_tools.study_march91('configs/output_filename.h5', redo=True)
```


# Dependencies

Python 3 is required to run this set of scripts, and the necessary dependencies can be installed via pip by running:

pip install —upgrade pip

pip install -r requirements.txt

# Configuration

## Overview

A configuration file is formatted like a CSV file with each line following the syntax:

**keyword**, **parameter**

where **keyword** is a string, typed without quotation marks, describing some physical parameter, and **parameter** is some value assigned to the variable represented by **keyword**. Blank lines and lines beginning with # are ignored.


Keywords can appear in any order, but all of the following keywords must be present: **species**, **orbit**, **duration to solve**, **reverse**, **store trajectory**, **store GC**, **find initial K**, **re-calculate invariants**, **year**, **month**, **day**, **Lmin**, **Lmax**, **nL**, **amin**, **amax**, **na**, **logmumin**, **logmumax**, **nmu**, **nphase\_gyro**, **nphase\_bounce**, **nphase\_drift**, **iphase\_gyro**, **iphase\_bounce**, **iphase\_drift**, **fieldpath**, **emin**, **emax**, **skipeveryn**, **continuefrom**, **override energy axis**.

The meaning of each keyword is indicated below, and examples of acceptable parameter values are indicated for each in bold font. Specifying a parameter value is optional for some keywords even though each keyword must appear, and in this case **parameter** can be left blank.

## Keyword Guide

- **species** describes the species of particle to trace: **p** for protons or **e** for electrons.

- **orbit** describes the type of orbit to simulate: **b** for a single bounce, **d** for a single drift, or **t** for a custom duration.

- **duration to solve**, when orbit type **t** is specified, this is the trajectory duration as an integer or float value in units of seconds, such as **180** for 180 seconds, etc. Otherwise its value is ignored but must be specified.

- **reverse** controls whether the particle is traced forward or backwards in time, accepted values are **n** or **y** respectively.

- **store trajectory** controls whether or not the trajectory of the particle is saved in the output file, accepted values are **y** or **n**. Files can potentially become large for long trajectories when **y** is specified.

- **store GC** controls whether or not the guiding centre of the trajectory is calculated and saved in the output file, accepted values are **y** or **n**. The **store trajectory** function must be enabled for this to work, and the guiding centre will overwrite the particle trajectory.

- **find initial K** controls whether or not the initial value of the second adiabatic invariant K is calculated and saved in the output file, accepted values are **y** or **n**. 

- **re-calculate invariants** controls whether or not the adiabatic invariants mu, K and L are re-calculated at the end of the trajectory for each particle, then saved in the output file. Accepted values are **y** or **n**. This can add considerable computation time depending on the number of particles, etc. Both the initial and final invariant values will be stored in the solution file for comparison.

- **year**, **month** and **day** describe the epoch of the IGRF coefficients used to calculate the dipole field moment. The parameter values must be integer, integer, integer respectively i.e. **2015**, **1**, **1** for 1st Jan. 2015.

- **Lmin**, **Lmax** and **nL** set up a grid in the L coordinate to populate with particles. The parameter values must be float, float, integer respectively i.e. **2.0**, **2.0**, **1** to consider particles only at L=2, or **2.0, 3.0**, **11** to evenly space particles between L=2 and L=3 at 11 coordinates.

- **amin**, **amax** and **na** set up a grid in the equatorial pitch angle coordinate to populate with particles. The parameter values must be float, float, integer respectively in units of degrees i.e. **90**, **90**, **1** to consider equatorially mirroring particles only, or **10.0, 90.0**, **11** to evenly space particles between aeq=10 and aeq=90 at 11 coordinates.

- **logmumin**, **logmumax** and **nmu** set up a grid in the mu coordinate to populate with particles. The parameter values must be float, float, integer respectively in units of log10(mu / 1MeV/G) i.e. **2**, **2**, **1** to consider 100MeV/G particles only, or **2, 3**, **11** to log space particles between mu=100MeV/G and mu=1000MeV/G at 11 coordinates.

- **nphase\_gyro**, **nphase\_bounce** and **nphase\_drift** control the phase distribution of particles across the particle grid along each of the three types of periodic motion. The parameter values must be integer, integer, integer respectively i.e. **1**, **1**, **1** to consider only a single particle at each of the adiabatic coordinates generated by the previously described grid options, or **1**, **1**, **24** to consider 24 particles distributed evenly in drift phase at each set of adiabatic coordinates, etc.

- **iphase\_gyro**, **iphase\_bounce** and **iphase\_drift** control the initial conditions of phase distribution along each of the three types of periodic motion, normalised between 0 and 1. The parameter values must be float, float, float respectively i.e. **0**, **0**, **0**. Changing from **0**, **0**, **0** to **0.5**, **0.5**, **0.5** will effectively shift the initial phase of each particle by 180 degrees along the gyro, bounce and drift path. In the case of non-zero values for iphase\_bounce, or if nphase\_bounce is greater than 1, the bounce trajectory must be pre-computed before the distribution can be initialised in space, since the path cannot be solved analytically, and this may add computation time.

- **fieldpath** is optional. This experimental feature can be used to load in time-varying electromagnetic fields. Leave blank.

- **emin** and **emax** control the maximum and minimum energy of particles to consider. The parameter values must be float, float respectively with units MeV i.e. **10**, **500** to skip any calculation below 10MeV or above 500MeV.

- **skipeveryn** controls the number of trajectory points to be stored in any output file. The parameter value must be an integer. Setting this above 1 will cause only every nth point to be output, useful to reducing the size of trajectory files. A recommended value is **10** or higher.

- **continuefrom** is optional. If a previous simulation is interrupted, this keyword can be used to continue from it by specifying the name of the output file, i.e. **pt\_solutions/pt\_20230117-091223\_solutions.h5**. Any trajectory that was already solved before will be skipped, and the file will be updated as new trajectories are solved.

- **override energy axis** is optional. This can be set to a list of values separated by commas, i.e. "2.5, 5, 6" to study the three energies 2.5, 5 and 6MeV. If it is set, this will replace the mu grid. This can be used to initialize particles at energies corresponding to spacecraft instrument energy channels.

# Solutions

Solutions are stored as .h5 files in the pt\_solutions/ directory, with a name generated based on the time at which they were run. This behaviour can be overridden by supplying the --runname argument, i.e. python pt\_run.py --config configs/example1.txt --runname custom\_name, where custom\_name will be the new name of the solution in pt\_solutions/.

The configuration is also stored for each solution with a corresponding name, i.e. the file pt\_20230117-105105\_solutions.h5 will be accompanied by pt\_20230117-105105\_config.txt, which is a backup of the configuration file used to launch the simulation.

Each .h5 file is a HDF5 database storing the properties of each particle, along with their solved trajectories in the MAG frame when this option is set. The structure of the .h5 file produced for the example1 solution (see Example Configuration) is:

```
Lmax <class 'h5py._hl.dataset.Dataset'>
Lmin <class 'h5py._hl.dataset.Dataset'>
amax <class 'h5py._hl.dataset.Dataset'>
amin <class 'h5py._hl.dataset.Dataset'>
continuefrom <class 'h5py._hl.dataset.Dataset'>
day <class 'h5py._hl.dataset.Dataset'>
duration to solve <class 'h5py._hl.dataset.Dataset'>
emax <class 'h5py._hl.dataset.Dataset'>
emin <class 'h5py._hl.dataset.Dataset'>
fieldpath <class 'h5py._hl.dataset.Dataset'>
find initial K <class 'h5py._hl.dataset.Dataset'>
iphase_bounce <class 'h5py._hl.dataset.Dataset'>
iphase_drift <class 'h5py._hl.dataset.Dataset'>
iphase_gyro <class 'h5py._hl.dataset.Dataset'>
logmumax <class 'h5py._hl.dataset.Dataset'>
logmumin <class 'h5py._hl.dataset.Dataset'>
month <class 'h5py._hl.dataset.Dataset'>
nL <class 'h5py._hl.dataset.Dataset'>
na <class 'h5py._hl.dataset.Dataset'>
nmu <class 'h5py._hl.dataset.Dataset'>
nphase_bounce <class 'h5py._hl.dataset.Dataset'>
nphase_drift <class 'h5py._hl.dataset.Dataset'>
nphase_gyro <class 'h5py._hl.dataset.Dataset'>
orbit <class 'h5py._hl.dataset.Dataset'>
override energy axis <class 'h5py._hl.dataset.Dataset'>
re-calculate invariants <class 'h5py._hl.dataset.Dataset'>
reverse <class 'h5py._hl.dataset.Dataset'>
skipeveryn <class 'h5py._hl.dataset.Dataset'>
species <class 'h5py._hl.dataset.Dataset'>
store GC <class 'h5py._hl.dataset.Dataset'>
store trajectory <class 'h5py._hl.dataset.Dataset'>
tracklist_ID <class 'h5py._hl.dataset.Dataset'>
tracklist_L <class 'h5py._hl.dataset.Dataset'>
tracklist_check <class 'h5py._hl.dataset.Dataset'>
tracklist_mu <class 'h5py._hl.dataset.Dataset'>
tracklist_pa <class 'h5py._hl.dataset.Dataset'>
tracklist_pb <class 'h5py._hl.dataset.Dataset'>
tracklist_pd <class 'h5py._hl.dataset.Dataset'>
tracklist_pg <class 'h5py._hl.dataset.Dataset'>
tracks <class 'h5py._hl.group.Group'>
year <class 'h5py._hl.dataset.Dataset'>
```


# Example Configuration

The following example configuration file is provided in the configs/ directory:

## example1.txt
Simulates two 100MeV/G protons with equatorial pitch angles of 90 degrees at L=4, separated by 180 degrees in drift phase, for one drift orbit. Stores the guiding center of both particles, and calculates the adiabatic invariants before and after the drift orbit.

This example can be run by executing from the top level directory of the repository: `python pt_run.py --config configs/example1.txt --runname example1`
