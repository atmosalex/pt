# **British Antarctic Survey Energetic Particle Tracer**

# Authorship

Developed by Alexander R. Lozinski (lozinskialexander@gmail.com).

Code in the boris\_solver\_example/ directory was contributed by Ravindra T. Desai and adapted to form part of the solver\_boris(...) function in pt\_fp.py where indicated by comments in that file.

# Usage

This set of scripts can be used to solve electron or proton particle trajectories initialised from a grid of adiabatic coordinates. The basic steps are:

1. Write a configuration file (see Configuration section)
2. Run python pt\_run.py --config configs/example1.txt, where configs/example1.txt is replaced by the local path to the desired configuration file. Particle trajectories will be solved and output to the pt\_solutions/ directory in a HDF5-format file.
3. Optionally, plot the trajectory (if stored) by running python pt\_plot.py --solution pt\_solutions/example1\_solutions.h5, where pt\_solutions/example1.h5 is replaced by the local path to the solution file.

The configuration file allows a great deal of flexibility, with the user able to control the initial distribution of particles in gryo, bounce and drift phase independently, and make use of advanced features such as re-calculating the values of adiabatic invariants following a given simulation to directly study non-adiabatic redistribution due to trapping limits, etc. To help familiarise the user, five example configuration files and their corresponding solutions are included with a brief description of each given in the Example Configurations section below.

# Dependencies

Python 3 is required to run this set of scripts, and the necessary dependencies can be installed via pip by running:

pip install —upgrade pip

pip install -r requirements.txt

# Configuration

A configuration file is formatted like a CSV file with each line following the syntax:

**keyword**, **parameter**

where **keyword** is a string, typed without quotation marks, describing some physical parameter, and **parameter** is some value assigned to the variable represented by **keyword**. Blank lines and lines beginning with # are ignored.


Keywords can appear in any order, but all of the following keywords must be present:

**species**, **orbit**, **duration to solve**, **store trajectory**, **store GC**, **find initial K**, **re-calculate invariants**, **year**, **month**, **day**, **lmin**, **lmax**, **lsize**, **amin**, **amax**, **asize**, **logmumin**, **logmumax**, **dlogmu**, **nphase\_gyro**, **nphase\_bounce**, **nphase\_drift**, **iphase\_gyro**, **iphase\_bounce**, **iphase\_drift**, **fieldpath**, **emin**, **emax**, **skipeveryn**, **continuefrom**.

The meaning of each keyword is indicated below, and examples of acceptable parameter values are indicated for each in bold font. Specifying a parameter value is optional for some keywords even though each keyword must appear, and in this case **parameter** can be let blank.

**species**

Describes the species of particle to trace: **p** for protons or **e** for electrons.

**orbit**

Describes the type of orbit to simulate: **b** for a single bounce, **d** for a single drift, or **t** for a custom duration.

**duration to solve**

If orbit type **t** is specified, this is the trajectory duration as an integer or float value in units of seconds, such as **180** for 180 seconds, etc.

**store trajectory**

Controls whether or not the trajectory of the particle is saved in the output file, accepted values are **y** or **n**. Files can potentially become large for long trajectories when **y** is specified.

**store GC**

Controls whether or not the guiding centre of the trajectory is calculated and saved in the output file, accepted values are **y** or **n**. The **store trajectory** function must be enabled for this to work, and the guiding centre will overwrite the particle trajectory.

**find initial K**

Controls whether or not the initial value of the second adiabatic invariant K is calculated and saved in the output file, accepted values are **y** or **n**. 

**re-calculate invariants**

Controls whether or not the adiabatic invariants mu, K and L are re-calculated at the end of the trajectory for each particle, then saved in the output file. Accepted values are **y** or **n**. This can add considerable computation time depending on the number of particles, etc. Both the initial and final invariant values will be stored in the solution file for comparison.

**year**

**month**

**day**

Describes epoch of the IGRF coefficients used to calculate the dipole field moment. The parameter values must be integer, integer, integer respectively i.e. **2015**, **1**, **1** for 1st Jan. 2015.

**lmin**

**lmax**

**lsize**

Sets up a grid in the L coordinate to populate with particles. The parameter values must be float, float, integer respectively i.e. **2.0**, **2.0**, **1** to consider particles only at L=2, or **2.0, 3.0**, **11** to evenly space particles between L=2 and L=3 at 11 coordinates.

**amin**

**amax**

**asize**

Sets up a grid in the equatorial pitch angle coordinate to populate with particles. The parameter values must be float, float, integer respectively in units of degrees i.e. **90**, **90**, **1** to consider equatorially mirroring particles only, or **10.0, 90.0**, **11** to evenly space particles between aeq=10 and aeq=90 at 11 coordinates.

**logmumin**

**logmumax**

**dlogmu**

Sets up a grid in the mu coordinate to populate with particles. The parameter values must be float, float, integer respectively in units of log10(mu / 1MeV/G) i.e. **2**, **2**, **1** to consider 100MeV/G particles only, or **2, 3**, **11** to evenly space particles between mu=100MeV/G and mu=1000MeV/G at 11 coordinates.

**nphase\_gyro**

**nphase\_bounce**

**nphase\_drift**

Controls the phase distribution of particles across the particle grid along each of the three types of periodic motion. The parameter values must be integer, integer, integer respectively i.e. **1**, **1**, **1** to consider only a single particle at each of the adiabatic coordinates generated by the previously described grid options, or **1**, **1**, **24** to consider 24 particles distributed evenly in drift phase at each set of adiabatic coordinates, etc.

**iphase\_gyro**

**iphase\_bounce**

**iphase\_drift**

Controls the initial conditions of phase distribution along each of the three types of periodic motion, normalised between 0 and 1. The parameter values must be float, float, float respectively i.e. **0**, **0**, **0**. Changing from **0**, **0**, **0** to **0.5**, **0.5**, **0.5** will effectively shift the initial phase of each particle by 180 degrees along the gyro, bounce and drift path. In the case of non-zero values for iphase\_bounce, or if nphase\_bounce is greater than 1, the bounce trajectory must be pre-computed before the distribution can be initialised in space, since the path cannot be solved analytically, and this may add computation time.

**fieldpath**

Optional. An experimental feature that can be used to load in time-varying electromagnetic fields. Leave blank.

**emin**

**emax**

Controls the maximum and minimum energy of particles to consider. The parameter values must be float, float respectively with units MeV i.e. 10, 500 to skip any calculation below 10MeV or above 500MeV.

**skipeveryn**

Controls the number of trajectory points to be stored in any output file. The parameter value must be an integer. Setting this above 1 will cause only every nth point to be output, useful to reducing the size of trajectory files. A recommended value is 10.

**continuefrom**

Optional. If a previous simulation is interrupted, this keyword can be used to continue from it by specifying the name of the output file, i.e. **pt\_solutions/pt\_20230117-091223\_solutions.h5**. Any trajectory that was already solved before will be skipped, and the file will be updated as new trajectories are solved.

# Solutions

Solutions are stored as .h5 files in the pt\_solutions/ directory, with a name generated based on the time at which they were run. This behaviour can be overridden by supplying the --runname argument, i.e. python pt\_run.py --config configs/example1.txt --runname custom\_name, where custom\_name will be the new name of the solution in pt\_solutions/.


The configuration is also stored for each solution with a corresponding name, i.e. the file pt\_20230117-105105\_solutions.h5 will be accompanied by pt\_20230117-105105\_config.txt, which is a backup of the configuration file used to launch the simulation.

A plaintext tracklist file will also accompany the .h5 file, containing a list of the solved particle trajectories in terms of each particle’s basic properties, i.e. pt\_20230117-105105\_tracklist.txt.

Each .h5 file is a HDF5 database storing the properties of each particle along with their solved trajectories. The structure of an .h5 file can be printed by running python pt\_plot.py --solution pt\_solutions/example1\_solutions.h5, where pt\_solutions/example1.h5 is replaced by the local path to the solution file. The output of this command for the example1 solution (see Example Configurations) is:

```
amax = 17.0
amin = 85.0
asize = 3
continuefrom = './pt\_solutions/pt\_20230117-105105\_solutions.h5'
day = 1dlogmu = 1
duration to solve = 0.0
emax = 400.0emin = 0.1
fieldpath = ''
find initial K = 'n'
iphase\_bounce = 0.0
iphase\_drift = 0.0
iphase\_gyro = 0.25
lmax = 2.0
lmin = 2.0
logmumax = 2.0
logmumin = 2.0
lsize = 1
month = 1
nphase\_bounce = 1
nphase\_drift = 1
nphase\_gyro = 1
orbit = 'b'
re-calculate invariants = 'n'
skipeveryn = 10
species = 'p'
store GC = 'n'
store trajectory = 'y'
tracklist\_ID = [0 1 2]
tracklist\_L = [2. 2. 2.]
tracklist\_check = [1 1 1]
tracklist\_mu = [100. 100. 100.]
tracklist\_pa = [85. 51. 17.]
tracklist\_pb = [0. 0. 0.]
tracklist\_pd = [0. 0. 0.]
tracklist\_pg = [0.25 0.25 0.25]
tracks:NXgroup
  0:NXgroup
    muenKalphaL0 = float64(5)
    muenKalphaL1 = float64(5)
    position = float64(4992x3)
    time = float64(4992)
  1:NXgroup
    muenKalphaL0 = float64(5)
    muenKalphaL1 = float64(5)
    position = float64(5818x3)
    time = float64(5818)
  2:NXgroup
    muenKalphaL0 = float64(5)
    muenKalphaL1 = float64(5)
    position = float64(9339x3)
    time = float64(9339)
year = 2015
```


# Example Configurations

The following example configurations files are provided in the configs/ directory, with brief descriptions below:

## example1.txt
Simulates three 100MeV/G protons with equatorial pitch angles of 17, 51 and 85 degrees at L=2. The protons bounce once, grouped closely in longitude.

## example2.txt
Simulates 12 100MeV/G protons at the three adiabatic coordinates of the example1 simulation, but distributed in gyro and bounce phase via nphase\_gyro and nphase\_bounce. However, they are still grouped closely in longitude.

## example3.txt
Simulates 3 100MeV/G protons with equatorial pitch angles of 88 degrees for a single drift period at L=2, evenly spaced in initial drift phase.

## example4.txt
The same as example3, but storing the guiding centres.

## example5.txt
Simulates a ~250MeV/G electron at L=6 with an equatorial pitch angle of 50 degrees for 30 seconds, storing only the guiding centre.
