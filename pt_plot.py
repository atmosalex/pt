import pt_tools
import sys
from math import pi, sqrt
import numpy as np
import field_tools
from matplotlib.cm import ScalarMappable
import matplotlib.pyplot as plt
import random
from mpl_toolkits.mplot3d import Axes3D
import numpy.linalg as linalg
from matplotlib import animation
import datetime
import argparse
import matplotlib
mu_conv = pt_tools.constants.G2T/pt_tools.constants.MeV2J

def colors(n, truerandom = False):
    """
    generate n colours for plotting
    """
    if not truerandom: random.seed(2004)
    ret = [] 
    r = int(random.random() * 256) 
    g = int(random.random() * 256) 
    b = int(random.random() * 256) 
    step = 256 / n 
    for i in range(n): 
        r += step 
        g += step 
        b += step 
        r = int(r) % 256 
        g = int(g) % 256 
        b = int(b) % 256
        rgb = (r,g,b) 
        ret.append([float(x)/255 for x in rgb])  
    return ret 

def interpolate_constant_dt(times, positions, dt_min=-1):
    """
    convert a particle position vector array to have constant dt, useful for animations
    """
    if dt_min <= 0:
        dt = np.roll(times, -1) - times
        dt = dt[:-1]
        dt_min = np.min(dt)
        
    #interpolate the position array to minimum dt
    nt = int(np.ceil((times[-1] - times[0])/dt_min))
    newtimes = np.linspace(times[0], times[-1], nt)

    newpositions = []
    for newtime in newtimes:
        idx1 = np.argmin(newtime >= times)
        if newtime == times[idx1]:
            idx0 = idx1
        else:
            idx0 = idx1 - 1
        frac = 1 - (times[idx1] - newtime)/(times[idx1] - times[idx0])
        #print(0, (newtime - times[idx0])/ (times[idx1] - times[idx0]), frac)
        newpositions.append((1-frac)*positions[idx0] + frac * positions[idx1])
    newpositions = np.array(newpositions) 

    return newtimes, newpositions, dt_min

def plot_positions(positionslist, seeEarth=True, filename=None, limit=-1, view_ele = None, view_azi = None):
    # set up figure and axes:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    RE = pt_tools.constants.RE

    if seeEarth:
        from spacepy import coordinates as coord
        from spacepy.time import Ticktock
        import datetime
        date= datetime.datetime(2015,1,1)

        bfield = Dipolefield(date.year)
        
        x_ed_GEO_ = bfield.get_eccentric_centre_GEO()
        #change them to the MAG frame:
        x_ed_GEO = coord.Coords(x_ed_GEO_/RE, 'GEO', 'car')
        x_ed_GEO.ticks = Ticktock(date, 'UTC')
        x_ed_MAG = x_ed_GEO.convert('MAG', 'car')
        #x_ed_MAG.ticks = Ticktock(epoch, 'UTC')
        #
        x_ed_GEO = RE * np.array([x_ed_GEO.x[0], x_ed_GEO.y[0], x_ed_GEO.z[0]])
        x_ed_MAG = RE * np.array([x_ed_MAG.x[0], x_ed_MAG.y[0], x_ed_MAG.z[0]])

        A = WGS84.M
        A1 = A[:,0]
        A2 = A[:,1]
        A3 = A[:,2]
        x_Earth_GEO = coord.Coords(np.array([A1,A2,A3]), 'GEO', 'car')
        x_Earth_GEO.ticks = Ticktock(np.array([date]*3), 'UTC')
        x_Earth_MAG = x_Earth_GEO.convert('MAG', 'car')
        A = x_Earth_MAG.data

        # rotate it
        # https://math.stackexchange.com/questions/1403126/what-is-the-general-equation-equation-for-rotated-ellipsoid
        ### find the rotation matrix and radii of the axes
        U, s, rotation = linalg.svd(A)
        radii = 1.0 / np.sqrt(s)

        ue, ve = np.mgrid[0:2 * np.pi:24j, 0:np.pi:10j]


        xe = radii[0] * np.cos(ue) * np.sin(ve) / constants.RE
        ye = radii[1] * np.sin(ue) * np.sin(ve) / constants.RE
        ze = radii[2] * np.cos(ve) / constants.RE


        #add off-centre
        xe += x_ed_MAG[0]/ constants.RE
        ye += x_ed_MAG[1]/ constants.RE
        ze += x_ed_MAG[2]/ constants.RE
        ax.plot_wireframe(xe, ye, ze, color="deepskyblue", zorder=2)

    # draw particle trajectory:
    colours = colors(len(positionslist))
    for idx, positions in enumerate(positionslist):
        #if idx != 3: continue
        positions = positions / RE
        if limit > 0:
            ax.plot3D(positions[:, 0][:limit], positions[:, 1][:limit], positions[:, 2][:limit], color=colours[idx], zorder=1, linewidth=0.4)
        else:
            ax.plot3D(positions[:, 0], positions[:, 1], positions[:, 2], color=colours[idx], zorder=1, linewidth=0.4)

    # axis labels:
    plt.xlabel('$X_{MAG}$ (RE)')
    plt.ylabel('$Y_{MAG}$ (RE)')
    ax.set_zlabel('$Z_{MAG}$ (RE)')

    limits = np.array([getattr(ax, f'get_{axis}lim')() for axis in 'xyz'])
    ax.set_box_aspect(np.ptp(limits, axis = 1))
    if (view_ele != None and view_azi != None):
        ax.view_init(elev=view_ele, azim=view_azi)

    # Get rid of colored axes planes
    # First remove fill
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    # Now set color to white (or whatever is "invisible")
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
    ax.grid(False)


    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=300, facecolor='w', edgecolor='w', orientation='portrait')
    else:
        plt.show()
    plt.close()


def plot_positions2D_birdseye(positionslist, seeEarth=True, filename=None, ring = -1):
    # set up figure and axes:
    fig, ax1 = plt.subplots()

    if seeEarth:
        circle1 = plt.Circle((0, 0), 1, color='grey')
        ax1.add_patch(circle1)
    if ring > 1:
        circle2 = plt.Circle((0, 0), ring, color='b', fill=False)
        ax1.add_patch(circle2)


    colours = colors(len(positionslist))

    # draw particle trajectory:
    colours = colors(len(positionslist))
    for idx, positions in enumerate(positionslist):
        #if idx==1:continue
        positions = positions / pt_tools.constants.RE
        ax1.scatter(positions[:, 0], positions[:, 1], color=colours[idx], marker = ".", zorder=1)

    # axis labels:
    plt.xlabel('$X_{\mathrm{MAG}}$ [$R_{\mathrm{E}}$]')
    plt.ylabel('$Y_{\mathrm{MAG}}$ [$R_{\mathrm{E}}$]')
    ax1.axis('equal')

    if filename:
        plt.savefig(filename, dpi=300, facecolor='w', edgecolor='w', orientation='portrait')
    else:
        plt.show()
    plt.close()

def plot_positions2D_side(positionslist, seeEarth=True, filename=None, ring = -1):
    # set up figure and axes:
    fig, ax1 = plt.subplots()

    if seeEarth:
        circle1 = plt.Circle((0, 0), 1, color='grey')
        ax1.add_patch(circle1)
    if ring > 1:
        circle2 = plt.Circle((0, 0), ring, color='b', fill=False)
        ax1.add_patch(circle2)


    colours = colors(len(positionslist))

    # draw particle trajectory:

    colours = colors(len(positionslist))
    for idx, positions in enumerate(positionslist):
        positions = positions / pt_tools.constants.RE
        ax1.plot(-positions[:, 0], positions[:, 2], color=colours[idx], zorder=len(positionslist)-idx)

    # axis labels:
    plt.xlabel('$X_{\mathrm{MAG}}$ [$R_{\mathrm{E}}$]')
    plt.ylabel('$Z_{\mathrm{MAG}}$ [$R_{\mathrm{E}}$]')
    ax1.axis('equal')

    if filename:
        plt.savefig(filename, dpi=300, facecolor='w', edgecolor='w', orientation='portrait')
    else:
        plt.show()
    plt.close()





#set up parser and arguments: ------------------------------------------------+
parser = argparse.ArgumentParser(description='Get configuration file')

parser.add_argument("--solution",type=str, required=True)

args = parser.parse_args()

fileh5 = args.solution
#-----------------------------------------------------------------------------+





poslist = []
invariants = []
skipeveryn_additionally =1



pre_pds = []
pre_Ls = []
post_Ls = []
pre_ens = []
post_ens = []
post_aeqs = []
post_Ks = []


resultfile = pt_tools.HDF5_pt(fileh5, existing=True)
metadata = resultfile.read_root()
ptids = resultfile.get_solved_ids()
tracklist = resultfile.get_existing_tracklist()

#
#resultfile.print_file_tree()
#

axes_invariants_idx = [1, 2, 4]
axes_invariants_idx = [4, 2, 1]
axes_invariants_idx = [4, 1, 2]
axes_invariants_idx = [4, 1, 3]
axes_invariants_idx = [4, 0, 2]
axes_invariants_labels = ['$\mu$ [MeV/G]',
                          'E [MeV]',
                          '$K$ [G$^{0.5}$R$_E$]',
                          '$\\alpha_{\\mathrm{eq}}$ [$^{\circ}$]',
                          '$L$']
axes_invariants_logspace = [True,
                            True,
                            False,
                            False,
                            False]

fig, ax = plt.subplots()
colormap = plt.cm.jet  # or any other colormap

xyc0 = []
xyc1 = []
# xyc0_lost = []
for ptid in ptids:
    checkcode = ptids[ptid]
    if checkcode != 1:
        # checkcode > 1 could be caused by a number of issues, see return statements in solve_trajectory(...)
        print("pt ID {} has incorrect check code".format(ptid))
        continue

    #pg, pb, pd = tracklist[ptid][3:] #starting phases

    time, pos = resultfile.read_track(ptid)
    muenKalphaL0, muenKalphaL1 = resultfile.read_invariants(ptid)
    muenKalphaL0[0] = muenKalphaL0[0] * mu_conv
    muenKalphaL1[0] = muenKalphaL1[0] * mu_conv
    muenKalphaL0[3] = muenKalphaL0[3] * 180/np.pi
    muenKalphaL1[3] = muenKalphaL1[3] * 180/np.pi

    if muenKalphaL1[2] < 0 and muenKalphaL1[0] > 0:
        # K = -1 but mu, etc., is valid when bounce orbits could not correctly be ID'd
        # however the particle may not have been 'lost'
        print("pt ID {} has invalid K1".format(ptid))
        # xyc0_lost.append([
        #     muenKalphaL0[axes_invariants_idx[0]],
        #     muenKalphaL0[axes_invariants_idx[1]],
        #     muenKalphaL0[axes_invariants_idx[2]]])
        continue
    else:
        xyc0.append([
            muenKalphaL0[axes_invariants_idx[0]],
            muenKalphaL0[axes_invariants_idx[1]],
            muenKalphaL0[axes_invariants_idx[2]]])
        xyc1.append([
            muenKalphaL1[axes_invariants_idx[0]],
            muenKalphaL1[axes_invariants_idx[1]],
            muenKalphaL1[axes_invariants_idx[2]]])

xyc0 = np.array(xyc0)
xyc1 = np.array(xyc1)
# xyc0_lost = np.array(xyc0_lost)

cmin = min([min(xyc0[:,2]), min(xyc1[:,2])])#, min(xyc0_lost[:,2])])
cmax = max([max(xyc0[:,2]), max(xyc1[:,2])])#, max(xyc0_lost[:,2])])
if axes_invariants_logspace[axes_invariants_idx[2]]:
    normfunc = matplotlib.colors.LogNorm
else:
    normfunc = matplotlib.colors.Normalize
normalize = normfunc(vmin=cmin, vmax=cmax)
fig.colorbar(matplotlib.cm.ScalarMappable(norm=normalize, cmap=colormap), ax=ax, aspect=25, shrink=0.6, label=axes_invariants_labels[axes_invariants_idx[2]], pad=0., panchor=(0, 0.5))


#plot changes in trapped coordinate:
for idx in range(len(xyc0)):
    x0, y0, c0 = xyc0[idx]
    x1, y1, c1 = xyc1[idx]
    arrowprops = dict(arrowstyle='<-', color=colormap(normalize(c0)), lw=0.75, ls='-')
    arrowprops = dict(arrowstyle='<-', color='black', lw=0.5, ls='-')
    #ax.scatter([x0, x1],[y0, y1], c=[colormap(normalize(c)) for c in [c0,c1]], marker='.', zorder=1)

    ax.scatter([x1], [y1], color=colormap(normalize(c1)), marker='.', zorder=1)
    ax.annotate('', xy=(x0, y0),
                xycoords='data',
                xytext=(x1, y1),
                textcoords='data',
                arrowprops=arrowprops,
                zorder = 2)
    #ax.scatter([x0], [y0], color='white', edgecolors='black', marker='.', zorder=3)
    ax.scatter([x0], [y0], color=colormap(normalize(c0)), edgecolors='black', marker='.', zorder=3)

# #plot lost coordinates:
# colors = xyc0_lost[:,2]
# colrgb_all = colormap(normalize(colors))
# for idx in range(len(xyc0_lost)):
#     x0, y0, c0 = xyc0[idx]
#     ax.scatter([x0],[y0], color=colormap(normalize(c0)), marker='x')

ax.set_xlabel(axes_invariants_labels[axes_invariants_idx[0]])
ax.set_ylabel(axes_invariants_labels[axes_invariants_idx[1]])
ax.set_xscale(['linear','log'][axes_invariants_logspace[axes_invariants_idx[0]]])
ax.set_yscale(['linear','log'][axes_invariants_logspace[axes_invariants_idx[1]]])
#plt.show()
plt.savefig("Figure.png", dpi=200)


sys.exit()

for ptid in ptids:
    mu = tracklist[ptid][0]
    aeq = tracklist[ptid][1]
    L = tracklist[ptid][2]
    pg, pb, pd = tracklist[ptid][3:]
    checkcode = ptids[ptid]

    if checkcode == 0: continue
    time, pos = resultfile.read_track(ptid)
    muenKalphaL0, muenKalphaL1 = resultfile.read_invariants(ptid)
    #print(muenKalphaL0)


    #print(muenKalphaL0[0]* mu_conv, muenKalphaL0[1:], muenKalphaL0[3]*180/pi)

    pre_pds.append(pd)
    pre_Ls.append(muenKalphaL0[4])
    pre_ens.append(muenKalphaL0[1])
    if checkcode == 1:
        #print(muenKalphaL1[0]* mu_conv, muenKalphaL1[1:], muenKalphaL1[3]*180/pi)
        post_Ls.append(muenKalphaL1[4])
        post_ens.append(muenKalphaL1[1])
        post_aeqs.append(muenKalphaL1[3])
        post_Ks.append(muenKalphaL1[2])
    else:
        print("","post-tracking invariants could not be evaluated")
        post_Ls.append(-1)
        post_ens.append(-1)
        post_aeqs.append(-1)
        post_Ks.append(-1)
    print()

    if len(time) < 2: skipeveryn_additionally = 1
    time_ = time[::skipeveryn_additionally]
    pos_ = pos[::skipeveryn_additionally]
    poslist.append(pos_)

# print()
# print("Printing 3D overview, click and drag to move around...")
# plot_positions(poslist, seeEarth = False, view_ele = 0, view_azi = -71)
# print()
# print("Printing 2D overview looking down towards Earth...")
# plot_positions2D_birdseye(poslist, seeEarth = True, ring = -1)
# print()
# print("Printing 2D overview side-on...")
# plot_positions2D_side(poslist, seeEarth = True, ring = -1)


