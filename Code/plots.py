# -*- coding: utf-8 -*-
"""
Created on Sun Sep  7 11:13:15 2014

@author: isaac
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import time

import matplotlib
from matplotlib.pyplot import gca

# remove this dependency
# import brewer2mpl
# bmap = brewer2mpl.get_map('Set1', 'Qualitative', 5).mpl_colors
bmap = [(0.8941176470588236, 0.10196078431372549, 0.10980392156862745),
        (0.21568627450980393, 0.49411764705882355, 0.7215686274509804),
        (0.30196078431372547, 0.6862745098039216, 0.2901960784313726),
        (0.596078431372549, 0.3058823529411765, 0.6392156862745098),
        (1.0, 0.4980392156862745, 0.0)]
almost_black = '#262626'

# fix the sucky defaults
matplotlib.rcParams['figure.facecolor'] = 'white'
matplotlib.rcParams['axes.color_cycle'] = bmap
matplotlib.rcParams['figure.figsize'] = [6.0, 5]
matplotlib.rcParams['figure.max_open_warning'] = 50
matplotlib.rcParams['figure.dpi'] = 80
matplotlib.rcParams['image.interpolation'] = 'nearest'
matplotlib.rcParams['image.cmap'] = 'bone'
matplotlib.rcParams['lines.linewidth'] = 1
matplotlib.rcParams['lines.markeredgewidth'] = 0
matplotlib.rcParams['grid.color'] = 'gray'
matplotlib.rcParams['grid.alpha'] = .55
matplotlib.rcParams['grid.linestyle'] = '-'
matplotlib.rcParams['grid.linewidth'] = 0.5

matplotlib.rcParams['savefig.dpi'] = 300
matplotlib.rcParams['savefig.format'] = u'pdf'
matplotlib.rcParams['savefig.format'] = u'pdf'

# makes it nicer for the paper
matplotlib.rcParams['ytick.major.size'] = 5
matplotlib.rcParams['xtick.major.size'] = 5
matplotlib.rcParams['legend.numpoints'] = 1


# a good sans-serif font
# matplotlib.rcParams['font.sans-serif'] = 'Lucida Grande'
# matplotlib.rcParams['mathtext.default'] = 'regular'  # this is ugly

matplotlib.rcParams['text.usetex'] = False
matplotlib.rcParams['font.family'] = 'serif'
# matplotlib.rcParams['font.serif'] = 'Palatino Linotype'  # also nice
matplotlib.rcParams['font.serif'] = 'Times New Roman'
matplotlib.rcParams['mathtext.fontset'] = 'stix'

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


# Function definitions

def tl(fig):
    fig.set_tight_layout(True)


def tickout(ax):
    ax.yaxis.set_tick_params(direction='out')


def rcj(ax, sp=('top', 'right'), show_ticks=True):
    """
    Removes "chartjunk", such as extra lines of axes and tick marks.

    If grid="y" or "x", will add a white grid at the "y" or "x" axes,
    respectively

    If ticklabels="y" or "x", or ['x', 'y'] will remove ticklabels from that
    axis

    This function was taken from prettyplotlib.utils.remove_chartjunk and
    changed...not sure from which version...
    """

    spines = sp
    if ax is None:
        ax = gca()

    all_spines = ['top', 'bottom', 'right', 'left', 'polar']
    for spine in spines:
        # The try/except is for polar coordinates, which only have a 'polar'
        # spine and none of the others
        try:
            ax.spines[spine].set_visible(False)
        except KeyError:
            pass

    # For the remaining spines, make their line thinner and a slightly
    # off-black dark grey
    for spine in all_spines:
        if spine not in spines:
            # The try/except is for polar coordinates, which only
            # have a 'polar' spine and none of the others
            try:
                ax.spines[spine].set_linewidth(0.5)
            except KeyError:
                pass

    x_pos = set(['top', 'bottom'])
    y_pos = set(['left', 'right'])
    xy_pos = [x_pos, y_pos]
    xy_ax_names = ['xaxis', 'yaxis']

    for ax_name, pos in zip(xy_ax_names, xy_pos):
        axis = ax.__dict__[ax_name]
        # axis.set_tick_params(color=almost_black)
        if show_ticks or axis.get_scale() == 'log':
            # if this spine is not in the list of spines to remove
            for p in pos.difference(spines):
                axis.set_tick_params(direction='out')
                axis.set_ticks_position(p)
                #                axis.set_tick_params(which='both', p)
        else:
            axis.set_ticks_position('none')


# %% FUNCTIONS FOR PLOTTING THE VELOCITY POLAR DIAGRAM

def setup_grid(npts, vxlim, vzlim, pitch, arng):
    """Velocity and angle of attack grid. Angle of attack grid has
    nans outside of the set angle range.
    """

    from eqns import calc_gamma
    vx = np.linspace(vxlim[0], vxlim[1], npts)
    vz = np.linspace(vzlim[0], vzlim[1], npts)
    VX, VZ = np.meshgrid(vx, vz)
    AL = pitch + calc_gamma(VX, VZ)
    badidx = (AL < arng[0]) | (AL > arng[1])
    AL[badidx] = np.nan

    return vx, vz, VX, VZ, AL


def test_seed_locations():
    a = np.arange(100).reshape(10, 10)
    for i in range(10):
        for j in range(10):
            if i > j:
                a[i, j] = 0
            else:
                a[i, j] = 1
    als = a.copy()
    left = als > np.roll(als, 1, 1)
    right = als > np.roll(als, -1, 1)
    top = als > np.roll(als, 1, 0)
    bottom = als > np.roll(als, -1, 0)
    boarders = left + right + top + bottom
    print boarders * 1


def seed_locations(nseed, vxlim, vzlim, pitch, arng):
    """Where to starting integration trajectories from.
    """

    vxs, vzs, VXs, VZs, als = setup_grid(nseed, vxlim, vzlim, pitch, arng)
    als[~np.isnan(als)] = 1
    als[np.isnan(als)] = 0

    # make sure we have a boarder (if no bad data in wedge)
    als[:, 0] = 0
    als[:, -1] = 0
    als[0, :] = 0
    als[-1, :] = 0

    # http://www.youtube.com/watch?v=IKfJRyoiBlY, 16:17 min
    left = als > np.roll(als, 1, 1)
    right = als > np.roll(als, -1, 1)
    top = als > np.roll(als, 1, 0)
    bottom = als > np.roll(als, -1, 0)
    boarders = left + right + top + bottom

    return VXs[boarders].flatten(), VZs[boarders].flatten()


def ps_traj_dp5(x0, tvec, odeargs, model, arng, vxlim, vzlim,
                rett=False, ignore_vxvz=False):
    """Be smart about integrating the trajectory. This takes the
    same arguments and order as ps_traj.

    We use the technique presented here:
    http://stackoverflow.com/questions/12926393/using-adaptive-step-
    #sizes-with-scipy-integrate-ode
    """

    # This solver requires a flipped x and t; needs an array returned.
    def swap_model(t, x, args):
        return np.array(model(x, t, args))

    from eqns import calc_gamma
    from scipy.integrate import ode
    import warnings

    pitch = odeargs[0]

    # possible backends are 'vode', 'dopri5', and 'dop853'
    solver = ode(swap_model)
    solver.set_integrator('dopri5', nsteps=1, rtol=1e-13, atol=1e-13)
    solver.set_initial_value(x0, tvec[0])
    solver.set_f_params(odeargs)
    solver._integrator.iwork[2] = -1  # suppress Fortran-printed warning

    soln, tvecs = [], []
    warnings.filterwarnings("ignore", category=UserWarning)
    while np.abs(solver.t) < np.abs(tvec[-1]):
        solver.integrate(tvec[-1], step=True)
        out = solver.y
        aoa = pitch + calc_gamma(out[2], out[3])
        if aoa < arng[0] or aoa > arng[1]:
            break

        if not ignore_vxvz:
            if out[2] < vxlim[0] or out[2] > vxlim[1]:
                break
            if out[3] > vzlim[0] or out[3] < vzlim[1]:
                break

        soln.append(out)
        tvecs.append(solver.t)
    warnings.resetwarnings()
    soln = np.array(soln)
    tvecs = np.array(tvecs)
    if len(soln) == 0:
        soln = np.array([4 * [np.nan]])
        tvecs = np.array(np.nan)

    if rett:
        return soln, tvecs
    else:
        return soln


def ps_traj(x0, tvec, odeargs, model, arng, vxlim, vzlim):
    """Return the phase space trajectory. This can be called by
    mutliproessing to (hopefully) run faster.
    """

    from eqns import calc_gamma
    soln = odeint(model, x0, tvec, (odeargs,))

    # get rid of the extrapolation region
    pitch = odeargs[0]
    al = pitch + calc_gamma(soln[:, 2], soln[:, 3])
    badidx = np.where((al < arng[0]) | (al > arng[1]))[0]
    if len(badidx) > 0:
        soln[badidx[0]:, :] = np.nan

    # get rid of solutions that come back into the plot domain
    ix = np.where((soln[:, 2] < vxlim[0]) | (soln[:, 2] > vxlim[1]))[0]
    iz = np.where((soln[:, 3] > vzlim[0]) | (soln[:, 3] < vzlim[1]))[0]
    if len(ix) > 0:
        soln[ix[0]:, :] = np.nan
    if len(iz) > 0:
        soln[iz[0]:, :] = np.nan

    return soln


def saddle_filler(x, z, o=.0001):
    """Fill in some ICs around a fixed point to globalize the phase
    space.
    """

    xx = np.r_[x + o, x - o, x - o, x + o]
    zz = np.r_[z + o, z - o, z + o, z - o]

    return xx, zz


def phase_plotter(afdict, pitch, lims, arng, tvec, ngrid=201,
                  nseed=41, nseed_skip=1, quiver=False, skip=10, seed=False,
                  timer=False, gamtest=None, extrap=None, traj=None,
                  seedloc=None, fig=None, ax=None, acc_contour=True,
                  nullcline_x=False, nullcline_z=False):

    # unpack the data
    cli, cdi, = afdict['cli'], afdict['cdi']
    clip, cdip = afdict['clip'], afdict['cdip']

    # upack axis limits
    vxlim, vzlim = lims

    from eqns import cart_eqns, cart_model

    if traj is None:
        traj = ps_traj_dp5

    VXorig, VZorig, VX, VZ, ALPHA = setup_grid(ngrid, vxlim, vzlim,
                                               pitch, arng)
    CL = cli(ALPHA.flatten()).reshape(ALPHA.shape)
    CD = cdi(ALPHA.flatten()).reshape(ALPHA.shape)
    dVX, dVZ = cart_eqns(VX, VZ, CL, CD)
    AMAG = np.hypot(dVX, dVZ)
    AX, AZ = dVX / AMAG, dVZ / AMAG

    # calculation the integral curves
    now = time.time()
    # vxs, vzs, VXs, VZs, als = setup_grid(nseed, vxlim, vzlim, pitch, arng)
    vxseed, vzseed = seed_locations(nseed, vxlim, vzlim, pitch, arng)
    vxseed, vzseed = vxseed[::nseed_skip], vzseed[::nseed_skip]

    if seedloc is not None:
        vxseed = np.r_[vxseed, seedloc[:, 0]]
        vzseed = np.r_[vzseed, seedloc[:, 1]]
    odeargs = (pitch, cli, cdi)
    solns = []
    for i in range(len(vxseed)):
        x0 = (0, 0, vxseed[i], vzseed[i])
        soln = traj(x0, tvec, odeargs, cart_model, arng, vxlim, vzlim)
        solns.append(soln)
    if timer:
        print('Elapsed time: {:.3f}'.format(time.time() - now))

    # equilibrium points
    if gamtest is not None:
        from eqns import pitch_bifurcation as pb
        from eqns import v_equil, vxvz_equil, tau_delta, classify_fp
        equil = pb([pitch], gamtest, cli, cdi, angle_rng=arng)
        thbar, gambar = equil.T
        vbar = v_equil(pitch + gambar, cli, cdi)
        vxbar, vzbar = vxvz_equil(vbar, gambar)

        td, ev = tau_delta(equil, cli, cdi, clip, cdip, arng)
        _, _, fp_class = classify_fp(td)

        possible_class = ['saddle point', 'unstable focus', 'unstable node',
                          'stable focus', 'stable node']
        bfbmap = [bmap[0], bmap[4], bmap[2], bmap[3], bmap[1]]

    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(5.5, 5.125))

    if quiver:
        ax.quiver(VX[::skip, ::skip], VZ[::skip, ::skip],
                  AX[::skip, ::skip], AZ[::skip, ::skip],
                  color='gray', pivot='middle', alpha=.8, edgecolor='gray',
                  linewidths=0 * np.ones(VX[::skip, ::skip].size))

    for i in range(len(solns)):
        ax.plot(solns[i][:, 2], solns[i][:, 3], '-', ms=2.25, c='gray',
                alpha=.95, lw=.5)  # color=bmap[1])

    if seed:
        ax.plot(vxseed, vzseed, 'o', ms=2.5)

    if gamtest is not None:
        for ii, fp_kind in enumerate(possible_class):
            idx = np.where(fp_class == fp_kind)[0]
            if len(idx) == 0:
                continue

            # now globalize the phase space
            if fp_kind == 'saddle point':
                #  or fp_kind == 'stable node' or fp_kind == 'unstable node':
                tvec = np.linspace(tvec[0], 3 * tvec[-1], 6 * len(tvec))
                vxpt, vzpt = vxbar[idx], vzbar[idx]
                solns = []
                for i in range(len(vxpt)):
                    vxs, vzs = saddle_filler(vxpt[i], vzpt[i])
                    for vx0, vz0 in zip(vxs, vzs):
                        ic = (0, 0, vx0, vz0)
                        sn1 = traj(ic, tvec, odeargs, cart_model,
                                   arng, vxlim, vzlim)
                        sn2 = traj(ic, -tvec, odeargs, cart_model,
                                   arng, vxlim, vzlim)
                        solns.append(sn1)
                        solns.append(sn2)

                # plot the globalized saddles
                for i in range(len(solns)):
                    ax.plot(solns[i][:, 2], solns[i][:, 3], '-',
                            c=bfbmap[ii])

            # plot the equilibrium point
            ax.plot(vxbar[idx], vzbar[idx], 'o', ms=7, c=bfbmap[ii],
                    zorder=200)

    if acc_contour:
        ax.contourf(VX, VZ, AMAG, [0, .1], colors=[bmap[3]], alpha=.2)
        # ax.contourf(VX, VZ, dVZ, [0, .1], colors=[bmap[4]], alpha=.2)

    if extrap is not None and extrap[0] is not None:
        ax.contourf(VX, VZ, np.array(ALPHA < extrap[0]).astype(np.int),
                    [.5, 1.5], colors='gray', alpha=.1)
    if extrap is not None and extrap[1] is not None:
        ax.contourf(VX, VZ, np.array(ALPHA > extrap[1]).astype(np.int),
                    [.5, 1.5], colors='gray', alpha=.1)

    # plot the nullclines
    if nullcline_x:
        ax.contour(VX, VZ, dVX, [0], colors=[bmap[3]], alpha=1, zorder=100)
    if nullcline_z:
        ax.contour(VX, VZ, dVZ, [0], colors=[bmap[3]], alpha=1, zorder=100)

    ax.set_xlim(vxlim[0], vxlim[1])
    ax.set_ylim(vzlim[1], vzlim[0])

    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    ax.set_xlabel(r"$\hat{v}_x$", fontsize=20)
    ax.set_ylabel(r"$\hat{v}_z    $", fontsize=20, rotation=0)
    ax.set_aspect('equal', adjustable='box')  # these need to be square
    ax.set_xticks([0, .25, .5, .75, 1, 1.25])
    ax.set_yticks([0, -.25, -.5, -.75, -1, -1.25])
    ax.set_xticklabels(['0', '', '', '', '', '1.25'])
    ax.set_yticklabels(['0', '', '', '', '', '-1.25'])
    [ttl.set_size(18) for ttl in ax.get_xticklabels()]
    [ttl.set_size(18) for ttl in ax.get_yticklabels()]
    rcj(ax, ['bottom', 'right'])
    tl(fig)

    return fig, ax


def four_plot(aoa, cl, cd, cot=True, sty='-', label=None, fig=None, axs=None):
    """Plot an overview of the aerodynamics.
    """

    if cot:
        gm = np.deg2rad(aoa[aoa > 0])
        ctgm = 1 / np.tan(gm)

    if fig is None or axs is None:
        fig, axs = plt.subplots(2, 2, figsize=(9, 7.5))

    ((ax1, ax2), (ax3, ax4)) = axs

    ax1.plot(aoa, cl, sty, label=label)
    ax2.plot(aoa, cd, sty, label=label)
    ax3.plot(aoa, cl / cd, sty, label=label)
    if cot:
        ax3.plot(np.rad2deg(gm), ctgm)
    ax4.axhline(0, color='gray', lw=.75)
    ax4.axvline(0, color='gray', lw=.75)
    ax4.plot(cd, cl, sty, label=label)

    # ax4.axis('equal')

    ax1.set_xlabel(r'$\alpha$ (deg)')
    ax1.set_ylabel(r'$C_L$')
    ax2.set_xlabel(r'$\alpha$ (deg)')
    ax2.set_ylabel(r'$C_D$')
    ax3.set_xlabel(r'$\alpha$' + ' (deg)')
    ax3.set_ylabel(r'$C_L/C_D$')
    ax4.set_xlabel(r'$C_D$')
    ax4.set_ylabel(r'$C_L$')

    # sns.despine()
    [rcj(ax) for ax in np.array(axs).flat]
    fig.set_tight_layout(True)
    return fig, ((ax1, ax2), (ax3, ax4))
