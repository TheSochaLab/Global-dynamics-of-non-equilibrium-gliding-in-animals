# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 12:44:25 2015

%reset -f
%clear
%pylab
%load_ext autoreload
%autoreload 2

@author: isaac
"""

from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

import plots
reload(plots)
from plots import bmap, rcj, tl


# %%

def add_arrow_to_line2D(axes, line, arrow_locs=[.05, .275, .5, .725, .95],
                        arrowstyle='-|>', arrowsize=1, transform=None):
    """
    arrow_locs=[0.2, 0.4, 0.6, 0.8],

    http://stackoverflow.com/a/27666700


    Add arrows to a matplotlib.lines.Line2D at selected locations.

    Parameters:
    -----------
    axes:
    line: list of 1 Line2D obbject as returned by plot command
    arrow_locs: list of locations where to insert arrows, % of total length
    arrowstyle: style of the arrow
    arrowsize: size of the arrow
    transform: a matplotlib transform instance, default to data coordinates

    Returns:
    --------
    arrows: list of arrows
    """

    import matplotlib.lines as mlines
    import matplotlib.patches as mpatches

    if (not(isinstance(line, list)) or not(isinstance(line[0],
                                           mlines.Line2D))):
        raise ValueError("expected a matplotlib.lines.Line2D object")
    x, y = line[0].get_xdata(), line[0].get_ydata()

    arrow_kw = dict(arrowstyle=arrowstyle, mutation_scale=10 * arrowsize)

    color = line[0].get_color()
    use_multicolor_lines = isinstance(color, np.ndarray)
    if use_multicolor_lines:
        raise NotImplementedError("multicolor lines not supported")
    else:
        arrow_kw['color'] = color

    linewidth = line[0].get_linewidth()
    if isinstance(linewidth, np.ndarray):
        raise NotImplementedError("multiwidth lines not supported")
    else:
        arrow_kw['linewidth'] = linewidth

    if transform is None:
        transform = axes.transData

    arrows = []
    s = np.cumsum(np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2))
    for loc in arrow_locs:
        n = np.searchsorted(s, s[-1] * loc)
        arrow_tail = (x[n], y[n])
        # arrow_tail = (np.mean(x[n - 1:n + 1]), np.mean(y[n - 1:n + 1]))
        arrow_head = (np.mean(x[n:n + 2]), np.mean(y[n:n + 2]))
        p = mpatches.FancyArrowPatch(arrow_tail, arrow_head,
                                     transform=transform, **arrow_kw)
        axes.add_patch(p)
        arrows.append(p)
    return arrows


# %% Start with a velocity of 1.7 m/s in horizontal direction
# rescale this initial condition

Ws = 29
c = .022
g = 9.81
rho = 1.2
eps = (rho * g / 2) * (c / Ws)

vdim = 1.7
vnon_res = vdim / np.sqrt(2 * Ws / rho) # use (4.1) from paper


# %%

def eom(X, t, cl, cd):
    x, z, vx, vz = X
    vm = np.sqrt(vx**2 + vz**2)
    dx = vx
    dz = vz
    dvx = -vm * (cl * vz + cd * vx)
    dvz = vm * (cl * vx - cd * vz) - 1

    return dx, dz, dvx, dvz

# Snake 1: best
cl = .64
cd = .21

# Case 5: Best
cl = .61
cd = .14

# Case 6: Average
cl = .54
cd = .29

clcd = cl / cd

t = np.linspace(0, 20, 401)
X0 = np.array([0, 0, vnon_res, 0])

# solve the equations
soln = odeint(eom, X0, t, args=(cl, cd))
x, z, vx, vz = soln.T

# calculate accelerations
_, _, acx, acz = eom([0, 0, vx, vz], 0, cl, cd)

# glide angles
gl = np.rad2deg(-np.arctan(vz / vx))
gl_eq = np.rad2deg(np.arctan(cd / cl))
gl_min = .95 * gl_eq
gl_max = 1.05 * gl_eq

v_equil = 1 / (cl**2 + cd**2)**.25
vx_equil = v_equil * np.cos(np.deg2rad(gl_eq))
vz_equil = -v_equil * np.sin(np.deg2rad(gl_eq))

# map-out the low accleration magnitude region
vxrng = np.linspace(0, 1.5, 1001)
vzrng = np.linspace(0, -1.5, 1001)
Vx, Vz = np.meshgrid(vxrng, vzrng)
_, _, Ax, Az = eom([0, 0, Vx, Vz], 0, cl, cd)
Am = np.hypot(Ax, Az)


# %% Velocity polar diagram, with acceleration region and equilibrium
# condition

fig, ax = plt.subplots()

vx_seed = np.r_[0, .25, .5, .75, 1, 1.25]
for vx0 in vx_seed:
    X0 = np.array([0, 0, vx0, 0])
    soln = odeint(eom, X0, t, args=(cl, cd))
    x, z, vx, vz = soln.T
    ln = ax.plot(vx, vz, color='gray', lw=1.25)
    add_arrow_to_line2D(ax, ln, arrow_locs=[.35], arrowsize=2, arrowstyle='->')

vz_seed = np.r_[-1, -1.25, -1.5]
for vz0 in vz_seed:
    X0 = np.array([0, 0, 0, vz0])
    soln = odeint(eom, X0, t, args=(cl, cd))
    x, z, vx, vz = soln.T
    ln = ax.plot(vx, vz, color='gray', lw=1.25)
    add_arrow_to_line2D(ax, ln, arrow_locs=[.35], arrowsize=2, arrowstyle='->')

ax.contourf(Vx, Vz, Am, [0, .1], colors=[bmap[3]], alpha=.2)

ax.plot([0, 1.5], [0, -np.tan(np.deg2rad(gl_eq)) * 1.5], color=bmap[1], lw=1)
ax.plot([0, 1.5], [0, -np.tan(np.deg2rad(gl_min)) * 1.5], color=bmap[2], lw=1)
ax.plot([0, 1.5], [0, -np.tan(np.deg2rad(gl_max)) * 1.5], color=bmap[2], lw=1)
ax.plot(vx_equil, vz_equil, 'o', ms=8, color=bmap[3])

ax.set_xlim(0, 1.5)
ax.set_ylim(-1.5, 0)

ax.xaxis.set_label_position('top')
ax.xaxis.tick_top()
ax.set_xlabel(r"$\hat{v}_x$", fontsize=20)
ax.set_ylabel(r"$\hat{v}_z    $", fontsize=20, rotation=0)
ax.set_aspect('equal', adjustable='box')  # these need to be square
ax.set_xticks([0, .25, .5, .75, 1, 1.25, 1.5])
ax.set_yticks([0, -.25, -.5, -.75, -1, -1.25, -1.5])
ax.set_xticklabels(['0', '', '', '', '', '', '1.5'])
ax.set_yticklabels(['0', '', '', '', '', '', '-1.5'])
[ttl.set_size(18) for ttl in ax.get_xticklabels()]
[ttl.set_size(18) for ttl in ax.get_yticklabels()]
rcj(ax, ['bottom', 'right'])
tl(fig)

fig.set_tight_layout(True)

fig.savefig('../Figures/figure_SI_const_clcd.pdf', transparent=True)

