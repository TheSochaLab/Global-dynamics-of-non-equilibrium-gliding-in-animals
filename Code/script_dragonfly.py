# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 12:37:11 2015

%reset -f
%pylab
%clear
%load_ext autoreload
%autoreload 2

@author: isaac
"""


from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline

# setup better plots
import plots
reload(plots)
from plots import bmap, rcj, tl

import squirrel
reload(squirrel)

import eqns
reload(eqns)


# %%

cl_raw = np.genfromtxt('Data/Wakeling1997/wakeling1997_cl.csv', delimiter=',')
cd_raw = np.genfromtxt('Data/Wakeling1997/wakeling1997_cd.csv', delimiter=',')

# first column of digitized data in radians
cl_raw[:, 0] = np.deg2rad(cl_raw[:, 0])
cd_raw[:, 0] = np.deg2rad(cd_raw[:, 0])

# data in paper evalued -15 to 45 deg, in 5 degree increments
aoa_deg = np.r_[-20, -10, 0, 10, 20, 30, 50, 70]
aoa_rad = np.deg2rad(aoa_deg)

# determine the actual experimental values
kind = 'linear'
Cl_raw = UnivariateSpline(cl_raw[:, 0], cl_raw[:, 1], s=0, k=1)
Cd_raw = UnivariateSpline(cd_raw[:, 0], cd_raw[:, 1], s=0, k=1)
Cl = Cl_raw(aoa_rad)
Cd = Cd_raw(aoa_rad)

# linear fit between points
Cl_fun = UnivariateSpline(aoa_rad, Cl, s=0, k=1)
Cd_fun = UnivariateSpline(aoa_rad, Cd, s=0, k=1)
Clprime_fun = Cl_fun.derivative()
Cdprime_fun = Cd_fun.derivative()

# fit a smoothing spline to experimental data
cl_fun = UnivariateSpline(aoa_rad, Cl, s=.0001, k=3)
cd_fun = UnivariateSpline(aoa_rad, Cd, s=.0001, k=3)
clprime_fun = cl_fun.derivative()
cdprime_fun = cd_fun.derivative()

# angles to evaluate spline over
angle_min, angle_max = aoa_rad.min(), aoa_rad.max()
alpha, dalpha = np.linspace(angle_min, angle_max, 1001, retstep=True)

# evaluate the spline
cl = cl_fun(alpha)
cd = cd_fun(alpha)
clprime = clprime_fun(alpha)
cdprime = cdprime_fun(alpha)

# evaluate the linear fit
Clf = Cl_fun(alpha)
Cdf = Cd_fun(alpha)
Clprime = Clprime_fun(alpha)
Cdprime = Cdprime_fun(alpha)


# %% Cl, Cd, and ClCd curves for paper (updated)

fig, ax = plt.subplots()
ax.axvline(0, color='gray', lw=1)
ax.axhline(0, color='gray', lw=1)
ax.plot(np.rad2deg(aoa_rad), Cl, 'o', ms=6, label=r'$C_L$')
ax.plot(np.rad2deg(aoa_rad), Cd, 's', ms=6, label=r'$C_D$')
ax.plot(np.rad2deg(aoa_rad), Cl / Cd, '^', ms=6, label=r'$C_L/C_D$')
ax.plot(np.rad2deg(alpha), cl, color=bmap[0], lw=1.5)
ax.plot(np.rad2deg(alpha), cd, color=bmap[1], lw=1.5)
ax.plot(np.rad2deg(alpha), cl / cd, color=bmap[2], lw=1.5)

ax.set_xlim(-15, 65)
ax.set_ylim(-2, 3)
#ax.set_ylim(-1, 4)
ax.legend(loc='upper right', frameon=False, fontsize=18)
ax.set_xlabel(r'$\alpha$', fontsize=18)
ax.set_ylabel('force coefficients', fontsize=18)

plt.draw()
# add degree symbol to angles
ticks = ax.get_xticklabels()
newticks = []
for tick in ticks:
    text = tick.get_text()
    newticks.append(text + u'\u00B0')
ax.set_xticklabels(newticks)

# ax.text(20, 2.5, 'flying fish', {'fontsize': 18})
ax.text(2, 2.75, 'dragonfly', {'fontsize': 18})

[ttl.set_size(18) for ttl in ax.get_xticklabels()]
[ttl.set_size(18) for ttl in ax.get_yticklabels()]

rcj(ax)
tl(fig)

fig.savefig('Figures/figure4e_dragonfly.pdf', transparent=True)


# %% Find equilibrium points

pitches = np.deg2rad(np.linspace(-45, 15, 4000))
gammas = np.deg2rad(np.linspace(10, 90, 1000))
angle_rng = (angle_min, angle_max)

df_equil_exp = eqns.pitch_bifurcation(pitches, gammas, Cl_fun, Cd_fun)
df_equil_spl = eqns.pitch_bifurcation(pitches, gammas, cl_fun, cd_fun,
                                      angle_rng=angle_rng)


# %% Classify the stability of fixed points

df_td_exp, df_ev_exp = eqns.tau_delta(df_equil_exp, Cl_fun, Cd_fun,
                                      Clprime_fun, Cdprime_fun)
df_td_spl, df_ev_spl = eqns.tau_delta(df_equil_spl, cl_fun, cd_fun,
                                      clprime_fun, cdprime_fun,
                                      angle_rng=angle_rng)


# %% Stability analysis for individual squirrels

df_nuni_exp, df_uni_exp, df_class_exp = eqns.classify_fp(df_td_exp)
df_nuni_spl, df_uni_spl, df_class_spl = eqns.classify_fp(df_td_spl)

possible_class = ['saddle point', 'unstable focus', 'unstable node',
                  'stable focus', 'stable node']
bfbmap = [bmap[0], bmap[4], bmap[2], bmap[3], bmap[1]]


# %% Spline bifurcation plot for paper

rd = np.rad2deg

gam_high = angle_rng[0] - pitches  # closer to 0
gam_low = angle_rng[1] - pitches  # closer to 90

fig, ax = plt.subplots()

ax.fill_between(rd(pitches), rd(gam_high), 0, color='gray', alpha=.1, lw=0)
ax.fill_between(rd(pitches), rd(gam_low), 90, color='gray', alpha=.1, lw=0)

ax.axvline(-10, color='gray')
ax.axvline(-25, color='gray')

for ii, fp_kind in enumerate(possible_class):
    idx = np.where(df_class_spl == fp_kind)[0]
    if len(idx) > 0:
        ax.plot(rd(df_equil_spl[idx, 0]), rd(df_equil_spl[idx, 1]), 'o',
                c=bfbmap[ii], ms=2.5, label=fp_kind)

_leg = ax.legend(loc='lower left', markerscale=3, fancybox=True, framealpha=.75,
                 frameon=True, fontsize=16)
_leg.get_frame().set_color('w')
ax.set_xlim(-30, 0)
ax.set_ylim(90, 30)
#ax.set_ylabel(r'$\gamma^*$, equilibrium glide angle', fontsize=18)
#ax.set_xlabel(r'$\theta$, pitch angle', fontsize=18)
ax.set_ylabel(r'$\gamma^*$    ', fontsize=18, rotation=0)
ax.set_xlabel(r'$\theta$', fontsize=18)

ax.text(-23, 32, 'dragonfly', {'fontsize': 18})

plt.draw()

# add degree symbol to angles
ticks = ax.get_xticklabels()
newticks = []
for tick in ticks:
    text = tick.get_text()
    newticks.append(text + u'\u00B0')
ax.set_xticklabels(newticks)

ticks = ax.get_yticklabels()
newticks = []
for tick in ticks:
    text = tick.get_text()
    newticks.append(text + u'\u00B0')
ax.set_yticklabels(newticks)

[ttl.set_size(18) for ttl in ax.get_xticklabels()]
[ttl.set_size(18) for ttl in ax.get_yticklabels()]

rcj(ax)
tl(fig)

fig.savefig('Figures/figure7e_bifurcation_dragonfly.pdf',
            transparent=True)


# %% Velocity polar diagram, pitch = -10

pitch = np.deg2rad(-10)

afdict = dict(cli=cl_fun, cdi=cd_fun, clip=clprime_fun, cdip=cdprime_fun)

lims = (vxlim, vzlim) = np.r_[0, 1.25], np.r_[0, -1.25]
tvec = np.linspace(0, 30, 351)

reload(plots)
from plots import phase_plotter as ppr

fig, ax = ppr(afdict, pitch, lims, angle_rng, tvec, ngrid=201,
              nseed=501, nseed_skip=25, quiver=False, skip=10, seed=False,
              timer=True, gamtest=gammas, extrap=None,
              traj=plots.ps_traj_dp5, fig=None, ax=None)

lab = 'dragonfly, ' + r'$\theta=$-10' + u'\u00B0'
ax.text(.05, -.1, lab, fontsize=16)

fig.savefig('Figures/figure6ei_vpd-10_dragonfly.pdf', transparent=True)


# %% Velocity polar diagram, pitch = -25

pitch = np.deg2rad(-25)

fig, ax = ppr(afdict, pitch, lims, angle_rng, tvec, ngrid=201,
              nseed=501, nseed_skip=25, quiver=False, skip=10, seed=False,
              timer=True, gamtest=gammas, extrap=None,
              traj=plots.ps_traj_dp5, fig=None, ax=None)

lab = 'dragonfly, ' + r'$\theta=$-25' + u'\u00B0'
ax.text(.05, -.1, lab, fontsize=16)

fig.savefig('Figures/figure6eii_vpd-25_dragonfly.pdf', transparent=True)
