# -*- coding: utf-8 -*-
"""
Created on Tue Sep 30 10:18:12 2014

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

from scipy.interpolate import UnivariateSpline

import eqns
reload(eqns)

# setup better plots
import plots
reload(plots)
from plots import bmap, rcj, tl


# %% Load in the data

def load_song(fname):
    data = np.genfromtxt('./Data/Song2008/{}.csv'.format(fname), delimiter=',')
    return data

cl_010 = load_song('cl_010mm')
cl_025 = load_song('cl_025mm')
cd_010 = load_song('cd_010mm')
cd_025 = load_song('cd_025mm')


# %% Interpolate them to the same degree vector

def fix_angles(data, mod_rng=False, low=-20, high=60):
    data[:, 0] = np.deg2rad(data[:, 0])
    low, high = np.deg2rad(low), np.deg2rad(high)

    if mod_rng:
        if data[0, 0] > low:
            print('Fixing low by {0:.6f}'.format(data[0, 0] - low))
            data[0, 0] = low
        if data[-1, 0] < high:
            print('Fixing high by {0:.6f}'.format(data[-1, 0] - high))
            data[-1, 0] = high

    return data

cl_10 = fix_angles(cl_010)
cl_25 = fix_angles(cl_025)
cd_10 = fix_angles(cd_010)
cd_25 = fix_angles(cd_025)


# %%

# spline fit
ss, kk = .03, 5
cl_fun_10 = UnivariateSpline(*cl_10.T, s=ss, k=kk)
cl_fun_25 = UnivariateSpline(*cl_25.T, s=ss, k=kk)
cd_fun_10 = UnivariateSpline(*cd_10.T, s=ss, k=kk)
cd_fun_25 = UnivariateSpline(*cd_25.T, s=ss, k=kk)
clprime_fun_10 = cl_fun_10.derivative()
clprime_fun_25 = cl_fun_25.derivative()
cdprime_fun_10 = cd_fun_10.derivative()
cdprime_fun_25 = cd_fun_25.derivative()

# linear interpolation
ss, kk = 0, 1
Cl_fun_10 = UnivariateSpline(*cl_10.T, s=ss, k=kk)
Cl_fun_25 = UnivariateSpline(*cl_25.T, s=ss, k=kk)
Cd_fun_10 = UnivariateSpline(*cd_10.T, s=ss, k=kk)
Cd_fun_25 = UnivariateSpline(*cd_25.T, s=ss, k=kk)
Clprime_fun_10 = Cl_fun_10.derivative()
Clprime_fun_25 = Cl_fun_25.derivative()
Cdprime_fun_10 = Cd_fun_10.derivative()
Cdprime_fun_25 = Cd_fun_25.derivative()

# evaluate the fits
angle_min = np.deg2rad(-20)
angle_max = np.deg2rad(60)
al_inp = np.deg2rad(np.arange(-20, 61, 2))
al_spl = np.linspace(angle_min, angle_max, 501)

cl10 = cl_fun_10(al_spl)
cl25 = cl_fun_25(al_spl)
cd10 = cd_fun_10(al_spl)
cd25 = cd_fun_25(al_spl)
Cl10 = cl_fun_10(al_inp)
Cl25 = cl_fun_25(al_inp)
Cd10 = cd_fun_10(al_inp)
Cd25 = cd_fun_25(al_inp)

clprime10 = clprime_fun_10(al_spl)
clprime25 = clprime_fun_25(al_spl)
cdprime10 = cdprime_fun_10(al_spl)
cdprime25 = cdprime_fun_25(al_spl)
Clprime10 = clprime_fun_10(al_inp)
Clprime25 = clprime_fun_25(al_inp)
Cdprime10 = cdprime_fun_10(al_inp)
Cdprime25 = cdprime_fun_25(al_inp)


# %% Look at the data

fig, ax = plt.subplots()
ax.plot(al_inp, Cl10, 'o')
ax.plot(al_inp, Cl25, 'o')
ax.plot(al_inp, Cd10, 'o')
ax.plot(al_inp, Cd25, 'o')
ax.plot(al_spl, cl10, c=bmap[0])
ax.plot(al_spl, cl25, c=bmap[1])
ax.plot(al_spl, cd10, c=bmap[2])
ax.plot(al_spl, cd25, c=bmap[3])
ax.axhline(0, color='gray', lw=.75)
ax.set_xlabel(r'$\alpha$', fontsize=16)
ax.set_ylabel('force coefficients')
rcj(ax)
tl(fig)


# %% Cl, Cd, and ClCd curves for paper (updated)
# use use the .25 mm membrane for the paper

rd = np.rad2deg
fig, ax = plt.subplots()
ax.axvline(0, color='gray', lw=1)
ax.axhline(0, color='gray', lw=1)
ax.plot(rd(al_inp), Cl25, 'o', ms=6, label=r'$C_L$')
ax.plot(rd(al_inp), Cd25, 's', ms=6, label=r'$C_D$')
ax.plot(rd(al_inp), Cl25 / Cd25, '^', ms=6, label=r'$C_L/C_D$')
ax.plot(rd(al_spl), cl25, color=bmap[0], lw=1.5)
ax.plot(rd(al_spl), cd25, color=bmap[1], lw=1.5)
ax.plot(rd(al_spl), cl25 / cd25, color=bmap[2])

ax.set_xlim(-15, 65)
ax.set_ylim(-2, 3)
ax.legend(loc='lower right', frameon=False, fontsize=18)
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

ax.text(35, 2.5, 'airfoil squirrel', {'fontsize': 18})

[ttl.set_size(18) for ttl in ax.get_xticklabels()]
[ttl.set_size(18) for ttl in ax.get_yticklabels()]

rcj(ax)
tl(fig)

fig.savefig('Figures/figure4a_airfoil_squirrel.pdf', transparent=True)


# %% Find equilibrium points

pitches = np.deg2rad(np.linspace(-25, 25, 4000))
gammas = np.deg2rad(np.linspace(0, 60, 1000))
arng = (angle_min, angle_max)

from eqns import pitch_bifurcation as ptbi
so_Equil_10 = ptbi(pitches, gammas, Cl_fun_10, Cd_fun_10, angle_rng=arng)
so_equil_10 = ptbi(pitches, gammas, cl_fun_10, cd_fun_10, angle_rng=arng)
so_Equil_25 = ptbi(pitches, gammas, Cl_fun_25, Cd_fun_25, angle_rng=arng)
so_equil_25 = ptbi(pitches, gammas, cl_fun_25, cd_fun_25, angle_rng=arng)


# %% Classify the stability of fixed points

from eqns import tau_delta as td
so_TD_10, so_EV_10 = td(so_Equil_10, Cl_fun_10, Cd_fun_10, Clprime_fun_10,
                        Cdprime_fun_10, angle_rng=arng)
so_TD_25, so_EV_25 = td(so_Equil_25, Cl_fun_25, Cd_fun_25, Clprime_fun_25,
                        Cdprime_fun_25, angle_rng=arng)
so_td_10, so_ev_10 = td(so_equil_10, cl_fun_10, cd_fun_10, clprime_fun_10,
                        cdprime_fun_10, angle_rng=arng)
so_td_25, so_ev_25 = td(so_equil_25, cl_fun_25, cd_fun_25, clprime_fun_25,
                        cdprime_fun_25, angle_rng=arng)

_, _, so_Class_10 = eqns.classify_fp(so_TD_10)
_, _, so_Class_25 = eqns.classify_fp(so_TD_25)
_, _, so_class_10 = eqns.classify_fp(so_td_10)
_, _, so_class_25 = eqns.classify_fp(so_td_25)

possible_class = ['saddle point', 'unstable focus', 'unstable node',
                  'stable focus', 'stable node']
bfbmap = [bmap[0], bmap[4], bmap[2], bmap[3], bmap[1]]


# %% Spline bifurcation plot (deg) for paper

rd = np.rad2deg

gam_high = arng[0] - pitches  # closer to 0
gam_low = arng[1] - pitches  # closer to 90

fig, ax = plt.subplots()

ax.fill_between(rd(pitches), rd(gam_high), 0, color='gray', alpha=.1, lw=0)
ax.fill_between(rd(pitches), rd(gam_low), 60, color='gray', alpha=.1, lw=0)

ax.axvline(0, color='gray')
ax.axvline(2, color='gray')

for ii, fp_kind in enumerate(possible_class):
    idx = np.where(so_class_25 == fp_kind)[0]
    if len(idx) == 0:
        continue
    ax.plot(rd(so_equil_25[idx, 0]), rd(so_equil_25[idx, 1]), 'o',
            c=bfbmap[ii], ms=2.5, label=fp_kind)

_leg = ax.legend(loc='upper right', markerscale=3, fancybox=True, framealpha=.75,
                 frameon=True, fontsize=16)
_leg.get_frame().set_color('w')
ax.set_xlim(-15, 15)
ax.set_ylim(60, 0)
#ax.set_ylabel(r'$\gamma^*$, equilibrium glide angle', fontsize=18)
#ax.set_xlabel(r'$\theta$, pitch angle', fontsize=18)
ax.set_ylabel(r'$\gamma^*$    ', fontsize=18, rotation=0)
ax.set_xlabel(r'$\theta$', fontsize=18)

ax.text(-13, 5, 'airfoil squirrel', {'fontsize': 18})

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

fig.savefig('Figures/figure6a_bifurcation_airfoil_squirrel.pdf',
            transparent=True)


# %% Plot the phase space, pitch = 0

#afdict_10 = dict(cli=cl_fun_10, cdi=cd_fun_10,
#                 clip=clprime_fun_10, cdip=cdprime_fun_10)

afdict_25 = dict(cli=cl_fun_25, cdi=cd_fun_25,
                 clip=clprime_fun_25, cdip=cdprime_fun_25)

pitch = 0

lims = (vxlim, vzlim) = np.r_[0, 1.25], np.r_[0, -1.25]

normalize = True
tvec = np.linspace(0, 30, 251)

import plots
reload(plots)
from plots import phase_plotter as ppr

fig, ax = ppr(afdict_25, pitch, lims, arng, tvec, ngrid=201,
              nseed=501, nseed_skip=25, quiver=False, skip=10, seed=False,
              timer=True, gamtest=gammas, traj=plots.ps_traj_dp5,
              fig=None, ax=None)

lab = 'airfoil squirrel, ' + r'$\theta=$0' + u'\u00B0'
ax.text(.05, -1, lab, fontsize=16)

fig.savefig('Figures/figure5ai_vpd0_airfoil_squirrel.pdf', transparent=True)


# %% Plot the phase space, pitch = 2

pitch = np.deg2rad(2)

fig, ax = ppr(afdict_25, pitch, lims, arng, tvec, ngrid=201,
              nseed=501, nseed_skip=25, quiver=False, skip=10, seed=False,
              timer=True, gamtest=gammas, traj=plots.ps_traj_dp5,
              fig=None, ax=None)

lab = 'airfoil squirrel, ' + r'$\theta=$2' + u'\u00B0'
ax.text(.05, -1, lab, fontsize=16)

fig.savefig('Figures/figure5aii_vpd2_airfoil_squirrel.pdf', transparent=True)


# %% Additional plots

fig, ax = plt.subplots()

for ii, fp_kind in enumerate(possible_class):

    idx = np.where(so_class_25 == fp_kind)[0]
    if len(idx) > 0:
        geq = so_equil_25[idx, 1]
        veq = eqns.v_equil(geq, cl_fun_25, cd_fun_25)
        vxeq = veq * np.cos(geq)
        vzeq = -veq * np.sin(geq)

        ax.plot(vxeq, vzeq, 'o', c=bfbmap[ii], ms=2, label=fp_kind,
                mec=bfbmap[ii])

ax.set_xlim(0, 1.5)
ax.set_ylim(-1.5, 0)
ax.set_xlabel(r"$\hat{v}_x$", fontsize=18)
ax.set_ylabel(r"$\hat{v}_z$", fontsize=18)
_leg = ax.legend(loc='best', markerscale=4, frameon=False, framealpha=.75)


ax.xaxis.set_label_position('top')
ax.xaxis.tick_top()
rcj(ax, ['bottom', 'right'])
tl(fig)

fig.savefig('Figures/appendix_bifurcation_airfoil_squirrel.pdf',
            transparent=True)
