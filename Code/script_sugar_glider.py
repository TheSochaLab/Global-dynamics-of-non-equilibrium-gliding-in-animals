# -*- coding: utf-8 -*-
"""
Created on Fri Nov 14 22:40:48 2014

%reset -f
%pylab
%clear
%load_ext autoreload
%autoreload 2

@author: isaac
"""

from __future__ import division

import numpy as np
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt

#import seaborn as sns
#sns.set('talk', 'ticks', rc={'mathtext.fontset': 'stixsans'})
#sns.set_context(rc={'lines.markeredgewidth': 0.1})

import eqns
reload(eqns)

import plots
reload(plots)
from plots import bmap, rcj, tl, four_plot



# %% Load in plate covered with fur data

d = np.genfromtxt('Data/Nachtigall1974/polar_0to90.csv',
                  delimiter=',', skip_header=1)
d = np.genfromtxt('Data/Nachtigall1974/polar_plate_with_fur.csv',
                  delimiter=',', skip_header=1)
aoapt, Cd, Cl = d.T
alphapt = np.deg2rad(aoapt)
arngpt = (alphapt.min(), alphapt.max())

fig, ((ax1, ax2), (ax3, ax4)) = four_plot(aoapt, Cl, Cd)
ax3.set_ylim(-2, 2.5)
#sns.despine()


# %% Spline fit to Cl, Cd curves

cl_fun = UnivariateSpline(alphapt, Cl, k=5, s=.01)
cd_fun = UnivariateSpline(alphapt, Cd, k=5, s=.01)
clprime_fun = cl_fun.derivative()
cdprime_fun = cd_fun.derivative()

Cl_fun = UnivariateSpline(alphapt, Cl, k=1, s=0)
Cd_fun = UnivariateSpline(alphapt, Cd, k=1, s=0)
Clprime_fun = Cl_fun.derivative()
Cdprime_fun = Cd_fun.derivative()

alpha = np.deg2rad(np.linspace(-15, 90, 251))

#eqns.save_spline(alpha, cl_fun(alpha), cd_fun(alpha),
#                 clprime_fun(alpha), cdprime_fun(alpha), 'nach_plate')


fig, axs = four_plot(np.rad2deg(alpha), cl_fun(alpha), cd_fun(alpha))
four_plot(np.rad2deg(alpha), Cl_fun(alpha), Cd_fun(alpha), fig=fig, axs=axs)
((ax1, ax2), (ax3, ax4)) = axs
ax3.set_ylim(-2, 2.5)
#sns.despine()


# %% Load in taxidermically preparted sugar glider

d4 = np.genfromtxt('Data/Nachtigall1974/F1_polar_4.5ms.csv',
                  delimiter=',', skip_header=1)
d1 = np.genfromtxt('Data/Nachtigall1974/F1_polar_10.1ms.csv',
                  delimiter=',', skip_header=1)

aoa4, Cde4, Cle4 = d4.T
aoa1, Cde1, Cle1 = d1.T

alpha4 = np.deg2rad(aoa4)
alpha1 = np.deg2rad(aoa1)


# %% Spline fit the data
# NOTE: we are using cl_fun_1 and cd_fun_1

# spline fit
ssl, ssd, kk = .0001, .0001, 3  # .001, .01, 5
cl_fun_1 = UnivariateSpline(alpha1, Cle1, s=ssl, k=kk)
cl_fun_4 = UnivariateSpline(alpha4, Cle4, s=ssl, k=kk)
cd_fun_1 = UnivariateSpline(alpha1, Cde1, s=ssd, k=kk)
cd_fun_4 = UnivariateSpline(alpha4, Cde4, s=ssd, k=kk)
clprime_fun_1 = cl_fun_1.derivative()
clprime_fun_4 = cd_fun_4.derivative()
cdprime_fun_1 = cd_fun_1.derivative()
cdprime_fun_4 = cd_fun_4.derivative()

# linear interpolation
ss, kk = 0, 1
Cl_fun_1 = UnivariateSpline(alpha1, Cle1, s=ss, k=kk)
Cl_fun_4 = UnivariateSpline(alpha4, Cle4, s=ss, k=kk)
Cd_fun_1 = UnivariateSpline(alpha1, Cde1, s=ss, k=kk)
Cd_fun_4 = UnivariateSpline(alpha4, Cde4, s=ss, k=kk)
Clprime_fun_1 = Cl_fun_1.derivative()
Clprime_fun_4 = Cd_fun_4.derivative()
Cdprime_fun_1 = Cd_fun_1.derivative()
Cdprime_fun_4 = Cd_fun_4.derivative()

# evaluate the fits
angle_min = np.deg2rad(aoa1.min())
angle_max = np.deg2rad(aoa1.max())
al_spl = np.linspace(angle_min, angle_max, 501)
aoa_spl = np.rad2deg(al_spl)

cl1 = cl_fun_1(al_spl)
cl4 = cl_fun_4(al_spl)
cd1 = cd_fun_1(al_spl)
cd4 = cd_fun_4(al_spl)

Cl1 = Cl_fun_1(al_spl)
Cl4 = Cl_fun_4(al_spl)
Cd1 = Cd_fun_1(al_spl)
Cd4 = Cd_fun_4(al_spl)

clprime1 = clprime_fun_1(al_spl)
clprime4 = clprime_fun_4(al_spl)
cdprime1 = cdprime_fun_1(al_spl)
cdprime4 = cdprime_fun_4(al_spl)

Clprime1 = Clprime_fun_1(al_spl)
Clprime4 = Clprime_fun_4(al_spl)
Cdprime1 = Cdprime_fun_1(al_spl)
Cdprime4 = Cdprime_fun_4(al_spl)


# %% Cl, Cd, and ClCd curves for paper

fig, ax = plt.subplots()
ax.axvline(0, color='gray', lw=1)
ax.axhline(0, color='gray', lw=1)
ax.plot(aoa1, Cle1, 'o', ms=6, label=r'$C_L$')
ax.plot(aoa1, Cde1, 's', ms=6, label=r'$C_D$')
ax.plot(aoa1, Cle1 / Cde1, '^', ms=6, label=r'$C_L/C_D$')
ax.plot(aoa_spl, cl1, color=bmap[0], lw=1.5)
ax.plot(aoa_spl, cd1, color=bmap[1], lw=1.5)
ax.plot(aoa_spl, cl1 / cd1, color=bmap[2], lw=1.5)

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

ax.text(5, 2.5, 'sugar glider', {'fontsize': 18})

[ttl.set_size(18) for ttl in ax.get_xticklabels()]
[ttl.set_size(18) for ttl in ax.get_yticklabels()]

rcj(ax)
tl(fig)

fig.savefig('Figures/figure4d_sugar_glider.pdf', transparent=True)


# %% Find equilibrium points

pitches = np.deg2rad(np.linspace(-25, 25, 4000))
gammas = np.deg2rad(np.linspace(0, 60, 1000))
arng = (angle_min, angle_max)

from eqns import pitch_bifurcation as ptbi
nt_Equil = ptbi(pitches, gammas, Cl_fun, Cd_fun, angle_rng=arngpt)
nt_equil = ptbi(pitches, gammas, cl_fun, cd_fun, angle_rng=arngpt)
nt_Equil_1 = ptbi(pitches, gammas, Cl_fun_1, Cd_fun_1, angle_rng=arng)
nt_equil_1 = ptbi(pitches, gammas, cl_fun_1, cd_fun_1, angle_rng=arng)
nt_Equil_4 = ptbi(pitches, gammas, Cl_fun_4, Cd_fun_4, angle_rng=arng)
nt_equil_4 = ptbi(pitches, gammas, cl_fun_4, cd_fun_4, angle_rng=arng)


# %% Classify the stability of fixed points

from eqns import tau_delta as td
nt_TD_1, nt_EV_1 = td(nt_Equil_1, Cl_fun_1, Cd_fun_1, Clprime_fun_1,
                      Cdprime_fun_1, angle_rng=arng)
nt_TD_4, nt_EV_4 = td(nt_Equil_4, Cl_fun_4, Cd_fun_4, Clprime_fun_4,
                      Cdprime_fun_4, angle_rng=arng)
nt_TD, nt_EV = td(nt_Equil, Cl_fun, Cd_fun, Clprime_fun,
                  Cdprime_fun, angle_rng=arngpt)
nt_td_1, nt_ev_1 = td(nt_equil_1, cl_fun_1, cd_fun_1, clprime_fun_1,
                      cdprime_fun_1, angle_rng=arng)
nt_td_4, nt_ev_4 = td(nt_equil_4, cl_fun_4, cd_fun_4, clprime_fun_4,
                      cdprime_fun_4, angle_rng=arng)
nt_td, nt_ev = td(nt_equil, cl_fun, cd_fun, clprime_fun,
                  cdprime_fun, angle_rng=arngpt)

_, _, nt_Class_1 = eqns.classify_fp(nt_TD_1)
_, _, nt_Class_4 = eqns.classify_fp(nt_TD_4)
_, _, nt_Class = eqns.classify_fp(nt_TD)
_, _, nt_class_1 = eqns.classify_fp(nt_td_1)
_, _, nt_class_4 = eqns.classify_fp(nt_td_4)
_, _, nt_class = eqns.classify_fp(nt_td)

#possible_class = ['saddle point', 'unstable spiral', 'unstable node',
#                  'stable spiral', 'stable node']
#bfbmap = [bmap[0], bmap[4], bmap[2], bmap[3], bmap[1]]

possible_class = ['saddle point', 'unstable focus',
                  'stable focus', 'stable node']
bfbmap = [bmap[0], bmap[4], bmap[3], bmap[1]]


# %% Spline bifurcation plot (deg) for paper

rd = np.rad2deg

gam_high = arng[0] - pitches  # closer to 0
gam_low = arng[1] - pitches  # closer to 90

fig, ax = plt.subplots()

#ax.plot(rd(pitches), rd(gam_low), c='gray')
#ax.plot(rd(pitches), rd(gam_high), c='r')

ax.fill_between(rd(pitches), rd(gam_high), 0, color='gray', alpha=.1, lw=0)
ax.fill_between(rd(pitches), rd(gam_low), 60, color='gray', alpha=.1, lw=0)

ax.axvline(0, color='gray')
ax.axvline(10, color='gray')

for ii, fp_kind in enumerate(possible_class):

    idx = np.where(nt_class_1 == fp_kind)[0]
    if len(idx) > 0:
        ax.plot(rd(nt_equil_1[idx, 0]), rd(nt_equil_1[idx, 1]), 'o',
                c=bfbmap[ii], ms=2.5, label=fp_kind)

_leg = ax.legend(loc='lower right', markerscale=3, fancybox=True, framealpha=.75,
                 frameon=True, fontsize=16, ncol=2)
_leg.get_frame().set_color('w')
ax.set_xlim(-15, 15)
ax.set_ylim(60, 0)
#ax.set_ylabel(r'$\gamma^*$, equilibrium glide angle', fontsize=18)
#ax.set_xlabel(r'$\theta$, pitch angle', fontsize=18)
ax.set_ylabel(r'$\gamma^*$    ', fontsize=18, rotation=0)
ax.set_xlabel(r'$\theta$', fontsize=18)

ax.text(-13, 5, 'sugar glider', {'fontsize': 18})

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

fig.savefig('Figures/figure6d_bifurcation_sugar_glider.pdf',
            transparent=True)


# %% Plot the equilibrium glide velocity on a polar plot

fig, ax = plt.subplots()

for ii, fp_kind in enumerate(possible_class):

    idx = np.where(nt_class_1 == fp_kind)[0]
    if len(idx) > 0:
        geq = nt_equil_1[idx, 1]
        veq = eqns.v_equil(geq, cl_fun_1, cd_fun_1)
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

fig.savefig('Figures/appendix_bifurcation_sugar_glider.pdf',
            transparent=True)


# %%

#afdict = dict(cli=cl_fun, cdi=cd_fun, clip=clprime_fun, cdip=cdprime_fun)
#angle_rng = arngpt  # for plate curves

afdict = dict(cli=cl_fun_1, cdi=cd_fun_1, clip=clprime_fun_1, cdip=cdprime_fun_1)
#afdict = dict(cli=cl_fun_4, cdi=cd_fun_4, clip=clprime_fun_4, cdip=cdprime_fun_4)

#afdict = dict(cli=Cl_fun_1, cdi=Cd_fun_1, clip=Clprime_fun_1, cdip=Cdprime_fun_1)

# used in old version of the paper
#afdict = dict(cli=Cl_fun_1, cdi=Cd_fun_1, clip=Clprime_fun_1, cdip=Cdprime_fun_1)

angle_rng = arng  # for taxodermically prepared specimens

pitch = 0

extrap = (None, None)  # (ale[0], ale[-1])
lims = (vxlim, vzlim) = np.r_[0, 1.25], np.r_[0, -1.25]

tvec = np.linspace(0, 30, 351)

reload(plots)
from plots import phase_plotter as ppr

fig, ax = ppr(afdict, pitch, lims, angle_rng, tvec, ngrid=201,
              nseed=501, nseed_skip=25, quiver=False, skip=25, seed=False,
              timer=True, gamtest=gammas, extrap=extrap,
              traj=plots.ps_traj_dp5, fig=None, ax=None)

lab = 'sugar glider, ' + r'$\theta=$0' + u'\u00B0'
ax.text(.05, -1, lab, fontsize=16)

fig.savefig('Figures/figure5di_vpd0_sugar_glider.pdf', transparent=True)


# %%

pitch = np.deg2rad(10)

fig, ax = ppr(afdict, pitch, lims, angle_rng, tvec, ngrid=201,
              nseed=501, nseed_skip=25, quiver=False, skip=25, seed=False,
              timer=True, gamtest=gammas, extrap=extrap,
              traj=plots.ps_traj_dp5, fig=None, ax=None)

lab = 'sugar glider, ' + r'$\theta=$10' + u'\u00B0'
ax.text(.05, -1, lab, fontsize=16)

fig.savefig('Figures/figure5dii_vpd10_sugar_glider.pdf', transparent=True)
