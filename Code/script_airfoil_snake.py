# -*- coding: utf-8 -*-
"""
Created on Thu Aug 14 14:16:45 2014

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
from scipy.io import loadmat

# setup better plots
import plots
reload(plots)
from plots import bmap, rcj, tl

import eqns
reload(eqns)


# %% Load in the provided data

data = loadmat('Data/Holden2014/aero_data_mod360.mat')

# get out the data
Clall = data['C_lift'].flatten()
Cdall = data['C_drag'].flatten()
alphaall = data['alpha'].flatten()


# %% "raw" experimental data

idx_exp = np.where((alphaall >= -np.deg2rad(12)) &
                   (alphaall <= np.deg2rad(61)))[0]

ale = alphaall[idx_exp]
Cle = Clall[idx_exp]
Cde = Cdall[idx_exp]
ClCde = Cle / Cde
#Clprimee = np.gradient(Cle, np.deg2rad(5))
#Cdprimee = np.gradient(Cde, np.deg2rad(5))

Cl_fun = UnivariateSpline(ale, Cle, k=1, s=0)
Cd_fun = UnivariateSpline(ale, Cde, k=1, s=0)
#ClCd_fun = interp1d(ale, ClCde, bounds_error=False)
Clprime_fun = Cl_fun.derivative()
Cdprime_fun = Cd_fun.derivative()

Clprimee = Clprime_fun(ale)
Cdprimee = Cdprime_fun(ale)


# %% "valid" region where date was recorded (-10 to 60 deg aoa)

idx_fit = np.where((alphaall >= -np.deg2rad(12)) &
                   (alphaall <= np.deg2rad(61)))[0]  # was 91

alf = alphaall[idx_fit]
Clf = Clall[idx_fit]
Cdf = Cdall[idx_fit]
ClCdf = Clf / Cdf
#Clprimef = np.gradient(Clf, 5)
#Cdprimef = np.gradient(Cdf, 5)

#s = .005
s = .0001
cl_fun = UnivariateSpline(alf, Clf, s=s, k=2)
cd_fun = UnivariateSpline(alf, Cdf, s=s, k=2)
clprime_fun = cl_fun.derivative()
cdprime_fun = cd_fun.derivative()

# numerically evaluate the spline
al = np.linspace(alf[0], alf[-1], 500)
cl = cl_fun(al)
cd = cd_fun(al)
clprime = clprime_fun(al)
cdprime = cdprime_fun(al)
clcd = cl / cd


# %% Cl, Cd, and ClCd curves for paper (updated)

fig, ax = plt.subplots()
ax.axvline(0, color='gray', lw=1)
ax.axhline(0, color='gray', lw=1)
ax.plot(np.rad2deg(alf), Clf, 'o', ms=6, label=r'$C_L$')
ax.plot(np.rad2deg(alf), Cdf, 's', ms=6, label=r'$C_D$')
ax.plot(np.rad2deg(alf), ClCdf, '^', ms=6, label=r'$C_L/C_D$')
ax.plot(np.rad2deg(al), cl, color=bmap[0], lw=1.5)
ax.plot(np.rad2deg(al), cd, color=bmap[1], lw=1.5)
ax.plot(np.rad2deg(al), clcd, color=bmap[2], lw=1.5)

ax.set_xlim(-15, 65)
ax.set_ylim(-2, 3)
ax.legend(loc='lower right', frameon=False, fontsize=18)
ax.set_xlabel(r'$\alpha$', fontsize=18)
ax.set_ylabel('force coefficients', fontsize=18)

fig.canvas.draw()
# add degree symbol to angles
ticks = ax.get_xticklabels()
newticks = []
for tick in ticks:
    text = tick.get_text()
    newticks.append(text + u'\u00B0')
ax.set_xticklabels(newticks)

ax.text(5, 2.5, 'airfoil snake', {'fontsize': 18})

[ttl.set_size(18) for ttl in ax.get_xticklabels()]
[ttl.set_size(18) for ttl in ax.get_yticklabels()]

rcj(ax)
tl(fig)

fig.savefig('Figures/figure4b_airfoil_snake.pdf', transparent=True)


# %% Intersections with spline data (for paper about pitch effects)

gamma = al
cgamma = 1 / np.tan(gamma)

pitch_array = np.deg2rad(np.array([-10, 10]))
_gamma_equil = np.deg2rad(np.linspace(10, 70, 1000))

fig, ax = plt.subplots()
ax.plot(np.rad2deg(gamma[gamma > 0]), cgamma[gamma > 0], c=bmap[2], lw=2,
        label=r'$\cot{\gamma}$')

for idx, pitch in enumerate(pitch_array):
    alpha = gamma + pitch
    drag = cd_fun(alpha)
    lift = cl_fun(alpha)
    ratio = lift / drag
    goodidx = np.where((alpha > al[0]) & (alpha < al[-1]))[0]
    lb_txt = r'$\theta = {:.0f}$'.format(np.rad2deg(pitch))
    lb_txt = lb_txt + u'\u00B0'
    _ln, = ax.plot(np.rad2deg(gamma[goodidx]), ratio[goodidx], lw=2,
                   label=lb_txt, c=bmap[idx])

    # find equilibrium points
    peq, geq = eqns.pitch_bifurcation([pitch], _gamma_equil, cl_fun, cd_fun,
                                      angle_rng=(al[0], al[-1])).T
    aeq = peq + geq
    ratio_eq = cl_fun(aeq) / cd_fun(aeq)
    _c = _ln.get_color()
    ax.plot(np.rad2deg(geq), ratio_eq, 'o', c=_c, mec=_c, ms=9)
#    for i in range(len(geq)):
#        ax.axvline(np.rad2deg(geq[i]), color=_c)


leg = ax.legend(loc='upper right', frameon=False, fontsize=18)
#ax.set_xlim(np.deg2rad(np.r_[-10, 90]))
ax.set_xlim(0, 60)
ax.set_ylim(0, 3)
ax.set_xlabel(r'$\gamma$, glide angle', fontsize=18)
ax.set_ylabel(r'$C_L/C_D(\gamma + \theta)$', fontsize=18)

fig.canvas.draw()
# add degree symbol to angles
ticks = ax.get_xticklabels()
newticks = []
for tick in ticks:
    text = tick.get_text()
    newticks.append(text + u'\u00B0')
ax.set_xticklabels(newticks)

[ttl.set_size(18) for ttl in ax.get_xticklabels()]
[ttl.set_size(18) for ttl in ax.get_yticklabels()]

rcj(ax)
tl(fig)

fig.savefig('Figures/figure2_effect_of_pitch.pdf', transparent=True)


# %% Find the glide angle and velocity at equilibrium (pitch of 0 deg)

peq, geq = eqns.pitch_bifurcation([0], _gamma_equil, cl_fun, cd_fun,
                                  angle_rng=(al[0], al[-1])).T

peq, geq = float(peq), float(geq)
veq = eqns.v_equil(geq, cl_fun, cd_fun)
vxeq, vzeq = eqns.vxvz_equil(veq, geq)

cleq, cdeq = cl_fun(geq), cd_fun(geq)

assert np.allclose(np.arctan(cdeq / cleq), geq)


# %% Find equilibrium points

pitches = np.deg2rad(np.linspace(-25, 25, 4000))
gammas = np.deg2rad(np.linspace(10, 70, 1000))
sn_angle_rng = (al[0], al[-1])

sn_equil_exp = eqns.pitch_bifurcation(pitches, gammas, Cl_fun, Cd_fun,
                                      angle_rng=sn_angle_rng)
sn_equil_spl = eqns.pitch_bifurcation(pitches, gammas, cl_fun, cd_fun,
                                      angle_rng=sn_angle_rng)


# %% Classify the stability of fixed points

sn_td_exp, sn_ev_exp = eqns.tau_delta(sn_equil_exp, Cl_fun, Cd_fun,
                                      Clprime_fun, Cdprime_fun,
                                      angle_rng=sn_angle_rng)
sn_td_spl, sn_ev_spl = eqns.tau_delta(sn_equil_spl, cl_fun, cd_fun,
                                      clprime_fun, cdprime_fun,
                                      angle_rng=sn_angle_rng)


# %% Classification of fixed points

sn_nuni_exp, sn_uni_exp, sn_class_exp = eqns.classify_fp(sn_td_exp)
sn_nuni_spl, sn_uni_spl, sn_class_spl = eqns.classify_fp(sn_td_spl)

possible_class = ['saddle point', 'unstable focus', 'unstable node',
                  'stable focus', 'stable node']
bfbmap = [bmap[0], bmap[4], bmap[2], bmap[3], bmap[1]]


# %% Acceleration along terminal manifold when we have a saddle point

sad_idx = np.where(sn_class_spl == 'saddle point')[0]
sad_pitch, sad_gamma = sn_equil_spl[sad_idx].T

# we have some double saddle points below theta =2 deg; remove these
sad_idx = np.where(sad_pitch >= np.deg2rad(2))[0]
sad_pitch, sad_gamma = sad_pitch[sad_idx], sad_gamma[sad_idx]

sad_aoa = sad_pitch + sad_gamma

dcl_fun = cl_fun.derivative()
ddcl_fun = dcl_fun.derivative()
dcd_fun = cd_fun.derivative()
ddcd_fun = dcd_fun.derivative()

# 2nd order spline, needs more to get higher derivatives
#dddcl_fun = ddcl_fun.derivative()
#dddcd_fun = ddcd_fun.derivative()

# evaluate force coefficients at the saddle
sad_cl = cl_fun(sad_aoa)
sad_dcl = dcl_fun(sad_aoa)
sad_ddcl = ddcl_fun(sad_aoa)
sad_ddcl = np.zeros_like(sad_aoa)

sad_cd = cd_fun(sad_aoa)
sad_dcd = dcd_fun(sad_aoa)
sad_ddcd = ddcd_fun(sad_aoa)
sad_dddcd = np.zeros_like(sad_aoa)

# place the values in a large array for export
sad_angles = np.c_[np.rad2deg(sad_pitch), sad_pitch, sad_gamma, sad_aoa]
sad_lift = np.c_[sad_cl, sad_dcl, sad_ddcl, sad_ddcl]
sad_drag = np.c_[sad_cd, sad_dcd, sad_ddcd, sad_ddcd]
sad_export = np.c_[sad_angles, sad_lift, sad_drag]


# %%

# save the data
import pandas as pd


node_idx = np.where(sn_class_spl == 'stable node')[0]

node_pitch, node_gamma = sn_equil_spl[node_idx].T

# nodes, select ones with saddles
node_idx_with_saddles = np.where(np.in1d(node_pitch, sad_pitch))[0]

node_pitch = node_pitch[node_idx_with_saddles]
node_gamma = node_gamma[node_idx_with_saddles]

# do the reverse to ensure we have the same number of values
sad_idx_with_nodes = np.where(np.in1d(sad_pitch, node_pitch))[0]

# too many indices...
node_idx_with_saddles = []
for i in np.arange(len(sad_pitch)):
    s_pitch = sad_pitch[i]
    idx = np.where(node_pitch == s_pitch)[0]
    if len(idx) == 0:
        continue
    elif len(idx) == 1:
        node_idx_with_saddles.append(idx)
    elif len(idx) > 1:
        for ii in np.arange(len(idx)):
            node_idx_with_saddles.append(idx[ii])

node_idx_with_saddles = np.array(node_idx_with_saddles)


# %% Spline bifurcation plot (deg) for paper

rd = np.rad2deg

gam_high = sn_angle_rng[0] - pitches  # closer to 0
gam_low = sn_angle_rng[1] - pitches  # closer to 90

fig, ax = plt.subplots()

ax.fill_between(rd(pitches), rd(gam_high), 0, color='gray', alpha=.1, lw=0)
ax.fill_between(rd(pitches), rd(gam_low), 60, color='gray', alpha=.1, lw=0)

ax.axvline(0, color='gray')
ax.axvline(5, color='gray')

for ii, fp_kind in enumerate(possible_class):
    idx = np.where(sn_class_spl == fp_kind)[0]
    if len(idx) > 0:
        ax.plot(rd(sn_equil_spl[idx, 0]), rd(sn_equil_spl[idx, 1]), 'o',
                c=bfbmap[ii], ms=2.5, label=fp_kind)

_leg = ax.legend(loc='lower left', markerscale=3, fancybox=True, framealpha=.75,
                 frameon=True, fontsize=16)
_leg.get_frame().set_color('w')
ax.set_xlim(-15, 15)
ax.set_ylim(60, 0)
#ax.set_ylabel(r'$\gamma^*$, equilibrium glide angle', fontsize=18)
#ax.set_xlabel(r'$\theta$, pitch angle', fontsize=18)
ax.set_ylabel(r'$\gamma^*$    ', fontsize=18, rotation=0)
ax.set_xlabel(r'$\theta$', fontsize=18)

ax.text(-13, 5, 'airfoil snake', {'fontsize': 18})

fig.canvas.draw()

[ttl.set_size(18) for ttl in ax.get_xticklabels()]
[ttl.set_size(18) for ttl in ax.get_yticklabels()]

rcj(ax)
tl(fig)

fig.savefig('Figures/figure6b_bifurcation_airfoil_snake.pdf',
            transparent=True)


# %% Velocity polar diagram, pitch = 0

afdict = dict(cli=cl_fun, cdi=cd_fun, clip=clprime_fun, cdip=cdprime_fun)

pitch = 0
arng = sn_angle_rng
extrap = (ale[0], ale[-1])
lims = (vxlim, vzlim) = np.r_[0, 1.25], np.r_[0, -1.25]

tvec = np.linspace(0, 30, 351)

reload(plots)
from plots import phase_plotter as ppr

fig, ax = ppr(afdict, pitch, lims, arng, tvec, ngrid=201,
              nseed=501, nseed_skip=25, quiver=False, skip=10, seed=False,
              timer=True, gamtest=gammas, extrap=extrap,
              traj=plots.ps_traj_dp5, fig=None, ax=None)

lab = 'airfoil snake, ' + r'$\theta=$0' + u'\u00B0'
ax.text(.05, -1, lab, fontsize=16)

fig.savefig('Figures/figure5bi_vpd0_airfoil_snake.pdf', transparent=True)


# %% Velocity polar diagram, pitch = 0 with Z nullcline

afdict = dict(cli=cl_fun, cdi=cd_fun, clip=clprime_fun, cdip=cdprime_fun)

pitch = 0
arng = sn_angle_rng
extrap = (ale[0], ale[-1])
lims = (vxlim, vzlim) = np.r_[0, 1.25], np.r_[0, -1.25]

tvec = np.linspace(0, 30, 351)

reload(plots)
from plots import phase_plotter as ppr

fig, ax = ppr(afdict, pitch, lims, arng, tvec, ngrid=201,
              nseed=501, nseed_skip=25, quiver=False, skip=10, seed=False,
              timer=True, gamtest=gammas, extrap=extrap,
              traj=plots.ps_traj_dp5, nullcline_z=True,
              fig=None, ax=None)

lab = 'airfoil snake, ' + r'$\theta=$0' + u'\u00B0'
ax.text(.05, -1, lab, fontsize=16)

fig.savefig('Figures/figure5bi_vpd0_nullcline_airfoil_snake.pdf',
            transparent=True)


# %% Velocity polar diagram, pitch = 5

pitch = np.deg2rad(5)

fig, ax = ppr(afdict, pitch, lims, arng, tvec, ngrid=201,
              nseed=501, nseed_skip=25, quiver=False, skip=10, seed=False,
              timer=True, gamtest=gammas, extrap=extrap,
              traj=plots.ps_traj_dp5, fig=None, ax=None)

lab = 'airfoil snake, ' + r'$\theta=$5' + u'\u00B0'
ax.text(.05, -1, lab, fontsize=16)

fig.savefig('Figures/figure5bii_vpd5_airfoil_snake.pdf', transparent=True)


# %% Velocity polar diagram, pitch = 5, with manifold approximations

man_folder = './Data/airfoil snake manifold/'
man_2 = np.genfromtxt(man_folder + 'manifold_2nd_order.csv', delimiter=',')
man_3 = np.genfromtxt(man_folder + 'manifold_3rd_order.csv', delimiter=',')

vx_2, vz_2 = man_2.T
vx_3, vz_3 = man_3.T

pitch = np.deg2rad(5)

fig, ax = ppr(afdict, pitch, lims, arng, tvec, ngrid=201,
              nseed=501, nseed_skip=25, quiver=False, skip=10, seed=False,
              timer=True, gamtest=gammas, extrap=extrap,
              traj=plots.ps_traj_dp5, fig=None, ax=None)

ax.plot(vx_2, vz_2, c=bmap[2], label='2nd-order approx.')
ax.plot(vx_3, vz_3, c=bmap[3], label='3rd-order approx.')
ax.legend(loc='lower right', frameon=True)
ax.set_xlim(.55, .8)
ax.set_ylim(-.525, -.275)

fig.savefig('Figures/figure5bii_inset_vpd5_airfoil_snake.pdf',
            transparent=False)


# %% Supplement figure - Acclerations along the terminal velocity manifold

gam_2 = -np.arctan(vz_2 / vx_2)
gam_3 = -np.arctan(vz_3 / vx_3)

ptc_2 = np.deg2rad(5)
ptc_3 = np.deg2rad(5)

aoa_2 = gam_2 + ptc_2
aoa_3 = gam_3 + ptc_3

cl_2 = cl_fun(aoa_2)
cd_2 = cd_fun(aoa_2)
cl_3 = cl_fun(aoa_3)
cd_3 = cd_fun(aoa_3)

ax_2, az_2 = eqns.cart_eqns(vx_2, vz_2, cl_2, cd_2)
ax_3, az_3 = eqns.cart_eqns(vx_3, vz_3, cl_3, cd_3)

a_2 = np.sqrt(ax_2**2 + az_2**2)
a_3 = np.sqrt(ax_3**2 + az_3**2)

xx_2 = np.arange(len(a_2))
xx_3 = np.arange(len(a_3))

# arbitrary shift the indices for plotting; saddle at zero, stable node at 1
xx_2 = (xx_2 - 150) / 150
xx_3 = (xx_3 - 150) / 150

fig, ax = plt.subplots()

ax.axhline(.1, color=bmap[3], lw=1, label='low acceleration contour')
ax.axvline(0, color=bmap[0], lw=1, ls='--', label='location of saddle point')
ax.axvline(.93, color=bmap[1], lw=1, ls='--', label='location of stable node')

ax.plot(xx_2, a_2, c=bmap[2], lw=2, label='2nd order approx.')
ax.plot(xx_3, a_3, c=bmap[3], lw=2, label='3rd order approx.')

ax.legend(loc='upper left', frameon=True)
ax.set_xlabel('distance along terminal velocity manifold')
ax.set_ylabel('acceleration magnitude')

rcj(ax)
tl(fig)

fig.savefig('Figures/figure_SI_acceleration_along_manifold.pdf',
            transparent=True)


# %% Figure 1 - show how VPD differs from time series approach

pitch = 0

ts = np.linspace(0, 30, 351)
vxseed, vzseed = np.r_[.4], np.r_[0]
odeargs = (pitch, cl_fun, cd_fun)
for i in range(len(vxseed)):
    x0 = (0, 0, vxseed[i], vzseed[i])
    soln = plots.ps_traj(x0, ts, odeargs, eqns.cart_model, arng,
                         vxlim, vzlim)

ntime = len(ts)

# unpack values
xs, zs, vxs, vzs = soln.T
gs = eqns.calc_gamma(vxs, vzs)

# just plot once the glide angle derivative is slow
idx = np.where(np.abs(np.gradient(gs)) >= 1e-4)[0]
xs, zs = xs[idx], zs[idx]
vxs, vzs = vxs[idx], vzs[idx]
gs = gs[idx]
ts = ts[idx]
accxs, acczs = np.zeros(len(ts)), np.zeros(len(ts))

for k in np.arange(len(ts)):
    x0 = (xs[k], zs[k], vxs[k], vzs[k])
    _, _, accxs[k], acczs[k] = eqns.cart_model(x0, ts[k], odeargs)

vmag = np.sqrt(vxs**2 + vzs**2)
accmag = np.sqrt(accxs**2 + acczs**2)

i0 = gs.argmax()

np.where(accmag <= 0.1)[0]
i1 = 15
i2 = 139
i3 = 147
ii = np.r_[i1, i2, i3]
ii = np.r_[140]  # , 147]  # end of bump in acceleration


# %% Plot time histories

fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True, figsize=(4.2, 8))

ax1.axhline(0, color='gray')
ax2.axhline(0, color='gray')
ax3.axhline(0, color='gray')
ax4.axhline(gs[-1], color='gray', ls=':')

lw = 1.5
ax1.plot(ts, xs, 'k', lw=lw)
ax1.plot(ts, zs, 'k--', lw=lw)
ax2.plot(ts, vxs, 'k', lw=lw)
ax2.plot(ts, vzs, 'k--', lw=lw)
ax3.plot(ts, accxs, 'k', label='horizontal', lw=lw)
ax3.plot(ts, acczs, 'k--', label='vertical', lw=lw)
ax4.plot(ts, gs, 'k', lw=lw)

# plot velocity and acceleration magnitudes
# ax3.plot(ts, accmag, 'k:', label='magnitude', lw=lw)
# ax2.plot(ts, vmag, 'k:', lw=lw)

kwargs = dict(marker='o', ms=7, mfc=None, mec='gray', mew=1, fillstyle='none')

ax1.plot(ts[i0], xs[i0], 'o', ms=7, c='gray')
ax1.plot(ts[i0], zs[i0], 'o', ms=7, c='gray')
ax1.plot(ts[ii], xs[ii], **kwargs)
ax1.plot(ts[ii], zs[ii], **kwargs)

ax2.plot(ts[i0], vxs[i0], 'o', ms=7, c='gray')
ax2.plot(ts[i0], vzs[i0], 'o', ms=7, c='gray')
ax2.plot(ts[ii], vxs[ii], **kwargs)
ax2.plot(ts[ii], vzs[ii], **kwargs)

ax3.plot(ts[i0], accxs[i0], 'o', ms=7, c='gray')
ax3.plot(ts[i0], acczs[i0], 'o', ms=7, c='gray')
ax3.plot(ts[ii], accxs[ii], **kwargs)
ax3.plot(ts[ii], acczs[ii], **kwargs)

ax4.plot(ts[i0], gs[i0], 'o', ms=7, c='gray')
ax4.plot(ts[ii], gs[ii], **kwargs)

ax3.legend(loc='lower right', fontsize=18)

for ax in [ax1, ax2, ax3, ax4]:
    ax.set_yticks([])
    ax.set_xticks([])

ttext = .5
ax1.text(ttext, .9 * np.r_[xs, zs].max(), 'position', fontsize=18)
ax2.text(ttext, .9 * np.r_[vxs, vzs].max(), 'velocity', fontsize=18)
ax3.text(ttext, .9 * np.r_[accxs, acczs].max(), 'acceleration',
         fontsize=18)
ax4.text(ttext, .85 * np.pi / 2, 'glide angle', fontsize=18)
ax4.set_xlabel('time', fontsize=18)

#ax1.set_ylabel('position', fontsize=18)
#ax2.set_ylabel('velocity', fontsize=18)
#ax3.set_ylabel('acceleration', fontsize=18)
#ax4.set_ylabel('glide angle', fontsize=18)

ax4.set_xlim(0, ts[-1])
ax4.set_ylim(0, np.pi / 2)

rcj(ax1)
rcj(ax2)
rcj(ax3)
rcj(ax4)
tl(fig)

fig.savefig('Figures/1abcd_time_histories.pdf', transparent=True)


# %% Plot x-z space

skip = 10

fig, ax = plt.subplots(figsize=(4.2, 4.))

ax.plot(xs, zs, 'k-x', lw=1.5, markevery=skip, mew=.75)

ax.plot(xs[i0], zs[i0], 'o', ms=7, c='gray')
ax.plot(xs[ii], zs[ii], **kwargs)

ax.set_xlabel(r'$x$', fontsize=20)
ax.set_ylabel(r'$z$    ', rotation=0, fontsize=20)

ax.set_yticks([])
ax.set_xticks([])

ax.set_aspect('equal', adjustable='box')
ax.margins(0, .03)

rcj(ax)
tl(fig)

fig.savefig('Figures/1e_position_space.pdf', transparent=True)


# %% Plot velocity polar diagram

afdict = dict(cli=cl_fun, cdi=cd_fun, clip=clprime_fun, cdip=cdprime_fun)

arng = sn_angle_rng
extrap = (ale[0], ale[-1])
lims = (vxlim, vzlim) = np.r_[0, 1.25], np.r_[0, -1.25]

tvec = np.linspace(0, 30, 351)

reload(plots)
from plots import phase_plotter as ppr

fig, ax = plt.subplots(figsize=(4.2, 4))

fig, ax = ppr(afdict, pitch, lims, arng, tvec, ngrid=201,
              nseed=501, nseed_skip=25, quiver=False, skip=10, seed=False,
              timer=True, gamtest=gammas, extrap=extrap,
              traj=plots.ps_traj_dp5, fig=fig, ax=ax, acc_contour=True)

ax.plot(vxs, vzs, 'kx-', lw=1.5, markevery=skip, mew=.75, ms=5)

ax.plot(vxs[i0], vzs[i0], 'o', ms=7, c='gray')
ax.plot(vxs[ii], vzs[ii], **kwargs)

ax.set_xticks([])
ax.set_yticks([])

ax.set_xlabel(r'$v_x$', fontsize=20)
ax.set_ylabel(r'$v_z$    ', fontsize=20, rotation=0)

fig.savefig('Figures/1f_velocity_space.pdf', transparent=True)

