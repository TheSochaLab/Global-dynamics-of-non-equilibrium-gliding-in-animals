# -*- coding: utf-8 -*-
"""
Created on Wed Aug 13 21:22:12 2014

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
import pandas as pd
from scipy.interpolate import UnivariateSpline
import time

# to load and save Hopf bifurcation simulations
import pickle

# setup better plots
import plots
reload(plots)
from plots import bmap, rcj, tl

import squirrel
reload(squirrel)

import eqns
reload(eqns)


# %% Load in all squirrel data

df = pd.read_excel('Data/Bahlman2013/Bahlman rsif20120794supp2.xls', 'Sheet1')
ntrials = int(len(df.columns) / 2)


# %% Run through all trials, smooth, and make a bunch of plots

fig1, ax1 = plt.subplots()
fig11, (ax12, ax13) = plt.subplots(2, 1, sharex=True)
fig2, ax2 = plt.subplots()
fig3, ax3 = plt.subplots()
fig4, ax4 = plt.subplots()
fig5, ax5 = plt.subplots()
fig6, (ax61, ax62) = plt.subplots(2, 1, sharex=True, figsize=(6, 6))

alpha = .85
index_save = []
dfpx, dfpz, dfvx, dfvz = [], [], [], []
dfax, dfaz, dfgam, dfmag = [], [], [], []
iskip = []#[19]

for i in np.arange(1, ntrials + 1):
    posd, tvec, dt = squirrel.load_run(i, df)
    veld = squirrel.calc_vel(posd, dt)
    acceld = squirrel.calc_accel(veld, dt)

    endcond = (posd[-1, 0] > 16) & (posd[-1, 0] < 18.5)
    startcond = (posd[0, 1] > -2) & (posd[0, 0] < 3.8)
    if endcond & startcond and i not in iskip:
        print('trial {0:3d}'.format(i))
        index_save.append(i)

        # try a savgol filter
        # pos, vel, acc = squirrel.svfilter(tvec, posd, 81, 4, mode='interp')

        # preform filtering
        pos, vel, acc = squirrel.moving_window_pts(posd, tvec, wn=81, deg=1)
        vel_new, acc, _ = squirrel.moving_window_pts(vel, tvec, wn=81, deg=1)

        # pos, vel, acc = squirrel.moving_window_pts(posd, tvec, wn=91,
        #                                            deg=2, drop_deg=False)
        # vel_new, acc, _ = squirrel.moving_window_pts(vel, tvec, wn=31,
        #                                              deg=2, drop_deg=True)

        # pos, vel, acc = squirrel.polyfit_all(posd, tvec, 10, 120)

        gamma = squirrel.calc_gamma(vel)
        velmag = squirrel.calc_vel_mag(vel)
        diffpos = 100 * (posd - pos) / posd  # error in position

        # fill up list of DataFrames
        df_return = squirrel.fill_df(pos, vel, acc, gamma, velmag, tvec, i)
        posx, posz, velx, velz, accx, accz, gamm, vmag = df_return
        dfpx.append(posx)
        dfpz.append(posz)
        dfvx.append(velx)
        dfvz.append(velz)
        dfax.append(accx)
        dfaz.append(accz)
        dfgam.append(gamm)
        dfmag.append(vmag)

        _ln, = ax1.plot(pos[:, 0], pos[:, 1], alpha=alpha)
        _col = _ln.get_color()
        ax1.text(pos[-1, 0], pos[-1, 1], '{0:3d}'.format(i), color=_col)
        ax12.plot(posd[:, 0], diffpos[:, 0], alpha=alpha, color=_col)
        ax13.plot(posd[:, 0], diffpos[:, 1], alpha=alpha, color=_col)
        ax2.plot(vel[:, 0], vel[:, 1], alpha=alpha)
        ax2.plot(vel[0, 0], vel[0, 1], 'o', color=bmap[4], alpha=.35)
        ax3.plot(gamma, velmag, alpha=alpha)
        ax3.plot(gamma[0], velmag[0], 'o', color=bmap[4], alpha=.35)
        ax4.plot(tvec, pos[:, 0], color=bmap[0], alpha=alpha)
        ax4.plot(tvec, pos[:, 1], color=bmap[1], alpha=alpha)
        ax5.plot(tvec, vel[:, 0], color=bmap[0], alpha=alpha)
        ax5.plot(tvec, vel[:, 1], color=bmap[1], alpha=alpha)
        ax62.plot(pos[:, 0], acc[:, 0] / 9.81, color=bmap[0], alpha=alpha)
        ax61.plot(pos[:, 0], acc[:, 1] / 9.81, color=bmap[1], alpha=alpha)

ax2.axis('equal')

ax1.set_xlim(2, 19)
ax12.set_xlim(2, 19)
ax1.set_ylim(ymax=0)
ax3.set_xlim(np.deg2rad(np.r_[-45, 60]))
ax4.set_xlim(xmax=2.55)
ax5.set_xlim(xmax=2.55)
ax62.set_ylim(-.8, .8)
ax61.set_ylim(-.4, 1)

ax1.set_ylabel('vertical distance, m')
ax1.set_xlabel('horizontal distance, m')
ax12.set_title('horizontal relative error, %', fontsize='small')
ax13.set_title('vertical relative error, %', fontsize='small')
ax13.set_xlabel('horizontal distance, m')
ax2.set_xlabel('horizontal velocity, m/s')
ax2.set_ylabel('sinking velocity, m/s')
ax3.set_xlabel('glide angle, rad')
ax3.set_ylabel('velocity magnitude, m/s')
ax4.set_xlabel('time, sec')
ax4.set_ylabel('distance, m')
ax5.set_xlabel('time, sec')
ax5.set_ylabel('velocity, m/s')
ax5.set_xlabel('time, sec')
ax5.set_ylabel(r'velocity, m/s')
ax61.set_ylabel(r'$a_z / g$', fontsize=18)
ax62.set_ylabel(r'$a_x / g$', fontsize=18)
ax62.set_xlabel('distance, m')

ax2.axhline(0, color='gray', alpha=.4)
ax62.axhline(0, color='gray', linestyle='-', linewidth=.75)
ax61.axhline(0, color='gray', linestyle='-', linewidth=.75)

ax4.text(.9, 7.5, 'horizontal', color=bmap[0])
ax4.text(.9, -4.1, 'vertical', color=bmap[1])
ax5.text(.325, 5, 'horizontal', color=bmap[0])
ax5.text(1.3, -3.4, 'vertical', color=bmap[1])

ax2.axis('equal')
rcj(ax2)

axes = [ax1, ax12, ax13, ax3, ax4, ax5, ax61, ax62]
figs = [fig1, fig11, fig2, fig3, fig4, fig5, fig6]
[rcj(aa) for aa in axes]
[tl(f) for f in figs]

# concat all the DataFrames
dfpx = pd.concat(dfpx, axis=1)
dfpz = pd.concat(dfpz, axis=1)
dfvx = pd.concat(dfvx, axis=1)
dfvz = pd.concat(dfvz, axis=1)
dfax = pd.concat(dfax, axis=1)
dfaz = pd.concat(dfaz, axis=1)
dfgm = pd.concat(dfgam, axis=1)
dfvm = pd.concat(dfmag, axis=1)

#fig1.savefig('figs/squirrel/fit_position.pdf')
#fig11.savefig('figs/squirrel/fit_errors.pdf')
#fig2.savefig('figs/squirrel/fit_glide_polar.pdf')
#fig3.savefig('figs/squirrel/fit_cart_coords.pdf')
#fig4.savefig('figs/squirrel/fit_xz.pdf')
#fig5.savefig('figs/squirrel/fit_vxvz.pdf')
#fig6.savefig('figs/squirrel/fit_accel.pdf')


# %% Lift and drag calcuation

# select out where the data is 'complete'
#cnt = dfpx.count(axis=1)
#index_keep = np.where(cnt == cnt.ix[0])[0]
#tvec = np.array(dfpx.index[index_keep])
#N = dfpx.shape[1]  # number of 'complete' trials
#
#gl_rad = dfgm.ix[tvec]
#gl_deg = np.rad2deg(gl_rad)
#vmag = dfvm.ix[tvec]

gl_rad = dfgm.copy()
gl_deg = np.rad2deg(gl_rad)
vmag = dfvm.copy()

# physical parameters
mass = .0927  # kg
Sarea = 0.0154  # m^2
length = .169  # m
rho = 1.204  # kg/m^3
den = .5 * rho * vmag**2 * Sarea
grav = 9.81

# divide by these to go from dimensional to non-dim and resacled
ms, ln, Sa = mass, length, Sarea
eps = eqns.calc_epsilon(mass, length, Sarea)
pret = np.sqrt(length / (grav * eps))
prep = length / eps
prev = np.sqrt(length * grav / eps)

#accx = dfax.ix[tvec]
#accz = dfaz.ix[tvec]
accx = dfax.copy()
accz = dfaz.copy()
Fx = mass * accx
Fz = mass * accz

lift = Fx * np.sin(gl_rad) + Fz * np.cos(gl_rad) + mass * grav * np.cos(gl_rad)
drag = Fz * np.sin(gl_rad) - Fx * np.cos(gl_rad) + mass * grav * np.sin(gl_rad)

Cl = lift / den
Cd = drag / den

# save the DataFrames
save_data = False
if save_data:
    Cl.to_csv('data/squirrel_Cl.csv')
    Cd.to_csv('data/squirrel_Cd.csv')
    gl_deg.to_csv('data/squirrel_gldeg.csv')
    dfvm.ix[tvec].to_csv('data/squirrel_vm.csv')
    dfpx.ix[tvec].to_csv('data/squirrel_px.csv')
    dfpz.ix[tvec].to_csv('data/squirrel_pz.csv')
    dfvx.ix[tvec].to_csv('data/squirrel_vx.csv')
    dfvz.ix[tvec].to_csv('data/squirrel_vz.csv')
    dfax.ix[tvec].to_csv('data/squirrel_ax.csv')
    dfaz.ix[tvec].to_csv('data/squirrel_az.csv')


# %% Non-dimensionalize and rescale the data and save it

eps = eqns.calc_epsilon(mass, length, Sarea)
pret = np.sqrt(length / (grav * eps))
prep = length / eps
prev = np.sqrt(length * grav / eps)

sq_dfvx = dfvx / prev
sq_dfvz = dfvz / prev

savedata = False
if savedata:
    sq_dfvx.to_csv('data/sq_dfvx.csv')
    sq_dfvz.to_csv('data/sq_dfvz.csv')

fig, ax = plt.subplots()
for col in sq_dfvx:
    sq_vx = sq_dfvx[col].dropna().values
    sq_vz = sq_dfvz[col].dropna().values
    ax.plot(sq_vx, sq_vz, c=bmap[2])
ax.set_xlabel(r'$\hat{v}_x$', fontsize=16)
ax.set_ylabel(r'$\hat{v}_z$', fontsize=16)
rcj(ax)
tl(fig)


# %% Acceleration plot

fig, ax = plt.subplots()
accx.plot(ax=ax, grid=False, legend=False, alpha=.85, c=bmap[0])
accz.plot(ax=ax, grid=False, legend=False, alpha=.85, c=bmap[1])
ax.legend((ax.get_lines()[0], ax.get_lines()[-1]),
          (r'$a_x$', r'$a_z$'),
          loc='best', frameon=False)
ax.set_ylabel(r'acceleration, $m/s^2$')
ax.set_xlabel('time, sec')
rcj(ax)
tl(fig)


# %% Lift and drag plot

fig, ax = plt.subplots()
lift.plot(ax=ax, grid=False, legend=False, alpha=.85, c=bmap[0])
drag.plot(ax=ax, grid=False, legend=False, alpha=.85, c=bmap[1])
ax.legend((ax.get_lines()[0], ax.get_lines()[-1]),
          ('lift', 'drag'),
          loc='best', frameon=False)
ax.axhline(0, color='gray', linestyle='-', linewidth=.75, alpha=.8)
ax.set_ylabel('aerodynamics forces, N')
ax.set_xlabel('time, sec')
rcj(ax)
tl(fig)


# %% Cl and Cd plot

fig, ax = plt.subplots()
Cl.plot(ax=ax, grid=False, legend=False, alpha=.85, c=bmap[0])
Cd.plot(ax=ax, grid=False, legend=False, alpha=.85, c=bmap[1])
ax.legend((ax.get_lines()[0], ax.get_lines()[-1]),
          (r'$C_L$', r'$C_D$'),
          loc='best', frameon=False)
ax.axhline(0, color='gray', linestyle='-', linewidth=.75, alpha=.8)
ax.set_ylabel('force coefficients')
ax.set_xlabel('time, sec')
ax.set_ylim(ymin=-.25)
rcj(ax)
tl(fig)


# %% Glide angle plot

# gl_std = gl_deg.std(axis=1)
# gl_mean = gl_deg.mean(axis=1)

fig, ax = plt.subplots()
gl_deg.plot(ax=ax, grid=False, legend=False, alpha=.85, c=bmap[0])
ax.set_xlabel('time, sec')
ax.set_ylabel('glide angle, deg')
rcj(ax)
tl(fig)


# %% radian Density of Cl and Cd plots

gam_cot = np.deg2rad(np.linspace(.1, 60, 100))
cot_gam = 1 / np.tan(gam_cot)

fig, ax = plt.subplots()
_ln3 = ax.plot(gl_rad, Cl / Cd, 'o', c=bmap[0], alpha=.35, ms=1.4,
               label=r'$C_L(\gamma) / C_D(\gamma)$')
_ln4 = ax.plot(gam_cot, cot_gam, c=bmap[1], lw=1.75, label=r'$\cot{\gamma}$')
ax.set_ylabel(r'$C_L/C_D$', fontsize='large')
ax.set_xlabel('glide angle, rad', fontsize='large')
ax.set_ylim(0, 7)
ax.set_xlim(np.deg2rad(np.r_[-10, 50]))
ax.legend((_ln3[0], _ln4[0]),
          (r'$C_L(\gamma) / C_D(\gamma)$', r'$\cot{\gamma}$'),
          loc='best', frameon=False)
rcj(ax)
tl(fig)


# %% degree Density of Cl and Cd plots

gam_cot = np.deg2rad(np.linspace(.1, 60, 100))
cot_gam = 1 / np.tan(gam_cot)

rd = np.rad2deg

fig, ax = plt.subplots()
_ln3 = ax.plot(rd(gl_rad), Cl / Cd, 'o', c=bmap[0], alpha=.35, ms=1.4)
_ln4 = ax.plot(rd(gam_cot), cot_gam, c=bmap[1], lw=1.75,
               label=r'$\cot{\gamma}$')
ax.set_ylabel('force coefficients', fontsize='large')
ax.set_xlabel('glide angle, deg', fontsize='large')
ax.set_ylim(0, 7)
ax.legend((_ln3[0], _ln4[0]),
          (r'$C_L(\gamma) / C_D(\gamma)$', r'$\cot{\gamma}$'),
          loc='best', frameon=False)
rcj(ax)
tl(fig)


# %% Binning the glide angles to get Cl/Cd curve

dgl_deg = 2
dgl_rad = np.deg2rad(dgl_deg)
#gl_bins = np.deg2rad(np.arange(-5, 46, dgl_deg, dtype=np.float64))
gl_bins = np.deg2rad(np.arange(10, 46, dgl_deg, dtype=np.float64))
gl_flattened = gl_rad.values.flatten()

# peform the binning
_output = squirrel.clcd_binning(gl_bins, gl_rad, Cl, Cd)
clcd_means, cl_means, cd_means, gl_means = _output

fig, ax = plt.subplots()
_ln1 = ax.plot(gl_rad, Cl / Cd, 'o', c=bmap[0], alpha=.15, ms=1.5)
_ln2 = ax.plot(gam_cot, cot_gam, c=bmap[1], lw=1.5, label=r'$\cot{\gamma}$')

# _ln3 = ax.plot(gl_means[:, 0], clcd_means[:, 0], c=bmap[3])
_ln3 = ax.errorbar(gl_means[:, 0], clcd_means[:, 0], yerr=clcd_means[:, 1],
                   xerr=gl_means[:, 1], color=bmap[3], lw=1.5)

for start in np.r_[gl_bins[0] - dgl_rad, gl_bins]:
    ax.axvline(start, color='gray', linestyle='-', linewidth=.75, alpha=.2)

ax.set_ylabel('force coefficient')
ax.set_xlabel('glide angle, rad')
ax.set_ylim(0, 7)
ax.set_xlim(np.deg2rad(np.r_[-10, 50]))
leg = ax.legend((_ln1[0], _ln2[0], _ln3[0]),
                (r'$C_L / C_D$', r'$\cot{\gamma}$', 'mean',), loc='best',
                frameon=True, framealpha=.85, fancybox=True)
leg.get_frame().set_edgecolor('gray')
rcj(ax)
tl(fig)


# %% Binning glide angle, limit the xlim

rd = np.rad2deg

fig, ax = plt.subplots()
_ln1 = ax.plot(rd(gl_rad), Cl / Cd, 'o', c=bmap[0], alpha=.35, ms=1.5)

_ln2 = ax.plot(rd(gam_cot), cot_gam, c=bmap[2])
_ln3 = ax.errorbar(rd(gl_means[:, 0]), clcd_means[:, 0], yerr=clcd_means[:, 1],
                   xerr=rd(gl_means[:, 1]), color=bmap[1], lw=1.5)

for start in rd(np.r_[gl_bins[0] - dgl_rad, gl_bins]):
    ax.axvline(start, color='gray', linestyle='-', linewidth=.5, alpha=.2)

ax.set_ylabel(r'$C_L/C_D$', fontsize=14)
ax.set_xlabel('glide angle, deg', fontsize=14)
ax.set_ylim(0, 4)
ax.set_xlim(rd(np.deg2rad(np.r_[9, 47])))
#leg = ax.legend((_ln1[0], _ln2[0], _ln3[0]),
#                (r'$C_L / C_D$', r'$\cot{\gamma}$', 'mean'), loc='best',
#                frameon=True, framealpha=.85, fancybox=True)
leg = ax.legend((_ln1[0], _ln3[0], _ln2[0]),
                ('individual', 'mean', r'$\cot{\gamma}$'), loc='best',
                frameon=True, framealpha=.85, fancybox=True)
leg.get_frame().set_edgecolor('gray')
[ttl.set_size(14) for ttl in ax.get_xticklabels()]
[ttl.set_size(14) for ttl in ax.get_yticklabels()]
rcj(ax)
tl(fig)


# %% Histogram of glide angle bins

fig, ax = plt.subplots()
ax.hist(gl_flattened, gl_bins, color=bmap[1], alpha=.85,
        edgecolor='w', linewidth=.75)
# ax.plot(gl_means[:, 0], map(len, all_indices), color=bmap[2])
ax.set_xlabel('glide angle, rad')
ax.set_ylabel('number of data points')
rcj(ax)
tl(fig)


# %% Cl mean plot

fig, ax = plt.subplots()
_ln3 = ax.plot(gl_rad, Cl, 'o', c=bmap[1], alpha=.45, ms=1.4)
_ln1 = ax.plot(gl_means[:, 0], cl_means[:, 0], c=bmap[0])
ax.errorbar(gl_means[:, 0], cl_means[:, 0], yerr=cl_means[:, 1],
            xerr=gl_means[:, 1], color=bmap[0])

ax.set_ylabel('force coefficient')
ax.set_xlabel('glide angle, rad')
for start in np.r_[gl_bins[0] - dgl_rad, gl_bins]:
    ax.axvline(start, color='gray', linestyle='-', linewidth=.75, alpha=.2)
ax.set_ylim(0, 5)
ax.set_xlim(np.deg2rad(np.r_[-10, 50]))
leg = ax.legend((_ln1[0], _ln3[0]),
                ('mean', r'$C_L$'), loc='best',
                frameon=True, framealpha=.85, fancybox=True)
leg.get_frame().set_edgecolor('gray')
rcj(ax)
tl(fig)


# %% Cd mean plot

fig, ax = plt.subplots()
_ln3 = ax.plot(gl_rad, Cd, 'o', c=bmap[1], alpha=.45, ms=1.4)
_ln1 = ax.plot(gl_means[:, 0], cd_means[:, 0], c=bmap[0])
ax.errorbar(gl_means[:, 0], cd_means[:, 0], yerr=cd_means[:, 1],
            xerr=gl_means[:, 1], color=bmap[0])

ax.set_ylabel('force coefficient')
ax.set_xlabel('glide angle, deg')
for start in np.r_[gl_bins[0] - dgl_rad, gl_bins]:
    ax.axvline(start, color='gray', linestyle='-', linewidth=.75, alpha=.2)
ax.set_ylim(0, 2)
ax.set_xlim(np.deg2rad(np.r_[-10, 50]))
leg = ax.legend((_ln1[0], _ln3[0]),
                ('mean', r'$C_D$'), loc='best',
                frameon=True, framealpha=.85, fancybox=True)
leg.get_frame().set_edgecolor('gray')
rcj(ax)
tl(fig)


# %% Extend the curves

nuse = 6
clfit = np.polyfit(gl_means[-nuse:, 0], cl_means[-nuse:, 0], 1)
cdfit = np.polyfit(gl_means[-nuse:, 0], cd_means[-nuse:, 0], 1)

glstart = gl_means[-1, 0] + dgl_rad
glstop = np.deg2rad(52)  #  1.5 * gl_means[-1, 0] + dgl_rad
glfill = np.arange(glstart, glstop, dgl_rad)

clfill = np.polyval(clfit, glfill)
cdfill = np.polyval(cdfit, glfill)


## don't extend the curves...
glstop = gl_means[-1, 0]
glfill = []
clfill = []
cdfill = []

gammas = np.r_[gl_means[:, 0], glfill]
cls = np.r_[cl_means[:, 0], clfill]
cds = np.r_[cd_means[:, 0], cdfill]


# %% Spline fit the Cl and Cd curves

# spline
cl_fun = UnivariateSpline(gammas, cls, k=3, s=.0001)
cd_fun = UnivariateSpline(gammas, cds, k=3, s=.0001)
clprime_fun = cl_fun.derivative()
cdprime_fun = cd_fun.derivative()

Cl_fun = UnivariateSpline(gammas, cls, k=1, s=0)
Cd_fun = UnivariateSpline(gammas, cds, k=1, s=0)
Clprime_fun = Cl_fun.derivative()
Cdprime_fun = Cd_fun.derivative()

gl_spl, dgl_spl = np.linspace(gammas[0], gammas[-1], 1001, retstep=True)
clspl = cl_fun(gl_spl)
cdspl = cd_fun(gl_spl)
clpspl = clprime_fun(gl_spl)
cdpspl = cdprime_fun(gl_spl)


# %% Check the interpolations

fig, ax = plt.subplots()
ax.plot(gl_means[:, 0], cl_means[:, 0], '-o', alpha=.85)
ax.plot(gl_means[:, 0], cd_means[:, 0], '-o', alpha=.85)
ax.plot(gl_means[:, 0], clcd_means[:, 0], '-o', c=bmap[3])
ax.plot(gammas, cls, 'o', c=bmap[0], alpha=.85)
ax.plot(gammas, cds, 'o', c=bmap[1], alpha=.85)
ax.plot(gammas, cls / cds, 'o', c=bmap[2], alpha=.85)
#ax.plot(gam_cot, cot_gam, c=bmap[2], lw=1.25, label=r'$\cot{\gamma}$')
ax.plot(gl_spl, clspl, c=bmap[0], label=r"$C_L(\gamma)$")
ax.plot(gl_spl, cdspl, c=bmap[1], label=r"$C_D(\gamma)$")
ax.plot(gl_spl, clspl / cdspl, '--', c=bmap[2],
        label=r"$C_L(\gamma)/C_D(\gamma)$")
#ax.legend(loc='best', frameon=False)
#ax.set_xlim(np.deg2rad(np.r_[-10, 50]))
ax.axvline(gl_means[-1, 0], color='gray', lw=.75)
ax.set_ylim(0, 3.5)
ax.set_ylabel('force coefficients')
ax.set_xlabel('glide angle, rad')
rcj(ax)
tl(fig)


# %% Check the interpolations (degrees)

rd = np.rad2deg

fig, ax = plt.subplots()
#ax.plot(rd(gl_means[:, 0]), cl_means[:, 0], '-o', alpha=.85)
#ax.plot(rd(gl_means[:, 0]), cd_means[:, 0], '-o', alpha=.85)
#ax.plot(rd(gl_means[:, 0]), clcd_means[:, 0], '-o', c=bmap[3])
#ax.plot(rd(gammas), cls, 'o', c=bmap[0], alpha=.85)
#ax.plot(rd(gammas), cds, 'o', c=bmap[1], alpha=.85)
#ax.plot(rd(gammas), cls / cds, 'o', c=bmap[2], alpha=.85)
#ax.plot(gam_cot, cot_gam, c=bmap[2], lw=1.25, label=r'$\cot{\gamma}$')
#lw = 3
lw = 1.5
ax.plot(rd(gl_spl), 1 / np.tan(gl_spl), c=bmap[3], lw=1,
        label=r'$\cot{\gamma)$')
ax.plot(rd(gl_spl), clspl, c=bmap[0], lw=lw, label=r"$C_L$")
ax.plot(rd(gl_spl), cdspl, c=bmap[1], lw=lw, label=r"$C_D$")
ax.plot(rd(gl_spl), clspl / cdspl, c=bmap[2], lw=lw, label=r"$C_L/C_D$")
#ax.plot(rd(gl_spl), clspl / cdspl, '-', c=bmap[2],
#        label=r"$C_L/C_D$")
ax.legend(loc='best', frameon=False)#, fontsize=18)
#ax.set_xlim(np.deg2rad(np.r_[-10, 50]))
#ax.axvline(rd(gl_means[-1, 0]), color='gray', lw=.75)
#ax.set_ylim(-1, 4)
#ax.set_xlim(-15, 70)
ax.set_ylim(0, 3)
ax.set_ylabel('force coefficients')#, fontsize=20)
ax.set_xlabel(r'$\alpha$, deg')#, fontsize=20)

#[ttl.set_size(20) for ttl in ax.get_xticklabels()]
#[ttl.set_size(20) for ttl in ax.get_yticklabels()]

rcj(ax)
tl(fig)


# %% Cl, Cd, and ClCd curves for paper (updated)

rd = np.rad2deg
gld = rd(gl_means[:, 0])

fig, ax = plt.subplots()
ax.axvline(0, color='gray', lw=1)
ax.axhline(0, color='gray', lw=1)
ax.plot(gld, cl_means[:, 0], 'o', ms=6, label=r"$C_L$")
ax.plot(gld, cd_means[:, 0], 's', ms=6, label=r"$C_D$")
ax.plot(gld, clcd_means[:, 0], '^', ms=6, label=r"$C_L/C_D$")
ax.plot(rd(gl_spl), clspl, c=bmap[0], lw=1.5)
ax.plot(rd(gl_spl), cdspl, c=bmap[1], lw=1.5)
ax.plot(rd(gl_spl), clspl / cdspl, c=bmap[2], lw=1.5)

ax.errorbar(rd(gl_means[:, 0]), cl_means[:, 0], yerr=cl_means[:, 1],
            color=bmap[0], lw=1, linestyle='')
ax.errorbar(rd(gl_means[:, 0]), cd_means[:, 0], yerr=cd_means[:, 1],
            color=bmap[1], lw=1, linestyle='')
ax.errorbar(rd(gl_means[:, 0]), clcd_means[:, 0], yerr=clcd_means[:, 1],
            color=bmap[2], lw=1, linestyle='')

#ax.errorbar(rd(gl_means[:, 0]), clcd_means[:, 0], yerr=clcd_means[:, 1],
#            xerr=rd(gl_means[:, 1]), color=bmap[2], lw=1)

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

ax.text(2, -1.75, 'kinematic squirrel', {'fontsize': 18})

[ttl.set_size(18) for ttl in ax.get_xticklabels()]
[ttl.set_size(18) for ttl in ax.get_yticklabels()]

rcj(ax)
tl(fig)

fig.savefig('Figures/figure7a_kinematic_squirrel.pdf', transparent=True)


# %% Check the interpolations (degrees) for SICB 2015

rd = np.rad2deg

fig, ax = plt.subplots()
ax.axvline(0, color='gray', lw=1)
ax.axhline(0, color='gray', lw=1)
ax.plot(rd(gl_spl), clspl, c=bmap[0], lw=3, label=r"$C_L$")
ax.plot(rd(gl_spl), cdspl, c=bmap[1], lw=3, label=r"$C_D$")
ax.plot(rd(gl_spl), clspl / cdspl, c=bmap[2], lw=3, label=r"$C_L/C_D$")
ax.legend(loc='lower right', frameon=False, fontsize=18)
ax.set_xlabel('angle of attack (deg)', fontsize=20)
ax.set_ylabel('force coefficients', fontsize=20)
[ttl.set_size(20) for ttl in ax.get_xticklabels()]
[ttl.set_size(20) for ttl in ax.get_yticklabels()]
ax.set_xlim(-15, 65)
ax.set_ylim(-2, 3)

rcj(ax)
tl(fig)


# %% Derivatives of Cl and Cd

fig, ax = plt.subplots()
ax.plot(gammas, Clprime_fun(gammas), '-o')
ax.plot(gammas, Cdprime_fun(gammas), '-o')
ax.plot(gl_spl, clpspl, c=bmap[0], label=r"$C_L'(\gamma)$")
ax.plot(gl_spl, cdpspl, c=bmap[1], label=r"$C_D'(\gamma)$")
ax.set_ylabel('derivative of force coefficients')
ax.set_xlabel('glide angle, rad')
ax.legend(loc='best', frameon=False)
rcj(ax)
tl(fig)


# %% Find equilibrium points

pitches = np.deg2rad(np.linspace(-25, 45, 4000))
gammas = np.deg2rad(np.linspace(15, 80, 1000))
sq_angle_rng = (gl_spl[0], gl_spl[-1])

sq_equil_exp = eqns.pitch_bifurcation(pitches, gammas, Cl_fun, Cd_fun,
                                      angle_rng=sq_angle_rng)
sq_equil_spl = eqns.pitch_bifurcation(pitches, gammas, cl_fun, cd_fun,
                                      angle_rng=sq_angle_rng)


# %% Plot the pitch bifurcation diagram

fig, ax = plt.subplots()
ax.plot(sq_equil_spl[:, 1], sq_equil_spl[:, 0], 'o', ms=1.5)
ax.plot(sq_equil_exp[:, 1], sq_equil_exp[:, 0], 'o', ms=1.5)
ax.set_xlabel('glide angle, rad')
ax.set_ylabel('pitch angle, rad')
rcj(ax)
tl(fig)


# %% Plot the pitch bifurcation diagram (degree)

sq_eq = rd(sq_equil_spl)

fig, ax = plt.subplots()
ax.plot(sq_eq[:, 0], sq_eq[:, 1], 'o', ms=1.5)
ax.set_xlabel('pitch angle (deg)')
ax.set_ylabel('glide angle at eq. (deg)')
rcj(ax)
tl(fig)



# %% Derivative values needed for Hopf bifrucation analysis of a parameter

# derivative functions (to be evaluated in radians)
clp_fun = cl_fun.derivative()
cdp_fun = cd_fun.derivative()
clpp_fun = clp_fun.derivative()
cdpp_fun = cdp_fun.derivative()
clppp_fun = clpp_fun.derivative()
cdppp_fun = cdpp_fun.derivative()


fig, axs = plt.subplots(2, 2, sharex=True, figsize=(9, 6))
ax1, ax2, ax3, ax4 = axs.flatten()

# plot vertical lines where we care about derivatives
pitch_colors = (bmap[2], bmap[3], bmap[4])
for ax in axs.flatten():
    rcj(ax)
    ax.grid(axis='y')
    [ttl.set_size(18) for ttl in ax.get_xticklabels()]
    [ttl.set_size(18) for ttl in ax.get_yticklabels()]
    for ii, pitch_test in enumerate(np.r_[-1, 5, 6]):
        index_for_pitch = np.abs(sq_eq[:, 0] - pitch_test).argmin()
        glide_ang_pitch = sq_eq[index_for_pitch, 1]
        # print sq_eq[index_for_pitch]
        ax.axvline(glide_ang_pitch, color=pitch_colors[ii], lw=1,
                   label=r'$\theta = {0}$'.format(pitch_test) + u'\u00B0')
        ax.axvline(glide_ang_pitch, color=pitch_colors[ii], lw=1)
        ax.axvline(glide_ang_pitch, color=pitch_colors[ii], lw=1)

# plot the derivative curves
ax1.plot(rd(gl_spl), cl_fun(gl_spl), label='lift')
ax1.plot(rd(gl_spl), cd_fun(gl_spl), label='drag')
ax2.plot(rd(gl_spl), clp_fun(gl_spl), c=bmap[0])
ax2.plot(rd(gl_spl), cdp_fun(gl_spl), c=bmap[1])
ax3.plot(rd(gl_spl), clpp_fun(gl_spl), c=bmap[0])
ax3.plot(rd(gl_spl), cdpp_fun(gl_spl), c=bmap[1])
ax4.plot(rd(gl_spl), clppp_fun(gl_spl), c=bmap[0])
ax4.plot(rd(gl_spl), cdppp_fun(gl_spl), c=bmap[1])

ax1.legend(loc='upper left', frameon=True, fontsize=12)
ax3.set_xlabel('$\gamma$', fontsize=18)
ax4.set_xlabel('$\gamma$', fontsize=18)
ax1.set_ylabel(r"$C_L$, $C_D$", fontsize=18)
ax2.set_ylabel(r"$C'_L$, $C'_D$", fontsize=18)
ax3.set_ylabel(r"$C''_L$, $C''_D$", fontsize=18)
ax4.set_ylabel(r"$C'''_L$, $C'''_D$", fontsize=18)

plt.draw()
for ax in [ax3, ax4]:
    ticks = ax.get_xticklabels()
    newticks = []
    for tick in ticks:
        text = tick.get_text()
        newticks.append(text + u'\u00B0')
    ax.set_xticklabels(newticks)

tl(fig)

fig.savefig('Data/kin_sq_coefficient_der/derivatives.pdf', transparent=True)


# %% Make a table of values

gah = np.zeros(3)  # glide_angle_hopfs
pitch_tests = np.r_[-1, 5, 6]
for ii, pitch_test in enumerate(pitch_tests):
    index_for_pitch = np.abs(sq_eq[:, 0] - pitch_test).argmin()
    glide_ang_pitch = sq_eq[index_for_pitch, 1]
    gah[ii] = np.deg2rad(glide_ang_pitch)


cl_hopfs = np.c_[pitch_tests, rd(gah), cl_fun(gah), clp_fun(gah), clpp_fun(gah), clppp_fun(gah)]
cd_hopfs = np.c_[pitch_tests, rd(gah), cd_fun(gah), cdp_fun(gah), cdpp_fun(gah), cdppp_fun(gah)]

np.savetxt('Data/kin_sq_coefficient_der/cl.csv', cl_hopfs, delimiter=',',
           header="pitch (deg), gamma* (deg), Cl, Cl', Cl'', Cl'''",
           fmt='%.6f')
np.savetxt('Data/kin_sq_coefficient_der/cd.csv', cd_hopfs, delimiter=',',
           header="pitch (deg), gamma* (deg), Cd, Cd', Cd'', Cd'''",
           fmt='%.6f')


# %% Classify the stability of fixed points

sq_td_exp, sq_ev_exp = eqns.tau_delta(sq_equil_exp, Cl_fun, Cd_fun,
                                      Clprime_fun, Cdprime_fun,
                                      angle_rng=sq_angle_rng)
sq_td_spl, sq_ev_spl = eqns.tau_delta(sq_equil_spl, cl_fun, cd_fun,
                                      clprime_fun, cdprime_fun,
                                      angle_rng=sq_angle_rng)


# %% Make tau, delta plot

fig, ax = plt.subplots()
ax.plot(sq_td_spl[:, 1], sq_td_spl[:, 0], 'o', ms=1.5, alpha=1)
ax.plot(sq_td_exp[:, 1], sq_td_exp[:, 0], 'o', ms=1.5, alpha=1)
ax.plot(sq_td_spl[0, 1], sq_td_spl[0, 0], '^', c=bmap[2], ms=6)
ax.axhline(0, color='gray', lw=.5)
ax.axvline(0, color='gray', lw=.5)
deltas = np.linspace(0, ax.get_xlim()[1], 1001)
taus = np.sqrt(4 * deltas)
ax.plot(deltas, taus, color='gray', lw=.5)
ax.plot(deltas, -taus, color='gray', lw=.5)
ax.set_xlabel(r'$\Delta$', fontsize=18)
ax.set_ylabel(r'$\tau$', fontsize=18)
ax.text(1.45, -2.35, r'$\tau^2 - 4 \Delta$', color='gray', fontsize=12)
rcj(ax)
tl(fig)


# %% Eigenvalue plot (Re vs. Im)

fig, ax = plt.subplots()
ax.plot(np.real(sq_ev_spl[:, 0]), np.imag(sq_ev_spl[:, 0]), 'o')
ax.plot(np.real(sq_ev_spl[:, 1]), np.imag(sq_ev_spl[:, 1]), 'o')
ax.spines['left'].set_position('zero')
ax.spines['right'].set_color('none')
ax.spines['bottom'].set_position('zero')
ax.spines['top'].set_color('none')
ax.spines['left'].set_smart_bounds(True)
ax.spines['bottom'].set_smart_bounds(True)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.set_xlabel(r'$Re(\lambda)$', fontsize=16)
ax.set_ylabel(r'$Im(\lambda)$', fontsize=16)
rcj(ax)
tl(fig)


# %% Eigenvalue plot (Re vs. Re, Im vs. Im)

fig, ax = plt.subplots()
ax.plot(np.real(sq_ev_spl[:, 0]), np.real(sq_ev_spl[:, 1]), 'o', label=r'$Re$')
ax.plot(np.imag(sq_ev_spl[:, 0]), np.imag(sq_ev_spl[:, 1]), 'o', label=r'$Im$')
ax.set_xlabel(r'$\lambda_1$', fontsize=16)
ax.set_ylabel(r'$\lambda_2$', fontsize=16)
ax.legend(loc='best', frameon=False)
rcj(ax)
tl(fig)


# %% Stability analysis

sq_nuni_exp, sq_uni_exp, sq_class_exp = eqns.classify_fp(sq_td_exp)
sq_nuni_spl, sq_uni_spl, sq_class_spl = eqns.classify_fp(sq_td_spl)

possible_class = ['saddle point', 'unstable focus', 'unstable node',
                  'stable focus', 'stable node']
bfbmap = [bmap[0], bmap[4], bmap[2], bmap[3], bmap[1]]


# %% Spline bifurcation plot

fig, ax = plt.subplots()
for ii, fp_kind in enumerate(possible_class):
    idx = np.where(sq_class_spl == fp_kind)[0]
    if len(idx) > 0:
        ax.plot(sq_equil_spl[idx, 1], sq_equil_spl[idx, 0], 'o', ms=2,
                label=fp_kind, c=bfbmap[ii], mec=bfbmap[ii])
    # sq_glide_ratio = 1 / np.tan(sq_equil_spl[idx, 1])
    # ax.plot(sq_glide_ratio, sq_equil_spl[idx, 0], 'o', ms=2, label=fp_kind)
_leg = ax.legend(loc='best', markerscale=4, fancybox=True, framealpha=.75)
_leg.get_frame().set_ec('w')
ax.axvspan(gl_means[-1, 0], np.min([glstop, sq_equil_exp[:, 1].max()]), color='gray', alpha=.1)
ax.set_xlabel(r'$\gamma^*$', fontsize=18)
# ax.set_xlabel(r'$GR$', fontsize=18)
ax.set_ylabel(r'$\theta$', fontsize=18)
ax.grid(False)
rcj(ax)
tl(fig)


# %% Load in the Hopf bifurcation data

# save the data
hgammas = np.genfromtxt('Data/Bahlman2013/hgammas.txt')
hpitches = np.genfromtxt('Data/Bahlman2013/hpitches.txt')
hopf_pitch = np.genfromtxt('Data/Bahlman2013/hopf_pitch.txt')

with open('Data/Bahlman2013/hopf_solutions.pkl', 'r') as fp:
    hopf_solutions = pickle.load(fp)


# %% Spline bifurcation plot (deg) for paper

rd = np.rad2deg

gam_high = sq_angle_rng[0] - pitches  # closer to 0
gam_low = sq_angle_rng[1] - pitches  # closer to 90

fig, ax = plt.subplots()

ax.fill_between(rd(pitches), rd(gam_high), 0, color='gray', alpha=.1, lw=0)
ax.fill_between(rd(pitches), rd(gam_low), 60, color='gray', alpha=.1, lw=0)

ax.axvline(0, color='gray')
ax.axvline(6, color='gray')

shallow = np.where((hgammas[:, 1] > hopf_pitch.max()) & (hpitches < np.deg2rad(4)))[0]
steep = np.where((hgammas[:, 0] > hopf_pitch.max()) & (hpitches < np.deg2rad(4)))[0]

ax.plot(rd(hpitches[shallow]), rd(hgammas[shallow, 0]), '-', c=bfbmap[1],
        lw=1.5)
ax.plot(rd(hpitches[steep]), rd(hgammas[steep, 1]), '-x', c=bfbmap[1], lw=1.5,
        label='unstable limit cycle')

for ii, fp_kind in enumerate(possible_class):
    idx = np.where(sq_class_spl == fp_kind)[0]
    if len(idx) > 0:
        ax.plot(rd(sq_equil_spl[idx, 0]), rd(sq_equil_spl[idx, 1]), 'o',
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

ax.text(-13, 5, 'kinematic squirrel', {'fontsize': 18})

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

fig.savefig('Figures/figure7b_bifurcation_kinematic_squirrel.pdf',
            transparent=True)


# %% Velocity bifurcation plot (could be for the paper...)


fig, ax = plt.subplots()

for ii, fp_kind in enumerate(possible_class):

    idx = np.where(sq_class_spl == fp_kind)[0]
    if len(idx) > 0:
        peq = sq_equil_spl[idx, 0]
        geq = sq_equil_spl[idx, 1]
        aleq = peq + geq
        veq = eqns.v_equil(aleq, cl_fun, cd_fun)
        vxeq = veq * np.cos(geq)
        vzeq = -veq * np.sin(geq)
        ax.plot(rd(peq), veq, 'o', c=bfbmap[ii], ms=2, mec=bfbmap[ii])

#        nos = np.ones(len(veq))
#        sk = 5
#        ax.plot(rd(peq), veq, rd(geq), 'o',
#                c=bfbmap[ii], ms=2, label=fp_kind, mec=bfbmap[ii])
#        ax.plot(rd(peq)[::sk], .755 * nos[::sk], rd(geq)[::sk], 'o',
#                c=bfbmap[ii], ms=2, mec=bfbmap[ii], alpha=.35)
#        ax.plot(rd(peq)[::sk], veq[::sk], 18 * nos[::sk], 'o',
#                c=bfbmap[ii], ms=2, mec=bfbmap[ii], alpha=.35)

#ax.set_ylim(ymax=.755)
ax.set_xlabel(r'$\theta$', fontsize=18)
ax.set_ylabel(r'$\hat{v}^*$', fontsize=18)
_leg = ax.legend(loc='best', markerscale=4, frameon=False, framealpha=.75)
rcj(ax)
tl(fig)


# %% Spline bifurcation plot (deg)

rd = np.rad2deg

fig, ax = plt.subplots()
for ii, fp_kind in enumerate(possible_class):
    idx = np.where(sq_class_spl == fp_kind)[0]
    if len(idx) > 0:
        # ax.plot(rd(sq_equil_spl[idx, 1]), rd(sq_equil_spl[idx, 0]), 'o', ms=2,
        #         label=fp_kind, c=bfbmap[ii])
        ax.plot(rd(sq_equil_spl[idx, 0]), rd(sq_equil_spl[idx, 1]), 'o', ms=2,
                    label=fp_kind, c=bfbmap[ii], mec=bfbmap[ii])
_leg = ax.legend(loc='best', markerscale=4, fancybox=True, framealpha=.75)
_leg.get_frame().set_ec('w')
#ax.axhspan(rd(gl_means[-1, 0]), rd(np.min([glstop, sq_equil_exp[:, 1].max()])),
#           color='gray', alpha=.1)
ax.set_ylabel(r'$\gamma^*$', fontsize=18)
# ax.set_xlabel(r'$GR$', fontsize=18)
ax.set_xlabel(r'$\theta$', fontsize=18)
ax.grid(False)

ax.axvline(0, color='gray')
ax.axvline(6, color='gray')
ax.axvline(-6, color='gray')
#ax.axvline(0, color='gray')
#ax.axvline(2.5, color='gray')
#ax.axvline(-2.5, color='gray')
#ax.axvline(5, color='gray')

[ttl.set_size(14) for ttl in ax.get_xticklabels()]
[ttl.set_size(14) for ttl in ax.get_yticklabels()]

rcj(ax)
tl(fig)


# %% 3D bifurcation plot

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.view_init(elev=35, azim=-136)

for ii, fp_kind in enumerate(possible_class):

    idx = np.where(sq_class_spl == fp_kind)[0]
    if len(idx) > 0:
        peq = sq_equil_spl[idx, 0]
        geq = sq_equil_spl[idx, 1]
        aleq = geq + peq
        veq = eqns.v_equil(geq, cl_fun, cd_fun)
        vxeq = veq * np.cos(geq)
        vzeq = -veq * np.sin(geq)

        nos = np.ones(len(veq))
        sk = 5
        ax.plot(rd(peq), veq, rd(geq), 'o',
                c=bfbmap[ii], ms=2, label=fp_kind, mec=bfbmap[ii])
        ax.plot(rd(peq)[::sk], .755 * nos[::sk], rd(geq)[::sk], 'o',
                c=bfbmap[ii], ms=2, mec=bfbmap[ii], alpha=.35)
        ax.plot(rd(peq)[::sk], veq[::sk], 18 * nos[::sk], 'o',
                c=bfbmap[ii], ms=2, mec=bfbmap[ii], alpha=.35)

ax.set_ylim(ymax=.755)
ax.set_xlabel(r'$\theta$', fontsize=18)
ax.set_zlabel(r'$\gamma^*$', fontsize=18)
ax.set_ylabel(r'$\hat{v}^*$', fontsize=18)
_leg = ax.legend(loc='best', markerscale=4, frameon=False, framealpha=.75)

tl(fig)


# %% Interpolated bifuration plot

fig, ax = plt.subplots()
for ii, fp_kind in enumerate(possible_class):
    idx = np.where(sq_class_exp == fp_kind)[0]
    if len(idx) > 0:
        ax.plot(sq_equil_exp[idx, 1], sq_equil_exp[idx, 0], 'o', ms=2,
                label=fp_kind, c=bfbmap[ii], mec=bfbmap[ii])
    # sq_glide_ratio = 1 / np.tan(sq_equil_exp[idx, 1])
    # ax.plot(sq_glide_ratio, sq_equil_exp[idx, 0], 'o', ms=2, label=fp_kind)
_leg = ax.legend(loc='best', markerscale=4, fancybox=True, framealpha=.75)
_leg.get_frame().set_ec('w')
ax.axvspan(gl_means[-1, 0], sq_equil_exp[:, 1].max(), color='gray', alpha=.1)
ax.set_xlabel(r'$\gamma^*$', fontsize=18)
# ax.set_xlabel(r'$GR$', fontsize=18)
ax.set_ylabel(r'$\theta$', fontsize=18)
ax.grid(False)
rcj(ax)
tl(fig)


# %% Plot the phase portrait, pitch = 0 deg

pitch = 0

afdict = dict(cli=cl_fun, cdi=cd_fun, clip=clprime_fun, cdip=cdprime_fun)
arng = sq_angle_rng
#arng = np.r_[.98, 1.02] * np.r_[arng]
lims = (vxlim, vzlim) = np.r_[0, 1.25], np.r_[0, -1.25]
tvec = np.linspace(0, 30, 351)
nseed = 8
seedloc = np.c_[.6 * np.ones(nseed), np.linspace(0, -.3, nseed)]
#seedloc = None

reload(plots)
from plots import phase_plotter as ppr

fig, ax = ppr(afdict, pitch, lims, arng, tvec, ngrid=201,
              nseed=501, nseed_skip=25, quiver=False, skip=10, seed=False,
              timer=True, gamtest=gammas, extrap=None,
              traj=plots.ps_traj_dp5, seedloc=seedloc,
              fig=None, ax=None)

# plot Hopf bifurcations
ixhopf = np.abs(np.rad2deg(hopf_pitch)).argmin()
ax.plot(hopf_solutions[ixhopf][-500:, 2], hopf_solutions[ixhopf][-500:, 3],
        c=bfbmap[1], lw=2)

lab = 'kinematic squirrel, ' + r'$\theta=$0' + u'\u00B0'
ax.text(.05, -1, lab, fontsize=16)

fig.savefig('Figures/figure7ci_vpd0_kinematic_squirrel.pdf', transparent=True)


# %% Plot phase portraint, pitch = 6 deg

pitch = np.deg2rad(6)

fig, ax = ppr(afdict, pitch, lims, arng, tvec, ngrid=201,
              nseed=501, nseed_skip=15, quiver=False, skip=10, seed=False,
              timer=True, gamtest=gammas, extrap=None,
              traj=plots.ps_traj_dp5, seedloc=None,
              fig=None, ax=None)

lab = 'kinematic squirrel, ' + r'$\theta=$6' + u'\u00B0'
ax.text(.05, -1, lab, fontsize=16)

fig.savefig('Figures/figure7cii_vpd6_kinematic_squirrel.pdf', transparent=True)


# %% Individualized simulations/phase portraits


# [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14]
# [ 8, 15, 17, 19, 22, 23, 27, 29, 33, 34, 45, 46, 50, 54, 59]

ix = 13

cols = index_save
col = cols[ix]

ms, ln, Sa = mass, length, Sarea
eps = eqns.calc_epsilon(ms, ln, Sa)
pret = np.sqrt(ln / (grav * eps))
prep = ln / eps
prev = np.sqrt(ln * grav / eps)

_out = eqns.clean_df(col, dfgm, Cl, Cd, dfpx, dfpz, dfvx, dfvz)
gm, cl, cd, px, pz, vx, vz, tv = _out
nn = np.arange(len(cl))
gm, tv = gm[nn], tv[nn]
px, pz = px[nn], pz[nn]
vx, vz = vx[nn], vz[nn]
cli, cdi, clpi, cdpi, dd = eqns.extend_clcd(gm, cl, cd, s=.0001, endper=0)

# non-dimensionalize and rescale the position and velocity
vhat = np.c_[vx, vz] / prev
phat = np.c_[px, pz] / prep

gm, cl, cd = dd['gm'], dd['cl'], dd['cd']

gmd = np.rad2deg(gm)
fig, ax = plt.subplots()
ax.plot(gmd, cl, 'o', c=bmap[1])
ax.plot(gmd, cd, 'o', c=bmap[2])
ax.plot(gmd, cli(gm), '-', c=bmap[4])
ax.plot(gmd, cdi(gm), '-', c=bmap[4])
#ax.plot(np.rad2deg(dd['gm']), dd['cl'], '-', c=bmap[0])
#ax.plot(np.rad2deg(dd['gm']), dd['cd'], '-', c=bmap[3])
ax.axvline(np.rad2deg(dd['gm_max']), color='gray')
rcj(ax)
tl(fig)

fig, ax = plt.subplots()
ax.plot(gmd, np.gradient(cl) / np.gradient(gm), 'o', c=bmap[1])
ax.plot(gmd, np.gradient(cd) / np.gradient(gm), 'o', c=bmap[2])
ax.plot(gmd, clpi(gm), '-', c=bmap[4])
ax.plot(gmd, cdpi(gm), '-', c=bmap[4])
ax.axvline(np.rad2deg(dd['gm_max']), color='gray')
rcj(ax)
tl(fig)


# %% VPD individual ix = 13

pitch = np.deg2rad(0)

afdict = dict(cli=cli, cdi=cdi, clip=clpi, cdip=cdpi)
arng = (gm.min(), gm.max())
tvec = np.linspace(0, 15, 551)
nseed = 8
seedloc = np.c_[.6 * np.ones(nseed), np.linspace(0, -.3, nseed)]

import plots
reload(plots)
from plots import phase_plotter as ppr

fig, ax = ppr(afdict, pitch, lims, arng, tvec, ngrid=201,
              nseed=101, nseed_skip=5, quiver=False, skip=25, seed=False,
              timer=True, gamtest=gammas, extrap=None, seedloc=seedloc,
              traj=plots.ps_traj_dp5, fig=None, ax=None)

# plot the trajectory on the vpd
ax.plot(vhat[:, 0], vhat[:, 1], '-x', lw=2, c=bmap[1], ms=7, markevery=10,
        mew=1)

lab = 'individual squirrel (trial 54)'
ax.text(.05, -1, lab, fontsize=16)

fig.savefig('Figures/figure7d_vpd_ix13_trial54_kinematic_squirrel.pdf',
            transparent=True)


# %% Individualized simulations/phase portraits


# [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14]
# [ 8, 15, 17, 19, 22, 23, 27, 29, 33, 34, 45, 46, 50, 54, 59]

ix = 4

cols = index_save
col = cols[ix]

ms, ln, Sa = mass, length, Sarea
eps = eqns.calc_epsilon(ms, ln, Sa)
pret = np.sqrt(ln / (grav * eps))
prep = ln / eps
prev = np.sqrt(ln * grav / eps)

_out = eqns.clean_df(col, dfgm, Cl, Cd, dfpx, dfpz, dfvx, dfvz)
gm, cl, cd, px, pz, vx, vz, tv = _out
nn = np.arange(len(cl))
gm, tv = gm[nn], tv[nn]
px, pz = px[nn], pz[nn]
vx, vz = vx[nn], vz[nn]
cli, cdi, clpi, cdpi, dd = eqns.extend_clcd(gm, cl, cd, s=.0001, endper=0)

# non-dimensionalize and rescale the position and velocity
vhat = np.c_[vx, vz] / prev
phat = np.c_[px, pz] / prep

gm, cl, cd = dd['gm'], dd['cl'], dd['cd']

gmd = np.rad2deg(gm)
fig, ax = plt.subplots()
ax.plot(gmd, cl, 'o', c=bmap[1])
ax.plot(gmd, cd, 'o', c=bmap[2])
ax.plot(gmd, cli(gm), '-', c=bmap[4])
ax.plot(gmd, cdi(gm), '-', c=bmap[4])
#ax.plot(np.rad2deg(dd['gm']), dd['cl'], '-', c=bmap[0])
#ax.plot(np.rad2deg(dd['gm']), dd['cd'], '-', c=bmap[3])
ax.axvline(np.rad2deg(dd['gm_max']), color='gray')
rcj(ax)
tl(fig)

fig, ax = plt.subplots()
ax.plot(gmd, np.gradient(cl) / np.gradient(gm), 'o', c=bmap[1])
ax.plot(gmd, np.gradient(cd) / np.gradient(gm), 'o', c=bmap[2])
ax.plot(gmd, clpi(gm), '-', c=bmap[4])
ax.plot(gmd, cdpi(gm), '-', c=bmap[4])
ax.axvline(np.rad2deg(dd['gm_max']), color='gray')
rcj(ax)
tl(fig)


# %% VPD individual, ix = 4

pitch = np.deg2rad(0)

afdict = dict(cli=cli, cdi=cdi, clip=clpi, cdip=cdpi)
arng = (gm.min(), gm.max())
tvec = np.linspace(0, 15, 551)
nseed = 8
seedloc = np.c_[.6 * np.ones(nseed), np.linspace(0, -.3, nseed)]

import plots
reload(plots)
from plots import phase_plotter as ppr

fig, ax = ppr(afdict, pitch, lims, arng, tvec, ngrid=201,
              nseed=101, nseed_skip=5, quiver=False, skip=25, seed=False,
              timer=True, gamtest=gammas, extrap=None, seedloc=seedloc,
              traj=plots.ps_traj_dp5, fig=None, ax=None)

# plot the trajectory on the vpd
ax.plot(vhat[:, 0], vhat[:, 1], '-x', lw=2, color=bmap[1], ms=7, markevery=10,
        mew=1)

lab = 'individual squirrel (trial 22)'
ax.text(.05, -1, lab, fontsize=16)

fig.savefig('Figures/figure7e_vpd_ix04_trial22_kinematic_squirrel.pdf',
            transparent=True)


# %% Hopf bifurcation by integrating backwards

# NOTE: This take a long time to run (~600 seconds)

if False:

    reload(plots)
    from plots import ps_traj_dp5
    idx = np.where(sq_class_spl == 'stable focus')[0]
    hopf_pitch, hopf_gam = sq_equil_spl[idx].T
    # hopf_pitch, hopf_gam = sq_equil_spl.T

    afdict = dict(cli=cl_fun, cdi=cd_fun, clip=clprime_fun, cdip=cdprime_fun)
    arng = sq_angle_rng
    vxlim, vzlim = np.r_[0, 1.25], np.r_[0, -1.25]

    vbar = eqns.v_equil(hopf_pitch + hopf_gam, afdict['cli'], afdict['cdi'])
    vxbar, vzbar = eqns.vxvz_equil(vbar, hopf_gam)

    solns, hgammas, hpitches = [], [], []
    tvec = np.linspace(0, -150, 700)
    now_total = time.time()
    for pitch, gam in zip(hopf_pitch[::1], hopf_gam[::1]):
        odeargs = (pitch, afdict['cli'], afdict['cdi'])
        vbar = eqns.v_equil(pitch + gam, afdict['cli'], afdict['cdi'])
        vxbar, vzbar = eqns.vxvz_equil(vbar, gam)
        x0 = (0, 0, vxbar, .99 * vzbar)

        now = time.time()
        soln, tsol = ps_traj_dp5(x0, tvec, odeargs, eqns.cart_model, arng,
                                 vxlim, vzlim, rett=True, ignore_vxvz=True)

        # if True:  # this will locate the limit cycle, even when go out of viewed range
        if np.allclose(tsol[-1], tvec[-1]):
            print('integration time: {0:.3f}'.format(time.time() - now))
            solns.append(soln)
            gam = eqns.calc_gamma(soln[:, 2], soln[:, 3])
            hgammas.append(np.r_[gam.min(), gam.max()])
            hpitches.append(pitch)
    print('total time: {0:.3f}'.format(time.time() - now_total))

    hgammas = np.array(hgammas)
    hpitches = np.array(hpitches)

    # save the data
    np.savetxt('Data/Bahlman2013/hgammas.txt', hgammas)
    np.savetxt('Data/Bahlman2013/hpitches.txt', hpitches)
    np.savetxt('Data/Bahlman2013/hopf_pitch.txt', hopf_pitch)

    # save the solutions for plotting on VPD
    with open('Data/Bahlman2013/hopf_solutions.pkl', 'w') as fp:
        pickle.dump(solns, fp)
