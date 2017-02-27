# -*- coding: utf-8 -*-
"""
Created on Tue Sep  2 20:33:55 2014

@author: isaac
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import newton


def calc_epsilon(m, L, S):
    """Non-dimensional gliding parameter, epsilon.

    Parameters
    ----------
    m : float
        mass in kg
    L : float
        length in m
    S : float
        area in m^2

    Returns
    -------
    epsilon : float
        Non-dimensional parameter
    """

    rho_air = 1.204
    return .5 * (rho_air * L * S) / m


def calc_gamma(vx, vz):
    """Calculate glide angle.

    Parameters
    ----------
    vx : float or array
        velocity in x-direction
    vz : float or array
        velocity in z-direction

    Returns
    -------
    gamma : float
        Glide angle in rad.
    """
    return -np.arctan2(vz, vx)


def cart_eqns(vx, vz, Cl, Cd):
    """Return dvx and dvz.

    Parameters
    ----------
    vx : float or array
        Velocity in horizontal direction
    vz : float or array
        Velocity in vertical direction
    Cl : float or array
        Current lift coefficient
    Cd : float or array
        Current drag coefficient

    Returns
    -------
    dvx, dvx : floats or arrays
        Derivatives of states.
    """
    vmag = np.sqrt(vx**2 + vz**2)
    dvx = -vmag * (Cl * vz + Cd * vx)
    dvz = vmag * (Cl * vx - Cd * vz) - 1

    return dvx, dvz


def cart_model(x, t, args):
    """Return derivative of states (dx, dz, dvx, dvz)

    Parameters
    ----------
    x : array
        State vector of px, pz, vx, vz.
    t : float
        Current time.
    args : tuple, theta, cli, cdi
        Pitch angle is in deg and cli and cdi are functions
        that return Cl and Cd given an angle of attack.

    Returns
    -------
    dpx, dpz, dvx, dvz : floats
        Derivatives of the current states
    """

    px, pz, vx, vz = x
    theta, cli, cdi = args
    gamma = calc_gamma(vx, vz)
    alpha = gamma + theta

    vmag = np.hypot(vx, vz)

    Cl = cli(alpha)
    Cd = cdi(alpha)

    dvx = -vmag * (Cl * vz + Cd * vx)
    dvz = vmag * (Cl * vx - Cd * vz) - 1

    return vx, vz, dvx, dvz


def polar_eqns(gam, vmag, Cl, Cd):
    """Return dgam and dvmag.

    Parameters
    ----------
    gam : float or array
        Glide angle in radians
    vmag : float or array
        Velocity magnitude
    Cl : float or array
        Current lift coefficient
    Cd : float or array
        Current drag coefficient

    Returns
    -------
    dgam, dvmag : floats or arrays
        Derivatives of states.
    """
    gam = np.deg2rad(gam)
    dv = -vmag**2 * Cd + np.sin(gam)
    dgam = -vmag * Cl + np.cos(gam) / vmag

    return np.rad2deg(dgam), dv


def polar_model(x, t, args):
    """Return derivative of states (dx, dz, dgam, dv)

    Parameters
    ----------
    x : array
        State vector of px, pz, gam, vz.
    t : float
        Current time.
    args : tuple, theta, cli, cdi
        Pitch angle is in deg and cli and cdi are functions
        that return Cl and Cd given an angle of attack.

    Returns
    -------
    dpx, dpz, dvx, dvz : floats
        Derivatives of the current states
    """

    gamma, vmag = x
    theta, cli, cdi = args
    alpha = gamma + theta

    # these take alpha in deg
    Cl = cli(alpha)
    Cd = cdi(alpha)

    # now we need gamma in rad for trig functions
    gamma = np.deg2rad(gamma)

    dv = -vmag**2 * Cd + np.sin(gamma)
    dgam = -vmag * Cl + np.cos(gamma) / vmag

    return np.rad2deg(dgam), dv


def v_equil(alpha, cli, cdi):
    """Calculate the equilibrium glide velocity.

    Parameters
    ----------
    alpha : float
        Angle of attack in rad
    cli : function
        Returns Cl given angle of attack
    cdi : function
        Returns Cd given angle of attack

    Returns
    -------
    vbar : float
        Equilibrium glide velocity
    """

    den = cli(alpha)**2 + cdi(alpha)**2
    return 1 / den**(.25)


def vxvz_equil(vbar, gambar):
    """Calculate the equilibrium glide velocities, vx and vz.

    Parameters
    ----------
    vbar : float
        Equilibrium glide velocity
    gambar : float
        Equilibrium glide angle in rad

    Returns
    -------
    vx : float
        Equilibrium glide velocity in x-direction
    vz : float
        Equilibrium glide velocity in z-direction
    """

    vx = vbar * np.cos(gambar)
    vz = -vbar * np.sin(gambar)
    return vx, vz


def jacobian_polar(alpha, vbar, gambar, cli, cdi, clpi, cdpi):
    """Jacobian matrix for the polar equations.

    Parameters
    ----------
    alpha : float
        Glide anlge in rad.
    vbar : float
        Equilibrium glide velocity.
    gambar : float
        Equilibrium glide angle in rad.
    cli, cdi, clpi, cdpi : functions
        Interpolation function for Cl, Cd, Cl', Cd'

    Returns
    -------
    A : array
        2 x 2 Jacobian matrix
    """

    a = -vbar * clpi(alpha) - np.sin(gambar) / vbar
    b = -cli(alpha) - np.cos(gambar) / vbar**2
    c = -vbar**2 * cdpi(alpha) + np.cos(gambar)
    d = -2 * vbar * cdi(alpha)

    return np.array([[a, b], [c, d]])


def sign_changes(arr):
    """Find intervals where the sign between elements changes.

    Parameters
    ----------
    arr : 1D numpy array
        Arry to find sign changes in.

    Returns
    -------
    intervals : 2D numpy array
        The (start, stop) indices where a sign changes occurs. The top
        part of the arry is negative to positive and the bottom
        part is positive to negative.

    Notes
    -----
    We don't do anything special if there is a zero (mostly because we don't
    expect to have identically zero values in the array).
    """

    neg2pos = np.where(np.diff(np.sign(arr)) == 2)[0]
    pos2neg = np.where(np.diff(np.sign(arr)) == -2)[0]
    neg2pos = np.c_[neg2pos, neg2pos + 1]
    pos2neg = np.c_[pos2neg, pos2neg + 1]
    intervals = np.r_[neg2pos, pos2neg]

    return intervals


def equil_gamma_newton(fun, guesses):
    """Find the equilibrium points using Newton's method.

    Parameters
    ----------
    fun : function
        Function that returns cot(gamma) - Cl/Cd
    guesses : array
        Initial guess array as input to newton

    Returns
    -------
    equil : 1D numpy array
        The equilibrium glide angles
    """

    nequil = len(guesses)
    equil = []  # np.zeros(nequil)
    for i in range(nequil):
        equil.append(newton(fun, guesses[i]))

    return np.array(equil)


def pitch_bifurcation(test_pitches, test_gammas, cli, cdi, angle_rng=None):
    """Find the equilibrium glide angle(s) for a particular pitch.

    Parameters
    ----------
    test_pitches : array
        Pitch angles in rad to iterate through and test
    test_gammas : array
        Glide angles in rad to check fro equilibrium over
    cli : function
        Function that returns Cl
    cdi : function
        Function that returns Cd
    angle_rng : list, default=None
        The (low, high) angle regions where the interpolation functions
        are valid. This is to prevent issues with extrapolating when
        using a spline.

    Returns
    -------
    all_roots : array, (nroots, 2)
        Array with the pitch angles and equilibrium glide angles.
    """

    # deal with extrapolation
    if angle_rng is None:
        goodidx = np.arange(len(test_gammas))

    test_cot = 1 / np.tan(test_gammas)
    all_equil = []
    for pitch in test_pitches:
        test_alpha = test_gammas + pitch
        if angle_rng is not None:
            al, ah = angle_rng
            goodidx = np.where((test_alpha >= al) & (test_alpha <= ah))[0]
            if len(goodidx) < 2:
                continue

        ratio = cli(test_alpha) / cdi(test_alpha)
        zero = test_cot[goodidx] - ratio[goodidx]
        zero_fun = interp1d(test_gammas[goodidx], zero)

        intervals = sign_changes(zero)
        guesses = test_gammas[goodidx][intervals].mean(axis=1)
        equil_gammas = equil_gamma_newton(zero_fun, guesses)

        equil_pitches = pitch * np.ones(len(equil_gammas))
        equil = np.c_[equil_pitches, equil_gammas]
        all_equil.append(equil)

    # get the list of arrays into one list
    all_roots = all_equil[0]
    for arr in all_equil[1:]:
        all_roots = np.r_[all_roots, arr]

    return all_roots


def tau_delta(equil, cli, cdi, clpi, cdpi, angle_rng=None):
    """Return the trace and determinant of the equilibrium points.

    Parameters
    ----------
    equil : array, (pitch, glide angle)
         Result from pitch_bifurcation
    cli, cdi, clpi, cdpi : functions
        Interpolation functions
    angle_rng : list, default=None
        The (low, high) angle regions where the interpolation functions
        are valid. This is to prevent issues with extrapolating when
        using a spline.

    Returns
    -------
    td : array
        Trace, determinant from eigenvalue equation
    eigvals : array
        Sorted eigenvectors
    """

    pitches, gammas = equil.T
    alphas = pitches + gammas
    vbars = v_equil(alphas, cli, cdi)

    if angle_rng is not None:
        al, ah = angle_rng
        bad_idx = (alphas <= al) & (alphas >= ah)
    else:
        bad_idx = np.array([False] * len(equil))

    td = np.zeros((equil.shape[0], 2))
    eigvals = np.zeros((equil.shape[0], 2), dtype=np.complex128)

    for i in range(len(equil)):
        jac = jacobian_polar(alphas[i], vbars[i], gammas[i],
                             cli, cdi, clpi, cdpi)
        eigs = np.linalg.eigvals(jac)
        eigs = np.sort(eigs)[::-1]

        tau, delta = np.trace(jac), np.linalg.det(jac)
        if not bad_idx[i]:
            td[i] = tau, delta
            eigvals[i] = eigs[0], eigs[1]
        else:
            td[i] = np.nan, np.nan
            eigvals[i] = np.nan, np.nan

    return td, eigvals


def classify_fp(td):
    """Classify the fixed points according to the tau-delta plot
    (see Strogatz, p. 137).

    Parameters
    ----------
    td : (n x 2) array
        Columns are taus and deltas for a fixed point

    Returns
    -------
    nunique : int
        Number of unique type of fixed point
    unique : array
        The types of fixed points we have (sorted)
    classification : array
        String for the type of fixed point
    """

    classification = np.zeros(len(td), dtype='|S15')
    for i, (tau, delta) in enumerate(td):
        if delta < 0:
            classification[i] = 'saddle point'
        elif delta == 0:
            classification[i] = 'non-isolated fixed point'
        elif tau == 0:
            classification[i] = 'center'
        elif np.abs(tau) > np.sqrt(4 * delta):
            if tau > 0:
                classification[i] = 'unstable node'
            elif tau < 0:
                classification[i] = 'stable node'
            else:
                print('Should be a node...')
        elif np.abs(tau) < np.sqrt(4 * delta):
            if tau > 0:
                classification[i] = 'unstable focus'
            elif tau < 0:
                classification[i] = 'stable focus'
            else:
                print('Should be a spiral')

    unique = np.sort(np.unique(classification))
    nunique = len(unique)

    return nunique, unique, classification
