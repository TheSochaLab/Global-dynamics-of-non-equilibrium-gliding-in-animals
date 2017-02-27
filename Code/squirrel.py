from __future__ import division


import numpy as np
from scipy import interpolate
import pandas as pd


def load_run(run_num, df):
    """Load in trial data.

    Parameters
    ----------
    run_num : int
        Which run to load.
    df : DataFrame
        The DataFrame loaded from the original excel file.

    Returns
    -------
    pos : array
        (x, z) positions
    tvec : array
        Time vector for the run.
    dt : float
        Sampling interval between data points.
    """

    # sampling rate
    # http://rsif.royalsocietypublishing.org/content/10/80/20120794/suppl/DC1
    if run_num <= 7:
        dt = 1 / 60.
    else:
        dt = 1 / 125.

    xkey = "'Caribou_Trial_{0:02d}_Xvalues'".format(run_num)
    zkey = "'Caribou_Trial_{0:02d}_Zvalues'".format(run_num)

    d = df[[xkey, zkey]]
    d = np.array(d)

    # get rid of nans and a bunch of junky zeros starting at row 301
    start_bad = np.where(np.isnan(d))[0]
    if len(start_bad) > 0:
        start_bad = start_bad[0]
        d = d[:start_bad]

    # get rid of zeros (if we get past rows 301...)
    start_bad = np.where(d == 0.)[0]
    if len(d) > 300 and len(start_bad) > 0:
        start_bad = start_bad[0]
        d = d[:start_bad]

    tvec = np.arange(0, len(d)) * dt

    return d, tvec, dt


def calc_vel(pos_data, dt):
    """Velocity in the x and z directions.

    Parameters
    ----------
    pos_data : array
        (x, z) position information
    dt : float
        Sampling rate

    Returns
    -------
    vel : array
        (vx, vz)
    """

    vx = np.gradient(pos_data[:, 0], dt)
    vy = np.gradient(pos_data[:, 1], dt)

    return np.c_[vx, vy]


def calc_accel(vel_data, dt):
    """Acceleration in the x and z directions.

    Parameters
    ----------
    vel_data : array
        (vx, vz) velocity data
    dt : float
        Sampling rate

    Returns
    -------
    accel : array
        (ax, az)
    """

    ax = np.gradient(vel_data[:, 0], dt)
    ay = np.gradient(vel_data[:, 1], dt)

    return np.c_[ax, ay]


def calc_vel_mag(vel_data):
    """Velocity magnitude.

    Parameters
    ----------
    vel_data : array
        (vx, vz) velocity data

    Returns
    -------
    vel_mag : array
        np.sqrt(vx**2 + vz**2)
    """

    return np.sqrt(vel_data[:, 0]**2 + vel_data[:, 1]**2)


def calc_gamma(vel_data):
    """Glide angle.

    Parameters
    ----------
    vel_data : array
        (vx, vz)

    Returns
    -------
    gamma : array
        Glide angle in rad
    """

    return -np.arctan2(vel_data[:, 1], vel_data[:, 0])


def splfit_all(data, tvec, k=5, s=.5):
    """Fit a spline to the data.
    """

    posx = interpolate.UnivariateSpline(tvec, data[:, 0], k=k, s=s)
    posz = interpolate.UnivariateSpline(tvec, data[:, 1], k=k, s=s)
    velx = posx.derivative(1)
    velz = posz.derivative(1)
    accx = posx.derivative(2)
    accz = posz.derivative(2)

    pos = np.c_[posx(tvec), posz(tvec)]
    vel = np.c_[velx(tvec), velz(tvec)]
    acc = np.c_[accx(tvec), accz(tvec)]

    return pos, vel, acc


def polyfit(data, tvec, intfun):
    """Fit a spline to the data.
    """

    posx = intfun(tvec, data[:, 0])
    posz = intfun(tvec, data[:, 1])
    velx = posx.derivative(1)
    velz = posz.derivative(1)
    accx = posx.derivative(2)
    accz = posz.derivative(2)

    pos = np.c_[posx(tvec), posz(tvec)]
    vel = np.c_[velx(tvec), velz(tvec)]
    acc = np.c_[accx(tvec), accz(tvec)]

    return pos, vel, acc


def polyfit_all(data, tvec, deg, wn=0):
    """Fit a spline to the data.
    TODO: this does not to the mirroring correctly!
    """

    start = data[:wn][::-1]
    stop = data[-wn:][::-1]
    datanew = np.r_[start, data, stop]
    tvecnew = np.r_[tvec[:wn][::-1], tvec, tvec[-wn:][::-1]]

    posx = np.polyfit(tvecnew, datanew[:, 0], deg)
    posz = np.polyfit(tvecnew, datanew[:, 1], deg)
    velx = np.polyder(posx, 1)
    velz = np.polyder(posz, 1)
    accx = np.polyder(posx, 2)
    accz = np.polyder(posz, 2)

    pos = np.c_[np.polyval(posx, tvec), np.polyval(posz, tvec)]
    vel = np.c_[np.polyval(velx, tvec), np.polyval(velz, tvec)]
    acc = np.c_[np.polyval(accx, tvec), np.polyval(accz, tvec)]

    return pos, vel, acc


def fill_df(pos, vel, acc, gamma, velmag, tvec, i):
    """Put one trial's data into a DataFrame.

    Parameters
    ----------
    pos : (n x 2) array
        x and z position data
    vel : (n x 2) array
        x and z velocity data
    acc : (n x 2) array
        x and z acceleration data
    gamma : (n x 1) array
        Glide angles in deg
    velmag : (n x 1) array
        Velocity magnitude
    tvec : (n x 1) array
        Time points
    i : int
        Trial number that becomes the column name

    Returns
    -------
    posx, posz, velx, velz, accx, accz, gamm, vmag : DataFrame
        Data in a DataFrame
    """

    posx = pd.DataFrame(pos[:, 0], index=tvec, columns=[str(i)])
    posz = pd.DataFrame(pos[:, 1], index=tvec, columns=[str(i)])

    velx = pd.DataFrame(vel[:, 0], index=tvec, columns=[str(i)])
    velz = pd.DataFrame(vel[:, 1], index=tvec, columns=[str(i)])

    accx = pd.DataFrame(acc[:, 0], index=tvec, columns=[str(i)])
    accz = pd.DataFrame(acc[:, 1], index=tvec, columns=[str(i)])

    gamm = pd.DataFrame(gamma, index=tvec, columns=[str(i)])
    vmag = pd.DataFrame(velmag, index=tvec, columns=[str(i)])

    return posx, posz, velx, velz, accx, accz, gamm, vmag


def window_bounds(i, n, wn):
    """Start and stop indices for a moving window.

    Parameters
    ----------
    i : int
        Current index
    n : int
        Total number of points
    wn : int, odd
        Total window size

    Returns
    -------
    start : int
        Start index
    stop : int
        Stop index
    at_end : bool
        Whether we are truncating the window
    """

    at_end = False
    hw = wn // 2

    start = i - hw
    stop = i + hw + 1

    if start < 0:
        at_end = True
        start = 0
    elif stop > n:
        at_end = True
        stop = n

    return start, stop, at_end


def moving_window_pts(data, tvec, wn, deg=2, drop_deg=False):
    """Perform moving window smoothing.

    Parameters
    ----------
    data : (n x 2) array
        Data to smooth and take derivatives of
    tvec : (n x 1) array
        Time vector
    wn : int, odd
        Total window size
    deg : int, default=2
        Polynomial degree to fit to data
    drop_deg : bool, default=False
        Whether to drop in interpolating polynomial at the
        ends of the time series, since the truncated window can
        negatively affect things.

    Returns
    -------
    spos : (n x 2) array
        x and z smoothed data
    svel : (n x 2) array
        First derivatives of smoothed data (velocity)
    sacc : (n x 2) array
        Second derivatives of smoothed data (acceleration)
    """

    deg_orig = deg
    posx, posz = data.T
    npts = len(posx)
    spos = np.zeros((npts, 2))
    svel = np.zeros((npts, 2))
    sacc = np.zeros((npts, 2))

    for i in range(npts):
        start, stop, at_end = window_bounds(i, npts, wn)
        if at_end and drop_deg:
            deg = deg_orig - 1
        else:
            deg = deg_orig

        t = tvec[start:stop]
        x = posx[start:stop]
        z = posz[start:stop]

        pfpx = np.polyfit(t, x, deg)
        pfpz = np.polyfit(t, z, deg)
        pfvx = np.polyder(pfpx, m=1)
        pfvz = np.polyder(pfpz, m=1)
        pfax = np.polyder(pfpx, m=2)
        pfaz = np.polyder(pfpz, m=2)

        tval = tvec[i]
        spos[i] = np.polyval(pfpx, tval), np.polyval(pfpz, tval)
        svel[i] = np.polyval(pfvx, tval), np.polyval(pfvz, tval)
        sacc[i] = np.polyval(pfax, tval), np.polyval(pfaz, tval)

    return spos, svel, sacc


def moving_window_pos(data, tvec, wn, deg=2):
    """Do a moving window of +/- wn, where wn is position.
    """

    xwn = wn
    hxwn = xwn / 2
    posx, posz = data.T
    npts = len(posx)
    spos = np.zeros((npts, 2))
    svel = np.zeros((npts, 2))
    sacc = np.zeros((npts, 2))

    for i in range(npts):

        ind = np.where((posx >= posx[i] - hxwn) & (posx <= posx[i] + hxwn))[0]
        t = tvec[ind]
        x = posx[ind]
        z = posz[ind]

        pfpx = np.polyfit(t, x, deg)
        pfpz = np.polyfit(t, z, deg)
        pfvx = np.polyder(pfpx, m=1)
        pfvz = np.polyder(pfpz, m=1)
        pfax = np.polyder(pfpx, m=2)
        pfaz = np.polyder(pfpz, m=2)

        tval = tvec[i]
        spos[i] = np.polyval(pfpx, tval), np.polyval(pfpz, tval)
        svel[i] = np.polyval(pfvx, tval), np.polyval(pfvz, tval)
        sacc[i] = np.polyval(pfax, tval), np.polyval(pfaz, tval)

    return spos, svel, sacc


def moving_window_spl(data, tvec, wn, s=.5):
    """Do a moving window of +/- wn on the data and
    take derivatves.
    """

    posx, posz = data.T
    npts = len(posx)
    spos = np.zeros((npts, 2))
    svel = np.zeros((npts, 2))
    sacc = np.zeros((npts, 2))

    for i in range(npts):
        start, stop, at_end = window_bounds(i, npts, wn)

        t = tvec[start:stop]
        x = posx[start:stop]
        z = posz[start:stop]

        px = interpolate.UnivariateSpline(t, x, k=5, s=s)
        pz = interpolate.UnivariateSpline(t, z, k=5, s=s)
        vx = px.derivative(1)
        vz = pz.derivative(1)
        ax = px.derivative(2)
        az = pz.derivative(2)

        tval = tvec[i]
        spos[i] = px(tval), pz(tval)
        svel[i] = vx(tval), vz(tval)
        sacc[i] = ax(tval), az(tval)

    return spos, svel, sacc


def svfilter(tvec, data, wn, order, mode='interp'):
    """Use a Savitzky-Golay to smooth position data and to
    calculate the derivatives.

    This blog post has a modification of this, which might have better
    high frequency filtering: http://bit.ly/1wjZKvk
    """

    from scipy.signal import savgol_filter
    x, z = data.T
    dt = np.diff(tvec).mean()

    px = savgol_filter(x, wn, order, mode=mode)
    pz = savgol_filter(z, wn, order, mode=mode)
    vx = savgol_filter(x, wn, order, mode=mode, deriv=1, delta=dt)
    vz = savgol_filter(z, wn, order, mode=mode, deriv=1, delta=dt)
    ax = savgol_filter(x, wn, order, mode=mode, deriv=2, delta=dt)
    az = savgol_filter(z, wn, order, mode=mode, deriv=2, delta=dt)

    return np.c_[px, pz], np.c_[vx, vz], np.c_[ax, az]


def clcd_binning(gl_bins, gl_rad, Cl, Cd):
    """Bin the lift and drag coefficient curves against glide angle
    to get average across all trajectories

    Parameters
    ----------
    gl_bins : array
        The different bins [left, right)
    gl_rad : DataFrame
        Glide angle data in radians
    Cl : DataFrame
        Lift coefficients
    Cd : DataFrame
        Drag coefficients

    Returns
    -------
    clcd_means : array, (n x 3)
        lift-to-drag ratio mean, std, stderror
    cl_means : array, (n x 3)
        same for lift coefficient
    cd_means : array, (n x 3)
        same for drag coefficient
    gl_means : array, (n x 3)
        same for glide angle

    Notes
    -----
    This uses a Taylor expansion for the Cl/Cd ratio statistics,
    becuase I guess using a standard ratio is biased.
    """

    nbins = len(gl_bins)
    gl_flattened = gl_rad.values.flatten()
    cl_flattened = Cl.values.flatten()
    cd_flattened = Cd.values.flatten()

    bins = np.digitize(gl_flattened, gl_bins)

    all_indices = []
    no_data = []
    cl_means = np.zeros((nbins, 3))
    cd_means = np.zeros((nbins, 3))
    clcd_means = np.zeros((nbins, 3))
    gl_means = np.zeros((nbins, 3))
    for idx in np.arange(nbins):

        # find relevent indices
        all_indices.append(np.where(bins == idx)[0])
        indices = np.where(bins == idx)[0]
        if len(indices) == 0:
            no_data.append(idx)
            continue

        # get out our data
        glsnip = gl_flattened[indices]
        clsnip = cl_flattened[indices]
        cdsnip = cd_flattened[indices]

        clcd_means[idx] = taylor_moments(clsnip, cdsnip)
        cl_means[idx] = simple_moments(clsnip)
        cd_means[idx] = simple_moments(cdsnip)
        gl_means[idx] = simple_moments(glsnip)

    # remove where we have no interpolation
    # clcd_means[no_data] = np.nan
    # cl_means[no_data] = np.nan
    # cd_means[no_data] = np.nan
    # gl_means[no_data] = np.nan
    return clcd_means[1:], cl_means[1:], cd_means[1:], gl_means[1:]


def taylor_moments(x, y):
    """Taylor series approximations to the moments of a ratio.

    See http://bit.ly/1uy8qND and http://bit.ly/VHPX4u
    and http://en.wikipedia.org/wiki/Ratio_estimator

    Parameters
    ----------
    x : 1D array
        Numerator of the ratio
    y : 1D array
        Denomenator of the ratio

    Returns
    -------
    tmean : float
        Mean of the ratio
    tstd : float
        STD of the ratio
    tserr : float
        Standard error of the ratio
    """

    n = len(x)
    ex = x.mean()
    ey = y.mean()
    varx = x.var()
    vary = y.var()
    cov = np.cov(x, y)[0, 1]

    tmean = ex / ey - cov / ey**2 + ex / ey**3 * vary
    tvar = varx / ey**2 - 2 * ex / ey**3 * cov + ex**2 / ey**4 * vary
    tstd = np.sqrt(tvar)

    return tmean, tstd, tstd / np.sqrt(n)


def simple_moments(x):
    """Moments for Cl and Cd curves.

    Parameters
    ----------
    x : 1D numpy array

    Returns
    -------
    mean, std, sterr
    """

    mean = x.mean()
    std = x.std()
    sterr = std / np.sqrt(len(x))

    return mean, std, sterr


def interpspl(data, npts, k=3, s=3):
    """Interpolate using splines.
    """

    tck, u = interpolate.splprep(data.T, k=k, s=s, nest=-1)
    datanew = interpolate.splev(np.linspace(0, 1, npts), tck)
    return np.array(datanew).T
