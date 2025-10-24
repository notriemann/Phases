import numpy as np
from astropy import units as u
from astropy.coordinates import EarthLocation, SkyOffsetFrame
from astropy.time import Time

"""TODO:
-Make functions so they have default values
"""


def Q_coeff(R, Psi_AR):
    """Calculates the quadratic coefficients which describe the ISS anisotropy.

    As defined in equation 4 of Rickett et al. 2014.

    Parameters
    ----------
    R : float
        A bounded parameter related to the axial ratio, with range 0 to 1
        where 0 describes a circle and 1 describes a line.
    Psi_AR: `~numpy.ndarray` or `~astropy.units.Quantity`
        Angle describing orientation of the major axis (in radians if not
        a Quantity).

    Returns
    -------
    coefficients : `~numpy.ndarray`
        quadratic coefficients a, b, and c.
    """
    Psi_AR = u.Quantity(Psi_AR, u.radian).value
    a = (1 - R * np.cos(2 * Psi_AR)) / np.sqrt(1 - R * R)
    b = (1 + R * np.cos(2 * Psi_AR)) / np.sqrt(1 - R * R)
    c = -2 * R * np.sin(2 * Psi_AR) / np.sqrt(1 - R * R)
    return np.array([a, b, c])


def OrbitMeanVel(PB, SMA, ECC):
    """Calculate the mean orbital velocity of the pulsar.

    Parameters
    ----------
    PB : `~astropy.units.Quantity`
        Orbital period of the pulsar (units of time).
    SMA: `~astropy.units.Quantity`
        Semi major axis (units of length).
    ECC: `~astropy.units.Quantity`
        Orbital eccentricity (dimensionless).

    Returns
    -------
    V0 : `~astropy.units.Quantity`
        The mean orbital velocity in km/s.
    """
    return (2 * np.pi * SMA / (PB * np.sqrt(1 - ECC * ECC))).to(u.km / u.s)


def SpatialScale(s0, s):
    """Calculates the unitless spatial scintillation scale in the pulsar frame.

    Parameters
    ----------
    s0 : `~astropy.units.Quantity`
        Mean diffractive scale (units of length).
    s : `~astropy.units.Quantity`
        Fractional distance from the pulsar to the scintillation screen
        (dimensionless).

    Returns
    -------
    sp : `~astropy.units.Quantity`
         Spatial scale in the pulsar frame.
    """
    return s0 / (1 - s)


def EarthVelocity(t, site, psr, rot):
    """Get the proper earth velocity in RA-DEC coordinates in the pulsar frame.

    For data taken from a given site, relative to the sun.

    Parameters
    ----------
    t : `~astropy.time.Time`
        Time of the observation.
    site : `~astropy.coordinates.EarthLocation` or string
        Location or name of the observatory where data was taken.
    psr : `~astropy.coordinates.SkyCoord`
        Position of the pulsar.
    rot : `~astropy.coordinates.Angle`
        Angle at which to set the offset frame.

    Returns
    -------
    v_earth : `~astropy.units.Quantity`
        Site XYZ velocities in units of km/s, with Y and Z giving the
        velocities in the RA and DEC directions.
    """
    psr_frame = SkyOffsetFrame(origin=psr, rotation=rot)
    if not isinstance(site, EarthLocation):
        site = EarthLocation.of_site(site)
    pos = site.get_gcrs(t).transform_to(psr_frame).cartesian
    vel = pos.differentials['s']
    return vel.d_xyz.to(u.km / u.s)


def PulsarBCVelocity(psr):
    """Calculate the proper motion of the barycentre of the pulsar binary system.
    Parameters
    ----------
    psr : `~astropy.coordinates.SkyCoord`
        Position of the pulsar.

    Returns
    -------
    pm_psr : `~astropy.coordinates.CartesianDifferential`
        3D pulsar barycentre proper motion.
    """
    psr_frame = SkyOffsetFrame(origin=psr, rotation=0 * u.deg)
    # if I put this into the pulsar frame it seems that dy and dz are the
    # Valpha and Vdelta
    pm_psr = psr.transform_to(psr_frame).cartesian.differentials['s']
    return pm_psr


def RotateVector(v, angle):
    """Rotate a vector from RA-DEC to pulsar orbital frame coordinates.

    Parameters
    ----------
    v : `~astropy.units.Quantity` or `~numpy.ndarray`
        Proper motions in velocity units in RA, Dec.  In km/s if an array.
    angle : `~astropy.coordinates.Angle` or float
        Rotation angle (radians if float).

    Returns
    -------

    Vxy : `~astropy.units.Quantity`
        Velocities in the X and Y directions.
    """
    v = u.Quantity(v, u.km / u.s)
    new_v = [np.sin(angle) * v[0] + np.cos(angle) * v[1],
             -np.cos(angle) * v[0] + np.sin(angle) * v[1]]
    return u.Quantity(new_v)


def SystemVel(t_start, t_end, t_nsteps, fitval, psr):
    """Calculates the system velocity in the pulsar frame.

    As defined in equation 6 of Rickett et al. 2014.

    Parameters
    ----------
    t_start : int
        Initial MJD of observation.
    t_end : int
        Final MJD of observation.
    t_nsteps : int
        Number of days to calculate the system velocity for.
    fitval : dict
        Physical physical parameters, which must include:
        s : fractional distance from the pulsar to the scintillation screen.
            (dimensionless
        VIS : Best fit velocities of the interstellar scintillation screen,
              in X and Y.
        Oangle : the angle needed to rotate RA/Dec coordinates onto x-y plane.
    psr : `~astropy.coordinates.SkyCoord`
        Position of the pulsar.

    Returns
    -------
    VC : `~astropy.units.Quantity`
        Shape ``t_nsteps, 2``, representing x and y tranverse system velocity.
    """
    times = Time(np.linspace(t_start, t_end, t_nsteps), format='mjd')

    # Calculate Earth velocity
    VE = u.Quantity(np.ones((t_nsteps, 2)), u.km/u.s)
    for i, t in enumerate(times):
        VE[i] = RotateVector(EarthVelocity(t, 'gbt', psr, 0*u.deg)[1:3],
                             fitval['Oangle'])

    # Calculate pulsar velocity
    # Uncomment next two lines if want to just use values from Rickett et al. 2014
    #VP = np.array([-17.8, 11.6]) * u.km / u.s
    #VP = RotateVector(VP, fitval['Oangle'])  # rotate pulsar velocity by Oangle

    # Comment these two lines if using values from Rickett et al. 2014
    vpsr = PulsarBCVelocity(psr)
    VP = RotateVector([vpsr.d_y,vpsr.d_z], fitval['Oangle'])  # rotate pulsar velocity by Oangle

    VC = u.Quantity(np.ones((t_nsteps, 2)), u.km/u.s)
    for i, v in enumerate(VE):
        VC[i] = np.add(np.add(VP, v * fitval['s'] / (1 - fitval['s'])),
                       -fitval['VIS'] / (1 - fitval['s']))

    return VC


def K_coeffs(t_start, fitval, psr, psr_m):
    """Calculates the orbital harmonic coefficients for the scintillation
    timescale.

    As defined in equation 10 of Rickett et al. 2014.

    Parameters
    ----------
    t_start : int, or tuple
        MJD of observation, or tuple of start, stop, N
    fitval : dict
        Physical physical parameters, which must include:
        s : fractional distance from the pulsar to the scintillation screen.
            (dimensionless
        VIS : Best fit velocities of the interstellar scintillation screen,
              in X and Y.
        Oangle : the angle needed to rotate RA/Dec coordinates onto x-y plane.
        R : dimensionless parameter related to the axial ratio, with range
            0 to 1, where 0 describes a circle and 1 describes a line.
        PsiAR : angle describing orientation of the major axis of screen
                from the x axis.
        s0 : the mean diffractive scale (units of length).
        i : inclination angle of the orbit.
    psr : `~astropy.coordinates.SkyCoord`
        Position of the pulsar.
    psr_ m : pint model
        Timing model for the  pulsar (determined from par file).

    Returns
    -------
    K0, KS, KC, KS2, KC2: `~astropy.units.Quantity`
         The orbital harmonic coefficients.
    """

    # convert projected semi major axis to actual value
    SMA = (psr_m.A1.quantity / psr_m.SINI.quantity)
    V0 = OrbitMeanVel(psr_m.PB.quantity, SMA, psr_m.ECC.quantity)
    if not isinstance(t_start, tuple):
        t_start = (t_start, t_start + 1, 1)
    VC = SystemVel(*t_start, fitval, psr).T
    Qabc = Q_coeff(fitval['R'], fitval['PsiAR'])
    omega = psr_m.OM.quantity
    ecc = psr_m.ECC.quantity
    sp = fitval['s0'] / (1 - fitval['s'])
    i = fitval['i']

    K0 = (0.5 * V0 * V0 * (Qabc[0] + Qabc[1] * (np.cos(i))**2)
          + Qabc[0] * (VC[0] - V0 * ecc * np.sin(omega))**2)
    K0 = K0 + Qabc[1] * (VC[1] + V0 * ecc * np.cos(omega) * np.cos(i))**2
    K0 = K0 + (Qabc[2] * (VC[0] - V0 * ecc * np.sin(omega))
               * (VC[1] + V0 * ecc * np.cos(omega) * np.cos(i)))
    K0 = K0.to(u.m * u.m / (u.s * u.s))

    KS = -V0 * (2 * Qabc[0] * (VC[0] - V0 * ecc * np.sin(omega))
                + Qabc[2] * (VC[1] + V0 * ecc * np.cos(i) * np.cos(omega)))
    KS = KS.to(u.m * u.m / (u.s * u.s))

    KC = V0 * np.cos(i) * (Qabc[2] * (VC[0] - V0 * ecc * np.sin(omega)) +
                           2 * Qabc[1] * (VC[1] + V0 * ecc * np.cos(i) * np.cos(omega)))
    KC = KC.to(u.m * u.m / (u.s * u.s))

    KS2 = -0.5 * Qabc[2] * V0 * V0 * np.cos(i)
    KS2 = KS2.to(u.m * u.m / (u.s * u.s))

    KC2 = 0.5 * V0 * V0 * (-Qabc[0] + Qabc[1] * (np.cos(i))**2)
    KC2 = KC2.to(u.m * u.m / (u.s * u.s))

    return [K / (sp * sp) for K in (K0, KS, KC, KS2, KC2)]


def TISS(K, phi):
    """Interstellar scintillation timescale.

    As defined in equation 9 of Rickett et al. 2014.

    Note that measured timescale values for MJD 52997, 53211, 53311,
    53467, 53560 are stored in .tiss5t files.  This function will only
    return the timescale as a function of phase for one set of orbital
    harmonic coefficients (i.e. one observation day).

    Parameters
    ----------
    K: list of `~astropy.units.Quantity`
        Orbital harmonic coefficients (K0, KS, KC, KS2, KC2).
    phi: `~astropy.units.Quantity` or float
        Orbital phase from the line of nodes (in radians if float).

    Returns
    -------
    TISS: interstellar scintillation timescale at phi, float

    """
    in_T = (K[0]
            + K[1] * np.sin(phi)
            + K[2] * np.cos(phi)
            + K[3] * np.sin(2 * phi) +
            + K[4] * np.cos(2 * phi))
    return np.sqrt(1 / in_T)


def k_norm(t_start, t_end, t_nsteps, fitval, psr, psr_m, bm):
    """Calculate the values of ux, uy, and w.

    See equation 13 of Rickett et al 2014.

    These variables combine to give normalized harmonic coefficients
    ks and k0, which are output by this function.

    Parameters
    ----------
    t_start : int
        Initial MJD of observation.
    t_end : int
        Final MJD of observation.
    t_nsteps : int
        Number of days to calculate the system velocity for.
    fitval : dict
        Physical physical parameters, which must include:
        s : fractional distance from the pulsar to the scintillation screen.
            (dimensionless
        VIS : Best fit velocities of the interstellar scintillation screen,
              in X and Y.
        Oangle : the angle needed to rotate RA/Dec coordinates onto x-y plane.
        R : dimensionless parameter related to the axial ratio, with range
            0 to 1, where 0 describes a circle and 1 describes a line.
        PsiAR : angle describing orientation of the major axis of screen
                from the x axis.
        i : inclination angle of the orbit.
    psr : `~astropy.coordinates.SkyCoord`
        Position of the pulsar.
    psr_ m : pint model
        Timing model for the pulsar (determined from par file).
    bm: pint binary instance
        For the pulsar.

    Returns
    -------
    ks, k0: float
        list of of dimensionless normalized harmonic coefficients, as defined
        in Eq. 14 of Rickett.

    """
    VC = SystemVel(t_start, t_end, t_nsteps, fitval, psr)
    # convert projected semi major axis to actual value
    SMA = (psr_m.A1.quantity / psr_m.SINI.quantity)
    V0 = OrbitMeanVel(psr_m.PB.quantity, SMA, psr_m.ECC.quantity)
    Qabc = Q_coeff(fitval['R'], fitval['PsiAR'])
    i = fitval['i']

    ux = (VC[:, 0] / V0 - psr_m.ECC.quantity * np.sin(bm.omega())).to_value(1)
    uy = (np.sqrt(Qabc[1] / Qabc[0]) * (VC[:, 1] / V0
                                        + (psr_m.ECC.quantity * np.cos(i)
                                           * np.cos(bm.omega())))).to_value(1)
    w = Qabc[2] / np.sqrt(Qabc[0] * Qabc[1])
    return [4 * ux + 2 * w * uy,
            -1 - 2 * ux * ux - 2 * w * ux * uy - 2 * uy * uy]