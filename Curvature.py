import numpy as np
from astropy.coordinates import SkyCoord
from pint.models import get_model
from pint import toa
from pint import simulation
from RickettEquations import SystemVel, OrbitMeanVel, RotateVector
from astropy import constants as const

import sys
sys.path.append('..')  # Add parent directory to the system path


def eta(t_start, t_end, t_nsteps, fitval, par, freq):
    """
    This function calculates the expected curvature of scintillation arcs.
    
    Parameters
    -----------
            t_start: float  
                The start of the epoch that the curvature is being calculated for in MJD
            t_end: float 
                The end of the epoch that the curvature is being calculated for in MJD
            t_nsteps: int
                Number of observations to calculate the curvature for
            fival: dict
                Physical parameters, which must include:

                i : `~astropy.units.Quantity`
                    the inclination angle of the orbit
                s : `~astropy.units.Quantity`
                    the fractional distance from the pulsar to the scintillation screen
                    (dimensionless)
                Oangle :  `~astropy.units.Quantity`
                   the angle needed to rotate RA/Dec coordinates onto x-y plane
                PsiAR : `~astropy.units.Quantity`
                    angle describing orientation of major axis of screen from the x axis
                VIS :  `~astropy.units.Quantity`
                    the best fit velocity of the interstellar scintillation screen,
                    in x and y directions
                dpsr: `~astropy.units.Quantity`
                    distance to the pulsar
            par : str
                filename for ATNF par file
            freq :  `~astropy.units.Quantity`
                frequency of interest
    Returns
    -------
            eta : `~astropy.units.Quantity`
                the expected curvature of the scintillation arcs, shape (t_nsteps,1)
                (dimensionless)
            VA : `~astropy.units.Quantity`
                The scintillation velocity parallel and perpendicular to the screen major axis,
                shape (t_nsteps,2)
"""
    
    psr_m = get_model(par)  # create model of pulsar
    
    psr = SkyCoord(ra=str(psr_m.RAJ.quantity), dec=str(psr_m.DECJ.quantity),
                   pm_ra_cosdec=psr_m.PMRA.quantity, pm_dec=psr_m.PMDEC.quantity, distance=fitval['dpsr'])
    
    # Calculate values from binary model for given time frame
    t = simulation.make_fake_toas_uniform(t_start, t_end, t_nsteps,
                           psr_m, freq=freq, obs="GBT")
    
    psr_m.delay(t)
    
    bm = psr_m.binary_instance
    
    phase = bm.nu()+bm.omega()
    
    # Calculate scintillation velocity
    VC = SystemVel(t_start, t_end, 1, fitval, psr)
    
    V0 = OrbitMeanVel(psr_m.PB.quantity, psr_m.A1.quantity /
                      psr_m.SINI.quantity, psr_m.ECC.quantity)
     
    VAx = VC[:, 0] - V0*psr_m.ECC.quantity * \
        np.sin(bm.omega()) - V0*np.sin(phase)
    VAy = VC[:, 1] + np.cos(fitval['i'])*(V0*psr_m.ECC.quantity *
                                          np.cos(bm.omega()) + V0*np.cos(phase))
    VA = RotateVector([VAx, VAy], -fitval['PsiAR'])

    eta = const.c*fitval['dpsr']*fitval['s'] / \
        (2*freq*freq*(1-fitval['s'])*VA[0]*VA[0])
    VA = [VA[1], -VA[0]]
    return eta, VA



def eta_vel(t_start, t_end, t_nsteps, fitval, par, freq):
    """
    This function calculates the expected Velocities .
    
    Parameters
    -----------
            t_start: float  
                The start of the epoch that the curvature is being calculated for in MJD
            t_end: float 
                The end of the epoch that the curvature is being calculated for in MJD
            t_nsteps: int
                Number of observations to calculate the curvature for
            fival: dict
                Physical parameters, which must include:

                i : `~astropy.units.Quantity`
                    the inclination angle of the orbit
                s : `~astropy.units.Quantity`
                    the fractional distance from the pulsar to the scintillation screen
                    (dimensionless)
                Oangle :  `~astropy.units.Quantity`
                   the angle needed to rotate RA/Dec coordinates onto x-y plane
                PsiAR : `~astropy.units.Quantity`
                    angle describing orientation of major axis of screen from the x axis
                VIS :  `~astropy.units.Quantity`
                    the best fit velocity of the interstellar scintillation screen,
                    in x and y directions
                dpsr: `~astropy.units.Quantity`
                    distance to the pulsar
            par : str
                filename for ATNF par file
            freq :  `~astropy.units.Quantity`
                frequency of interest
    Returns
    -------

            VA : `~astropy.units.Quantity`
                The pulsar velocity in RA, and DEC
                shape (t_nsteps,2)
            phase: `~astropy.units.Quantity`
                The orbital phase as defined in Rickett 2014
"""
    
    psr_m = get_model(par)  # create model of pulsar
    
    
    psr = SkyCoord(ra=str(psr_m.RAJ.quantity), dec=str(psr_m.DECJ.quantity),
                   pm_ra_cosdec=psr_m.PMRA.quantity, pm_dec=psr_m.PMDEC.quantity, distance=fitval['dpsr'])
    
    # Calculate values from binary model for given time frame
    t = simulation.make_fake_toas_uniform(t_start, t_end, t_nsteps,
                           psr_m, freq = freq, obs="mk")
    
    psr_m.delay(t)
    
    bm = psr_m.binary_instance
    
    phase = bm.nu()+bm.omega()
    
    # mean velocity
    V0 = OrbitMeanVel(psr_m.PB.quantity, psr_m.A1.quantity /
                      psr_m.SINI.quantity, psr_m.ECC.quantity)
    
    
    VC = SystemVel(t_start, t_end, 1, fitval, psr)

    
    
    #Vx and Vy 
    VAx = VC[:, 0] - V0*psr_m.ECC.quantity * \
        np.sin(bm.omega()) - V0*np.sin(phase)
    VAy = VC[:, 1] + np.cos(fitval['i'])*(V0*psr_m.ECC.quantity *
                                          np.cos(bm.omega()) + V0*np.cos(phase))
    
    #VAxy = [VAx, VAy]
    #derotating them into ra / dec
    #VA = deRotateVector([VAx, VAy], fitval['Oangle'])
    
    VA = RotateVector([VAx, VAy], -fitval['PsiAR'])

    VA = [VA[1], -VA[0]]

    return VA, phase, bm.nu(), VC


def get_time_phase(t_start, t_end, t_nsteps, par, freq):
    """
    This function calculates the expected Velocities .
    
    Parameters
    -----------
            t_start: float  
                The start of the epoch that the curvature is being calculated for in MJD
            t_end: float 
                The end of the epoch that the curvature is being calculated for in MJD
            t_nsteps: int
                Number of observations to calculate the curvature for
            par : str
                filename for ATNF par file
            freq :  `~astropy.units.Quantity`
                frequency of interest
    Returns
    -------

            t
            phase: `~astropy.units.Quantity`
                The orbital phase as defined in Rickett 2014
    """
    
    psr_m = get_model(par)  # create model of pulsar
    
    # Calculate values from binary model for given time frame
    t = simulation.make_fake_toas_uniform(t_start, t_end, t_nsteps,
                           psr_m, freq = freq, obs="mk")
    
    psr_m.delay(t)
    bm = psr_m.binary_instance
    
    phase = bm.nu()+bm.omega()
    
    
    psr = SkyCoord(ra=str(psr_m.RAJ.quantity), dec=str(psr_m.DECJ.quantity),
                   pm_ra_cosdec=psr_m.PMRA.quantity, pm_dec=psr_m.PMDEC.quantity, distance=fitval['dpsr'])
    
    # Calculate values from binary model for given time frame
    t = simulation.make_fake_toas_uniform(t_start, t_end, t_nsteps,
                           psr_m, freq = freq, obs="mk")
    
    psr_m.delay(t)
    
    bm = psr_m.binary_instance
    
    phase = bm.nu()+bm.omega()
    


    return t, phase, bm.nu()
