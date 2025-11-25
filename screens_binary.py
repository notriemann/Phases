#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from astropy import units as u
from astropy import constants as const
from astropy.coordinates import (
    CartesianRepresentation, CylindricalRepresentation,
    UnitSphericalRepresentation)
from astropy.visualization import quantity_support
from scipy.signal.windows import tukey

import sys
sys.path.append('..')  # Add parent directory to the system path

import screens
from screens.fields import dynamic_field
from screens.dynspec import DynamicSpectrum as DS
from screens.conjspec import ConjugateSpectrum as CS
from screens.screen import Source, Screen1D, Telescope
from screens.fields import phasor


from pint.models import get_model
from RickettTables import fitvals
from RickettEquations import *
from Curvature import *

from astropy.coordinates import SkyCoord, EarthLocation, SkyOffsetFrame
from astropy.time import Time
from matplotlib.colors import LogNorm, SymLogNorm
import matplotlib.animation as animation


# import scintools2.scintools.ththmod as thth
# from scintools2.scintools.dynspec import BasicDyn, Dynspec

from scipy.sparse.linalg import eigsh
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit, minimize
from scipy.sparse.linalg import svds
from scipy.interpolate import griddata


from Funcs_DP import *
from Funcs_DP_Orbsim import *
from Funcs_DP_Sspec import *
from Funcs_DP_thth import *
from Aux_funcs import *


def axis_extent(*args):
    result = []
    for a in args:
        x = a.squeeze().value
        result.extend([x[0] - (dx:=x[1]-x[0])/2, x[-1]+dx/2])
    return result

def evolve_sys( freq, d_p, d_s, p_x, e_x, s_x, s_n, s_mu):
    
    #setting pulsar source
    pulsar = Source(pos = p_x)
    
    #setting observation
    telescopes = Telescope(pos = e_x)
    #setting the line of sight
    obs_lineofsight = telescopes.observe( source=pulsar, distance=d_p)
    
    #setting 1d screen 
    scr = Screen1D(normal=s_n, p=s_x,
                magnification=s_mu)
    
    #setting observation with screen
    obs_scr1_pulsar = scr.observe(source=pulsar, distance=d_p-d_s)
    obs_scr1 = telescopes.observe(source=obs_scr1_pulsar, distance=d_s)
    
   
    #getting delays and taudot
    tau_t = np.hstack([ obs_lineofsight.tau.ravel() , 
                        obs_scr1.tau.ravel() ])
    
    #calculating dynamic spectrum
    ph = phasor(freq, tau_t[:, np.newaxis, np.newaxis])
    brightness = np.hstack([ obs_lineofsight.brightness.ravel() ,
                             obs_scr1.brightness.ravel()
                               ])
    
    dynwave = ph * brightness[:, np.newaxis, np.newaxis]

    dynspec = np.abs(dynwave.sum(axis=0))**2
    
    return dynspec.T, tau_t, brightness
    
def iterate_evolve(t, f, d_p, d_s, pul_v, ear_v, scr_v, p_pos, e_pos, screen_pos, scr_normal):
    #storage arrays
    dyns = []
    vs = []
    t_orb = []
    vsp = []
    taut = []
    dtaudt = []
    
    screen_pos1 = np.copy(screen_pos)
    p_pos1 = np.copy(p_pos)
    e_pos1 = np.copy(e_pos)

    for j in range(len(t)):

        dt = t[1] - t[0]

        #use the current position vector
        pulsar_pos_tmp = CartesianRepresentation( p_pos1 )
        earth_pos_tmp = CartesianRepresentation( e_pos1 )

        #screen vector, positions and magnifications
        amp_val = screen_pos.value #+ ( np.random.rand(len(scr1_pos)) - 0.5 )
        scr_magnification =  np.exp(-amp_val**2 ) * np.exp(1j * amp_val)
        N1 = len( scr_magnification)
        scr_magnification /= scr_magnification[N1 // 2]

        scrn_vec = np.array([ scr_normal.x.value, scr_normal.y.value, scr_normal.z.value ]) 

        #iterating the dynamic spectrum
        ds_tmp = evolve_sys(  
                            freq = f, 
                            d_p = d_p, 
                            d_s = d_s, 
                            p_x = pulsar_pos_tmp,  
                            e_x = earth_pos_tmp,  
                            s_x = screen_pos1,  
                            s_n = scr_normal, 
                            s_mu = scr_magnification
                            )

        #appending relevant quantities
        dyns += [ds_tmp[0]]

        #update the position vectors
        p_pos1 += ( pul_v  * dt ).to(u.au)
        e_pos1 += ( ear_v  * dt ).to(u.au)
        screen_pos1 += (scr_v * dt).to(u.au)


    dyns = np.array(dyns).T
    dyns = dyns.reshape(len(f),j+1)

    return dyns


def orbital_evolver(t, nu1, phase1, Omp, Pb, Vpx, Vpy, ip ):
    
    ecc = 0.088
    sma = 1.410090245 * u.s * const.c
    
    fv = (1 - ecc**2) / (1 + ecc * np.cos(nu1))
    
    rx = ( sma * fv * ( np.cos(phase1) * np.sin(Omp) - np.cos(Omp) * np.sin(phase1) * np.cos(ip) ) + Vpx * t ).to(u.Mm)
    ry = ( sma * fv * ( np.sin(phase1) * np.sin(Omp) * np.cos(ip) + np.cos(Omp) * np.cos(phase1)  ) + Vpy * t ).to(u.Mm)
    
    rx -= rx[0]
    ry -= ry[0]
    
    return rx.to(u.au), ry.to(u.au)

def iterate_evolve_orb(t, f, nu1, phase1, d_p, d_s, ear_v, scr_v, p_pos, e_pos, screen_pos, scr_normal, Pb, Omp, ip, Vpx, Vpy, sig1, amp1, rand1 = False):
    #storage arrays
    dyns = []
    vsp = []
    rsp = []
    eta = []
    
    screen_pos1 = np.copy(screen_pos)
    p_pos1 = np.copy(p_pos)
    e_pos1 = np.copy(e_pos)
    
    #keplerian parameters
    ecc = 0.088
    sma = 1.410090245 * u.s * const.c
    
    rx, ry = orbital_evolver(t, nu1, phase1, Omp, Pb, Vpx, Vpy, ip )
    
    rx -= rx[0]
    ry -= ry[0]
    
    vx = np.gradient( rx.to(u.km).value , t.to(u.s).value)
    vy = np.gradient( ry.to(u.km).value , t.to(u.s).value)
    
    scrn_vec = np.array([ scr_normal.x.value, scr_normal.y.value, scr_normal.z.value ]) 
    
    #screen vector, positions and magnifications
    amp_val = screen_pos.value #
    
    #if no phase was given compute a phase for the line of images
    if rand1 == True: 
        rand1 = np.exp( 2j*np.pi*np.random.uniform(size=amp_val.shape) )
        
    elif rand1 == False:
        rand1 = np.exp( 2j*np.pi* amp_val )
        

    for j in range(len(t)):

        dt = t[1] - t[0]

        #use the current position vector
        pulsar_pos_tmp = CartesianRepresentation( np.array([rx[j].value, ry[j].value, 0.])* u.au )
        earth_pos_tmp = CartesianRepresentation( e_pos1 )

        #screen vector, positions and magnifications
        amp_val = screen_pos.value #+ ( np.random.rand(len(scr1_pos)) - 0.5 )
        scr_magnification =  np.exp(-amp_val**2 / sig1**2 ) * rand1
        scr_magnification /= np.max( np.abs(  scr_magnification ) )
        scr_magnification *= amp1 

        #iterating the dynamic spectrum
        ds_tmp = evolve_sys(  
                            freq = f, 
                            d_p = d_p, 
                            d_s = d_s, 
                            p_x = pulsar_pos_tmp,  
                            e_x = earth_pos_tmp,  
                            s_x = screen_pos1,  
                            s_n = scr_normal, 
                            s_mu = scr_magnification
                            )

        #appending relevant quantities
        dyns += [ds_tmp[0]]
        
        #update the position vectors
#         p_pos1 += ( pul_v  * dt ).to(u.au)
        e_pos1 += ( ear_v  * dt ).to(u.au)
        screen_pos1 += (scr_v * dt).to(u.au)
        
        pv_tmp = np.array([ vx[j], vy[j], 0.]) 
        r_tmp = np.array([ rx[j].to(u.Mm).value, ry[j].to(u.Mm).value, 0. ])
        
        vp_proj = np.dot(pv_tmp, scrn_vec ) * u.km / u.s
        rp_proj = np.dot(r_tmp, scrn_vec ) * u.Mm
        ve_proj = np.dot(ear_v.value, scrn_vec ) * ear_v.unit
        
        s = 1 - d_s/d_p
        
        vsp += [ ( scr_v / s - (1-s) / s *  vp_proj - ve_proj).to(u.km / u.s).value ]
        rsp += [ ( (scr_v / s - ve_proj) * t[j]  - (1-s) / s *  rp_proj  ).to(u.Mm).value]
        eta += [( d_p * d_s / (d_p - d_s) * const.c / (2 * np.mean(f)**2 * (vsp[-1] * u.km / u.s)**2 ) ).to(u.s**3).value]


    dyns = np.array(dyns).T
    dyns = dyns.reshape(len(f),j+1)
    
    
    fv = (1 - ecc**2) / (1 + ecc * np.cos(nu1))
    
    pv_tmp = np.array([ Vpx.value, Vpy.value, 0.]) 
    vp_proj = np.dot(pv_tmp, scrn_vec ) * u.km / u.s
    xi_ang = (np.arctan2( scrn_vec[1],  scrn_vec[0] )*u.rad).to(u.deg) - 90 * u.deg
    delt_Om = xi_ang + Omp
    delta = np.arctan( -np.tan(delt_Om) * np.cos(ip) )
    
    A1 = - (1-s) / s *  vp_proj - ve_proj + scr_v / s
    B1 = - (1-s) / s * sma * fv * np.cos(delt_Om)
    C1 = - (1-s) / s * sma * fv * np.cos(ip) * np.sin(delt_Om) 
    
    A2 = Pb / (sma * np.sqrt( np.cos(delt_Om)**2 + np.sin(delt_Om)**2 * np.cos(ip)**2))
    A = A2 * A1 / (1 - s) * s
    
    B = (1-s) / s * sma * np.sqrt( np.cos(delt_Om)**2 + (np.cos(ip) * np.sin(delt_Om) )**2 )
    B = np.sqrt( ((1-s) / s * sma  * np.cos(delt_Om))**2 
               + ((1-s) / s * sma  * np.cos(ip) * np.sin(delt_Om) )**2)
    
    phys_params = [ A1, B1, C1, A.to(u.m/u.m), delta, B]

    return dyns, np.array(vsp) * u.km / u.s, np.array(rsp) * u.Mm, np.array(eta) * u.s**3, phys_params


def evolve_sys2( freq, d_p, d_s, d_s2, p_x, e_x, s_x, s_x2, s_n, s_n2, s_mu, s_mu2, spower, interaction = False):
    
    #setting pulsar source
    pulsar = Source(pos = p_x)
    
    #setting observation
    telescopes = Telescope(pos = e_x)
    #look at what this line does specifically
    los = telescopes.observe(source=pulsar, distance = d_p)
    
    #setting 1d screen 
    scr = Screen1D(normal=s_n, p=s_x,
                magnification=s_mu)
    #setting second screen
    scr2 = Screen1D(normal=s_n2, p=s_x2,
                magnification=s_mu2)
    
    #--------------Setting the interaction between the screens------------------------------
    
    #---------------------------------------------------------------------------------------
    #Interacting screens block
    #---------------------------------------------------------------------------------------
        #setting observation with screen1 
        #source -> screen1 -> obs
    if interaction == True:
        obs_scr1_pulsar = scr.observe(source=pulsar, distance=d_p-d_s)
        obs = telescopes.observe(source=obs_scr1_pulsar, distance=d_s)

        #setting observation with screen2
        #source -> screen2 -> screen1 -> obs
        obs_scr2_pulsar = scr2.observe(source=pulsar, distance=d_p-d_s2)
        obs_scr2_pulsar2 = scr.observe(source=obs_scr2_pulsar, distance=d_s2-d_s)
        obs2 = telescopes.observe(source=obs_scr2_pulsar2, distance=d_s)
    
    
    #---------------------------------------------------------------------------------------
    #Independent screens block
    #---------------------------------------------------------------------------------------
        #setting observation with screen1 
        #source -> screen1 -> obs
    elif interaction == False:
        obs_scr1_pulsar = scr.observe(source=pulsar, distance=d_p-d_s)
        obs = telescopes.observe(source=obs_scr1_pulsar, distance=d_s)

        #setting observation with screen2
        #source -> screen2 -> obs
        obs_scr2_pulsar2 = scr2.observe(source=pulsar, distance=d_p-d_s2)
        obs2 = telescopes.observe(source=obs_scr2_pulsar2, distance=d_s2)
    
    #--------------End of interactions between screens------------------------------
    
    obs2.tau
    bool_on_lens2 = obs2.source.pos.x.ravel() < 7. * u.au
    #getting delays and taudot
    tau0 = np.hstack([los.tau.ravel(), 
                      obs.tau.ravel(),
                      obs2.tau.ravel()[bool_on_lens2]])

    #calculating dynamic spectrum
    ph = phasor(freq, tau0[:, np.newaxis, np.newaxis])
    
    ptot = (  np.sum( np.abs(los.brightness.ravel()) ) 
            + np.sum( np.abs(obs.brightness.ravel()) ) 
            + np.sum( np.abs(obs2.brightness.ravel()[bool_on_lens2]) ) )
    
    
    
    p1_norm = ptot * (1. - spower[0] - spower[1]) / np.sum( np.abs(los.brightness.ravel()) ) * 1e1
    p2_norm = ptot *  spower[0] / np.sum( np.abs(obs.brightness.ravel()) ) * 1e1
    p3_norm = ptot * spower[1] / np.sum( np.abs(obs2.brightness.ravel()[bool_on_lens2]) ) * 1e1
    
    brightness = np.hstack([ los.brightness.ravel() * p1_norm,
                             obs.brightness.ravel() * p2_norm,
                             obs2.brightness.ravel()[bool_on_lens2] * p3_norm
                           ])
    
    dynwave = ph * brightness[:, np.newaxis, np.newaxis]
    dynspec = np.abs(dynwave.sum(axis=0))**2
    
    return dynspec.T, tau0, brightness


def iterate_evolve_orb2(t, f, nu1, phase1, d_p, d_s, ear_v, p_pos, e_pos, 
                        scr_v, screen_pos, scr_normal, 
                        d_s2, scr_v2, screen2_pos2, scr_normal2,
                        Pb, Omp, ip, Vpx, Vpy, spower = [0.09, 0.01]):
    #storage arrays
    dyns = []
    vsp = []
    rsp = []
    eta = []
    
    vsp2 = []
    rsp2 = []
    
    screen_pos1 = np.copy(screen_pos)
    screen_pos2 = np.copy(screen2_pos2)
    p_pos1 = np.copy(p_pos)
    e_pos1 = np.copy(e_pos)
    
    #keplerian parameters
    ecc = 0.088
    sma = 1.410090245 * u.s * const.c
    
    rx, ry = orbital_evolver(t, nu1, phase1, Omp, Pb, Vpx, Vpy, ip )
    
    rx -= rx[0]
    ry -= ry[0]
    
    vx = np.gradient( rx.to(u.km).value , t.to(u.s).value)
    vy = np.gradient( ry.to(u.km).value , t.to(u.s).value)
    
    scrn_vec = np.array([ scr_normal.x.value, scr_normal.y.value, scr_normal.z.value ]) 
    scrn_vec2 = np.array([ scr_normal2.x.value, scr_normal2.y.value, scr_normal2.z.value ]) 
    
    for j in range(len(t)):

        dt = t[1] - t[0]

        #use the current position vector
        pulsar_pos_tmp = CartesianRepresentation( np.array([rx[j].value, ry[j].value, 0.])* u.au )
        earth_pos_tmp = CartesianRepresentation( e_pos1 )

        #screen vector, positions and magnifications
        amp_val = screen_pos.value #+ ( np.random.rand(len(scr1_pos)) - 0.5 )
        scr_magnification =  np.exp(-amp_val**2 ) * np.exp(1j * amp_val)
        scr_magnification /= scr_magnification[len(amp_val) // 2] 
        
        #screen vector, positions and magnifications
        amp_val2 = screen_pos2.value #+ ( np.random.rand(len(scr1_pos)) - 0.5 )
        scr_magnification2 =  np.exp(-amp_val2**2 ) * np.exp(1j * amp_val2)
        scr_magnification2 /= scr_magnification2[len(amp_val2) // 2] 

        #iterating the dynamic spectrum
        ds_tmp = evolve_sys2(  
                            freq = f, 
                            d_p = d_p, 
                            d_s = d_s,
                            d_s2 = d_s2,
                            p_x = pulsar_pos_tmp,  
                            e_x = earth_pos_tmp,  
                            s_x = screen_pos1, 
                            s_x2 = screen_pos2,
                            s_n = scr_normal,
                            s_n2 = scr_normal2,
                            s_mu = scr_magnification,
                            s_mu2 = scr_magnification2,
                            spower = spower
                            )
        #appending relevant quantities
        dyns += [ds_tmp[0]]
        
        
        
        #update the position vectors
#         p_pos1 += ( pul_v  * dt ).to(u.au)
        e_pos1 += ( ear_v  * dt ).to(u.au)
        screen_pos1 += (scr_v * dt).to(u.au)
        screen_pos2 += (scr_v2 * dt).to(u.au)
        
        pv_tmp = np.array([ vx[j], vy[j], 0.]) 
        r_tmp = np.array([ rx[j].to(u.Mm).value, ry[j].to(u.Mm).value, 0. ])
        
        vp_proj = np.dot(pv_tmp, scrn_vec ) * u.km / u.s
        rp_proj = np.dot(r_tmp, scrn_vec ) * u.Mm
        ve_proj = np.dot(ear_v.value, scrn_vec ) * ear_v.unit
        
        vp_proj2 = np.dot(pv_tmp, scrn_vec2 ) * u.km / u.s
        rp_proj2 = np.dot(r_tmp, scrn_vec2 ) * u.Mm
        ve_proj2 = np.dot(ear_v.value, scrn_vec2 ) * ear_v.unit
        
        s = 1 - d_s/d_p
        s2 = 1. - d_s2/d_p
        
        vsp += [ ( scr_v / s - (1-s) / s *  vp_proj - ve_proj).to(u.km / u.s).value ]
        rsp += [ ( (scr_v / s - ve_proj) * t[j]  - (1-s) / s *  rp_proj  ).to(u.Mm).value]
        eta += [( d_p * d_s / (d_p - d_s) * const.c / (2 * np.mean(f)**2 * (vsp[-1] * u.km / u.s)**2 ) ).to(u.s**3).value]
        
        vsp2 += [ ( scr_v2 / s2 - (1-s2) / s2 *  vp_proj2 - ve_proj2).to(u.km / u.s).value ]
        rsp2 += [ ( (scr_v2 / s2 - ve_proj2) * t[j]  - (1-s2) / s2 *  rp_proj2  ).to(u.Mm).value]


    dyns = np.array(dyns).T
    dyns = dyns.reshape(len(f),j+1)
    
#     print( np.array( np.abs( ds_tmp[2]) ) )
#     print( [np.abs(scr_magnification), np.abs( scr_magnification2 )] )
    
    fv = (1 - ecc**2) / (1 + ecc * np.cos(nu1))
    
    #params for screen 1--------------------------------------------------------------------------
    pv_tmp = np.array([ Vpx.value, Vpy.value, 0.]) 
    vp_proj = np.dot(pv_tmp, scrn_vec ) * u.km / u.s
    xi_ang = (np.arctan2( scrn_vec[1],  scrn_vec[0] )*u.rad).to(u.deg) - 90 * u.deg
    delt_Om = xi_ang + Omp
    delta = np.arctan( -np.tan(delt_Om) * np.cos(ip) )
    
    A1 = - (1-s) / s *  vp_proj - ve_proj + scr_v / s
    B1 = - (1-s) / s * sma * fv * np.cos(delt_Om)
    C1 = - (1-s) / s * sma * fv * np.cos(ip) * np.sin(delt_Om) 
    
    A2 = Pb / (sma * np.sqrt( np.cos(delt_Om)**2 + np.sin(delt_Om)**2 * np.cos(ip)**2))
    A = A2 * A1 / (1 - s) * s
    
    B = (1-s) / s * sma * np.sqrt( np.cos(delt_Om)**2 + (np.cos(ip) * np.sin(delt_Om) )**2 )
    B = np.sqrt( ((1-s) / s * sma  * np.cos(delt_Om))**2 
               + ((1-s) / s * sma  * np.cos(ip) * np.sin(delt_Om) )**2)
    
    phys_params = [ A1, B1, C1, A.to(u.m/u.m), delta, B]

    #params for screen 2--------------------------------------------------------------------------
    screen2_params = [np.array(rsp2) * u.Mm, np.array(vsp2) * u.km / u.s]
    
    vp_proj2 = np.dot(pv_tmp, scrn_vec2 ) * u.km / u.s
    xi_ang2 = (np.arctan2( scrn_vec2[1],  scrn_vec2[0] )*u.rad).to(u.deg) - 90 * u.deg
    delt_Om2 = xi_ang2 + Omp
    delta2 = np.arctan( -np.tan(delt_Om2) * np.cos(ip) )
    
    A12 = - (1-s2) / s2 *  vp_proj2 - ve_proj2 + scr_v2 / s2
    B12 = - (1-s2) / s2 * sma * fv * np.cos(delt_Om2)
    C12 = - (1-s2) / s2 * sma * fv * np.cos(ip) * np.sin(delt_Om2) 
    
    A22 = Pb / (sma * np.sqrt( np.cos(delt_Om2)**2 + np.sin(delt_Om2)**2 * np.cos(ip)**2))
    A_2 = A22 * A12 / (1 - s2) * s2
    
#     B2 = (1-s2) / s2 * sma * np.sqrt( np.cos(delt_Om2)**2 + (np.cos(ip) * np.sin(delt_Om2) )**2 )
    B_2 = np.sqrt( ((1-s2) / s2 * sma  * np.cos(delt_Om2))**2 
               + ((1-s2) / s2 * sma  * np.cos(ip) * np.sin(delt_Om2) )**2)
    
    phys_params2 = [ A12, B12, C12, A_2.to(u.m/u.m), delta2, B_2]
    
    return dyns, np.array(vsp) * u.km / u.s, np.array(rsp) * u.Mm, np.array(eta) * u.s**3, phys_params, screen2_params, phys_params2




def geometry_evolve_orb2(t, f, nu1, phase1, d_p, d_s, ear_v, p_pos, e_pos, 
                        scr_v, scr_normal, 
                        d_s2, scr_v2, scr_normal2,
                        Pb, Omp, ip, Vpx, Vpy):
    #storage arrays
    dyns = []
    vsp = []
    rsp = []
    eta = []
    
    vsp2 = []
    rsp2 = []
    
    p_pos1 = np.copy(p_pos)
    e_pos1 = np.copy(e_pos)
    
    #keplerian parameters
    ecc = 0.088
    sma = 1.410090245 * u.s * const.c
    
    rx, ry = orbital_evolver(t, nu1, phase1, Omp, Pb, Vpx, Vpy, ip )
    
    rx -= rx[0]
    ry -= ry[0]
    
    vx = np.gradient( rx.to(u.km).value , t.to(u.s).value)
    vy = np.gradient( ry.to(u.km).value , t.to(u.s).value)
    
    scrn_vec = np.array([ scr_normal.x.value, scr_normal.y.value, scr_normal.z.value ]) 
    scrn_vec2 = np.array([ scr_normal2.x.value, scr_normal2.y.value, scr_normal2.z.value ]) 
    
    for j in range(len(t)):

        dt = t[1] - t[0]

        #use the current position vector
        pulsar_pos_tmp = CartesianRepresentation( np.array([rx[j].value, ry[j].value, 0.])* u.au )
        earth_pos_tmp = CartesianRepresentation( e_pos1 )        
        
        
        #update the position vectors
#         p_pos1 += ( pul_v  * dt ).to(u.au)
        e_pos1 += ( ear_v  * dt ).to(u.au)
        
        pv_tmp = np.array([ vx[j], vy[j], 0.]) 
        r_tmp = np.array([ rx[j].to(u.Mm).value, ry[j].to(u.Mm).value, 0. ])
        
        vp_proj = np.dot(pv_tmp, scrn_vec ) * u.km / u.s
        rp_proj = np.dot(r_tmp, scrn_vec ) * u.Mm
        ve_proj = np.dot(ear_v.value, scrn_vec ) * ear_v.unit
        
        vp_proj2 = np.dot(pv_tmp, scrn_vec2 ) * u.km / u.s
        rp_proj2 = np.dot(r_tmp, scrn_vec2 ) * u.Mm
        ve_proj2 = np.dot(ear_v.value, scrn_vec2 ) * ear_v.unit
        
        s = 1 - d_s/d_p
        s2 = 1. - d_s2/d_p
        
        vsp += [ ( scr_v / s - (1-s) / s *  vp_proj - ve_proj).to(u.km / u.s).value ]
        rsp += [ ( (scr_v / s - ve_proj) * t[j]  - (1-s) / s *  rp_proj  ).to(u.Mm).value]
        eta += [( d_p * d_s / (d_p - d_s) * const.c / (2 * np.mean(f)**2 * (vsp[-1] * u.km / u.s)**2 ) ).to(u.s**3).value]
        
        vsp2 += [ ( scr_v2 / s2 - (1-s2) / s2 *  vp_proj2 - ve_proj2).to(u.km / u.s).value ]
        rsp2 += [ ( (scr_v2 / s2 - ve_proj2) * t[j]  - (1-s2) / s2 *  rp_proj2  ).to(u.Mm).value]


    
    fv = (1 - ecc**2) / (1 + ecc * np.cos(nu1))
    
    #params for screen 1--------------------------------------------------------------------------
    pv_tmp = np.array([ Vpx.value, Vpy.value, 0.]) 
    vp_proj = np.dot(pv_tmp, scrn_vec ) * u.km / u.s
    xi_ang = (np.arctan2( scrn_vec[1],  scrn_vec[0] )*u.rad).to(u.deg) - 90 * u.deg
    delt_Om = xi_ang + Omp
    delta = np.arctan( -np.tan(delt_Om) * np.cos(ip) )
    
    A1 = - (1-s) / s *  vp_proj - ve_proj + scr_v / s
    B1 = - (1-s) / s * sma * fv * np.cos(delt_Om)
    C1 = - (1-s) / s * sma * fv * np.cos(ip) * np.sin(delt_Om) 
    
    A2 = Pb / (sma * np.sqrt( np.cos(delt_Om)**2 + np.sin(delt_Om)**2 * np.cos(ip)**2))
    A = A2 * A1 / (1 - s) * s
    
    B = (1-s) / s * sma * np.sqrt( np.cos(delt_Om)**2 + (np.cos(ip) * np.sin(delt_Om) )**2 )
    B = np.sqrt( ((1-s) / s * sma  * np.cos(delt_Om))**2 
               + ((1-s) / s * sma  * np.cos(ip) * np.sin(delt_Om) )**2)
    
    phys_params = [ A1, B1, C1, A.to(u.m/u.m), delta, B]

    #params for screen 2--------------------------------------------------------------------------
    screen2_params = [np.array(rsp2) * u.Mm, np.array(vsp2) * u.km / u.s]
    
    vp_proj2 = np.dot(pv_tmp, scrn_vec2 ) * u.km / u.s
    xi_ang2 = (np.arctan2( scrn_vec2[1],  scrn_vec2[0] )*u.rad).to(u.deg) - 90 * u.deg
    delt_Om2 = xi_ang2 + Omp
    delta2 = np.arctan( -np.tan(delt_Om2) * np.cos(ip) )
    
    A12 = - (1-s2) / s2 *  vp_proj2 - ve_proj2 + scr_v2 / s2
    B12 = - (1-s2) / s2 * sma * fv * np.cos(delt_Om2)
    C12 = - (1-s2) / s2 * sma * fv * np.cos(ip) * np.sin(delt_Om2) 
    
    A22 = Pb / (sma * np.sqrt( np.cos(delt_Om2)**2 + np.sin(delt_Om2)**2 * np.cos(ip)**2))
    A_2 = A22 * A12 / (1 - s2) * s2
    
#     B2 = (1-s2) / s2 * sma * np.sqrt( np.cos(delt_Om2)**2 + (np.cos(ip) * np.sin(delt_Om2) )**2 )
    B_2 = np.sqrt( ((1-s2) / s2 * sma  * np.cos(delt_Om2))**2 
               + ((1-s2) / s2 * sma  * np.cos(ip) * np.sin(delt_Om2) )**2)
    
    phys_params2 = [ A12, B12, C12, A_2.to(u.m/u.m), delta2, B_2]
    
    
    return np.array(vsp) * u.km / u.s, np.array(rsp) * u.Mm, np.array(eta) * u.s**3, phys_params, screen2_params, phys_params2


def curvature_estimate( f, d_p, d_s, d_s2, v1, v2, t, nu, phase, A, delta, plots = True ):

    eta_scr1_base = (d_p * d_s / (d_p - d_s) * const.c / 2 / np.mean(f)**2 / v1**2 ).to(u.s**3)
    eta_scr2_base = (d_p * d_s2 / (d_p - d_s2) * const.c / 2 / np.mean(f)**2 / v2**2 ).to(u.s**3)
    
    
    ife = generate_n_minus_1_x_2_array( peaks(Ad_projection_unitless(t = t.to(u.hour).value,
                                             nu = nu,
                                           phase = phase.value.astype(np.float64), 
                                           A = A , 
                                           delta = delta ))[-1] )

    res_etas_1 = []
    res_etas_2 = []

    if plots == True: 
        plt.figure(figsize = (10,5))
        
        
    for i in range(len(ife)):

        e1 = ife[i][0]
        e2 = ife[i][-1]
        
        #computing the expected curvatures for screen 1 and screen2
        res_etas_1 += [np.median( eta_scr1_base[e1:e2] * np.gradient( Ad_projection_unitless(t = t.to(u.hour).value,
                                             nu = nu,
                                           phase = phase.value.astype(np.float64), 
                                           A = A , 
                                           delta = delta ) * 100,
                t.to(u.s).value)[e1:e2]**2 )]

        res_etas_2 += [np.median(eta_scr2_base[e1:e2] * np.gradient( Ad_projection_unitless(t = t.to(u.hour).value,
                                             nu = nu,
                                           phase = phase.value.astype(np.float64), 
                                           A = A , 
                                           delta = delta ) * 100,
                t.to(u.s).value)[e1:e2]**2)]
        
        
        #if true, plot the resmapled expected curvature for screen 1 (flat) and the resulting curvature from
        #from screen 2 after resampling with screen 1 parameters
        if plots == True : 
            plt.subplot(1,len(ife),i+1)
            if i == 0:
                plt.xlabel('Time (hour)', fontsize = 16)
                plt.ylabel('Curvature (Mm$^2$s)', fontsize = 16)

            plt.plot(t.to(u.hour)[e1:e2],  eta_scr1_base[e1:e2] * np.gradient( Ad_projection_unitless(t = t.to(u.hour).value,
                                                 nu = nu,
                                               phase = phase.value.astype(np.float64), 
                                               A = A , 
                                               delta = delta ) * 100,
                    t.to(u.s).value)[e1:e2]**2  )

            plt.plot(t.to(u.hour)[e1:e2],  eta_scr2_base[e1:e2] * np.gradient( Ad_projection_unitless(t = t.to(u.hour).value,
                                                 nu = nu,
                                               phase = phase.value.astype(np.float64), 
                                               A = A , 
                                               delta = delta ) * 100,
                    t.to(u.s).value)[e1:e2]**2  )

            plt.ylim( res_etas_1[-1].value * 1e-5, res_etas_1[-1].value * 1e6)
            plt.yscale('log')
            if i > 0:
                plt.yticks([])


    return res_etas_1, res_etas_2