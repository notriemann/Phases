#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.sparse.linalg import eigsh
from scipy.optimize import curve_fit
import astropy.constants as const
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from scipy.signal import convolve2d
from scipy.stats import norm

# from scintools2.scintools.ththmod import fft_axis, ext_find
from scintools.ththmod import fft_axis, ext_find

from astropy import units as u
from astropy import constants as const
from astropy.coordinates import (
    CartesianRepresentation, CylindricalRepresentation,
    UnitSphericalRepresentation)
from astropy.visualization import quantity_support


import sys
sys.path.append('..')  # Add parent directory to the system path

import screens
from screens.fields import dynamic_field
from screens.dynspec import DynamicSpectrum as DS
from screens.conjspec import ConjugateSpectrum as CS
from screens.screen import Source, Screen1D, Telescope
from screens.fields import phasor

def axis_extent(*args):
    result = []
    for a in args:
        x = a.squeeze().value
        result.extend([x[0] - (dx:=x[1]-x[0])/2, x[-1]+dx/2])
    return result


def orbital_dp_evolver(t, nu1, phase1, Omp, Pb, Vpx, Vpy, ip ):
    
    """
    Function to calculate the orbital motion + integrated proper motion (times distance) as the total motion of the double pulsar
    """
    
    ecc = 0.088
    sma = 1.410090245 * u.s * const.c
    
    fv = (1 - ecc**2) / (1 + ecc * np.cos(nu1))
    
    rx = ( sma * fv * ( np.cos(phase1) * np.sin(Omp) - np.cos(Omp) * np.sin(phase1) * np.cos(ip) ) + Vpx * t ).to(u.Mm)
    ry = ( sma * fv * ( np.sin(phase1) * np.sin(Omp) * np.cos(ip) + np.cos(Omp) * np.cos(phase1)  ) + Vpy * t ).to(u.Mm)
    
    rx -= rx[0]
    ry -= ry[0]
    
    return rx.to(u.au), ry.to(u.au)

    
def evolve_sys_wave( freq, d_p, d_s, d_s2, p_x, e_x, s_x, s_x2, s_n, s_n2, s_mu, s_mu2, spower, 
                    interaction = False, int_threshold = 0.1, mag_plot = False):
    
    
    """
    Function to calculate the dynamic wavefield of a pulsar binary. 
    This function takes into account the interactions of having two screens interfering with each other. As in 
    (source -> screen2 -> obs) + (source -> screen1 -> obs) + (source -> screen2 -> screen1 -> observer )
    
    Inputs:
    freq (array-like): frequency array with astropy units
    d_p (float): distance to the pulsar with astropy units
    d_s (float): distance to the first screen with astropy units
    d_s2 (float): distance to the second screen with astropy units
        NOTE: screen2 must be further away than screen1
    p_x (cartesian astropy coords): position of the pulsar
    e_x (cartesian astropy coords): position of the earth
    s_x (array-like): position of the images from screen 1 on the line of images with astropy units
    s_x2 (cartesian astropy coords): position of the images from screen 2 on the line of images with astropy units
    s_n (screen 1 normal vector in cylindrical coords): orientation of the line of images from screen 1
    s_n2 (screen 1 normal vector in cylindrical coords): orientation of the line of images from screen 2
    s_mu (complex array-like): magnifications of the images of screen 1
    s_mu2 (complex array-like): magnifications of the images of screen 2
    spower (array-like): array containing multipliers for the total power of each screen as in
        [power from line of sight, power from screen1, power from screen2, power from interactions]
        in general, it should take the form [1, pwr1, pwr2, 1] as the los and interactions should be independent
    interaction (boolean): this is to set interactions as source -> screen2 -> screen1 -> observer 
    set as false if one treats them as independet from (source -> screen2 -> obs) + (source -> screen1 -> obs)
    int_threshold (float): number that sets a filter on all interaction images that have a lower brightness than max|mu_1| * max|mu_2| * int_threshold
    mag_plot (boolean): True if one wants to plot the images in |magnifications| as a function of delay 
    
    
    Returns:
    dynamic wavefield (array-like):  (dynwave.sum(axis=0)).T
    geometric delay of each image (array-like):  tau0
    brightness of each image(array-like): brightness
    
    """
    
    
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
        #source -> screen1 -> obs

        obs_scr1_pulsar = scr.observe(source=pulsar, distance=d_p-d_s)
        obs = telescopes.observe(source=obs_scr1_pulsar, distance=d_s)
        
        #source -> screen2 -> obs
        obs_scr2_pulsar2 = scr2.observe(source=pulsar, distance=d_p-d_s2)
        obs2 = telescopes.observe(source=obs_scr2_pulsar2, distance=d_s2)

        #setting observation with screen2
        #source -> screen2 -> screen1 -> obs
        obs_scr2_pulsar = scr2.observe(source=pulsar, distance=d_p-d_s2)
        obs_scr2_pulsar2 = scr.observe(source=obs_scr2_pulsar, distance=d_s2-d_s)
        obs3 = telescopes.observe(source=obs_scr2_pulsar2, distance=d_s)
        
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
    
    if interaction == True: 


        
        #setting up the total power as the incoherent sum of the magnifications 
        #P = sum( |magnification of each image|)
        ptot = (  np.sum( np.abs(los.brightness.ravel()) ) 
                + np.sum( np.abs(obs.brightness.ravel()) ) 
                + np.sum( np.abs(obs2.brightness.ravel()) )
                + np.sum( np.abs(obs3.brightness.ravel()) ) )
        

        #normalization factors such that the power that screen1 and screen2 get correspond to the indices of spower
        p0_norm = ptot * spower[0] / np.sum( np.abs(los.brightness.ravel()) ) 
        p1_norm = ptot * spower[1] / np.sum( np.abs(obs.brightness.ravel()) ) 
        p2_norm = ptot * spower[2] / np.sum( np.abs(obs2.brightness.ravel()) ) 
        p3_norm = ptot * spower[3] / np.sum( np.abs(obs3.brightness.ravel()) ) 
        
        
        p0_norm = 1.
        p1_norm = spower[1] / np.max( np.abs(obs.brightness.ravel()) ) 
        p2_norm = spower[2] / np.max( np.abs(obs2.brightness.ravel()) ) 
        p3_norm = spower[1] * spower[2] * spower[3] / np.max( np.abs(obs3.brightness.ravel()) ) 
        
        
        
#         plt.plot( los.tau.ravel(), np.abs(los.brightness.ravel()), 'o', label = 'LOS')
#         plt.plot( obs.tau.ravel(), np.abs(obs.brightness.ravel()), 'o', label = 'scr1')
#         plt.plot( obs2.tau.ravel(), np.abs(obs2.brightness.ravel()), 'o', label = 'scr2')
#         plt.plot( obs3.tau.ravel(), np.abs(obs3.brightness.ravel()), 'o', label = 'scr interaction')
        
#         plt.xlabel('Delay ' + str( (los.tau.ravel()).unit) )
#         plt.ylabel('Magnifications ')
#         plt.yscale('log')
#         plt.legend()
#         plt.title('Original')
#         plt.show()
        
        if mag_plot == True:
            
            print("Wave info")
            print("--------------------------------")
            print("pwr los before: ", np.sum( np.abs(los.brightness.ravel())) )
            print("pwr scr1 before: ", np.sum( np.abs(obs.brightness.ravel()) )  )
            print("pwr scr2 before: ", np.sum( np.abs(obs2.brightness.ravel()) )  )  
            print("pwr scr3 before: ", np.sum( np.abs(obs3.brightness.ravel()) )  ) 
            print("Total power: ", ptot)
            print("Max los: ", np.max( np.abs( los.brightness.ravel() ) ) )
            print("Max sc1: ", np.max(np.abs(obs.brightness.ravel() ) ) )
            print("Max sc2: ", np.max(np.abs(obs2.brightness.ravel() ) ) )
            print("Max sc3: ", np.max(np.abs(obs3.brightness.ravel() ) ) )
            print("Predicted Max sc3: ", np.max(np.abs(obs.brightness.ravel() ) ) * np.max(np.abs(obs2.brightness.ravel() ) ))
            
            plt.plot( los.tau.ravel(), np.abs(los.brightness.ravel() * p0_norm ), 'o', label = 'LOS')
            plt.plot( obs.tau.ravel(), np.abs(obs.brightness.ravel() * p1_norm ), 'o', label = 'scr1', markersize = 3.5)
            plt.plot( obs2.tau.ravel(), np.abs(obs2.brightness.ravel() * p2_norm), 'o', label = 'scr2', markersize = 3.5)
            plt.plot( obs3.tau.ravel(), np.abs(obs3.brightness.ravel() * p3_norm), '+', label = 'scr interaction', markersize = 3)
            
            plt.xlabel('Delay ' + str( (los.tau.ravel()).unit) )
            plt.ylabel('Magnifications ')
            plt.yscale('log')
            plt.legend()
            plt.title('Renormalized')
            plt.axhline( y = int_threshold, ls = '--', c = 'r')
            plt.show()
            
            plt.plot( los.tau.ravel(), np.abs(los.brightness.ravel()  ), 'o', label = 'LOS')
            plt.plot( obs.tau.ravel(), np.abs(obs.brightness.ravel()  ), 'o', label = 'scr1', markersize = 3.5)
            plt.plot( obs2.tau.ravel(), np.abs(obs2.brightness.ravel() ), 'o', label = 'scr2', markersize = 3.5)
            plt.plot( obs3.tau.ravel(), np.abs(obs3.brightness.ravel() ), '+', label = 'scr interaction', markersize = 3)
            
            plt.xlabel('Delay ' + str( (los.tau.ravel()).unit) )
            plt.ylabel('Magnifications ')
            plt.yscale('log')
            plt.legend()
            plt.title('Original ')
            plt.axhline( y = int_threshold, ls = '--', c = 'r')
            plt.show()
            
            print("--------------------------------")
            print("pwr los after: ", np.sum( np.abs(np.abs(los.brightness.ravel() * p0_norm )) ) )
            print("pwr scr1 after: ", np.sum( np.abs(obs.brightness.ravel() * p1_norm ) )  )
            print("pwr scr2 after: ", np.sum( np.abs(obs2.brightness.ravel() * p2_norm) )  )  
            print("pwr scr3 after: ", np.sum( np.abs(obs3.brightness.ravel() * p3_norm) )  ) 
            print("Max los: ", np.max(np.abs( los.brightness.ravel() * p0_norm ) ) )
            print("Max sc1: ", np.max(np.abs( obs.brightness.ravel() * p1_norm ) ) )
            print("Max sc2: ", np.max(np.abs( obs2.brightness.ravel() * p2_norm ) ) )
            print("Max sc3: ", np.max(np.abs( obs3.brightness.ravel() * p3_norm ) ) )
            print("Predicted Max sc3: ", np.max(np.abs(obs.brightness.ravel() * p1_norm ) ) * np.max( np.abs(obs2.brightness.ravel() * p2_norm ) ))
              
#         print("pwr los after: ", np.sum( np.abs(los.brightness.ravel() * p0_norm  ) ) )
#         print("pwr scr1 after: ", np.sum( np.abs(obs.brightness.ravel() * p1_norm ) ) )
#         print("pwr scr2 after: ", np.sum( np.abs(obs2.brightness.ravel() * p2_norm) ) )  
#         print("pwr scr3 after: ", np.sum( np.abs(obs3.brightness.ravel() * p3_norm) ) )
              
              
#         print("max los after: ", np.max( np.abs(los.brightness.ravel() * p0_norm  ) ) )
#         print("max scr1 after: ", np.max( np.abs(obs.brightness.ravel() * p1_norm ) ) )
#         print("max scr2 after: ", np.max( np.abs(obs2.brightness.ravel() * p2_norm) ) )  
#         print("max scr3 after: ", np.max( np.abs(obs3.brightness.ravel() * p3_norm) ) )
                
        
#             plt.xlabel('Delay ' + str( (los.tau.ravel()).unit) )
#             plt.ylabel('Magnifications ')
#             plt.yscale('log')
#             plt.legend()
#             plt.title('Renormalized')
#             plt.axhline( y = int_threshold, ls = '--', c = 'r')
#             plt.show()
        
#         plt.plot( [0.], np.abs(los.brightness.ravel() * p0_norm ), 'bo')
#         plt.plot( s_x, np.abs(obs.brightness.ravel() * p1_norm ), 'o')
#         plt.plot( s_x2, np.abs(obs2.brightness.ravel() * p2_norm ), 'o')
#         plt.plot( [0.], np.max(np.abs(obs3.brightness.ravel() * p3_norm)), '+')
#         plt.ylabel('Magnifications ')
#         plt.yscale('log')
        
#         plt.axhline( y = int_threshold, ls = '--', c = 'r')
#         plt.show()
        
        boolean_mag = np.where( np.abs(obs3.brightness.ravel() * p3_norm) > int_threshold )[0]
        
        if mag_plot == True: 
            print("LOS pwr: ", np.abs(los.brightness.ravel() * p0_norm ))
            print( "Interaction number of images: ", np.array(obs3.brightness.ravel()).shape)
            print( "Interaction number of images above threshold:  ", np.array(obs3.brightness.ravel())[boolean_mag].shape)
        
        #getting delays 
        tau0 = np.hstack([los.tau.ravel(), 
                          obs.tau.ravel(),
                          obs2.tau.ravel(),
                          obs3.tau.ravel()[boolean_mag]  ])

        
        #calculating dynamic spectrum
        ph = phasor(freq, tau0[:, np.newaxis, np.newaxis], linear_axis = 0)
        
        brightness = np.hstack([ los.brightness.ravel() * p0_norm,
                                 obs.brightness.ravel() * p1_norm,
                                 obs2.brightness.ravel() * p2_norm,
                                 obs3.brightness.ravel()[boolean_mag] * p3_norm
                               ])
        
    elif interaction == False :
        

        #getting delays 
        tau0 = np.hstack([los.tau.ravel(), 
                          obs.tau.ravel(),
                          obs2.tau.ravel()])

        #calculating dynamic spectrum
        ph = phasor(freq, tau0[:, np.newaxis, np.newaxis], linear_axis = 0 )
        
        #setting up the total power as the incoherent sum of the magnifications 
        #P = sum( |magnification of each image|)
        ptot = (  np.sum( np.abs(los.brightness.ravel()) ) 
                + np.sum( np.abs(obs.brightness.ravel()) ) 
                + np.sum( np.abs(obs2.brightness.ravel()) ) )
        
#         print( "LOS elements")
#         print(los.brightness.ravel())
#         print( np.sum( np.abs(los.brightness.ravel())) )
#         print( np.sum( np.abs(obs.brightness.ravel())) )
#         print( np.sum( np.abs(obs2.brightness.ravel())) )
#         print("-----------")


        #normalization factors such that the power that screen1 and screen2 get correspond to the indices of spower
        p1_norm = ptot * spower[0] / np.sum( np.abs(los.brightness.ravel()) ) 
        p2_norm = ptot * spower[1] / np.sum( np.abs(obs.brightness.ravel()) ) 
        p3_norm = ptot * spower[2] / np.sum( np.abs(obs2.brightness.ravel()) ) 

        brightness = np.hstack([ los.brightness.ravel() * p1_norm,
                                 obs.brightness.ravel() * p2_norm,
                                 obs2.brightness.ravel() * p3_norm
                               ])
    


    dynwave = ph * brightness[:, np.newaxis, np.newaxis]


    return (dynwave.sum(axis=0)).T, tau0, brightness
    
    
def full_screens_simulator(t, f, nu1, phase1, d_p, d_s, ear_v, p_pos, e_pos, 
                        scr_v, screen_pos, scr_normal, sig1, 
                        d_s2, scr_v2, screen2_pos2, scr_normal2, sig2, 
                        Pb, Omp, ip, Vpx, Vpy, spower = [0.1, 0.3, 0.3, 0.3], 
                        ds_computation = True, mag_plots = True, custom_mag_ind_1 = False, custom_mag_ind_2 = False, 
                        interaction = False, int_threshold = 0.1 , test_case = False, rand1 = True, rand2 = True):
    """
    Function to calculate the dynamic wavefield of a pulsar binary. 
    This function takes into account the interactions of having two screens interfering with each other. As in 
    (source -> screen2 -> obs) + (source -> screen1 -> obs) + (source -> screen2 -> screen1 -> observer )
    
    t (array-like): time array with astropy units
    f (array-like): frequency array with astropy units
    nu1 (array-like): array of values of the true anomaly 
    phase1 (array-like): array of values of true anomaly + longitude of periastron
    d_p (float): distance to the pulsar with astropy units
    d_s (float): distance to the first screen with astropy units
    ear_v (float): earth velocities in ra,dec 
    p_pos (cartesian astropy coords): position of the pulsar
    e_pos (cartesian astropy coords): position of the earth
    scr_v (float): speed of the screen 1
    screen_pos (array-like): position of the images from screen 1 on the line of images with astropy units
    scr_normal (screen 1 normal vector in cylindrical coords): orientation of the line of images from screen 1
    sig1 (float): tunable gaussian variance for the image magnifications of scr1
    Norm1 (float): tunable amplitude for the image magnifications of scr1
    d_s2 (float): distance to the second screen with astropy units
        NOTE: screen2 must be further away than screen1
    scr_v2 (float): speed of the screen 2
    screen_pos2 (array-like): position of the images from screen 2 on the line of images with astropy units
    scr_normal2 (screen 1 normal vector in cylindrical coords): orientation of the line of images from screen 2
    sig2 (float): tunable gaussian variance for the image magnifications of scr2
    Norm2 (float): tunable amplitude for the image magnifications of scr2
    Pb (float): orbital period in astropy units
    Omp (float): longitude of ascending node in astropy units
    ip (float): orbital inclination in astropy units
    Vpc, Vpy: pulsar velocity in the sky (proper motion * distance)
    spower (array-like): array containing multipliers for the total power of each screen as in
        [power from line of sight, power from screen1, power from screen2, power from interactions]
        in general, it should take the form [1, pwr1, pwr2, 1] as the los and interactions should be independent
    ds_computation (boolean): if one wishes to compute the yn wavefield or return an array of ones (only for getting geometry params)
    mag_plots (boolean): True if one wants to plot the images in |magnifications| as a function of delay 
    custom_mag_ind_1 and custom_mag_ind_2: custom indeces and multipliers for the magnifications of the images in screen_pos and screen+pos2
    interaction (boolean): this is to set interactions as source -> screen2 -> screen1 -> observer 
    set as false if one treats them as independet from (source -> screen2 -> obs) + (source -> screen1 -> obs)
    rand1, and rand2 (array-like): random phase to be assigned to each image. If no array is given then some random complex numbers will be assigned
    
    Returns
    
    dyns (Array-like): dynamic wavefield
    np.array(vsp) * u.km / u.s (array-like): effective velocity from scr1 
    np.array(rsp) * u.Mm (array-like): effective motion from scr1
    np.array(eta) * u.s**3 (array-like): curvature as a function of time 
    phys_params (array like): physical params from scr1 
    screen2_params (list): [rsp, vsp] for scr2 wrapped in a list
    phys_params2 (array-like): physical params from scr2
    
    """
    
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
    
    rx, ry = orbital_dp_evolver(t, nu1, phase1, Omp, Pb, Vpx, Vpy, ip )
    
    rx -= rx[0]
    ry -= ry[0]
    
    
    #getting pulsar orbital velocities + proper motion
    vx = np.gradient( rx.to(u.km).value , t.to(u.s).value)
    vy = np.gradient( ry.to(u.km).value , t.to(u.s).value)
    
    #getting screen vector in xyz
    scrn_vec = np.array([ scr_normal.x.value, scr_normal.y.value, scr_normal.z.value ]) 
    scrn_vec2 = np.array([ scr_normal2.x.value, scr_normal2.y.value, scr_normal2.z.value ]) 
    
    
    
    #screen1 vector, positions and magnifications
    amp_val = screen_pos.value #+ ( np.random.rand(len(scr1_pos)) - 0.5 )
    #screen2 vector, positions and magnifications
    amp_val2 = screen_pos2.value #+ ( np.random.rand(len(scr1_pos)) - 0.5 )
    
    #if no phase was given compute a phase for the line of images
    if rand1 == True: 
        rand1 = np.exp( 2j*np.pi*np.random.uniform(size=amp_val.shape) )
        
    elif rand1 == False:
        rand1 = np.exp( 2j*np.pi* amp_val )
        
    if rand2 == True: 
        rand2 = np.exp( 2j*np.pi*np.random.uniform(size=amp_val2.shape) )
        
    elif rand2 == False: 
        rand2 = np.exp( 2j*np.pi* amp_val2 )
    

    #line of images magnidication dependence and a random phase
    scr_magnification =  np.exp(-(amp_val / sig1 )**2 ) * rand1 
    #normalize
    scr_magnification /= np.sqrt( (np.abs(scr_magnification)**2).sum() ) 
    #custom indices with custom multiplier
    if isinstance(custom_mag_ind_1, bool) == False:
        scr_magnification[custom_mag_ind_1[1]] *= custom_mag_ind_1[0] 
    

    #line of images magnidication dependence and a random phase
    scr_magnification2 =  np.exp(-(amp_val2 / sig2 )**2 ) * rand2 
    #normalize
    scr_magnification2 /= np.sqrt( (np.abs(scr_magnification2)**2).sum() ) 
    #custom indices with custom multiplier
    if isinstance(custom_mag_ind_2, bool) == False:
        scr_magnification2[custom_mag_ind_2[1]] *= custom_mag_ind_2[0] 

    
    #plot distribution of images with magnifications
    if mag_plots == True:
        
        los_ = np.array([1.])
        
        #setting up the total power as the incoherent sum of the magnifications 
        #P = sum( |magnification of each image|)
        ptot = (  los_  
                + np.sum( np.abs(scr_magnification) )  
                + np.sum( np.abs(scr_magnification2) ) )


        #normalization factors such that the power that screen1 and screen2 get correspond to the indices of spower
        p0_norm = ptot * spower[0] / np.sum( los_ ) 
        p1_norm = ptot * spower[1] / np.sum( np.abs(scr_magnification) ) 
        p2_norm = ptot * spower[2] / np.sum( np.abs(scr_magnification2) )
        
        
        
        plt.figure(figsize=(12., 3.))
        plt.plot( [0.], np.abs( los_ * p0_norm), 'bo', label = 'LOS' )
        

        plt.semilogy( amp_val, np.abs(scr_magnification * p1_norm)  , '+', label = 'Screen 1')
        plt.semilogy( amp_val2, np.abs(scr_magnification2 * p2_norm), '+', label = 'Screen 2')
        
        
#         print("Info inside loop")
#         print("-----------------------------------")
#         print("Num of im1: ", scr_magnification.shape[0])
#         print("Num of im2: ", scr_magnification2.shape[0])
#         print("Pwr of im1 before: ", np.sum( np.abs(scr_magnification) ))
#         print("Pwr of im2 before: ", np.sum( np.abs(scr_magnification2) ))
        
#         print('LOS final : ', np.abs( los_ * p0_norm)  )
#         print("Total power: ", ptot)
#         print('LOS after : ', np.abs( los_ * p0_norm)  )
#         print("Pwr of im1 after: ", np.sum( np.abs(scr_magnification * p1_norm)) )
#         print("Pwr of im2 after: ", np.sum( np.abs(scr_magnification2 * p2_norm)) )
        
#         print("Max of im1: ", np.max(np.abs(scr_magnification * p1_norm)))
#         print("Max of im2: ", np.max(np.abs(scr_magnification2 * p2_norm)))
#         print("Max of interactions: ", np.max(np.abs(scr_magnification * p1_norm)) * np.max( np.abs(scr_magnification2 * p2_norm) ) * spower[3]  )
        
        
        plt.plot( [0.], 
                 np.max(np.abs(scr_magnification * p1_norm)) * np.max( np.abs(scr_magnification2 * p2_norm) ) * spower[3] / ptot, '+', label = 'Interactions cap' )
        
        plt.xlabel(r"$\theta\ (AU)$")
        plt.legend()
        plt.ylabel("A")
        plt.show()
    
    for j in range(len(t)):

        dt = t[1] - t[0]
        
        print(f"Processing: {j + 1}/{len(t)}", end='\r', flush=True)
        

        #use the current position vector
        pulsar_pos_tmp = CartesianRepresentation( np.array([rx[j].value, ry[j].value, 0.])* u.au )
        earth_pos_tmp = CartesianRepresentation( e_pos1 )
        
        
        if ds_computation == True: 
            
            

            #iterating the dynamic spectrum
            ds_tmp = evolve_sys_wave(  
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
                                spower = spower,
                                interaction = interaction,
                                int_threshold = int_threshold,
                                mag_plot = test_case
                                )
            #appending relevant quantities
            dyns += [ds_tmp[0]]
        else: 
            
            dyns += [np.ones(len(f))]

        
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
        
        
        if test_case == True:
                
            if j == 0:
                break


    dyns = np.array(dyns).T
    dyns = dyns.reshape(len(f),j+1)
    

    
    fv = (1 - ecc**2) / (1 + ecc * np.cos(nu1))
    
    #params for screen 1--------------------------------------------------------------------------
    pv_tmp = np.array([ Vpx.value, Vpy.value, 0.]) 
    vp_proj = np.dot(pv_tmp, scrn_vec ) * u.km / u.s
    xi_ang = -((np.arctan2( scrn_vec[1],  scrn_vec[0] )*u.rad).to(u.deg) - 90 * u.deg)
    delt_Om = xi_ang - Omp
    delta = np.arctan( np.tan(delt_Om) * np.cos(ip) )
    
    A1 = - (1-s) / s *  vp_proj - ve_proj + scr_v / s
    B1 = - (1-s) / s * sma * fv * np.cos(delt_Om)
    C1 = (1-s) / s * sma * fv * np.cos(ip) * np.sin(delt_Om) 
    
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
    C12 = (1-s2) / s2 * sma * fv * np.cos(ip) * np.sin(delt_Om2) 
    
    A22 = Pb / (sma * np.sqrt( np.cos(delt_Om2)**2 + np.sin(delt_Om2)**2 * np.cos(ip)**2))
    A_2 = A22 * A12 / (1 - s2) * s2
    
#     B2 = (1-s2) / s2 * sma * np.sqrt( np.cos(delt_Om2)**2 + (np.cos(ip) * np.sin(delt_Om2) )**2 )
    B_2 = np.sqrt( ((1-s2) / s2 * sma  * np.cos(delt_Om2))**2 
               + ((1-s2) / s2 * sma  * np.cos(ip) * np.sin(delt_Om2) )**2)
    
    phys_params2 = [ A12, B12, C12, A_2.to(u.m/u.m), delta2, B_2]
    
    return dyns, np.array(vsp) * u.km / u.s, np.array(rsp) * u.Mm, np.array(eta) * u.s**3, phys_params, screen2_params, phys_params2

def needle(aaa, scale = 1.):
    """
    Function to either compress or enlarge the spacing of an array keeping the last endpoints of the array fixed. As in the first and last element of aaa, and depending on the scale factor the array will be more dense around the center of aaa or dense at the start/end
    """
    
    aa2 = np.copy(aaa)
    aa2[aa2>0] = ( aa2[ aa2>0 ] / aa2[-1])**scale * aa2[aa2>0]
    
    aa2[aa2<0] = ( aa2[ aa2<0 ] / aa2[0])**scale * aa2[aa2<0]
    
    return aa2


def model_returner( dp, ds, xi, omp, vism, inc, vpa, vpd):
    
    """
    Function to calculate the values of xi, A2 s / (1-s) and C from the overleaf given the scintillometry observables
    """
    
    #orb period
    Pbp = 2.45 * u.hour
    #semimajor axis
    semim = (1.410090245 * u.s * const.c).to(u.Mm)
    #s value from scintillation
    s_val = 1. - ds / dp
    
    #A_M value from overleaf (22)
    A_two = Pbp / (semim * np.sqrt( np.cos(xi - omp)**2 + np.cos(inc)**2 * np.sin(xi - omp)**2  ))
    #C value from overleaf (24)
    C_val = A_two * ( - (vpa * np.sin(xi) + vpd * np.cos(xi))  + 1. /(1 - s_val) * vism )
    
    

    return xi.to(u.rad).value, (A_two * s_val / (1. - s_val)).to(u.s / u.km).value, C_val.to(u.m / u.m).value


def model_returner_delta( dp, ds, xi, omp, vism, inc, vpa, vpd):
    
    """
    Function to calculate the values of xi, A2 s / (1-s) and C from the overleaf given the scintillometry observables
    """
    
    #orb period
    Pbp = 2.45 * u.hour
    #semimajor axis
    semim = (1.410090245 * u.s * const.c).to(u.Mm)
    #s value from scintillation
    s_val = 1. - ds / dp
    
    #A_M value from overleaf (22)
    A_two = Pbp / (semim * np.sqrt( np.cos(xi - omp)**2 + np.cos(inc)**2 * np.sin(xi - omp)**2  ))
    #C value from overleaf (24)
    C_val = A_two * ( - (vpa * np.sin(xi) + vpd * np.cos(xi))  + 1. /(1 - s_val) * vism )
    
    #computing the value of delta from (19)
    delt_Om = xi - omp
    delta = np.arctan( np.tan(delt_Om) * np.cos(inc) )

    return xi.to(u.rad).value, (A_two * s_val / (1. - s_val)).to(u.s / u.km).value, C_val.to(u.m / u.m).value, delta.value