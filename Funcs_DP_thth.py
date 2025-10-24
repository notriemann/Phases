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

from Funcs_DP import *

# import scintools.ththmod as thth
# from scintools.dynspec import BasicDyn, Dynspec
from scintools.ththmod import fft_axis, ext_find

def result_interpreter( ththresults, sample_datas):
    
    """
    Function to take in the list that followed from the parallelized
    computation of curvatures, and reshapes the curvatures into the same 
    shape as param_array which is an array containing the phases for 
    all A
    """
    # Calculate the dimensions of the sample data
    A_num = len(sample_datas)
    k_num = len(sample_datas[0][0])
    N_num = len(sample_datas[0][0][0])
    
     # Calculate the total number of curvatures to group over
    reg_num = k_num * N_num
    big_N = int( len(ththresults) / reg_num )
    
    
    etas = np.array( ththresults )[:,0]
    etas_err = np.array( ththresults )[:,1]
    
    res_tmp = []
    res_err_tmp = []
    
    #group them into a list with the same number of elements as the total number
    #of allowed delta values
    for dindex in range(big_N):
        res_tmp += [etas[0 + dindex * reg_num : (dindex) * reg_num + reg_num]]
        res_err_tmp += [etas_err[0 + dindex * reg_num : (dindex) * reg_num + reg_num]]
     
    res_tmp2 = []
    res_err_tmp2 = []
    
    ccounter = 0
    
    #reshape the list into the same shape as param_arr
    for aindex in range(A_num):
        res_tmp3 = []
        res_err_tmp3 = []
        for dindex in range(len(sample_datas[aindex])):
            res_tmp3 += [res_tmp[ccounter]]
            res_err_tmp3 += [res_err_tmp[ccounter]]
            ccounter += 1
        
        res_tmp2 += [res_tmp3]
        res_err_tmp2 += [res_err_tmp3]  
        
    return res_tmp2, res_err_tmp2

def thth_curvature_fitter( t, dyns, freqs, N, etamin, etamax, elims, fw, npad, tau_lim = 1000. * u.us, plots = 0):
    dspec = 0
    timet = 0
    
    if (t[1] -  t[0] < 0):
        dspec= np.flip(dyns, 1) 
        timet= np.flip( t.astype(np.float64) )
    else:
        dspec= dyns
        timet = t.astype(np.float64)
    
    #loading a dynamicspec object
    bDyne = BasicDyn(
    name="AR",
    header=["AR"],
    times=timet.value,
    freqs=freqs.value,
    dyn=dspec,
    nsub=timet.shape[0],
    nchan=freqs.shape[0],
    dt=(timet[1] - timet[0]).value,
    df=(freqs[1] - freqs[0]).value,
    )
    
    #putting our data object into scintools format 
    dyn = Dynspec()
    dyn.load_dyn_obj(dyn=bDyne, verbose = False, process=False)
    
    #prep dyn for thetatheta
    dyn.prep_thetatheta(#cwf=101,
#                     cwf = freqs.shape[0],
#                     cwt = fd.shape[0], 
                    edges_lim=elims,
                    eta_min=etamin,
                    eta_max=etamax,
                    nedge = N ,
                    verbose=False,
                    fw=fw,
                    tau_lim = tau_lim,
                    fitting_proc = 'incoherent',
                    npad = npad)
    
    #calculate curvature and error
#     ts = time.time()
    if plots == 0:
        dyn.fit_thetatheta()
        
        #in case of getting infinite retry params
        if np.isnan( ((dyn.ththeta).to(u.s**3)).value ) == True:
            
            for saver_ind in range(3):
                    #prep dyn for thetatheta
                dyn.prep_thetatheta(
                                edges_lim=elims,
                                eta_min=etamin,
                                eta_max=etamax,
                                nedge = N, #int( N * (1 + (saver_ind+1)/5) ),
                                verbose=False,
                                fw=fw * (3 - saver_ind)/4,
                                tau_lim = tau_lim,
                                fitting_proc = 'incoherent',
                                npad = npad)
                if np.isnan( ((dyn.ththeta).to(u.s**3)).value ) == True and saver_ind == 2:
                    return ((dyn.ththeta).to(u.s**3)).value, ((dyn.ththetaerr).to(u.s**3)).value
                
                dyn.fit_thetatheta()
                
                if np.isnan( ((dyn.ththeta).to(u.s**3)).value ) == False:
                    return ((dyn.ththeta).to(u.s**3)).value, ((dyn.ththetaerr).to(u.s**3)).value
        
        #in case of getting negative curvatures retry params
        elif ((dyn.ththeta).to(u.s**3)).value  < 0:
            
            for saver_ind in range(3):
                    #prep dyn for thetatheta
                dyn.prep_thetatheta(
                                edges_lim=elims,
                                eta_min=etamin,
                                eta_max=etamax,
                                nedge = N, #int( N * (1 + (saver_ind+1)/5) ),
                                verbose=False,
                                fw=fw * (3 - saver_ind)/4,
                                tau_lim = tau_lim,
                                fitting_proc = 'incoherent',
                                npad = npad)
                if ((dyn.ththeta).to(u.s**3)).value  < 0 and saver_ind == 2:
                    return ((dyn.ththeta).to(u.s**3)).value, ((dyn.ththetaerr).to(u.s**3)).value
                
                dyn.fit_thetatheta()
                
                if ((dyn.ththeta).to(u.s**3)).value  > 0:
                    return ((dyn.ththeta).to(u.s**3)).value, ((dyn.ththetaerr).to(u.s**3)).value
            
            
        else:
        
            return ((dyn.ththeta).to(u.s**3)).value, ((dyn.ththetaerr).to(u.s**3)).value
        
    else:
        dyn.thetatheta_single(cf=0, ct=0, verbose=True)
        
        
        

def thth_curvature_fitter_coherent( t, dyns, freqs, N, etamin, etamax, elims, fw, npad, tau_lim = 1000. * u.us, plots = 0):
    dspec = 0
    timet = 0
    
    if (t[1] -  t[0] < 0):
        dspec= np.flip(dyns, 1) 
        timet= np.flip( t.astype(np.float64) )
    else:
        dspec= dyns
        timet = t.astype(np.float64)
    
    #loading a dynamicspec object
    bDyne = BasicDyn(
    name="AR",
    header=["AR"],
    times=timet.value,
    freqs=freqs.value,
    dyn=dspec,
    nsub=timet.shape[0],
    nchan=freqs.shape[0],
    dt=(timet[1] - timet[0]).value,
    df=(freqs[1] - freqs[0]).value,
    )
    
    #putting our data object into scintools format 
    dyn = Dynspec()
    dyn.load_dyn_obj(dyn=bDyne, verbose = False, process=False)
    
    #prep dyn for thetatheta
    dyn.prep_thetatheta(#cwf=101,
#                     cwf = freqs.shape[0],
#                     cwt = fd.shape[0], 
                    edges_lim=elims,
                    eta_min=etamin,
                    eta_max=etamax,
                    nedge = N ,
                    verbose=False,
                    fw=fw,
                    tau_lim = tau_lim,
                    fitting_proc = 'standard',
                    npad = npad)
    
    #calculate curvature and error
#     ts = time.time()
    if plots == 0:
        dyn.fit_thetatheta()
        
        #in case of getting infinite retry params
        if np.isnan( ((dyn.ththeta).to(u.s**3)).value ) == True:
            
            for saver_ind in range(3):
                    #prep dyn for thetatheta
                dyn.prep_thetatheta(
                                edges_lim=elims,
                                eta_min=etamin,
                                eta_max=etamax,
                                nedge = N, #int( N * (1 + (saver_ind+1)/5) ),
                                verbose=False,
                                fw=fw * (3 - saver_ind)/4,
                                tau_lim = tau_lim,
                                fitting_proc = 'standard',
                                npad = npad)
                if np.isnan( ((dyn.ththeta).to(u.s**3)).value ) == True and saver_ind == 2:
                    return ((dyn.ththeta).to(u.s**3)).value, ((dyn.ththetaerr).to(u.s**3)).value
                
                dyn.fit_thetatheta()
                
                if np.isnan( ((dyn.ththeta).to(u.s**3)).value ) == False:
                    return ((dyn.ththeta).to(u.s**3)).value, ((dyn.ththetaerr).to(u.s**3)).value
        
        #in case of getting negative curvatures retry params
        elif ((dyn.ththeta).to(u.s**3)).value  < 0:
            
            for saver_ind in range(3):
                    #prep dyn for thetatheta
                dyn.prep_thetatheta(
                                edges_lim=elims,
                                eta_min=etamin,
                                eta_max=etamax,
                                nedge = N, #int( N * (1 + (saver_ind+1)/5) ),
                                verbose=False,
                                fw=fw * (3 - saver_ind)/4,
                                tau_lim = tau_lim,
                                fitting_proc = 'standard',
                                npad = npad)
                if ((dyn.ththeta).to(u.s**3)).value  < 0 and saver_ind == 2:
                    return ((dyn.ththeta).to(u.s**3)).value, ((dyn.ththetaerr).to(u.s**3)).value
                
                dyn.fit_thetatheta()
                
                if ((dyn.ththeta).to(u.s**3)).value  > 0:
                    return ((dyn.ththeta).to(u.s**3)).value, ((dyn.ththetaerr).to(u.s**3)).value
            
            
        else:
        
            return ((dyn.ththeta).to(u.s**3)).value, ((dyn.ththetaerr).to(u.s**3)).value
        
    else:
        dyn.thetatheta_single(cf=0, ct=0, verbose=True)
        
def region_thth_prep( tx, dynsx, freqs, cwf, cwt, N, etamin, etamax, elims, fw, npad, tau_lim = 1000. * u.us, tau_mask = 0. * u.us, fitting_proc = 'standard' ):
    
    """
    Function that takes in a dynamic spectrum array with time, and frequency and puts it into a dynspec object with 
    theta_theta prep given the original parameters. It returns a dynspec class object with the data loaded
    """

    dspec = 0
    timet = 0
    
    if (tx[1] -  tx[0] < 0):
        dspec= np.flip(dynsx, 1) 
        timet= np.flip( tx.astype(np.float64) )
    else:
        dspec= dynsx
        timet = tx.astype(np.float64)
    
    #loading a dynamicspec object
    bDyne = BasicDyn(
    name="AR",
    header=["AR"],
    times=timet.value,
    freqs=freqs.value,
    dyn=dspec,
    nsub=timet.shape[0],
    nchan=freqs.shape[0],
    dt=(timet[1] - timet[0]).value,
    df=(freqs[1] - freqs[0]).value,
    )
    
    #putting our data object into scintools format 
    dyn = Dynspec()
    dyn.load_dyn_obj(dyn=bDyne, verbose = False, process=False)

    
    dyn.prep_thetatheta(#cwf=101,
                    cwf = cwf,
                    cwt= cwt, 
                    edges_lim = elims,
                    eta_min= etamin,
                    eta_max=etamax,
                    nedge = N ,
                    verbose=False,
                    fw=fw,
                    tau_lim = tau_lim,
                    fitting_proc = fitting_proc,
                    npad = npad,
                    tau_mask = tau_mask)
    
    return dyn


def curvature_predictor_n(A1, d1, A_real, d_real, t_u, nu_u, phase_u, 
                          base_curv, base_index = 'default', n = 3 , ylimbot=0, ylimtop=1):
    
    """
    Function that takes a given A1, d1 and produces a curvature prediction as a function of resampled time given the real params A_real and d_real
    It takes time as t_u, true anomaly as nu_u, and orbital phase as phase_u. The base curvature is simply given as the expected curvature it should have at base_index (as in all curvatures will be calculated as relative ratios with respect to this curvature. n is the number of splittings per region. ylimbot/top are simply y-axis limits. 
    """
    
    
    tare = base_curv / base_curv.value
    
    #generating the array of indeces for all regions given the resampled test params A1 and d1
    ife = generate_n_minus_1_x_2_array( peaks(Ad_projection_unitless(t = t_u.to(u.hour).value,
                                             nu = nu_u,
                                           phase = phase_u.value.astype(np.float64), 
                                           A = A1 , 
                                           delta = d1 ))[-1] )
    #getting the real effective motion
    real_r = Ad_projection_unitless(t = t_u.to(u.hour).value,
                                           nu = nu_u,
                                           phase = phase_u.value.astype(np.float64), 
                                           A = A_real , 
                                           delta = d_real )
    
    
    #getting the real effective velocity
    real_v = np.gradient( real_r, t_u.to(u.hour).value )
    #getting the real curavture
    real_curvature = tare * real_v**(-2.)
    
    #getting the tested effective velocity
    test_r = Ad_projection_unitless(t = t_u.to(u.hour).value,
                                             nu = nu_u,
                                           phase = phase_u.value.astype(np.float64), 
                                           A = A1 , 
                                           delta = d1 )
    #getting the tested effective velocity
    test_v = np.gradient( test_r, t_u.to(u.hour).value )
    
    res_etas_1 = []
    res_etas_2 = []
    
    
    if base_index == 'default':
        
        fac = base_curv.value 
        
    else:
        #factor_calculator
        e1 = ife[base_index][0]
        e2 = ife[base_index][-1]
        
        #getting the test velocity
        fac = base_curv.value / np.median( real_curvature[e1:e2] * ( test_v**2. )[e1:e2] ).value
    
    
    for i in range(len(ife)):
        
        #getting the start and end indeces of each region
        e1 = ife[i][0]
        e2 = ife[i][-1]
        

        res_etas_1 += [ np.median( real_curvature[e1:e2] * (test_v**2.)[e1:e2] ) * fac ]
        
        plt.subplot(1,len(ife),i+1)
#         plt.title(f' $\eta_{i+1} = $' + d2str(res_etas_1[-1].value,4) + '$s^3$')
    
        if i == 0:
            plt.xlabel('Resampled unitless time', fontsize = 16)
            plt.ylabel('Curvature ($s^3$)', fontsize = 16)
            
#         t_res_interval = real_r[e1:e2] 
        t_res_interval = test_r[e1:e2] 
        eta_res_interval = real_curvature[e1:e2] * test_v[e1:e2]**2 * fac

        plt.plot(t_res_interval,  
                 eta_res_interval, label = 'Resampled curvature'  )
        
        plt.plot(t_res_interval,  
                 real_curvature[e1:e2] * real_v[e1:e2]**2 * fac, label = 'Real curvature'  )


#         plt.ylim( res_etas_1[-1].value * 1e-5, res_etas_1[-1].value * 1e6)
        plt.ylim(ylimbot, ylimtop)
        
        #getting the entire length of resampled time and then dividing by the number of partitions
        tmin = min(t_res_interval[0], t_res_interval[-1] )
        tmax = max(t_res_interval[0], t_res_interval[-1] )
        t_delt = (tmax - tmin)/n
        
        #plotting vertical lines for each partition
        for j in range(n-1):
            plt.axvline( x = tmin + t_delt * (1 + j), c = 'r', lw = 0.5, ls = '--')
        #and setting ticks
        if i > 0:
            plt.yticks([])
        #if resampled time is decreasing then flip arrays
        if (t_res_interval[1] - t_res_interval[0]) < 0:
            eta_res_interval = np.flip(eta_res_interval)
            t_res_interval = np.flip(t_res_interval)
    
#         eta_len = len(eta_res_interval) // 3
        
#         res_t_2 = [  np.mean( (t_res_interval[0], t_res_interval[eta_len]) ) ,
#                          np.mean( (t_res_interval[eta_len], t_res_interval[eta_len*2]) ) ,
#                          np.mean( (t_res_interval[eta_len*2], t_res_interval[-1]) ) 
#                         ]

        #setting storage arrays
        res_etas_tt = []
        res_t_2 = []
        
        #looping over all partitions
        for tt in range(n):
            
            #storring the values of each partition in resampled time/curvature
            #starting at half of the first partition (tmin + t_delt * 0.5) and then iterating for the delta_time
            res_t_2 += [ tmin + t_delt * 0.5 + t_delt * (tt) ]
            #getting the index of the time in which this occurs and then using that index for the curvature
            ind1 = np.abs(t_res_interval - (tmin + t_delt * 0.5 + t_delt * (tt)) ).argmin()
            res_etas_tt += [eta_res_interval[ind1].value]

        #storing all the curvatures
        res_etas_2 += [res_etas_tt]
        
        #plotting the curavture-resampled time realtion for the first section
        plt.plot(res_t_2, np.array( res_etas_tt ), 'ro' )
        if i == 0:
            plt.legend()
                
        
    return res_etas_1, res_etas_2