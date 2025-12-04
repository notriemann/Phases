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

from Funcs_DP import *
from Funcs_DP_Orbsim import *

def interpolate_h(t, h, t_new, kind='quadratic'):
    """
    Interpolate h(t) over new time values t_new.
    
    Parameters:
        t (array-like): The original time values.
        h (array-like): The corresponding h(t) values.
        t_new (array-like): The new time values to interpolate over.
        kind (str): Type of interpolation ('linear', 'quadratic', 'cubic', etc.).
        
    Returns:
        np.ndarray: Interpolated h values at t_new.
    """
    # Create the interpolation function
    interpolation_function = interp1d(t, h, kind=kind, fill_value="extrapolate")
    
    # Evaluate the interpolated function at t_new
    h_interpolated = interpolation_function(t_new)
    
    return h_interpolated

def find_indices_over_same_length( pos1, pos2 ):
    """
    Function to find the indeces of the regions that
    pos1 and pos2 have in common
    """
    
    #get the ranges that each array spans
    range1 = np.array([pos1[0], pos1[-1]])
    range2 = np.array([pos2[0], pos2[-1]])
    
    #find indices of the position that are inside the other's region
    indices1 = find_indices_inside_range(range2, pos1)
    indices2 = find_indices_inside_range(range1, pos2)
    
    return indices1, indices2

def find_indices_just_outside(s, smin, smax):
    """
    Find the indices of elements in the array `s` that are within the range [smin, smax],
    and also include the indices of one value below and above the range, if they exist.

    Parameters:
        s (numpy.ndarray): Input NumPy array.
        smin (float): Minimum value of the range.
        smax (float): Maximum value of the range.

    Returns:
        numpy.ndarray: An array containing the indices of elements within the range,
                       and one value below and above the range, if present.
    """
    # Create a boolean mask for elements within the range
    mask = (s >= smin) & (s <= smax)
    
    # Use np.where to get the indices where the mask is True
    indices_within_range = np.where(mask)[0]
    
    # Find the indices just outside the range
    if indices_within_range.size > 0:
        first_index = indices_within_range[0]
        last_index = indices_within_range[-1]

        # Include the index just below the range, if it exists
        if first_index > 0:
            indices_within_range = np.insert(indices_within_range, 0, first_index - 1)
        
        # Include the index just above the range, if it exists
        if last_index < len(s) - 1:
            indices_within_range = np.append(indices_within_range, last_index + 1)

    return indices_within_range

def vertical_rolling_average(A, b):
    """
    Filter that smooths matrix A with strengh b over the vertical axis (rolling average with b rows by convolving with an array of ones)
    Parameters:
        A (array-like): initial matrix
        b (float): strength of smoothing
        
    Returns:
        result (array-like): smoothed matrix
    
    """
    
    # Define the vertical kernel for convolution
    kernel = np.ones((b, 1), dtype=np.float64) / b
    
    # Apply convolution along the rows (vertical direction)
    result = convolve2d(A, kernel, mode='same', boundary='symm')
    
    return result

def horizontal_rolling_average(A, b):
    
    """
    Filter that smooths matrix A with strengh b over the horizontal axis (rolling average with b rows by convolving with an array of ones)
    Parameters:
        A (array-like): initial matrix
        b (float): strength of smoothing
        
    Returns:
        result (array-like): smoothed matrix
    
    """
    # Define the vertical kernel for convolution
    kernel = np.ones(( 1, b), dtype=np.float64) / b
    
    # Apply convolution along the rows (vertical direction)
    result = convolve2d(A, kernel, mode='same', boundary='symm')
    
    return result

def find_as_des_segments(y):
    """
    Finds indices that mark the boundaries between purely ascending and descending regions in the dataset.
    
    Parameters:
    y : array-like
        The input data array.
        
    Returns:
    segment_indices : list
        A list of indices [a, b, c, ...] such that y[a:b], y[b:c], ... are purely ascending or descending.
    """
    # Calculate the first difference
    diff_y = np.diff(y)
    
    # Find where the sign of the difference changes
    sign_changes = np.sign(diff_y[:-1]) != np.sign(diff_y[1:])
    
    # Get the indices where the sign changes
    segment_indices = np.where(sign_changes)[0] + 1  # +1 to account for the index shift due to diff
    
    return segment_indices


def resampler_for_similar_regions(A, delta, nu, phase, time, dyns, freq, Nres ):
    """
    Function that takes data (time, dynamic spec. dyns, frequency freq) and input variables A, delta
    in order to compute the set of resampled dynamic specturm for each similar tunring point region

    By turning point this simply means about making a plot of position on the screen as a function of time,
    which looks like a sinusoid, and then evaluating the regions around the same turning points in remapped time
    that travel the same distance over the remapped position. This then returns

    Parameters:
       A (float): scalar value of A
       delta (float): scalar value of delta
       nu (array-like): true anomaly
       phase (array-like): true anomaly + longitude of periastron
       time (array-like): array of time
       freq (array-like): array of frequencies
       dyns (nxm array-like): array or matrix of the dynamic spectrum
       delta_y (foat): scalar value of the remapped separation
       
    Returns:
        datas_pos (list of arrays): remapped position on the screen
        datas_t (list of arrays): remapped time (rescaled datas_pos by a constant so it's in time units)
        datas_dyns (list of arrays): remapped time 

    """
    #compute the position on the screen as a function of time, phase, nu, A and delta
    
    #compute lenspos relative to B = 100 Mm (Megameters)
    lenspos = Ad_projection_unitless(t = time, 
                                        nu = nu, 
                                        phase = phase, 
                                        A = A, 
                                        delta = delta).value * 100.

    
    #get the indices of the turning points of the lenspos-time curve
    turning_ind = find_as_des_segments( lenspos )
    turning_ind_fill = np.concatenate(([0], turning_ind, [len(lenspos)]))
    
    datas_pos = []
    datas_t = []
    datas_dyns = []
    datas_dyns_N = []
    
    datas_conj = []
    datas_fd = []
    
    #get the indices of the data that cross the same position on the screen around each turning point
    for it in range(len(turning_ind)):
        
        #compute the region before and after the turning point
        pos1 = lenspos[ turning_ind_fill[ it ] : turning_ind_fill[it+1] ]
        pos2 = lenspos[ turning_ind_fill[it+1] : turning_ind_fill[it+2] ]

#         ind_before, ind_after = find_similar_regions( pos1, pos2 )
        ind_before, ind_after = find_indices_over_same_length( pos1, pos2 )

        #getting the remapped coordinates and dynamic spectrum
#         res_before, dyn_before = resampler( pos1[ind_before], 
#                                            dyns[:, turning_ind_fill[ it ] : turning_ind_fill[it+1]][:,ind_before], 
#                                            delta_y)

#         res_after, dyn_after = resampler( pos2[ind_after], 
#                                            dyns[:, turning_ind_fill[it+1] : turning_ind_fill[it+2]][:,ind_after], 
#                                            delta_y)
        
        res_before, dyn_before = resampler2( pos1[ind_before], 
                                           dyns[:, turning_ind_fill[ it ] : turning_ind_fill[it+1]][:,ind_before], 
                                           Nres)
        
        
        res_after, dyn_after = resampler2( pos2[ind_after], 
                                           dyns[:, turning_ind_fill[it+1] : turning_ind_fill[it+2]][:,ind_after],
                                           Nres)
    

#         if len(res_after) != len(res_before):
#             raise ValueError("The number of elements of region 1 and 2 are different")
            
        #store remapped position
        datas_pos += [ [res_before * u.Mm, res_after * u.Mm]]
        
        #compute and store remapped time 
        datas_t += [ [(res_before * u.Mm / (const.c / 1e4) ).to(u.s), 
                    (res_after * u.Mm / (const.c / 1e4) ).to(u.s)]]
        
        #store remapped dynspec
        datas_dyns += [ [dyn_before, dyn_after]]
        #store sizes of original dspec
        datas_dyns_N += [[dyns[:, turning_ind_fill[ it ] : turning_ind_fill[it+1]][:,ind_before].size,
                          dyns[:, turning_ind_fill[it+1] : turning_ind_fill[it+2]][:,ind_after ].size ]]
        
        
        #compute the conjugate variables

        conjspec_before = np.fft.fftshift( np.fft.fft2( dyn_before ) )
        conjspec_after = np.fft.fftshift( np.fft.fft2( dyn_after ) )
        
        fd_conj_before = np.fft.fftshift(np.fft.fftfreq( datas_t[-1][0].size, datas_t[-1][0][1]-datas_t[-1][0][0] )).to(u.mHz)
        fd_conj_after = np.fft.fftshift(np.fft.fftfreq( datas_t[-1][-1].size, datas_t[-1][-1][1]-datas_t[-1][-1][0] )).to(u.mHz)
        
        datas_conj += [ [conjspec_before, conjspec_after] ]
        datas_fd += [[ fd_conj_before, fd_conj_after ]]
        
    tau_conj = np.fft.fftshift(np.fft.fftfreq( freq.size, freq[1]-freq[0] )).to(u.us)
        
    return datas_pos, datas_t, datas_dyns, datas_dyns_N, datas_fd, datas_conj, tau_conj




def resampler_for_general_regions(A, delta, nu, phase, time, dyns, freq, Nres ):
    """
    Function that takes data (time, dynamic spec. dyns, frequency freq) and input variables A, delta
    in order to compute the set of resampled dynamic specturm that lie in-between each turning point

    By turning point this simply means about making a plot of position on the screen as a function of time,
    which looks like a sinusoid, and then evaluating the regions around the same turning points in remapped time
    that travel the same distance over the remapped position. This then returns

    Parameters:
       A (float): scalar value of A
       delta (float): scalar value of delta
       nu (array-like): true anomaly
       phase (array-like): true anomaly + longitude of periastron
       time (array-like): array of time
       freq (array-like): array of frequencies
       dyns (nxm array-like): array or matrix of the dynamic spectrum
       delta_y (foat): scalar value of the remapped separation
       
    Returns:
        datas_pos (list of arrays): remapped position on the screen
        datas_t (list of arrays): remapped time (rescaled datas_pos by a constant so it's in time units)
        datas_dyns (list of arrays): remapped time 

    """
    #compute the position on the screen as a function of time, phase, nu, A and delta
    
    #compute lenspos relative to B = 100 Mm (Megameters)
    lenspos = Ad_projection_unitless(t = time, 
                                        nu = nu, 
                                        phase = phase, 
                                        A = A, 
                                        delta = delta).value * 100.

    
    #get the indices of the turning points of the lenspos-time curve
    turning_ind = find_as_des_segments( lenspos )
    turning_ind_fill = np.concatenate(([0], turning_ind, [len(lenspos)]))
    
    datas_pos = []
    datas_t = []
    datas_dyns = []
    
    datas_conj = []
    datas_fd = []
    
    #get the indices of the data that cross the same position on the screen around each turning point
    for it in range(len(turning_ind)+1):

        
        res_, dyn_ = resampler2( lenspos[turning_ind_fill[ it ] : turning_ind_fill[it+1]], 
                                   dyns[:, turning_ind_fill[ it ] : turning_ind_fill[it+1]], 
                                   Nres)
    

            
        #store remapped position
        datas_pos += [ res_ * u.Mm]
        
        #compute and store remapped time 
        datas_t += [ (res_ * u.Mm / (const.c / 1e4) ).to(u.s)]
        
        #store remapped dynspec
        datas_dyns += [ dyn_ ]
        
        
        #compute the conjugate variables
        conjspec_ = np.fft.fftshift( np.fft.fft2( dyn_ ) )
        
        fd_conj_ = np.fft.fftshift(np.fft.fftfreq( datas_t[-1].size, datas_t[-1][1]-datas_t[-1][0] )).to(u.mHz)
        
        datas_conj += [ conjspec_ ]
        datas_fd += [ fd_conj_ ]
        
    tau_conj = np.fft.fftshift(np.fft.fftfreq( freq.size, freq[1]-freq[0] )).to(u.us)
        
    return datas_pos, datas_t, datas_dyns, datas_fd, datas_conj, tau_conj




def resampler_dimensionless(A, delta, nu, phase, time, dyns, freq, Nres ):
    """
    Function that takes data (time, dynamic spec. dyns, frequency freq) and input variables A, delta
    in order to compute the set of resampled dynamic specturm that lie in-between each turning point

    By turning point this simply means about making a plot of position on the screen as a function of time,
    which looks like a sinusoid, and then evaluating the regions around the same turning points in remapped time
    that travel the same distance over the remapped position. This then returns

    Parameters:
       A (float): scalar value of A
       delta (float): scalar value of delta
       nu (array-like): true anomaly
       phase (array-like): true anomaly + longitude of periastron
       time (array-like): array of time
       freq (array-like): array of frequencies
       dyns (nxm array-like): array or matrix of the dynamic spectrum
       delta_y (foat): scalar value of the remapped separation
       
    Returns:
        datas_pos (list of arrays): remapped position on the screen
        datas_t (list of arrays): remapped time (rescaled datas_pos by a constant so it's in time units)
        datas_dyns (list of arrays): remapped time 

    """
    #compute the position on the screen as a function of time, phase, nu, A and delta
    
    #compute lenspos relative to B = 100 Mm (Megameters)
    lenspos = Ad_projection_unitless(t = time, 
                                        nu = nu, 
                                        phase = phase, 
                                        A = A, 
                                        delta = delta).value 

    
    #get the indices of the turning points of the lenspos-time curve
    turning_ind = find_as_des_segments( lenspos )
    turning_ind_fill = np.concatenate(([0], turning_ind, [len(lenspos)]))
    
    datas_pos = []
    datas_t = []
    datas_dyns = []
    
    datas_conj = []
    datas_kd = []
    
    #get the indices of the data that cross the same position on the screen around each turning point
    for it in range(len(turning_ind)+1):

        
        res_, dyn_ = resampler2( lenspos[turning_ind_fill[ it ] : turning_ind_fill[it+1]], 
                                   dyns[:, turning_ind_fill[ it ] : turning_ind_fill[it+1]], 
                                   Nres)
    

            
        #store remapped position
        datas_pos += [ res_ * u.one]
        
        #store remapped dynspec
        datas_dyns += [ dyn_ ]
        
        #compute the conjugate variables
        conjspec_ = np.fft.fftshift( np.fft.fft2( dyn_ ) )
        
        fd_conj_ = np.fft.fftshift(np.fft.fftfreq( datas_pos[-1].size, datas_pos[-1][1]-datas_pos[-1][0] ))
        
        datas_conj += [ conjspec_ ]
        datas_kd += [ fd_conj_ * u.one]
        
    tau_conj = np.fft.fftshift(np.fft.fftfreq( freq.size, freq[1]-freq[0] )).to(u.us)
        
    return datas_pos, datas_dyns, datas_kd, datas_conj, tau_conj


def chi_pre_process( data_fd, data_cs, data_tau, 
                    tau_min = 0., tau_max = np.inf, fd_lim = 15, fd_noise = 10,
                    N_int = 201, alpha = 1., sstrength = 3, hstrength = 1, 
                    nsep = 4, cutmid = 1, 
                    kind = 'quadratic', diagnostic_plot = False):
    
    """
    Function to calculate the secondary spectra from conjugate spectra,
    crop in fd and tau by the amounts (tau_min and tau_max), (-fd_lim, fd_lim)
    
    Applies a vertical smoothing filter (in order to smooth out the granularities so as the
    secondary spectra subtraction mostly focuses on the shape of parabolae) with strength sstrength
    and makes an interpolation of the resulting secondary spectra in fd with N_int elements such that 
    all secondary spectra given as result have the same fd array.
    
    In addition it removes a cutmid number of data points around
    the fd = 0 coordinate in order to remove artifcats of secondary spectrum that just lie as a vertical line or points on fd = 0
    and to do a noise estimate cutoff through fd_noise, where any point outside the range of fd noise is taken as the noise floor
    for the secondary spectrum. A horizontal averaging filter can also be applied, default is 1, which means no filter. It also
    divides takes an nsep number of divisions in order to alculate how to equally divide the secondary spectrum in taud into
    nsep equal chunks. Only the division length in taud is calculated here. The actual indices will be taken in another function.
    
    Prameters:
        data_fd (array-like): list of regions with arrays of their fds
        data_cs (array-like): list of regions with arrays of their conjugate spectra
        data_tau (array-like): array of taud 
        tau_min (float): bottom limit of taud to crop
        tau_max (float): upper limit of taud to crop
        fd_lim (float): symmetric limit on fd to crop and interpolate
        N_int (float): number of elements on fd that each interpolated sec.spec would have
        alpha (float): exponent of the division from the secondary spectra computation |CS|^2 / max(|CS|)^alpha (only relevant for noise purposes)
        sstrength (float): strength of the vertical filter. It simply means how many rows to roll and average
        kind (float): kind of interpolation
        
    Returns:
        ss_eval_arr (array-like): list of processed secondary spectra
        fd_evaluate (array-like): list of the cropped fd values for each spectra
        data_tau[tau_ind] (array-like): array of the cropped taud values
        tau_split (array-like): array consisting of the minimum value of taud, delta taud divisions and how many divisions
        fd_noise (float): number outlining when noise region begins in fd
    
    
    """
    
    #storage arrays for fd, conjugate spec, and tau
    fd_eval_arr = []
    ss_eval_arr = []
    
    #crop in tau and fd before interpolating
    tau_ind = find_indices_within_range(data_tau.value, tau_min, tau_max )
    
    for region in range(len(data_cs)):
        
        #compute the secondary spectra
        ss_before = np.abs( data_cs[region][0] )**2 / np.max( np.abs( data_cs[region][0] )**alpha )
        ss_after = np.abs( data_cs[region][-1] )**2 / np.max( np.abs( data_cs[region][-1] )**alpha )

        #apply a vertical filter for smoothing
        ss_before = vertical_rolling_average(ss_before , sstrength)
        ss_after = vertical_rolling_average(ss_after , sstrength)


        #orient the secondary spectra in fd such that it runs from negative fd to positive
        if data_fd[region][0][1] > data_fd[region][0][0] :

            fd_in_before = find_indices_just_outside(data_fd[region][0].value, -fd_lim, fd_lim )
            fd_in_after = find_indices_just_outside(np.flip( data_fd[region][-1].value), -fd_lim, fd_lim )
            
            fd_bef = data_fd[region][0][fd_in_before]
            fd_aft = np.flip(data_fd[region][-1])[fd_in_after]
            
            ss_after = np.flip( ss_after, axis = 1)
            
        else:
            fd_in_before = find_indices_just_outside(np.flip( data_fd[region][0].value), -fd_lim, fd_lim )
            fd_in_after = find_indices_just_outside(data_fd[region][-1].value, -fd_lim, fd_lim )
            
            fd_bef = np.flip(data_fd[region][0])[fd_in_before]
            fd_aft = data_fd[region][-1][fd_in_after]
            
            ss_before = np.flip( ss_before, axis = 1)

        #interpolate along the fd axis
        if N_int <= min(len(fd_bef), len(fd_aft)):
            
            #in the case the interpolation number is short just fix it to the current resolution + 100
            N_int = min(len(fd_bef), len(fd_aft)) + 100
            
        fd_evaluate = np.linspace(-fd_lim, fd_lim, N_int ) * fd_bef.unit
        
        
        #interpolate sec spec before
        ssint_before = interp1d(fd_bef, ss_before[tau_ind][:,fd_in_before], kind=kind, axis= 1)
        ssint_after = interp1d(fd_aft, ss_after[tau_ind][:,fd_in_after], kind=kind, axis= 1)

        #New and oriented sec spec
        ss1 = ssint_before(fd_evaluate)
        ss2 = ssint_after(fd_evaluate)
        
        if ( len(fd_evaluate) - 1 ) % 2 == 1:
            raise ValueError("Number of fd values isn't an odd number ")
        
        #getting the total horizontal length from fd for taking data from the central region
        total_length = ( len(fd_evaluate) - 1 ) // 2
        
        ss1 = horizontal_rolling_average( ss1  , hstrength) 
        ss2 = horizontal_rolling_average( ss2  , hstrength) 
        
        ss1[:, total_length - cutmid : total_length + cutmid] = 0.
        ss2[:, total_length - cutmid : total_length + cutmid] = 0.
        
        ss_eval_arr += [[ss1, ss2]]
        
        
    #separation length in tau
    dtaus = ( data_tau[tau_ind][-1] - data_tau[tau_ind][0] ).value / nsep
    
    
    
    if diagnostic_plot == True: 
        
        for reg in range(len(ss_eval_arr)):

            plt.figure(figsize= (10,5) )
            
            for j in range(2):
                
                if (j == 0) and (reg == 0):
                    smax = np.max(ss_eval_arr[0][0])
                    smin = np.median(ss_eval_arr[0][0])
            
                plt.subplot(1,2,j+1)
                secondary_spectrum_plotter_time2( fd_evaluate, 
                                                data_tau[tau_ind], 
                                                ss_eval_arr[reg][j], 
                                                smin, 
                                                smax , 
                                                15, 
                                                None, 
                                                None)

                #plot separation lines
                for sep_ind in range(nsep-1):    
                    plt.axhline(y = dtaus * (sep_ind+1) + tau_min, c = 'r')

                #plot fd noise level separation
                plt.axvline(x = fd_lim, c = 'r')
                plt.axvline(x = -fd_lim, c = 'r')

                plt.axvline(x = -fd_noise, c = 'r')
                plt.axvline(x = fd_noise, c = 'r')
            
            plt.suptitle('Turning point number ' + str(reg + 1), fontsize = 16)
            plt.show()
            
    tau_split = [tau_min, dtaus, nsep]
    
    
    return ss_eval_arr, fd_evaluate, data_tau[tau_ind], tau_split, fd_noise

    
    
def chi_measure_original(data_fd, data_ss, data_ds, data_tau, data_fd_noise, data_tau_split ):
    """
    Function to compute the likeness of parabolae by subtracting secondary spectra. Different measures for the chi are attempted,
    hence the general form of chi_measure_attempt()
    """
    
    chi_storage = []
    N_storage = 0.
    
    taumin, taustep, ntau = data_tau_split
    
    
    for reg in range(len(data_ss)):
        
        #storage array for the different splittings in taud
        chi_stor_split = np.zeros(ntau)
        
        #taud indices inside the given range
        
        #computing the noise estimate:
        noise_left_ind = find_indices_inside_range([data_fd.value[0], -data_fd_noise], data_fd.value)
        noise_right_ind = find_indices_inside_range([data_fd_noise, data_fd.value[-1]], data_fd.value)
        
        #appending and taking the std of the left and right side of the noise floor for each region
        noise_bef = np.std( np.append(data_ss[reg][0][:,noise_left_ind], data_ss[reg][0][:,noise_right_ind], 1) )
        noise_aft = np.std( np.append(data_ss[reg][-1][:,noise_left_ind], data_ss[reg][-1][:,noise_right_ind], 1) )
        
        #computing the data size of each dynamic spectrum from which these sec spec were generated
        N_bef = data_ds[reg][0].size
        N_aft = data_ds[reg][-1].size
        
        #computing the data values from the sum of the original dynamic spectrum
        DS_val_bef = np.sum(data_ds[reg][0])
        DS_val_aft = np.sum(data_ds[reg][-1])
        
        #computing subtraction
        chi_reg = np.square( data_ss[reg][0] - data_ss[reg][-1]  ) 
        
        #norm_factor (N_bef + N_aft)
        nfac = (N_bef + N_aft) / ( noise_bef**2 + noise_aft**2 )
        
        #storing the total data used
        N_storage += (N_bef + N_aft)
        
        #getting the indices of the desired split
        for j in range(ntau):
            
            ss_tau_ind = find_indices_inside_range([taumin + taustep * float(j), taumin + taustep * float(1 + j)], data_tau.value)
            #ss_tau_ind = find_indices_within_range(data_tau.value, taumin + taustep * float(j), taumin + taustep * float(1 + j) )
            
            chi_stor_split[j] = nfac * np.sum( chi_reg[ss_tau_ind] )
            
        chi_storage += [ chi_stor_split ]
        
    return np.sum( chi_storage, axis = 0 ) / N_storage
    
def chi_measure_var(data_fd, data_ss, data_ds, data_tau, data_fd_noise, data_tau_split, full_chi = False ):
    """
    Function to compute the likeness of parabolae by subtracting secondary spectra. Different measures for the chi are attempted,
    hence the general form of chi_measure_attempt()
    """
    
    chi_storage = []
    N_storage = 0.
    
    taumin, taustep, ntau = data_tau_split
    
    
    for reg in range(len(data_ss)):
        
        #storage array for the different splittings in taud
        chi_stor_split = np.zeros(ntau)
        
        #taud indices inside the given range
        
        #computing the noise estimate:
        noise_left_ind = find_indices_inside_range([data_fd.value[0], -data_fd_noise], data_fd.value)
        noise_right_ind = find_indices_inside_range([data_fd_noise, data_fd.value[-1]], data_fd.value)
        
        #computing the data size of each dynamic spectrum from which these sec spec were generated
        N_bef = data_ds[reg][0]
        N_aft = data_ds[reg][-1]
        
        #appending and taking the std of the left and right side of the noise floor for each region
        noise_bef = np.std( np.append(data_ss[reg][0][:,noise_left_ind], data_ss[reg][0][:,noise_right_ind], 1)  ) * N_bef
        noise_aft = np.std( np.append(data_ss[reg][-1][:,noise_left_ind], data_ss[reg][-1][:,noise_right_ind], 1)  ) * N_aft
        
        
        #computing the data values from the sum of the original dynamic spectrum
#         DS_val_bef = np.sum(data_ds[reg][0])
#         DS_val_aft = np.sum(data_ds[reg][-1])
        
        #computing subtraction
        chi_reg = np.square( data_ss[reg][0]*N_bef - data_ss[reg][-1]*N_aft  ) 
        
        #norm_factor (N_bef + N_aft)
        nfac = 1. / ( noise_bef**2 + noise_aft**2 )

        
        #storing the total data used
#         N_storage += (N_bef + N_aft)
        
        #getting the indices of the desired split
        for j in range(ntau):
            
            ss_tau_ind = find_indices_inside_range([taumin + taustep * float(j), taumin + taustep * float(1 + j)], data_tau.value)
            #ss_tau_ind = find_indices_within_range(data_tau.value, taumin + taustep * float(j), taumin + taustep * float(1 + j) )
            
            chi_stor_split[j] = nfac * np.sum( chi_reg[ss_tau_ind] )
            
        chi_storage += [ chi_stor_split ]
        
    if full_chi == True:
        return chi_storage
    else:
        return np.sum( chi_storage, axis = 0 ) 
    
    
def chi_from_ad_parallel(aind, dind, a_array, d_array, d_ds, d_ts, d_fs, d_nu, d_phase,
                      Nres, N_int, tau_min, tau_max, fd_lim, fd_noise, alpha, vstrength, nsep, cutmid, 
                      kind = 'quadratic', chi_func = chi_measure_var):
    
    """
    Function to get a single value of chi given the previous function. 
    This is a function wrapper for the previous functions where chi_func serves as a stand-in for 
    calculating chi (with wahtevr measure)
    """
    
    #Getting the remapped data and conjugate data
    d_pos, d_tres, d_dyn, d_dyn_N, f_fd, f_cs, f_tau = resampler_for_similar_regions(A = a_array[aind][dind], 
                                                          delta = d_array[aind][dind], 
                                                          nu = d_nu, 
                                                          phase = d_phase, 
                                                          time = d_ts.value, 
                                                          dyns = d_ds, 
                                                          freq = d_fs,  
                                                          Nres = Nres )

    #prepping secondary spectra for subtraction
    c_ss, c_fd, c_tau, c_dtau_split, c_fd_noise = chi_pre_process( data_fd = f_fd , 
                                                                              data_cs = f_cs, 
                                                                              data_tau = f_tau, 
                                                                              tau_min = tau_min, 
                                                                              tau_max = tau_max, 
                                                                              fd_lim = fd_lim,
                                                                              fd_noise = fd_noise,
                                                                              N_int = N_int, 
                                                                              alpha = alpha, 
                                                                              sstrength = vstrength,
                                                                              nsep = nsep,
                                                                              cutmid = cutmid,
                                                                              kind = kind,
                                                                              diagnostic_plot = False)

    #getting the result for chi for one (A,delta)
    chi_res = chi_func (data_fd = c_fd, 
                         data_ss = c_ss, 
                         data_ds = d_dyn_N,
                         data_tau = c_tau,
                         data_fd_noise = c_fd_noise,
                         data_tau_split = c_dtau_split)
    
    return aind, dind, chi_res

def Ad_extractor_from_chi(chi_array, Avals, dvals, nsep ):
    
    """
    Function to extract the inferred A, delta from a grid of A, delta, the array of chi values, and the separations
    """
    
    adata = []
    
    for j in range(nsep):
        aind = np.where( np.nanmin( chi_array[j] ) == chi_array[j])[0][0]
        dind = np.where( np.nanmin( chi_array[j] ) == chi_array[j])[-1][0]
        
        adata+= [(Avals[aind], dvals[aind][dind])]
        
    return adata

def quad_2d(coords, c0, c1, c2, c3, c4, c5):
    a, d = coords
    return c0 + c1*a + c2*d + c3*a**2 + c4*a*d + c5*d**2

def two_d_parabola_fit( chi_data, ind_dummy, a_paramarr, d_paramarr, nsep):
    
    a_data = np.array( [ a_paramarr[i][j] for i, j in ind_dummy])
    d_data = np.array( [ d_paramarr[i][j] for i, j in ind_dummy])
    
    A_res_par = []
    d_res_par = []
    A_res_par_sig = []
    d_res_par_sig = []
    
    for ns in range(nsep):
        
        # Construct the design matrix for the quadratic model
        X = np.column_stack([
            np.ones_like(a_data),  # c0
            a_data,                # c1 * a
            d_data,                # c2 * d
            a_data**2,             # c3 * a^2
            a_data * d_data,       # c4 * a * d
            d_data**2              # c5 * d^2
        ])

        # Solve for coefficients using least squares
        # Fit the function
        popt, pcov = curve_fit(quad_2d, (a_data, d_data), np.array(chi_data)[:,ns])


        # Extract coefficients
        c0, c1, c2, c3, c4, c5 = popt

        # Extract the covariance matrix elements
        var_c1, var_c2 = pcov[1, 1], pcov[2, 2]  # Variances of c1 and c2
        var_c3, var_c4, var_c5 = pcov[3, 3], pcov[4, 4], pcov[5, 5]  # Variances of c3, c4, c5
        cov_c3c4, cov_c3c5, cov_c4c5 = pcov[3, 4], pcov[3, 5], pcov[4, 5]  # Covariances

        # Solve for (a_min, d_min)
        H = np.array([[2 * c3, c4], [c4, 2 * c5]])
        b = np.array([-c1, -c2])
        a_min, d_min = np.linalg.solve(H, b)

        # Compute the error on (a_min, d_min)
        H_inv = np.linalg.inv(H)  # Inverse Hessian (uncertainty propagation)
        cov_a_d = H_inv @ np.array([[var_c1, 0], [0, var_c2]]) @ H_inv.T  # Propagate uncertainties

        sigma_a = np.sqrt(cov_a_d[0, 0])  # Uncertainty in a_min
        sigma_d = np.sqrt(cov_a_d[1, 1])  # Uncertainty in d_min
        
        
        A_res_par += [a_min]
        d_res_par += [d_min]
        A_res_par_sig += [sigma_a]
        d_res_par_sig += [sigma_d]
        
#         plt.figure(figsize=(8, 6))
#         sc = plt.scatter(a_data, d_data, c=np.array(chi_data)[:,ns], cmap='viridis', marker='s')
#         plt.colorbar(sc, label='Chi-Squared Value')
#         plt.xlabel('a parameter')
#         plt.ylabel('d parameter')
#         plt.title('Chi-Squared Heatmap')
#         plt.show()

    return A_res_par, d_res_par, A_res_par_sig, d_res_par_sig