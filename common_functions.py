#!/usr/bin/env pAAython
# coding: utf-8

from scipy import stats
import numpy as np
import math
from random import randint, sample, choice


# remove nans

def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isnan(y), lambda z: z.nonzero()[0]


def correct_motion_artifact(neuron_signals, idx_MA, inplace=False):
    """ 
        Correct one or several motion artifact(s) by nearest point averaging. 

        :input neuron_signals: a single trace or a list of traces (or np array [n_cells, n_timesteps])
        :input idx_MA: an integer or a list of integers.
        
        /!\ For single time-point artifacts.
        /!\ If neuron_traces contains several cells (2d array): correct idx_MA for all neurons.
        /!\ Does not work for first or last timesteps.
    """
    
    if not inplace:
        neuron_signals = neuron_signals.copy()
        
    if idx_MA is None:
        return neuron_signals
    
    if type(idx_MA) == int:
        idx_MA = [idx_MA]
        
    if len(neuron_traces.shape) == 1:
        neuron_signals = [neuron_signals]
    
    for n in range(len(neuron_traces)):
        for i in idx_MA:
            neuron_signals[n, i] = (neuron_signals[n, i-1] + neuron_signals[n, i+1]) / 2 

    return neuron_signals



def shift_signal(neuron_signal, idx=None):
    """
        Shift the calcium signal.

        :input neuron_signal: calcium trace to shift.
        :input idx (optional): index at which to shift. If None, random. If array, random selection of index among possible values.
    """
            
    if idx is None:
        idx = randint(0, len(neuron_signal)-1)
        
    if not type(idx) == int: 
        idx = choice(idx)

    neuron_signal_sh = np.roll(neuron_signal, idx)

    return neuron_signal_sh
            
            
def shuffle_signal(neuron_signal, t_cycles, inplace=False):
    """
        Shuffle calcium signal keeping the stimulus structure.
        for hindbrain data: 
            t_cycles = np.array([round(5.81*(5+i*15)) for i in range(20)])

        :input neuron_signal: calcium trace to shuffle.
        :input t_cycle: start times of stimulus cycles.
    """
    
    if not inplace:
        neuron_signal = neuron_signal.copy()
        
    idx_t_cycles = [i for i in range(len(t_cycles)-1)]
    subset = sample(idx_t_cycles, len(t_cycles)-1)

    signal_shuffled = np.zeros((t_cycles[-1] - t_cycles[0]))
    start = 0

    for idx in subset:
        cycle = neuron_signal[t_cycles[subset[idx]]:t_cycles[subset[idx]+1]]
        end = start + len(cycle)
        signal_shuffled[start:end] = cycle
        start = end
    
    return signal_shuffled



def get_GC_sig(gc, fstat, threshold_f):
    """ 
        Get the significant links among all pairs.

        :input gc: Granger causality matrix of all pairs (NxN)
        :input Fstat: F-statistics of all pairs (NxN)
        :input threshold_F: threshold of significance. Can be a single value (int)
                             or a matrix (NxN) for customized threshold for each pair

        gc[i,j] is significant if fstat[i,j] > threshold_f or fstat[i,j] > threshold_f[i,j]

    """
    mask = fstat > threshold_f
    gc_sig = gc.copy()
    gc_sig[~mask] = np.nan
    return gc_sig


def get_mean_GC(gc):
    """
        Get the mean value of Granger causality values.
        Can be calculate on GC or GC_sig. 

        :input gc: Granger causality matrix of all pairs (NxN)

    """
    n_cells = len(gc)
    np.fill_diagonal(gc, np.nan)
    return np.nanmean(gc)


def get_percent_sig(gc_sig):
    """
        Get percentage of significant Granger causality links.

        :input gc_sig: significant Granger causality matrix of all pairs (NxN)

    """
    n_cells = len(gc_sig)
    n_sig = len([i for j in gc_sig for i in j if i>0]) 
    p_sig = n_sig / (n_cells * (n_cells-1))
    return p_sig



def get_GC_from_Fstat(fstat, nparams_full, nparams_reduced, t_regr):
    """ 
        Calulate GC matrix from F-statistics value of one pair or one matrix of NxN pairs.

        :params fstat: F-statistics of each pair (int or NxN)
        :params nparams_full: number of parameters in the full model (n_rois*n_lags + 1)
        :params nparams_reduced: number of parameters in the reduced model ( (n_rois-1)*n_lags + 1)
        :params t_regr: number of timesteps used for GC (n_timesteps - n_lags)
    """
    # For hindbrain:
    #     t_ regr = n_timesteps - n_lags = 1653 because 1656 timesteps taken not 1744
    #     BVGC
    #         nparams_full = 2 * n_lags + 1 = 7
    #         nparams_reduced = n_lags + 1 = 4
    #     cBVGC
    #         nparams_full = 3 * n_lags + 1 = 10
    #         nparams_reduced = 2 * n_lags + 1 = 7
    #     MVGC
    #         nparams_full = n_rois * n_lags + 1 = 61
    #         nparams_reduced = (n_rois - 1) * n_lags + 1 = 58
    #     cMVGC
    #         nparams_full =  (n_rois + 1) * n_lags + 1 = 64
    #         nparams_reduced = n_rois * n_lags + 1 = 61
    
    gc = np.log((fstat * (nparams_full - nparams_reduced +0.0) / (t_regr - nparams_full) + 1.0) * (t_regr - nparams_full) / (t_regr - nparams_reduced))

    if type(gc) is np.ndarray:
        np.fill_diagonal(gc, np.nan)  
    
    return gc


def bivariateGC(signals, n_lags=3, pval=0.01, tau=1, verbose=False):
    """ 
        Calculate the bivariate Granger causality between each pair of signals.
    
        :input signals: variables in rows and observations in columns --> shape = n_rois x n_timesteps 
        :input n_lags: number of past time steps to include in model (order)
        :input pval: significance level for the F test. The lower it is, the higher threshold_F (does not change GC)
        :input tau (default 1): number of time steps between lags --> keep past values at times: [t-tau*i for i in range(n_lags)]
               (tau=1 for GC: keep all values up to n_lags, don't skip any)
        :input verbose (default False): set to True to display result and threshold

        :output GC_sig: significant values of Granger causality matrix
        :output GC: Granger causality matrix
        :output F_stat: F statistics of the GC test
        :output threshold_F: threshold for significance.
    """
    (n_rois, n_timesteps) = signals.shape
    n_timesteps_regr = n_timesteps - n_lags
    n_pairs = n_rois * (n_rois - 1)

    threshold_F = stats.f.ppf(1 - pval / n_pairs, n_lags, n_timesteps_regr - 2 * n_lags - 1)  # statistical threshold
    # Bonferroni corrected 1 - pval/n_pairs
    # pval = significance level

    Fstat = np.zeros((n_rois, n_rois))  # matrix of all F_xy
    GC = np.zeros((n_rois, n_rois))  # matrix of all GC_xy
    GC_sig = np.zeros((n_rois, n_rois))  # matrix of all significant GC_xy (if F_xy >= threshold_F)

    signals_lagged = lag_signals(signals, n_lags + 1, tau=1)  # n_lags+1 --> n_lags + present

    for i, x in enumerate(signals):  # for each column (each roi)
        x_lagged = signals_lagged[i]
        x_past = x_lagged[:, :-1]  # past of signal x (lagged)

        for j, y in enumerate(signals):
            if i != j:
                y_lagged = signals_lagged[j]
                y_present = np.expand_dims(y_lagged[:, -1], axis=1)
                # current value of signal y (lagged)
                y_past = y_lagged[:, :-1]  # past of signal y (lagged)
                xy_past = np.concatenate((x_past, y_past), axis=1)  # both past concatenated
                reduced_model = np.concatenate((y_past, y_present), axis=1)  # y's past and current value
                full_model = np.concatenate((xy_past, y_present), axis=1)  # x and y's past and current y value

                # Covariances
                entr_reduced = entr(reduced_model.T) - entr(y_past.T)
                entr_full = entr(full_model.T) - entr(xy_past.T)

                # residual sum of squares
                RSS_reduced = np.exp(entr_reduced) 
                RSS_full = np.exp(entr_full) 
                
                sigma_reduced = RSS_reduced / (n_timesteps_regr - n_lags - 1)
                sigma_full = RSS_full / (n_timesteps_regr - 2 * n_lags - 1) 
                
                GC_xy = math.log(sigma_reduced / sigma_full) # GC value
                GC[i, j] = GC_xy
                
                F_xy = (n_timesteps_regr - 2 * n_lags - 1) / n_lags * (RSS_reduced - RSS_full) / RSS_full
                Fstat[i, j] = F_xy

                if F_xy > threshold_F:
                    GC_sig[i, j] = GC_xy
                else:
                    GC_sig[i, j] = np.nan

    np.fill_diagonal(GC, np.nan)
    np.fill_diagonal(GC_sig, np.nan)
    np.fill_diagonal(Fstat, np.nan)
    
    
    if verbose:
        print("F statistics:", Fstat)
        print("F threshold:", threshold_F)
        print("Significant GC values:", GC)

    return GC_sig, GC, Fstat, threshold_F



def multivariateGC(signals, n_lags=3, pval=0.01, tau=1, verbose=False):
    """ 
        Calculate the multivariate Granger causality between each pair of signals.

        :input signals: variables in rows and observations in columns --> shape = n_rois x n_timesteps 
        :input n_lags: number of past time steps to include in model (order)
        :input pval: significance level for the F test. The lower it is, the higher threshold_F (does not change GC)
        :input tau: number of time steps between lags --> keep past values at times: [t-tau*i for i in range(n_lags)]
               (tau=1 for GC: keep all values up to n_lags, don't skip any)
        :input verbose (default False): set to True to display result and threshold

        :output GC_sig: significant values of Granger causality matrix
        :output GC: Granger causality matrix
        :output F_stat: F statistics of the GC test
        :output threshold_F: threshold for significance.
    """
    (n_rois, n_timesteps) = signals.shape
    n_timesteps_regr = n_timesteps - n_lags
    n_pairs = n_rois * (n_rois - 1)

    threshold_F = stats.f.ppf(1 - pval / n_pairs, n_lags,
                              n_timesteps_regr - n_rois * n_lags - 1)  # statistical threshold 
    
    # Bonferroni corrected 1 - pval/n_pairs
    # pval = significance level

    Fstat = np.zeros((n_rois, n_rois))  # matrix of all F_xy
    GC = np.zeros((n_rois, n_rois))  # matrix of all GC_xy
    GC_sig = np.zeros((n_rois, n_rois))  # matrix of all significant GC_xy (if F_xy >= threshold_F)

    signals_lagged = lag_signals(signals, n_lags + 1, tau=1)  # n_lags+1 --> n_lags + present
    
    for i, x in enumerate(signals):  # for each column (each roi)
        x_lagged = signals_lagged[i]
        x_past = x_lagged[:, :-1]  # past of signal x (lagged)

        for j, y in enumerate(signals):
            if i != j:
                y_lagged = signals_lagged[j]
                y_present = np.expand_dims(y_lagged[:, -1], axis=1)  # current value of signal y (lagged)
                y_past = y_lagged[:, :-1]  # past of signal y (lagged)
                
                z_indices = [k for k in range(n_rois) if k not in [i, j]]
                
                z_lagged = signals_lagged[z_indices]
                z_past = np.concatenate(z_lagged[:, :, :-1], axis=1)

                yz_past = np.concatenate((y_past, z_past), axis=1)  # past without x
                xyz_past = np.concatenate((x_past, yz_past), axis=1)  # all past
                reduced_model = np.concatenate((y_present, yz_past), axis=1)  # past without x and current y value
                full_model = np.concatenate((xyz_past, y_present), axis=1)  # x, y, z's past and current y value

                # Covariances
                entr_reduced = entr(reduced_model.T) - entr(yz_past.T)
                entr_full = entr(full_model.T) - entr(xyz_past.T)
                
                # residual sum of squares
                RSS_reduced = np.exp(entr_reduced) 
                RSS_full = np.exp(entr_full) 
                
                sigma_reduced = RSS_reduced / (n_timesteps_regr - (n_rois - 1) * n_lags - 1)
                sigma_full = RSS_full / (n_timesteps_regr - n_rois * n_lags - 1) 
                
                GC_xy = math.log(sigma_reduced / sigma_full) # GC value
                GC[i, j] = GC_xy
                
                F_xy = (n_timesteps_regr - n_rois * n_lags - 1) / n_lags * (RSS_reduced - RSS_full) / RSS_full
                Fstat[i, j] = F_xy

                if F_xy > threshold_F:
                    GC_sig[i, j] = GC_xy
                else:
                    GC_sig[i, j] = np.nan

    
    np.fill_diagonal(GC, np.nan)
    np.fill_diagonal(GC_sig, np.nan)
    np.fill_diagonal(Fstat, np.nan)
    
    if verbose:
        print("F statistics:", Fstat)
        print("F threshold:", threshold_F)
        print("Significant GC values:", GC)

    return GC_sig, GC, Fstat, threshold_F



def conditionedGC(signals, Z, multi=False, n_lags=3, pval=0.01, tau=1, verbose=False):
    # signals should not be in Z: for multi + tail angle regressor --> all neurons in signals, ta_reg in Z
    
    """ 
        Calculate the Granger causality between each pair of signals, 
        conditioned on the signals in Z.

        :input signals: variables in rows and observations in columns --> shape = n_rois x n_timesteps 
        :input n_lags: number of past time steps to include in model (order)
        :input pval: significance level for the F test. The lower it is, the higher threshold_F (does not change GC)
        :input tau: number of time steps between lags --> keep past values at times: [t-tau*i for i in range(n_lags)]
               (tau=1 for GC: keep all values up to n_lags, don't skip any)
        :input verbose (default False): set to True to display result and threshold

        :output GC_sig: significant values of Granger causality matrix
        :output GC: Granger causality matrix
        :output F_stat: F statistics of the GC test
        :output threshold_F: threshold for significance.
    """
    (n_rois, n_timesteps) = signals.shape
    n_timesteps_regr = n_timesteps - n_lags
    n_pairs = n_rois * (n_rois-1)
    
    if multi:
        n_dof = n_rois
    else:
        n_dof = 2

    threshold_F = stats.f.ppf(1 - pval / n_pairs, n_lags,
                              n_timesteps_regr - (n_dof + len(Z)) * n_lags - 1) 
    
    # Bonferroni corrected 1 - pval/n_pairs
    # pval = significance level


    Fstat = np.zeros((n_rois, n_rois))  # matrix of all F_xy
    GC = np.zeros((n_rois, n_rois))  # matrix of all GC_xy
    GC_sig = np.zeros((n_rois, n_rois))  # matrix of all significant GC_xy (if F_xy >= threshold_F)

    signals_lagged = lag_signals(signals, n_lags + 1, tau)  
    z_lagged = lag_signals(Z, n_lags + 1, tau)
    
    for i, x in enumerate(signals):  # for each column (each roi)
        x_lagged = signals_lagged[i]
        x_past = x_lagged[:, :-1]  # past of signal x (lagged)
        
        for j, y in enumerate(signals):
            if i != j:
                y_lagged = signals_lagged[j]
                y_present = np.expand_dims(y_lagged[:, -1], axis=1)  # current value of signal y (lagged)
                y_past = y_lagged[:, :-1]  # past of signal y (lagged)

                z_past = np.concatenate(z_lagged[:, :, :-1], axis=1)
                print(z_lagged.shape)
                
                if multi:
                    zmulti_indices = [k for k in range(n_rois) if k not in [i,j]]
                    zmulti_lagged = signals_lagged[zmulti_indices]
                    zmulti_past = np.concatenate(zmulti_lagged[:, :, :-1], axis=1)
                    z_past = np.concatenate((z_past, zmulti_past), axis=1)

                yz_past = np.concatenate((y_past, z_past), axis=1)  # past without x
                xyz_past = np.concatenate((x_past, yz_past), axis=1)  # all past
                reduced_model = np.concatenate((y_present, yz_past), axis=1)  # past without x and current y value
                full_model = np.concatenate((xyz_past, y_present), axis=1)  # x, y, z's past and current y value

                print(z_past.shape, yz_past.shape, xyz_past.shape, reduced_model.shape, full_model.shape)
                
                # Covariances
                entr_reduced = entr(reduced_model.T) - entr(yz_past.T)
                entr_full = entr(full_model.T) - entr(xyz_past.T)
                
                # residual sum of squares
                RSS_reduced = np.exp(entr_reduced) 
                RSS_full = np.exp(entr_full) 
                
                sigma_reduced = RSS_reduced / (n_timesteps_regr -
                                               (n_dof + len(Z) - 1) * n_lags - 1)
                sigma_full = RSS_full / (n_timesteps_regr - (n_dof + len(Z)) * n_lags - 1) 
                
                GC_xy = math.log(sigma_reduced / sigma_full) # GC value
                GC[i, j] = GC_xy

                F_xy = (n_timesteps_regr - (n_dof + len(Z)) * n_lags - 1) / n_lags * (RSS_reduced - RSS_full) / RSS_full
                Fstat[i, j] = F_xy

                if F_xy > threshold_F:
                    GC_sig[i, j] = GC_xy
    
    np.fill_diagonal(GC, np.nan)
    np.fill_diagonal(GC_sig, np.nan)
    np.fill_diagonal(Fstat, np.nan)
    
    if verbose:
        print("F statistics:", Fstat)
        print("F threshold:", threshold_F)
        print("Significant GC values:", GC)

    return GC_sig, GC, Fstat, threshold_F


def bvgc_2_signals(signal1, signal2, n_lags=3, pval=0.01, n_pairs=1, tau=1, verbose=False):
    """ 
        Calculate bivariate Granger Causality between two neurons, from signal1 to signal2. 

        :input signal1: trace of neuron 1 (n_timesteps,)
        :input signal2: trace of neuron 2 (n_timesteps,)
        :input n_lags: number of lags for GC
        :input pval: p-value for significance of link
        :input n_pairs: total number of pairs (= number of tests for significance) to Bonferroni correct the threshold
        :input tau (default = 1): separation between lags for the embedding e.g. if tau=2 and n_lags=3: lag_signal uses n_lags past 
                timepoints separated by tau points i.e. signal at t = -5, -3, -1 
                --> keep tau=1 for GC: use first n_lags past points
        :input verbose (default = False): print results if True

        gc is significant if fstat > threshold_f 
    
    """
    # single BVGC value from neuron 1 to neuron 2
    (n_rois, n_timesteps) = (2, len(signal1))
    n_timesteps_regr = n_timesteps - n_lags
    threshold_F = stats.f.ppf(1 - pval/n_pairs, n_lags, n_timesteps_regr - 2 * n_lags - 1) 
    
    x_lagged = lag_signals([signal1], n_lags + 1, tau=1)[0]  # n_lags+1 --> n_lags + present
    y_lagged = lag_signals([signal2], n_lags + 1, tau=1)[0]

    x_past = x_lagged[:, :-1]  # past of signal x (lagged)

    y_present = np.expand_dims(y_lagged[:, -1], axis=1)  # current value of signal y (lagged)
    y_past = y_lagged[:, :-1]  # past of signal y (lagged)
    xy_past = np.concatenate((x_past, y_past), axis=1)  # both past concatenated

    reduced_model = np.concatenate((y_past, y_present), axis=1)  # y's past and current value
    full_model = np.concatenate((xy_past, y_present), axis=1)  # x and y's past and current y value

    # Covariances
    entr_reduced = entr(reduced_model.T) - entr(y_past.T)
    entr_full = entr(full_model.T) - entr(xy_past.T)

    # residual sum of squares
    RSS_reduced = np.exp(entr_reduced) 
    RSS_full = np.exp(entr_full) 

    sigma_reduced = RSS_reduced / (n_timesteps_regr - n_lags - 1)
    sigma_full = RSS_full / (n_timesteps_regr - 2 * n_lags - 1) 

    GC = math.log(sigma_reduced / sigma_full) # GC value
    Fstat = (n_timesteps_regr - 2 * n_lags - 1) / n_lags * (RSS_reduced - RSS_full) / RSS_full

    if Fstat > threshold_F :
        GC_sig = GC
    else:
        GC_sig = np.nan

    if verbose:
        print("F statistics:", Fstat)
        print("F threshold:", threshold_F)
        print("Significant GC values:", GC)

    return GC_sig, GC, Fstat, threshold_F


def cbvgc_2_signals(signal1, signal2, signal3, n_lags=3, pval=0.01, n_pairs=1, tau=1, verbose=False):
    """ 
        Calculate conditional bivariate Granger Causality between two neurons, from signal1 to signal2,
        conditioned on signal3. 

        :input signal1: trace of neuron 1 (n_timesteps,)
        :input signal2: trace of neuron 2 (n_timesteps,)
        :input signal3: trace of neuron 3 or other conditioning variable e.g. stimulus (n_timesteps,)
        :input n_lags: number of lags for GC
        :input pval: p-value for significance of link
        :input n_pairs: total number of pairs (= number of tests for significance) to Bonferroni correct the threshold
        :input tau: separation between lags for the embedding e.g. if tau=2 and n_lags=3: lag_signal uses n_lags past 
                timepoints separated by tau points i.e. signal at t = -5, -3, -1 
                --> keep tau=1 for GC: use first n_lags past points
        :input verbose (default False): print results if True

        gc is significant if fstat > threshold_f 
    
    """
    # single cBVGC value from neuron 1 to neuron 2 conditioned on signal3 = stimulus
    (n_rois, n_timesteps) = (3, len(signal1))
    n_timesteps_regr = n_timesteps - n_lags
    threshold_F = stats.f.ppf(1 - pval/n_pairs, n_lags, n_timesteps_regr - n_rois * n_lags - 1) 
   
    x_lagged = lag_signals([signal1], n_lags + 1, tau=1)[0]  # n_lags+1 --> n_lags + present
    y_lagged = lag_signals([signal2], n_lags + 1, tau=1)[0]
    z_lagged = lag_signals([signal3], n_lags + 1, tau=1)[0]
    
    x_past = x_lagged[:, :-1]  # past of signal x (lagged)

    y_present = np.expand_dims(y_lagged[:, -1], axis=1)  # current value of signal y (lagged)
    y_past = y_lagged[:, :-1]  # past of signal y (lagged)
    z_past = z_lagged[:, :-1] 
    
       
    yz_past = np.concatenate((y_past, z_past), axis=1)  # past without x
    xyz_past = np.concatenate((x_past, yz_past), axis=1)  # all past
    
    reduced_model = np.concatenate((y_present, yz_past), axis=1)  # past without x and current y value
    full_model = np.concatenate((xyz_past, y_present), axis=1)  # x, y, z's past and current y value
        
    # Covariances
    entr_reduced = entr(reduced_model.T) - entr(yz_past.T)
    entr_full = entr(full_model.T) - entr(xyz_past.T)

    # residual sum of squares
    RSS_reduced = np.exp(entr_reduced) 
    RSS_full = np.exp(entr_full) 

    sigma_reduced = RSS_reduced / (n_timesteps_regr - (n_rois - 1) * n_lags - 1)
    sigma_full = RSS_full / (n_timesteps_regr - n_rois * n_lags - 1) 

    GC = math.log(sigma_reduced / sigma_full) # GC value
    Fstat = (n_timesteps_regr - n_rois * n_lags - 1) / n_lags * (RSS_reduced - RSS_full) / RSS_full

    if Fstat > threshold_F:
        GC_sig = GC
    else:
        GC_sig = np.nan

    if verbose:
        print("F statistics:", Fstat)
        print("F threshold:", threshold_F)
        print("Significant GC values:", GC)

    return GC_sig, GC, Fstat, threshold_F


def mvgc_2_signals(signal1, signal2, Z, n_lags=3, pval=0.01, n_pairs=1, tau=1, verbose=False):
    """ 
        Caluclate multivariate Granger Causality between two neurons, from signal1 to signal2,
        conditioned on Z. 

        :input signal1: trace of neuron 1 (n_timesteps,)
        :input signal2: trace of neuron 2 (n_timesteps,)
        :input Z: traces of conditioning variables: other neurons, can include stimulus, etc. (n_z, n_timesteps)
        :input n_lags: number of lags for GC
        :input pval: p-value for significance of link
        :input n_pairs: total number of pairs (= number of tests for significance) to Bonferroni correct the threshold
        :input tau: separation between lags for the embedding e.g. if tau=2 and n_lags=3: lag_signal uses n_lags past 
                timepoints separated by tau points i.e. signal at t = -5, -3, -1 
                --> keep tau=1 for GC: use first n_lags past points
        :input verbose (default False): print results if True

        gc is significant if fstat > threshold_f 
    
    """
    # single MVGC value from neuron 1 to neuron 2 conditioned on Z = all other neurons 
    # (Z can also include stimulus for cMVGC)
    (n_rois, n_timesteps) = (2+len(Z), len(signal1))
    n_timesteps_regr = n_timesteps - n_lags
    threshold_F = stats.f.ppf(1 - pval/n_pairs, n_lags, n_timesteps_regr - n_rois * n_lags - 1)  # statistical threshold 

    x_lagged = lag_signals([signal1], n_lags + 1, tau=1)[0]  # n_lags+1 --> n_lags + present
    y_lagged = lag_signals([signal2], n_lags + 1, tau=1)[0]
    z_lagged = lag_signals(Z, n_lags + 1, tau=1)
    
    
    x_past = x_lagged[:, :-1]  # past of signal x (lagged)

    y_present = np.expand_dims(y_lagged[:, -1], axis=1)  # current value of signal y (lagged)
    y_past = y_lagged[:, :-1]  # past of signal y (lagged)
    z_past = np.concatenate(z_lagged[:, :, :-1], axis=1)
            
    yz_past = np.concatenate((y_past, z_past), axis=1)  # past without x
    xyz_past = np.concatenate((x_past, yz_past), axis=1)  # all past
    
    reduced_model = np.concatenate((y_present, yz_past), axis=1)  # past without x and current y value
    full_model = np.concatenate((xyz_past, y_present), axis=1)  # x, y, z's past and current y value
        
    # Covariances
    entr_reduced = entr(reduced_model.T) - entr(yz_past.T)
    entr_full = entr(full_model.T) - entr(xyz_past.T)

    # residual sum of squares
    RSS_reduced = np.exp(entr_reduced) 
    RSS_full = np.exp(entr_full) 

    sigma_reduced = RSS_reduced / (n_timesteps_regr - (n_rois - 1) * n_lags - 1 )
    sigma_full = RSS_full / (n_timesteps_regr - n_rois * n_lags - 1) 

    GC = math.log(sigma_reduced / sigma_full) # GC value
    Fstat = (n_timesteps_regr - n_rois * n_lags - 1) / n_lags * (RSS_reduced - RSS_full) / RSS_full

    if Fstat > threshold_F:
        GC_sig = GC
    else:
        GC_sig = np.nan

    if verbose:
        print("F statistics:", Fstat)
        print("F threshold:", threshold_F)
        print("Significant GC values:", GC)

    return GC_sig, GC, Fstat, threshold_F




def lag_signals(signals, n_lags, tau=1):
    """ 
        Create matrix of lagged data sequence signal (lag embedding). 
        Creates a matrix of dimension n_lags and tau lags.
      
        :input signals: matrix of the signals to be embedded
        :input n_lags: embedding dimension = np.shape(embedded_signal, 2)
        :input tau (default = 1): number of lags for the embedding 

        :output signals_lagged: matrix of embedded signals (shape: (n_rois, n_timesteps + tau - n_lags*tau, embed_dim))
    """
    
    (n_rois, n_timesteps) = np.shape(signals)
    signals_lagged = np.zeros((n_rois, n_timesteps + tau - n_lags * tau, n_lags))

    for i, x in enumerate(signals):
        signals_lagged[i] = np.array([x[np.arange(0, n_timesteps, tau)[:n_lags] + i]
                                        for i in range(n_timesteps + tau - n_lags * tau)])
    return signals_lagged



def entr(xy):
    """
        Entropy of a gaussian variable.
        This function computes the entropy of a gaussian variable for a 2D input.
    """
    # manually compute the covariance (faster)
    n_r, n_c = xy.shape
    xy = xy - xy.mean(axis=1, keepdims=True)
    out = np.empty((n_r, n_r), xy.dtype, order='C')
    np.dot(xy, xy.T, out=out)
    out /= (n_c - 1)
    # compute entropy using the slogdet in numpy rather than np.linalg.det
    # nb: the entropy is the logdet
    (sign, h) = np.linalg.slogdet(out)
    if not sign > 0:
        raise ValueError(f"Can't estimate the entropy properly of the input "
                         f"matrix of shape {xy.shape}. Try to increase the "
                         "step")
    return h


def get_angle(center_from, center_to, degree360=True):
    """ 0 = caudal to rostral """
    
    x_from = center_from[0]
    y_from = center_from[1]
    x_to = center_to[0]
    y_to = center_to[1]
    
#     angle = (math.atan2(y_to - y_from, x_to - x_from)) 
    angle = (math.atan2(y_to - y_from, x_to - x_from) + np.pi/2)  # to reorient 0 to top
    
    if angle >= np.pi:
         angle = angle - 2*np.pi
            
        
    if degree360:
        angle = angle * 180 / np.pi
        if angle < 0:
            angle = 360 + angle
        
    return angle


def get_phi(gc_sig, cell_centers, degree360=False):
    n_cells = len(cell_centers)
    phi_list = []
    
    for i in range(n_cells):
        for j in range(n_cells):
            if gc_sig[i,j] > 0:
                angle = get_angle(cell_centers[i], cell_centers[j], degree360)
                phi_list.append(angle)
                
    return phi_list


def flatten(l):
    return [item for sublist in l for item in sublist]


# ipsi-contra and rostro-caudal ratios functions (for motoneuron dataset)
    
# get ratio from df
def get_ratio(df, loc, sig=True, gc_type='dff', ratio_type='ipsi'):
    """ 
        Return either ipsi ratio or RC ratio, for all links or significant links only.
        Can choose on which GC matrix the ratio is calculated (default: dff).

        Ipsi-lateral ratio: 
        number of ipsi-lateral links / (number of ipsi-lateral links + number of contra-lateral links)

        Rostro-caudal ratio:
        number of rostral-to-caudal links / (number of rostral-to-caudal links + number of caudal-to-rostral links)

        Function specific to how the dataframe is built. For generic ratio calculation use:
            get_ratio_from_GC
        where only GC matrix and middle index are necessary.

        :input df: motoneuron dataframe with fish/trials as index (pandas DataFrame)
        :input loc: index of fish/trial to calculate the ratio for (int)
        :input sig: to calculate the ratio on significant links only or on all links (bool)
        :input gc_type: type of fluorescence trace used to calculate GC (string)
        :input ratio_type: ipsi-lateral or rostrocaudal (string)
    
    """
    fish = df.loc[loc].Fish
    trace = df.loc[loc].Trace
    mid = int(df.loc[loc].mid)
    
    if sig:  # Get ratio for significant links.
        if gc_type == 'raw':
            gc = df.loc[loc].GC_sig_raw
        elif gc_type == 'dt':
            gc = df.loc[loc].GC_sig_dt
        elif gc_type == 'f_smooth':
            gc = df.loc[loc].GC_sig_f_smooth
        elif gc_type == 'dfdt_smooth':
            gc = df.loc[loc].GC_sig_dfdt_smooth
        elif gc_type == 'disc_f':
            gc = df.loc[loc].GC_sig_disc_f
        else:
            if gc_type != 'dff':
                print('Unknown \'gc_type\' param: GC on DF/F is returned.')
            gc = df.loc[loc].GC_sig
        
    else:  # Get ratio for all links
        if gc_type == 'raw':
            gc = df.loc[loc].GC_raw
        elif gc_type == 'dt':
            gc = df.loc[loc].GC_dt
        elif gc_type == 'f_smooth':
            gc = df.loc[loc].GC_f_smooth
        elif gc_type == 'dfdt_smooth':
            gc = df.loc[loc].GC_dfdt_smooth
        elif gc_type == 'disc_f':
            gc = df.loc[loc].GC_disc_f
        else:
            if gc_type != 'dff':
                print('Unknown \'gc_type\' param: GC on DF/F is returned.')
            gc = df.loc[loc].GC
    
    
    return get_ratio_from_GC(gc, mid, ratio_type)

    


# get ratio from GC matrix
def get_ratio_from_GC(gc, mid, ratio_type='ipsi'):
    """
        Return either ipsi ratio or RC ratio, for GC matrix in param.

        Ipsi-lateral ratio: 
        number of ipsi-lateral links / (number of ipsi-lateral links + number of contra-lateral links)

        Rostro-caudal ratio:
        number of rostral-to-caudal links / (number of rostral-to-caudal links + number of caudal-to-rostral links)


        :input gc: Granger causality matrix, significant or not (NxN)
        :input mid: index of neuron separating left-right side (int). Note: neurons must be organized from
                     top left --> bottom left --> top right --> bottom right.
        :input ratio_type: 'ipsi' if ipsi-lateral, and 'RC' if rostrocaudal (string)
    """
    gc_ipsi_left = gc[:mid, :mid]
    gc_ipsi_right = gc[mid:, mid:]
    
    if ratio_type == 'RC':
        gc_RC_left = gc_ipsi_left[np.triu_indices(len(gc_ipsi_left), k=1)]
        gc_CR_left = gc_ipsi_left[np.tril_indices(len(gc_ipsi_left), k=-1)]
        gc_RC_right = gc_ipsi_right[np.triu_indices(len(gc_ipsi_right), k=1)]
        gc_CR_right = gc_ipsi_right[np.tril_indices(len(gc_ipsi_right), k=-1)]

        num_RC_left = len(gc_RC_left[gc_RC_left>0])
        num_CR_left = len(gc_CR_left[gc_CR_left>0])
        num_RC_right = len(gc_RC_right[gc_RC_right>0]) 
        num_CR_right = len(gc_CR_right[gc_CR_right>0]) 

        strength_RC_left = np.nansum(gc_RC_left)
        strength_CR_left = np.nansum(gc_CR_left)
        strength_RC_right = np.nansum(gc_RC_right)
        strength_CR_right = np.nansum(gc_CR_right)

        RC_number = num_RC_left + num_RC_right
        CR_number = num_CR_left + num_CR_right
        RC_strength = strength_RC_left + strength_RC_right
        CR_strength = strength_CR_left + strength_CR_right

        if RC_number == 0 and CR_number == 0:
            print("No links.")
            return np.nan, np.nan, np.nan
        else:
            if RC_number == 0:
                print("No R-->C links.")
                RC_mean_strength = 0
            else:
                RC_mean_strength = RC_strength / RC_number
                
            if CR_number == 0:
                print("No C-->R links.")
                CR_mean_strength = 0
            else:
                CR_mean_strength = CR_strength / CR_number
            
            ratio = RC_mean_strength / (RC_mean_strength + CR_mean_strength)
            return ratio
    
    else:
        if ratio_type != 'ipsi':
            print('Unknown \'ratio_type\' param: ipsi ratio is returned.')
        
        num_ipsi_left = len(gc_ipsi_left[gc_ipsi_left>0])
        num_ipsi_right = len(gc_ipsi_right[gc_ipsi_right>0]) 
        strength_ipsi_left = np.nansum(gc_ipsi_left)
        strength_ipsi_right = np.nansum(gc_ipsi_right)

        ipsi_number = num_ipsi_left + num_ipsi_right
        ipsi_strength = strength_ipsi_left + strength_ipsi_right

        gc_contra_from_left = gc[:mid, mid:]
        gc_contra_from_right = gc[mid:, :mid]
        num_contra_from_left = len(gc_contra_from_left[gc_contra_from_left>0])
        num_contra_from_right = len(gc_contra_from_right[gc_contra_from_right>0]) 
        strength_contra_from_left = np.nansum(gc_contra_from_left)
        strength_contra_from_right = np.nansum(gc_contra_from_right)

        contra_number = num_contra_from_left + num_contra_from_right
        contra_strength = strength_contra_from_left + strength_contra_from_right

        if ipsi_number == 0 and contra_number == 0:
            print("No links.")
            return np.nan, np.nan, np.nan
        else:
            if ipsi_number == 0:
                print("No ipsi links.")
                ipsi_mean_strength = 0
            else:
                ipsi_mean_strength = ipsi_strength / ipsi_number

            if contra_number == 0:
                print("No contra links.")
                contra_mean_strength = 0
            else:
                contra_mean_strength = contra_strength / contra_number

            ratio = ipsi_mean_strength / (ipsi_mean_strength + contra_mean_strength)
            return ratio