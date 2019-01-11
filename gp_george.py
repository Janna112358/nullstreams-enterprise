#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 2018

@author: jgoldstein

example: http://dfm.io/george/current/user/quickstart


PTA data problem:
    PTA data consists of a set of TOA residuals for each pulsar, observed at
    unevenly sampled times over a time span of ~5-15 years. Usually these data
    are analysed in the time domain. However, if we want to implement the null
    stream method, we need data points with either each pulsar observation at 
    the same times, or at the same frequencies. Since the time spans do not all
    overlap (so time coincident measurements would have to be extrapolated), 
    I want to try this in the frequency domain. So, the problem is to get a
    frequency domain representation of unevenly sampled data, at given 
    evenly sampled frequencies (so they can be the same for each pulsar).
    
    The current plan to do this is as follows (but no idea whether this is a 
    good idea): Estimate (interpolate) TOA residuals at a set of evenly sampled
    times, so that the fft of these points gives the desired FD representation.
    We can use a Gaussian Process (GP) to do this estimate. The GP assumes a
    prior distribution (set by the kernel/covariance function) of functions. We
    then provide a training set of the unevenly sampled data points (with some 
    error), which is used as a condition for the GP's functions (they have to
    go through the given points). We obtain a posterior distribution of
    functions which can be used to predict a given set of target points (the 
    evenly sampled points). Any such (finite) set of target points follows a
    join gaussian distribution for which we can compute the covariance matrix.
    
    Note on the frequency range we can use: Since the true Nyquist frequency 
    of the unevenly sampled times is practically infinite (>> PTA frequencies 
    of interest), it's set either by the integration time of the observations
    (1/Tint ~ 1/30 minutes = 5.5e-4 Hz) or we can choose to cut if off based
    on our astrophysical prior (fGW < 1e-6 Hz or something). This then gives us
    the desired time step for our target, evenly sampled times dt = 2/fmax. The
    minimum frequency is determined by the total time span fmin=1/T, which we
    can't really change anyway.
    
    Discussion points:
        - Is the output of the GP suitable for our analysis? I.e. a set of 
        points with a join guassian distribution.
        - The GP is clearly not the best method for the toy problem, since we 
        know we're sampling a sinusoid there (fitting a sinusoid would be a lot
        better). How does this compare to the actual PTA data?
        - Computational feasability?
        - The GP requires a choice of kernal (= covariance function for the 
        prior of the GP) and a value (length scale?). What should we use?
        
    Notes:
        - compare with "normal" interpolation (e.g. cubic spline)
        - spline interpolation: spline as reverse fft of frequency pass
            something like this done in LISA to get exact time delays for 
            TDI calculation (?)
        
"""

import numpy as np
import george
from george.kernels import ExpSquaredKernel
import matplotlib.pyplot as plt
from jannasutils import isIterable

def george_example(seed=None, ndata=10):
    if seed is not None:
        np.random.seed(seed)
    
    # Generate some fake noisy data.
    x = 10 * np.sort(np.random.rand(ndata))
    yerr = 0.2 * np.ones_like(x)
    y = np.sin(x) + yerr * np.random.randn(len(x))

    # Set up the Gaussian process.
    kernel = ExpSquaredKernel(1.0)
    gp = george.GP(kernel)

    # Pre-compute the factorization of the matrix.
    gp.compute(x, yerr)
    
    # Compute the log likelihood.
    print(gp.lnlikelihood(y))

    t = np.linspace(0, 10, 500)
    mu, cov = gp.predict(y, t)
    #std = np.sqrt(np.diag(cov))
    
    realy = np.sin(t)
    fig = plot_gp_stuff(t, mu, cov, realy, x, y, yerr)
    return fig

def plot_gp_stuff(x_data, y_data, y_err, x_gp, y_gp, cov, true_signal,
                  x_name='input variable', y_name='output variable'):
    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(121)
    if true_signal is not None:
        ax.plot(x_gp, true_signal, 'b', alpha=0.5, label='signal')
    ax.plot(x_gp, y_gp, 'k--', label='predicted')
    std = np.sqrt(np.diag(cov))
    ax.plot(x_gp, y_gp + std, 'k:')
    ax.plot(x_gp, y_gp - std, 'k:')
    ax.fill_between(x_gp, y_gp-std, y_gp+std, color='k', alpha=0.2)
    ax.errorbar(x_data, y_data, yerr=y_err, fmt='r.', label='data')
    ax.legend(loc='best')
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    ax.set_title('GP data and estimate')
    
    ax2 = fig.add_subplot(122)
    im = ax2.imshow(cov.T)
    plt.colorbar(im, ax=ax2)
    ax2.set_title('GP covariance matrix')
    
    fig.tight_layout()
    return fig


def gp_estimate(x, x_data, y_data, y_err, kernel_value=1.0):
    """
    Use a simple george GP (exponential-squared kernel) to estimate values 
    at points x given data at points (x_data, y_data).
    
    Parameters
    ----------
    x: NumPy Array
        x-coordinates to get predicted values at
    x_data: NumPy Array
        x-coordiantes of the data points
    y_data: NumPy Array
        y-coordinates of the data points (same length as x_data)
    y_err: float NumPy Array
        error on the y values of the data points
        if float, use the same error for all points
        if array, needs to be the same length as x_data and y_data
        
    Returns
    -------
    NumPy Array:
        predicted values (same shape as x)
    NumPy Array:
        covaraince matrix on the predicted points (NxN if x has N points)
    """
    if not isIterable(y_err):
        y_err = np.full_like(y_data, y_err)
    
    kernel = ExpSquaredKernel(kernel_value)
    gp = george.GP(kernel)
    gp.compute(x_data, y_err)
    
    print('GP log likelihood is {}'.format(gp.lnlikelihood(y_data)))
    
    mu, cov = gp.predict(y_data, x)
    return mu, cov

def round_to_p2(x):
    if x < 1:
        raise ValueError('Value must be greater or equal to 1')
    p = np.log2(x)
    return int(2 ** np.ceil(p))

def get_target_times(ti, tf, fmax=0.1):
    """
    Calculate desired target times given initial time, final time and
    maximum frequency.
    
    Time step is calculated using fmax, then the number of points is rounded up
    to get a power of two.
    
    Parameters
    ----------
    ti, tf: float
        initial and final time (days)
    fmax: float
        maximum frequency used to determine time step
        default = 0.1 (days^-1)
    """
    T = tf - ti
    # target time step is given by desired fmax
    # adjust to get a power of two for the number of samples
    Dt_try = 2 * (1/fmax)
    n = round_to_p2(T / Dt_try + 1)
    #new time step will be Dt = T / (n - 1)
    target_times = np.linspace(ti, tf, num=n, endpoint=True)
    return target_times    

def example_signal(t, f0):
    return np.sin(2*np.pi*f0*t)

def toy_problem(yerr=0.0, seed=None):
    ## Time window / frequency range values etc
    t0 = 0
    t1 = 15 * 365.25 # 15 years
    T = t1 - t0
    fmin = 1/T
    # so that we can easily see the signal in the FT, take an integer multiple of fmin
    fGW = 5 * (1/T)
    fmax = 0.1
    # roughly 20 days between observation (this is also about the cadence to get to fmax)
    obs_step = 20 
    
    ## Construct random unevenly sampled observation times and signal
    n_obs = int(T / obs_step)
    # choose obs times randomly from oversampled times
    many_times = np.linspace(t0, t1, num=50*n_obs)
    if seed is not None:
        np.random.seed(seed)
    obs_times = np.sort(np.random.choice(many_times, size=(n_obs), replace=False))
    obs_signal = example_signal(obs_times, fGW)
    obs_sn = obs_signal + yerr * np.random.randn(n_obs) # signal + noise
    
    ## Use gp to estimate signal at target times
    target_times = get_target_times(t0, t1, fmax)
    gp_nonoise, cov_nn = gp_estimate(target_times, obs_times, obs_signal, 0.0, kernel_value=T)
    gp_wnoise, cov = gp_estimate(target_times, obs_times, obs_sn, yerr, kernel_value=T)
    
    # signal at target times for comparison
    true_signal = example_signal(target_times, fGW)
    fig1 = plot_gp_stuff(obs_times, obs_signal, yerr, target_times, gp_wnoise, 
                         cov, true_signal, x_name='time', y_name='TOA residual')
    
    ## FFT of gp estimated points, compare with densesly sampled times and true fGW
    ft_gp_nonoise = np.fft.rfft(gp_nonoise)
    ft_gp_wnoise= np.fft.rfft(gp_wnoise)
    ft_true = np.fft.rfft(true_signal)
    freqs = np.fft.rfftfreq(n=len(target_times), d=target_times[1] - target_times[0])
    
    fig2 = plt.figure()
    ax = fig2.add_subplot(121)
    ax.plot(freqs, ft_true, 'b-', label='true signal')
    ax.plot(freqs, ft_gp_nonoise, 'm-.', label='gp no noise')
    ax.plot(freqs, ft_gp_wnoise, 'k--', label='gp w/ noise')
    ymin, ymax = ax.get_ylim()
    ax.vlines(fGW, ymin, ymax, color='g', zorder=0, label='fGW')
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel('frequency')
    ax.legend(loc='best')
    ax.set_title('rfft')
    
    ax2 = fig2.add_subplot(122)
    ax2.set_yscale('log')
    ft_true_sq = ft_true * np.conj(ft_true)
    ft_gp_nonoise_sq = ft_gp_nonoise * np.conj(ft_gp_nonoise)
    ft_gp_wnoise_sq = ft_gp_wnoise * np.conj(ft_gp_wnoise)
    ax2.plot(freqs, ft_true_sq, 'b-', label='true signal')
    ax2.plot(freqs, ft_gp_nonoise_sq, 'm-.', label='gp no noise')
    ax2.plot(freqs, ft_gp_wnoise_sq, 'k--', label='gp w/ noise')
    ymin, ymax = ax2.get_ylim()
    ax2.vlines(fGW, ymin, ymax, color='g', zorder=0, label='fGW')
    new_ymin = min(ft_true_sq[1:]) / 10
    ax2.set_ylim(new_ymin, ymax)
    ax2.set_xlabel('frequency')
    ax2.legend(loc='best')
    ax2.set_title('$|\mathrm{rfft}|^2$')
    
    fig2.tight_layout()
    
    fig1.savefig('/home/jgoldstein/Documents/GP_fig1.pdf')
    fig2.savefig('/home/jgoldstein/Documents/GP_fig2.pdf')
    return fig1, fig2
    
    
    