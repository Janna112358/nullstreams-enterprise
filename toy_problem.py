#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 15:49:12 2019

@author: jgoldstein

inspace toy problem
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import periodogram

from inspace.interpolation import get_target_times, sinc_interpolation
from inspace.gp_george import gp_estimate, plot_gp_stuff

# using time units of days
YEAR = 365.25

def example_signal(t, f0):
    return np.sin(2*np.pi*f0*t)

def make_example_data(ti=0, T=15*YEAR, obs_step=20, f_signal=1./YEAR, yerr=0.1, seed=None):
    ## Time window / frequency range values etc
    tf = ti + T
    
    ## Construct random unevenly sampled observation times and signal
    n_obs = int(T / obs_step)
    # choose obs times randomly from oversampled times
    many_times = np.linspace(ti, tf, num=50*n_obs)
    if seed is not None:
        np.random.seed(seed)
    obs_times = np.sort(np.random.choice(many_times, size=(n_obs), replace=False))
    obs_signal = example_signal(obs_times, f_signal)
    obs_sn = obs_signal + yerr * np.random.randn(n_obs) # signal + noise
    
    return obs_times, obs_signal, obs_sn

def toy_problem(ti=0, T=15*YEAR, obs_step=20, yerr=0.0, seed=None):    
    tf = ti + T
    fmin = 1/T
    # so that we can easily see the signal in the FT, 
    #take an integer multiple of fmin for the signal frequency
    fGW = 5 * (1/T)
    fmax = 10 * fGW
    obs_times, obs_signal, obs_sn = make_example_data(ti=ti, T=T, 
                        obs_step=obs_step, f_signal=fGW, yerr=yerr, seed=seed)
    target_times = get_target_times(ti, tf, fmax)
    
    ## Use gp to estimate signal at target times
    gp_nonoise, cov_nn = gp_estimate(target_times, obs_times, obs_signal, 0.0, kernel_value=T)
    gp_wnoise, cov = gp_estimate(target_times, obs_times, obs_sn, yerr, kernel_value=T)
    
    ## Use sinc interpolation to estimate signal at target times
    # time scale for sinc interpolation (based on fmax)
    TNy = 1 / (2*fmax)
    sinc_nonoise = sinc_interpolation(target_times, obs_times, obs_signal, TNy)
    sinc_wnoise = sinc_interpolation(target_times, obs_times, obs_sn, TNy)
    
    # actual signal at target times for comparison
    true_signal = example_signal(target_times, fGW)
    
    fig = plt.figure(figsize=(10, 6))
    ax1 = fig.add_subplot(121)
    ax1.set_xlabel('time')
    ax1.errorbar(obs_times, obs_sn, yerr=yerr, fmt='k.', linestyle='-', label='data')
    ax1.plot(target_times, true_signal, c='gray', label='true signal')
    ax1.plot(target_times, gp_nonoise, 'c--', label='gp no noise')#marker='^')
    ax1.plot(target_times, gp_wnoise, 'b-.', label='gp with noise')#marker='v')
    ax1.plot(target_times, sinc_nonoise, 'm--', label='sinc no noise')#marker='<')
    ax1.plot(target_times, sinc_wnoise, 'r-.', label='sinc with noise')# marker='>')
    ax1.legend(loc='best')
    
    ax2 = fig.add_subplot(122)
    ax2.set_xlabel('frequency')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_ylim(1e-4, 1e4)
    obs_fs = 1/obs_step
    target_fs = 1/(target_times[1] - target_times[0])
    ax2.plot(*periodogram(obs_sn, fs=obs_fs), c='k', label='data')
    ax2.plot(*periodogram(true_signal, fs=target_fs), c='gray', label='true signal')
    ax2.plot(*periodogram(gp_nonoise, fs=target_fs), 'c--', label='gp no noise')#marker='^')
    ax2.plot(*periodogram(gp_wnoise, fs=target_fs), 'b-.', label='gp with noise')#marker='v')
    ax2.plot(*periodogram(sinc_nonoise, fs=target_fs), 'm--', label='sinc no noise')#marker='<')
    ax2.plot(*periodogram(sinc_wnoise, fs=target_fs), 'r-.', label='sinc with noise')# marker='>')

    ax2.legend(loc='best')
    
    fig.tight_layout()
    fig.savefig('/home/jgoldstein/Documents/projects/pta-nullstreams/interpolation_plot.pdf')
    
#    fig2 = plt.figure()
#    ax = fig2.add_subplot(121)
#    ax.plot(freqs, ft_true, 'b-', label='true signal')
#    ax.plot(freqs, ft_gp_nonoise, 'm-.', label='gp no noise')
#    ax.plot(freqs, ft_gp_wnoise, 'k--', label='gp w/ noise')
#    ymin, ymax = ax.get_ylim()
#    ax.vlines(fGW, ymin, ymax, color='g', zorder=0, label='fGW')
#    ax.set_ylim(ymin, ymax)
#    ax.set_xlabel('frequency')
#    ax.legend(loc='best')
#    ax.set_title('rfft')
#    
#    ax2 = fig2.add_subplot(122)
#    ax2.set_yscale('log')
#    ft_true_sq = ft_true * np.conj(ft_true)
#    ft_gp_nonoise_sq = ft_gp_nonoise * np.conj(ft_gp_nonoise)
#    ft_gp_wnoise_sq = ft_gp_wnoise * np.conj(ft_gp_wnoise)
#    ax2.plot(freqs, ft_true_sq, 'b-', label='true signal')
#    ax2.plot(freqs, ft_gp_nonoise_sq, 'm-.', label='gp no noise')
#    ax2.plot(freqs, ft_gp_wnoise_sq, 'k--', label='gp w/ noise')
#    ymin, ymax = ax2.get_ylim()
#    ax2.vlines(fGW, ymin, ymax, color='g', zorder=0, label='fGW')
#    new_ymin = min(ft_true_sq[1:]) / 10
#    ax2.set_ylim(new_ymin, ymax)
#    ax2.set_xlabel('frequency')
#    ax2.legend(loc='best')
#    ax2.set_title('$|\mathrm{rfft}|^2$')
#    
#    fig2.tight_layout()
#    
#    fig1.savefig('/home/jgoldstein/Documents/GP_fig1.pdf')
#    fig2.savefig('/home/jgoldstein/Documents/GP_fig2.pdf')
    
    
    