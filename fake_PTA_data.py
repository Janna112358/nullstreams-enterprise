#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 14:38:17 2018

@author: jgoldstein
"""

# try to get some simulated PTA data like in Jeff Hazboun's github https://github.com/Hazboun6/pta_simulations

import numpy as np
import glob, os
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 2.5 * 72

import astropy
from astropy.time import Time

import enterprise
from enterprise.pulsar import Pulsar

import enterprise_extensions
from enterprise_extensions import models, model_utils

import libstempo as T2, libstempo.toasim as LT, libstempo.plot as LP
from ephem import Ecliptic, Equatorial



# can use LT.fakepulsar to create a fake libstempo tempopulsar
# LT.fakepulsar(parfile, obstimes, toaerr, [optional params])

# first need to get par files
# downloaded IPTA DR2 in /Documents/data/DR2
# maybe try to use .par files from DR2/release/VersionB/... for a bunch of pulsars?

source = '/home/jgoldstein/Documents/data/DR2/release/VersionB'

def fake_obs_times(source, cadence=20):
    """
    For all pulsars in source, generate some fake observation times

    Read start and finish from the pulsar .par file. Then pick random times
    with a given average cadence (in days).
    
    Parameters
    source: str
        path to pulsars with .par files in 'pulsar'/'pulsar'.IPTADR2.par
    cadence: scalar
        default = 20
        average cadence (in days) for fake observations
        
    Returns
    -------
    list: 
        pulsar names
    NumPy array:
        observation times in MJD for each pulsar
    """
    pulsars = os.listdir(source)
    observation_times = []
    
    for p in pulsars:
        parfile = os.path.join(source, p, '{}.IPTADR2.par'.format(p))
        # read start and end of the observation from parfile, then get some random obs times
        with open(parfile) as parf:
            for line in parf:
                if 'START' in line:
                    start = float(line.split()[1])
                elif 'FINISH' in line:
                    finish = float(line.split()[1])
                    break
                
        num_obs = int((finish - start) / cadence)
        obs = np.sort(np.random.randint(start, high=finish, size=num_obs))
        observation_times.append(obs)
    
    return pulsars, observation_times
    
    
            
    