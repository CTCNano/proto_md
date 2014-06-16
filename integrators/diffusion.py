'''
@author: Andrew
Created on Jan 6, 2013
Last modified Feb 17, 2014
'''

import numpy as n
from numpy.fft import fft, ifft, fftshift
import correlation as corr

def diffusion(obj):
    """
    @param obj: An object which has cg_positions, cg_velocities and cg_forces attribites. 
                This is very convienient as both the System and the Timestep objects
                have these attributes. 
    @return: an n x n of diffusion coefficients, where n_cg is the dimensionality
    of given velocity (or force) arrays.
    
    
   
    """
    
    # 


    cg_shape = obj.cg_positions.shape
    return n.diagflat(n.ones(cg_shape[1]*cg_shape[3])*stokes(obj.temperature, r=5))

def diff_from_vel(obj):
    """
    @param src: nensemble * nsubsystem * nframe * ncg
    @return: the diffusion tensor
    """
    avg_velocities = n.mean(obj.cg_velocities, axis = 0)
    Ncg = avg_velocities.shape[2]
    dt = obj.dt

    dtensor = n.zeros((avg_velocities.shape[0]*Ncg, avg_velocities.shape[0]*Ncg), 'f')
    
    for ss in n.arange(avg_velocities.shape[0]):
        for cg in n.arange(Ncg):
            dtensor[ss*Ncg + cg, ss*Ncg + cg] = diff_coef_from_corr(avg_velocities[ss,:,cg],avg_velocities[ss,:,cg],dt)

    return dtensor

def diff_coef_from_corr(vi, vj, dt, int_points = 4):
    """
    calculate the diffusion coefficient for a pair of time series vi, vj
    @param vi: an n * nt array of time series
    @param vj: an n * nt array of time series
    sampled at interval dt
    """
    
    # the velocity correlation function for a time average.
    corr = n.min(n.array([vi.shape[1],vj.shape[1]]))

    for i in n.arange(vi.shape[0]):
        corr += corr.FFT_Correlation(vi[i,:],vj[i,:])
        
    # average correlation func        
    corr /= float(vi.shape[0])      
    
    # inaccurate integration (for only 4 points)  
    # Best to use orthogonal polynomials for fitting
    # the ACs, but for now keep this for comparison
    # with snw. 
    return n.trapz(corr[:int_points],dx=dt)
