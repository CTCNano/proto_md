"""
Created on Jan 6, 2013

@author: andy

Functions to calculate dynamic properties of cg variables. 
Most functions here can take either a System or Timestep as arguments. 
This is convienient for analysis/debugging purposes
"""

from . import diffusion
from numpy import mean, dot, newaxis


def mean_force(obj):
    # 1st makes a n_subsystem x n_cg matrix, then flattens it (row major),
    # finally, [:,newaxis] reshapes it to a column vector.
    # so ordering of flattend array is [[subsystem.0.cgs, subsystem.1.cgs, ..., subsystem.n.cgs]].
    return mean(mean(obj.cg_forces, axis=0), axis=1).flatten()[:, newaxis]


def diffusion_force(obj):

    D = diffusion.diff_from_vel(obj)

    # self.cg_forces: nensembe x n_subsystem x n_step x n_cg
    # averate over all ensembles, md timesteps,
    # calculate a (n_subsystem x n_cg) x 1 column vector
    f = mean_force(obj)

    return dot(D, f)
