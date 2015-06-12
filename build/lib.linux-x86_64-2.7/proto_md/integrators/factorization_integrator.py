'''
Created on Jan 30, 2013

@author: andy
'''

import integrator
import proto_md.dynamics as dynamics
import numpy as np

class FactorizationIntegrator(integrator.Integrator):
    def cg_step(self):
        """
        perform a forward euler step (the most idiotic and unstable of all possible integrators)
        to integrate the state variables. 
        
        reads self.cg_momenta to advance the cg state variables. 
        
        X[n+1] = X[n] + dt*dX/dt[n], and dX/dt is cg_moment/cg_mass.
        """
        
        # forward euler 
        # cg_velocities: nensemble x n subsystem x n_step x n_cg
        
        # inner mean: average over ensembles - results in a  n subsystem x n_step x n_cg array
        # next mean: average over steps - results in a n subsystem x n_cg array
        # flatten(): makes a 1D array 
        # [:,np.newaxis]: makes a 1xN column vector.
        avg_velocities = np.mean(np.mean(self.system.cg_velocities, axis = 0), axis = 1).flatten()[:,np.newaxis]
        self.RHS.append(avg_velocities)
        self.RHS.pop(0)

        cg_translate = np.zeros(avg_velocities.shape)

        for rhs in self.RHS:
                if rhs is not None:
                        cg_translate += rhs

        self.system.translate(cg_translate)
