"""
Created on Jan 30, 2013

@author: andy
"""
from . import integrator
from .. import dynamics


class LangevinIntegrator(integrator.Integrator):
    def cg_step(self):
        """
        perform a forward euler step (the most idiotic and unstable of all possible integrators)
        to integrate the state variables.

        reads self.cg_velocities (or possibly self.cg_forces) to calculat the diffusion matrix,
        averages cg_forces, calculates D*f and notifies each subsystem with it's respective part
        of this vector to translate.

        X[n+1] = X[n] + dt*dX/dt[n], and dX/dt is D*f.
        """

        # forward euler
        # result = coordinate_ops + beta() * dt * Df
        Df = dynamics.diffusion_force(self.system)

        # Df is a column vector, change it to a row vector, as the subsystems
        # expect a length(ncg) row vector.
        Df = Df.transpose()

        # per euler step,
        # translate = dt * beta * dX/dt
        cg_translate = self.system.dt * self.system.beta * Df

        self.system.translate(cg_translate)
