'''
Created on Jan 30, 2013

@author: andy
'''

import logging

class Integrator(object):
    def __init__(self, system, *args):
        self.system = system
        self.args = args
	self.hist_steps = 1
	self.RHS = [None] * self.hist_steps

    def run(self):
        """
        Run the simulation for as many timesteps as specified by the configuration.
        
        This will automatically start at the last completed timestep and proceed untill
        all the specified timesteps are completed.
        """
        last = self.system.last_timestep
        start = last.timestep + 1 if last else 0
        del last
        end = self.system.cg_steps
        
        logging.info("running timesteps {} to {}".format(start, end))
        
        for _ in range(start, end):
            self.system.begin_timestep()
            self.atomistic_step()
            self.cg_step()
            self.system.end_timestep()
            
        logging.info("completed all {} timesteps".format(end-start))
        
    def cg_step(self):
        raise NotImplementedError
    
    def atomistic_step(self):
        """
        Performs a timestep on the atomistic scale using system.universe object as the 
        starting state. 
        
        This entails vacuum minimizing, optionally solvating and solvent minimizing, 
        equilibriating and finally running a series of MD runs to populate the
        cg_positions, cg_forces and cg_velocities state variables, and 
        saving these values to the current timestep. 
        
        Different types of integrators may want to override this method, as they 
        may not have need of equilibration, minimization, and so forth. 
        
        This method requires a current_timestep, as such, it must be called between
        begin_timestep and end_timestep.
        
        @precondition: self.system.universe contains a starting atomic structure, and the
        current_timestep exists. 
        
        @postcondition: self.system.universe contains a equilibriated structure, 
        cg_positions, cg_forces and cg_velocities are populated with md values, 
        and these values are saved to the current_timestep. 
        """

	# save velocities before energy min
	velo = self.system.universe.atoms.velocities.copy()

        # first minimize in vacuum, in either case, 
        # fixes problems with langevin bond deformation.
        self.system.minimize()

	self.system.universe.trajectory.ts.has_velocities = True
	self.system.universe.atoms.set_velocities(velo)
        
        if self.system.should_solvate:
            with self.system.solvate() as sol:
                with self.system.minimize(**sol) as mn:
                    with self.system.equilibrate(**mn) as eq:
                        self.system.md(**eq)
        else:
            self.system.equilibrate()
            self.system.md()
