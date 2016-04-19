'''
Created on January Jul 30, 2013

@author: Andrew Abi Mansour

The continuum field variable approach.

'''

import subsystems
import numpy as np
import os
import logging
import FieldVars as FV
import MDAnalysis as md

class ContinuumSubsystem(subsystems.SubSystem):
    """
    A set of CG variables.
    """
    def __init__(self, System, Sel, **args):
        """
        Create a continuum subsystem.
        The code is based on a C++-Python hybrid approach.
        
        @param system: an object (typically the system this subsystem belongs to)
        which has the following attributes:
        box: an array describing the periodic boundary conditions.

        note, the system MAY be None, such as when the config is created, so don't
        access it yet.
    
        @param NumNodes: number of nodes for the entire macromolecule
        @param Widths: a kernel matrix of widths of size N_atom x N_dim
        """
        
        self.System = System
        self.select = Sel
        self.CG_step = 0
        self.args = args
	self.Newton_Partial = args['NewtonPart']
	self.nNodes = args['NumNodes_x'] * args['NumNodes_y'] * args['NumNodes_z']
        
    def __del__(self):
    	try:
            del self.COMM
    	except:
	    pass

    def NumNodes(self):
        return self.CppFV.GetAdjNumNodes()
        
    def universe_changed(self, universe):
    	""" 
        universe changed, so select our atom group
        """

	self.atoms = universe.selectAtoms(self.select)

	if self.atoms.masses.min() == 0:
		self.atoms.set_masses(39.948)

        # check to see if atoms is valid
        if len(self.atoms) <= 0:
            raise ValueError("The select statement '{}' is syntactically valid but returned a zero sized AtomGroup".format( \
                self.select))

	self.universe = universe

        self.CppFV = FV.FieldVar(self.atoms.positions, self.atoms.masses, self.args)
        self.COMM = self.CppFV.GetCOMM()

    def translate(self, dCG):
        """ 
	translates the atomic positions from a given vectory of dCG positions,
        where dCG is a finite change in the CG velocities times dt,
        and then adds the residuals for better accuracy.

        @param CG: a length N_cg 1D array.
        """

        logging.info('translating continuum SS ..')
        self.atoms.positions = self.FineScale(self.atoms.positions, self.CG +  dCG[:self.NumNodes()])

	print 'COM of protein is {}'.format(self.atoms.centerOfMass())

    def frame(self):
        """ 
        undocumented ....
        @param pos, vel, force
        """
        pos, vel, forces = self.atoms.positions, self.atoms.velocities, self.atoms.forces
	
	if self.NumNodes() < self.nNodes:

		CG  = np.zeros(self.nNodes)
		CGv = np.zeros(self.nNodes)
		CGf = np.zeros(self.nNodes)

		CG[:self.NumNodes()], CGv[:self.NumNodes()], CGf[:self.NumNodes()] = self.ComputeCG_Pos(pos), self.ComputeCG_Mom(pos, vel), self.ComputeCG_For(pos, vel, forces)		
	else:
	        CG, CGv, CGf = self.ComputeCG_Pos(pos), self.ComputeCG_Mom(pos, vel), self.ComputeCG_For(pos, vel, forces)

	return CG, CGv, CGf
                
    def minimized(self):
        pass
    
    def equilibrated(self):
        """
        this is called just after the structure is equilibriated, this is the starting struct
        for the MD runs, this is to calculate basis.
        """
        logging.info('equilibrating ss ...')
        
        if self.CG_step%self.CppFV.GetFreqUpdate() == 0:

	    if self.CG_step >= 1:
		logging.info('Updating grid ...')
            	self.PetscError = self.CppFV.Py_UpdateGrid(self.atoms.positions)
	    else:

		self.box = self.atoms.bbox()

            	# Expand the box by +/- 10% of its size
            	self.box[0,:] -= self.box[0,:] * 0.1
            	self.box[1,:] += self.box[1,:] * 0.1

            	self.CppFV.SetBox(self.box)
		self.PetscError = self.CoarseScale(self.atoms.positions)

	    self.Assemble = True
	else:
	    self.Assemble = False

	self.CG = self.ComputeCG_Pos(self.atoms.positions)
	self.CG_step += 1

    def md(self):
        """
        This is called right after the MD trajectories are processed
        """
        
    def ComputeCG_Pos(self, pos):
        return self.CppFV.Py_ComputeCG_Pos(pos, self.CppFV.GetAdjNumNodes())
        
    def ComputeCG_Mom(self, pos, vel):
        return self.CppFV.Py_ComputeCG_Vel(pos, vel, self.CppFV.GetAdjNumNodes())
    
    def ComputeCG_For(self, pos, vel, forces):
        return self.CppFV.Py_ComputeCG_For(pos, vel, forces, self.CppFV.GetAdjNumNodes())
    
    def CoarseScale(self, Coords):
        return self.CppFV.Py_CoarseGrain(Coords)
    
    def FineScaleVelo(self, r, Vels, Mom):
        logging.info('fine graining ...')

        Mom = Mom[:self.NumNodes()]

        self.PetscError, Vels = self.CppFV.Py_FineGrainMom(Mom.copy(), Mom.copy(), Mom.copy(), Vels[:,0], Vels[:,1], Vels[:,2], r, self.atoms.n_atoms * Vels.shape[1])
        Vels = np.reshape(Vels, (Vels.shape[0]/3, 3))

	return Vels

    def FineScale(self, Coords, CG):
	logging.info('fine graining ...')

	CG = CG[:self.NumNodes()]

        self.PetscError, Coords = self.CppFV.Py_FineGrain(CG, Coords[:,0], Coords[:,1], Coords[:,2], self.atoms.n_atoms * Coords.shape[1], self.Assemble)

        return np.reshape(Coords, (Coords.shape[0]/3, 3))

def ContinuumSubsystemFactory(system, selects, **args):
    """
    create a list of ContinuumSubsystems.

    @param system: the system that the subsystem belongs to, this may be None
                   when the simulation file is created. 
    @param selects: A list of MDAnalysis selection strings, one for each
                    subsystem. 
    @param args: a list of variable length. See below.
    """
    
    # Default [optional] args
    if 'Tol' not in args:
        args['Tol'] = 0.01
    
    if 'Scaling' not in args:
        args['Scaling'] = 1.0
        
    if 'Extend' not in args:
        args['Extend'] = False
        
    if 'NewtonPart' not in args:
        args['NewtonPart'] = 1
        
    if 'Threshold' not in args:
        args['Threshold'] = 10
        
    if 'FreqUpdate' not in args:
        args['FreqUpdate'] = 100
    
    # Check for mandatory args
    if 'NumNodes_x' in args and 'NumNodes_y' in args and 'NumNodes_z' in args and 'Resolution' in args:
        pass
    else:
        raise ValueError("invalid subsystem args")
            
    # test to see if the generated selects work
    [system.universe.selectAtoms(select) for select in selects]

    return (args['NumNodes_x'] * args['NumNodes_y'] * args['NumNodes_z'], [ContinuumSubsystem(system, select, **args) for select in selects])
        
        

