'''
Created on January 27, 2013

@author: Andrew-AM

The Space warping type of coarse grained variable.

'''

import subsystems
import numpy as np
from scipy.special import legendre
from scipy.linalg import qr
from numpy.linalg import norm, solve
import logging

class SpaceWarpingSubsystem(subsystems.SubSystem):
    """
    A set of CG variables.
    """
    def __init__(self, system, pindices, select, freq):
        """
        Create a legendre subsystem.
        @param system: an object (typically the system this subsystem belongs to)
        which has the following attributes:
        box: an array describing the periodic boundary conditions.

        note, the system MAY be None, such as when the config is created, so don't
        access it yet.

        @param pindices: a N*3 array of Legendre polynomial indices.
        @param select: a select string used for Universe.selectAtoms which selects
        the atoms that make up this subsystem.
        """
        self.system = system

        # select statement to get atom group
        self.select = select

        # polynomial indices, N_cg x 3 matrix.
        self.pindices = pindices
	self.numnodes = pindices.shape[0] * pindices.shape[1]

        # we need to be called with a valid universe
        self.universe_changed(system.universe)
        
        self.CG_step = 0
        
        # How often should the reference struct be updated
        self.Freq_Update = freq

        logging.info("created LegendreSubsystem, pindices: {}, select: {}".
                     format(pindices, select))

    def NumNodes(self):
	return self.numnodes

    def universe_changed(self, universe):
        """
        universe changed, so select our atom group
        """
        self.atoms = universe.selectAtoms(self.select)

        # check to see if atoms is valid
        if len(self.atoms) <= 0:
            raise ValueError( \
                "The select statement '{}' is syntactically valid but returned a zero sized AtomGroup".format( \
                self.select))

        # call some stuff on the atomGroup to see if its valid
        self.atoms.bbox()

    def frame(self):
        """
        Returns a 3 tuple of CG variables, each one is a row vector of size n_cg
        """

        CG = self.ComputeCG(self.atoms.positions)
        CG_Vel = self.ComputeCG_Vel(self.atoms.velocities())
        CG_For = self.ComputeCG_Forces(self.atoms.forces)

        CG = np.reshape(CG.T,(CG.shape[0]*CG.shape[1]))
        CG_Vel = np.reshape(CG_Vel.T,(CG_Vel.shape[0]*CG_Vel.shape[1]))
        CG_For = np.reshape(CG_For.T,(CG_For.shape[0]*CG_For.shape[1]))

        return (CG,CG_Vel,CG_For)

    def translate(self, dCG):
        """
        translates the atomic positions from a given vectory of dCG positions,
        where dCG is a finite change in the CG positions over a short increment \Delta t:

	r(t + \Delta) = r(t) + U * dCG + O(\Delta^2)
 
        @param CG: a length N_cg 1D array.
        """
        self.atoms.positions += self.ComputeCGInv(dCG)

    def minimized(self):
        pass

    def equilibriated(self):
        """
        This is called just after the structure is equilibriated, this is the starting struct
        for the MD runs, this is to calculate basis.
        """
        if self.CG_step%self.Freq_Update == 0:
	    logging.info('Updating ref structure and constructing new basis functions...')
	    boxboundary = self.atoms.bbox()
	    self.box = (boxboundary[1,:] - boxboundary[0,:]) * 1.1 # 110% of the macromolecular box

            self.basis = self.Construct_Basis(self.atoms.positions - self.atoms.centerOfMass())
	    self.ref_coords = self.atoms.positions.copy()
	    self.ref_com = self.atoms.centerOfMass()

    def md(self):
	"""
	this is called right after MD trajectories are processed.
	"""
        self.CG_step += 1

    def ComputeCGInv(self,CG):
        """
        Computes atomic positions from CG positions
        Using the simplest scheme for now
        @param CG: 3*n_cg x 1 array
        @return: a n_atom x 3array
        """
        NCG = CG.shape[0]/3

        x = np.dot(self.basis,CG[:NCG])
        y = np.dot(self.basis,CG[NCG:2*NCG])
        z = np.dot(self.basis,CG[2*NCG:3*NCG])

	return np.array([x,y,z]).T

    def ComputeCG(self,var):
        """
        Computes CG momenta or positions
        CG = U^t * Mass * var
        var could be atomic positions or velocities
        """
        Utw = self.basis.T * self.atoms.masses()

        CG = solve(np.dot(Utw, self.basis), np.dot(Utw,var - self.ref_coords))

	return CG
        
    def ComputeCG_Vel(self,vel):
        """
        Computes CG momenta or positions
        CG = U^t * Mass * var
        var could be atomic positions or velocities
        """

        Utw = self.basis.T * self.atoms.masses()	
        Pi = solve(np.dot(Utw, self.basis), np.dot(Utw,vel))

	return Pi

    def ComputeCG_Forces(self, atomic_forces):
        """
        Computes CG forces = U^t * <f>
        for an ensemble average atomic force <f>
        """
	Utw = self.basis.T * self.atoms.masses()

        return solve(np.dot(Utw, self.basis), np.dot(Utw,atomic_forces))

    def Construct_Basis(self,coords):
        """
        Constructs a matrix of orthonormalized legendre basis functions
        of size Natoms x NCG.
        """
        logging.info('Performing QR decomposition ...')

        Masses = self.atoms.masses()[:,np.newaxis]

	# normalize coords to [-1,1]        
        ScaledPos = (coords - coords.mean(axis=0)) / self.box

        # grab the masses, and make it a column vector
        Basis = np.zeros([ScaledPos.shape[0], self.pindices.shape[0]],'f')

        for i in xrange(self.pindices.shape[0]):
            k1, k2, k3 = self.pindices[i,:]
            px = legendre(k1)(ScaledPos[:,0])
            py = legendre(k2)(ScaledPos[:,1])
            pz = legendre(k3)(ScaledPos[:,2])
            Basis[:,i] = px * py * pz

        return Basis

def QR_Decomp(V):
    """
    Performs QR decomposition on a matrix V to produce an orthonormalized
    matrix Q..
    """
    Q,R = qr(V, mode='economic')

    return Q

def poly_indexes(kmax):
    """
    Create 2D array of Legendre polynomial indices with index sum <= kmax.

    For example, if kmax is 1, the this returns
    [[0, 0, 0],
     [1, 0, 0],
     [0, 1, 0],
     [0, 0, 1]]
    Note, the sum of each row is less than or equal to 1.
    """
    indices = []

    for n in range(kmax + 1):
        for i in range(n+1):
            for j in range(n+1-i):
                indices.append([n-i-j, j, i])

    return np.array(indices, 'i')

def SpaceWarpingSubsystemFactory(system, selects, **args):
    """
    create a list of LegendreSubsystems.

    @param system: the system that the subsystem belongs to, this may be None
                   when the simulation file is created.
    @param selects: A list of MDAnalysis selection strings, one for each
                    subsystem.
    @param args: a list of length 1 or 2. The first element is kmax, and
                 the second element may be the string "resid unique", which can be
                 thought of as an additional selection string. What it does is
                 generate a subsystem for each residue. So, for example, select
                 can be just "resname not SOL", to strip off the solvent, then
                 if args is [kmax, "resid unique"], an seperate subsystem is
                 created for each residue.
    """
    kmax, freq = 0, 1000

    try:
	kmax = args['kmax']
    except:
	raise ValueError("invalid subsystem args")

    if 'freq' in args:
	freq = args['freq']
	logging.info('Ref structure will be updated every {} CG time steps'.format(freq))

    # test to see if the generated selects work
    [system.universe.selectAtoms(select) for select in selects]

    # create the polynomial indices
    pindices = poly_indexes(kmax)

    # the number of CG variables
    # actually, its sufficient to just say nrows * 3 as the
    # number of columns had better be 3.
    ncg = pindices.shape[0] * pindices.shape[1]

    return [SpaceWarpingSubsystem(system, pindices, select, freq) for select in selects], ncg
