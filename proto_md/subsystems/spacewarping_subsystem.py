"""
Created on January 27, 2013

@author: Andrew-AM

The SWM polynomial type of coarse grained variable.

"""

"""
Created on January 27, 2013

@author: Andrew-AM
"""


from . import subsystems
import numpy as np
from scipy.special import legendre
from scipy.linalg import qr
from numpy.linalg import norm, solve
import logging


class SpaceWarpingSubsystem(subsystems.SubSystem):
    """
    A set of CG variables.
    """

    def __init__(self, **kwargs):
        """
        Create a SWM subsystem.
        @param system: an object (typically the system this subsystem belongs to)
        which has the following attributes:
        box: an array describing the periodic boundary conditions.

        note, the system MAY be None, such as when the config is created, so don't
        access it yet.

        @param pindices: a N*3 array of Legendre polynomial indices.
        @param select: a select string used for Universe.select_atoms which selects
        the atoms that make up this subsystem.
        """
        self.system = kwargs.get("system")

        # select statement to get atom group
        self.select = kwargs.get("select")

        # polynomial indices, N_cg x 3 matrix.
        self.pindices = kwargs.get("pindices")

        # we need to be called with a valid universe
        if self.system is not None:
            self.universe_changed(self.system.universe)

        self.cgStep = 0

        # How often should the reference struct be updated
        self.freq_update = kwargs.get("freq", 1000)

        if self.pindices is None:
            raise ValueError(
                f"{self.__name__} requires pindices (polymer indices) to be supplied."
            )

        logging.info(
            "created SpaceWarpingSubsystem, pindices: {}, select: {}".format(
                self.pindices, self.select
            )
        )

    def universe_changed(self, universe):
        """
        universe changed, so select our atom group
        """
        self.atoms = universe.select_atoms(self.select)

        # check to see if atoms is valid
        if len(self.atoms) <= 0:
            raise ValueError(
                "The select statement '{}' is syntactically valid but returned a zero sized AtomGroup".format(
                    self.select
                )
            )

        # call some stuff on the atomGroup to see if its valid
        self.atoms.bbox()

    def frame(self):
        """
        Returns a 3 tuple of CG variables, each one is a row vector of size n_cg
        """
        cg = self.computeCG_pos(self.atoms.positions)
        try:
            cgVel = self.computeCG_vel(self.atoms.velocities)
        except Exception:
            cgVel = cg * 0
        try:
            cgFor = self.computeCG_forces(self.atoms.forces)
        except Exception:
            cgFor = cg * 0

        cg = np.reshape(cg.T, (cg.shape[0] * cg.shape[1]))
        cgVel = np.reshape(cgVel.T, (cgVel.shape[0] * cgVel.shape[1]))
        cgFor = np.reshape(cgFor.T, (cgFor.shape[0] * cgFor.shape[1]))

        return (cg, cgVel, cgFor)

    def translate(self, dCG):
        """
        translates the atomic positions from a given vectory of dCG positions,
        where dCG is a finite change in the CG velocities times dt,
        and then adds the residuals for better accuracy.

        @param CG: a length 3*N_cg 1D array.
        """
        ndim = 3
        nCG = len(dCG) / ndim
        assert (
            nCG.is_integer()
        ), f"dCG must be of length (nCG*ndim). Supplied nCG = {len(dCG)}!"
        nCG = int(nCG)
        self.atoms.positions += self.computeCG_inv(dCG.reshape(nCG, ndim))

    def minimized(self):
        pass

    def md(self):
        self.cgStep += 1

    def equilibrated(self, scale=1.1):
        """
        This is called just after the structure is equilibriated, this is the starting struct
        for the MD runs, this is to calculate basis.
        @param scale: factor to scale the macromolecular box with.
        """
        if self.cgStep % self.freq_update == 0:
            logging.info(
                "Updating ref structure and constructing new basis functions..."
            )
            boxboundary = self.atoms.bbox()
            # 110% of the macromolecular box
            self.box = (boxboundary[1, :] - boxboundary[0, :]) * scale
            self.ref_com = self.atoms.center_of_mass()

            self.basis = self.construct_basis(self.atoms.positions - self.ref_com)
            self.ref_coords = self.atoms.positions.copy()

    def computeCG_inv(self, cg):
        """
        Computes atomic positions from CG positions
        Using the simplest scheme for now
        @param CG: n_cg x 3 array
        @return: n_atom x 3 array
        """
        return self.ref_com + np.dot(self.basis, cg)

    def computeCG_pos(self, pos):
        """
        Computes CG positions
        CG = inverse(Utw * self.basis) * Utw * (pos - pos_c)
        """
        Utw = self.basis.T * self.atoms.masses
        cg = solve(np.dot(Utw, self.basis), np.dot(Utw, pos - self.ref_com))

        return cg

    def computeCG_vel(self, vel):
        """
        Computes CG velocities
        CG = inverse(Utw * self.basis) * Utw * vel
        """
        Utw = self.basis.T * self.atoms.masses
        cg_vel = solve(np.dot(Utw, self.basis), np.dot(Utw, vel))

        return cg_vel

    def computeCG_forces(self, forces):
        """
        Computes CG forces
        CG = inverse(Utw * self.basis) * Utw * forces
        """
        Utw = self.basis.T * self.atoms.masses
        cg_forces = solve(np.dot(Utw, self.basis), np.dot(Utw, forces))

        return cg_forces

    def construct_basis(self, coords):
        """
        Constructs a matrix of orthonormalized legendre basis functions
        of size Natoms x NCG.
        """
        logging.info("Constructing basis ...")

        # normalize coords to [-1,1]
        scaledPos = (coords - self.ref_com) / self.box

        # grab the masses, and make it a column vector
        basis = np.zeros([scaledPos.shape[0], self.pindices.shape[0]], "f")

        for i in range(self.pindices.shape[0]):
            k1, k2, k3 = self.pindices[i, :]
            px = legendre(k1)(scaledPos[:, 0])
            py = legendre(k2)(scaledPos[:, 1])
            pz = legendre(k3)(scaledPos[:, 2])
            basis[:, i] = px * py * pz

        return basis


def poly_indexes(kmax):
    """
    Create 2D array of Legendre polynomial indices with index sum <= psum.

    For example, if psum is 1, the this returns
    [[0, 0, 0],
     [1, 0, 0],
     [0, 1, 0],
     [0, 0, 1]]
    Note, the sum of each row is less than or equal to 1.
    """
    indices = []

    for n in range(kmax + 1):
        for i in range(n + 1):
            for j in range(n + 1 - i):
                indices.append([n - i - j, j, i])

    return np.array(indices, "i")


def SpaceWarpingSubsystemFactory(system=None, selects=["all"], **kwargs):
    """
    create a list of LegendreSubsystems.
    @param system: the system that the subsystem belongs to, this may be None
                   when the simulation file is created.
    @param selects: A list of MDAnalysis selection strings, one for each
                    subsystem.
    @param kwargs: a list of length 1 or 2. The first element is kmax, and
                 the second element may be the string "resid unique", which can be
                 thought of as an additional selection string. What it does is
                 generate a subsystem for each residue. So, for example, select
                 can be just "resname not SOL", to strip off the solvent, then
                 if args is [kmax, "resid unique"], an seperate subsystem is
                 created for each residue.
    """
    kmax = kwargs.get("kmax", 0)
    freq = kwargs.get("freq", 1000)

    if freq:
        logging.info(
            "Ref structure will be updated every {} CG time steps".format(freq)
        )

    # test to see if the generated selects work
    if system:
        [system.universe.select_atoms(select) for select in selects]

    # create the polynomial indices
    pindices = poly_indexes(kmax)

    # the number of CG variables
    # actually, its sufficient to just say nrows * 3 as the
    # number of columns had better be 3.
    ncg = pindices.shape[0] * pindices.shape[1]

    return (
        ncg,
        [
            SpaceWarpingSubsystem(
                system=system, pindices=pindices, select=select, freq=freq
            )
            for select in selects
        ],
    )
