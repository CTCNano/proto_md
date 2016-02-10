"""
@group Units: As proto_md uses MDAnalysis, it therefore uses the MDAnalysis units which are:
    force: kJ/(mol*A) - Note that MDA will convert forces to native units (kJ/(mol*A), even
    though gromcas uses kJ/(mol*nm).
    position: Angstroms,
    velocity: Angstrom/ps - velocities are automatically converted to MDAnalysis units
    (i.e. from Gromacs nm/ps to Angstrom/ps in MDAnalysis)
@group environment variables:
    PROTO_DEBUG: if set to , temporary directores are NOT deleted, this might be usefull
    for debugging MD issues.

Config Dictionary Specification
{
    struct: name of structure file (typically a PDB), required,
    protein: name of protein section, optional.
    top: name of topology file, optional.
    top_args:    a dictionary of optional additional arguments passed to topology and pdb2gmx, these may include
                 dirname: directory where top file is generated,
                 posres: name of position restraint include file,
                 and may also include any arguments accepted by pdb2gmx, see:
                 http://manual.gromacs.org/current/online/pdb2gmx.html

    md_nensemble: a optional number specifying how many

    md:         a dictionary of parameters used for the md run(s), these include
                nsteps: number of md steps
                multi: how many simulations to do for ensemble averaging. optional, defaults to 1
                       (I know, this is a probably not the best key name, but this is
                        the argname that mdrun takes, so used for consistency)

    "equilibriate":{"nsteps":1000},

    @group hdf file structure:
    All state variables of the system are saved in the hdf file, this is usefull for
    restarting or debugging. All information required for a restart is saved in the hdf
    file. This means that ONLY the hdf file is required for a restart, nothing else.

    The key names will correspond to the file name.

    All source files are copied into the "src_files" group.

    The files used by the sytem are stored in the "files" group, however,
    all items in this group are soft links to either one of the source files, or
    new files located in a timestep group.

    The "/files" group will contain at a minimum:
        struct.gro
        topol.top
        index.ndx


    "/timesteps" is a group which contains all timesteps, this has subgroups
    names "0", "1", ... "N".

    "/current_timestep" is a soft link to the timestep group currently being processed.
    This will not exist on a newly created file.

    "/prev_timestep" is a soft link to the previously completed timestep. This is
    used for restarts. It will not exist on newly created files.

    The "/timesteps" group has a series of timestep groups named 0, 1, ... N.
    Each one of these timestep subgroups contains the following data

        struct.gro: this is typically a soft link to the struct.gro
        in "/src_files". This is only used as a way to store the non-coordinate
        attributes such as segment and residue info. The coordinates in this
        pdb are NOT USED to store coordinate info.

"""


import logging
logging.basicConfig(format='%(asctime)s:%(levelname)s:%(funcName)s:%(message)s', filename='proto.log',level=logging.DEBUG)


from numpy import array, zeros, reshape, conjugate, \
                  real, max, fromfile, uint8
import numpy.random
import MDAnalysis                       #@UnresolvedImport
import dynamics

import h5py                             #@UnresolvedImport
import md
import util
import time

# change MDAnalysis table to read carbon correctly
import MDAnalysis.topology.tables       #@UnresolvedImport
MDAnalysis.topology.tables.atomelements["C0"]="C"

import config
from config import *



class Timestep(object):
    """
    each timestep is storred in a hdf group, this class wraps
    the group and provides properties to access the values.
    """
    ATOMIC_MINIMIZED_POSITIONS = "atomic_minimized_positions"
    ATOMIC_EQUILIBRIATED_POSITIONS = "atomic_equilibriated_positions"
    ATOMIC_STARTING_POSITIONS = "atomic_starting_positions"
    ATOMIC_FINAL_POSITIONS = "atomic_final_positions"
    TIMESTEP_BEGIN = "timestep_begin"
    TIMESTEP_END = "timestep_end"
    CG_POSITIONS = "cg_positions"
    CG_VELOCITIES = "cg_velocities"
    CG_FORCES = "cg_forces"
    CG_TRANSLATE = "cg_translate"

    def __create_property(self, name):
        def getter(self):
            return self._group[name][()]
        def setter(self,val):
            try:
                del self._group[name]
            except KeyError:
                pass
            self._group[name] = val
            self._group.file.flush()

        # construct property attribute and add it to the class
        setattr(self.__class__, name, property(fget=getter, fset=setter))

    def __init__(self, group):
        self._group = group
        self.__create_property(Timestep.ATOMIC_MINIMIZED_POSITIONS)
        self.__create_property(Timestep.ATOMIC_EQUILIBRIATED_POSITIONS)
        self.__create_property(Timestep.ATOMIC_STARTING_POSITIONS)
        self.__create_property(Timestep.ATOMIC_FINAL_POSITIONS)
        self.__create_property(Timestep.TIMESTEP_END)
        self.__create_property(Timestep.TIMESTEP_BEGIN)
        self.__create_property(Timestep.CG_POSITIONS)
        self.__create_property(Timestep.CG_VELOCITIES)
        self.__create_property(Timestep.CG_FORCES)
        self.__create_property(Timestep.CG_TRANSLATE)

    @property
    def timestep(self):
        """
        the index of the timestep
        """
        return int(self._group.name.split("/")[-1])

    def create_universe(self):
        data = self._group[STRUCT_PDB][()]

        with tempfile.NamedTemporaryFile(suffix=".gro") as f:
            # EXTREMLY IMPORTANT to flush the file,
            # NamedTemporaryFile returns an OPEN file, and the array writes to this file,
            # so we need to flush it, as this file handle remains open. Then the MDAnalysis
            # universe opens another handle and reads its contents from it.
            data.tofile(f.file)
            f.file.flush()
            u = MDAnalysis.Universe(f.name)
            return u

    def flush(self):
        self._group.file.flush()

class System(object):


    """
    @ivar subsystems: a list of subsystems, remains constant so long
                      as the topology does not change.
    """

    # define the __slots__ class variable, this helps preven typos by raising an error if a
    # ivar is set that is not one of these here.
    __slots__ = ["hdf", "config", "universe", "_box", "ncgs", "subsystems", "cg_positions", "cg_velocities", 
		 "cg_forces", "concentration", "anion", "cation"]

    def __init__(self, fid, mode="r"):
        """ Create a system object

        Args:
        fid: file name of a configuration hdf file

        mode: the mode to open the file, defaults to "r" - read only. For running a simulation, it should
        be "a" -  Read/write if exists.

        For debugging, mode should be 'd', this mode opens the file in read/write mode but
        does not clear any timesteps.
        """

        logging.info("creating System, fid={}".format(fid))

        fmode = 'a' if (mode == 'a' or mode == 'd') else 'r'
        self.hdf = h5py.File(fid, fmode)
        self.config = self.hdf[CONFIG].attrs
        self._box = self.config[BOX]
	self.concentration = self.config[CONCENTRATION]
	self.anion = self.config[ANION]
	self.cation = self.config[CATION]

        # if there is a current timestep, keep it around for debugging purposes
        if self.hdf.id.links.exists(CURRENT_TIMESTEP):
            print("WARNING, found previous \"current_timestep\" key, this means that a previous simulation likely crashed")
            logging.warn("found previous \"current_timestep\" key, this means that a previous simulation likely crashed")


        # load the universe object from either the last timestep, or from the src_files
        # its expensive to create a universe, so keep it around for the lifetime
        # of the system
        self.universe = self._create_universe()

        # load the subsystems
        # this list will remain constant as long as the topology remains constant.
        logging.info("creating subsystems")

        factory = util.get_class(self.config[SUBSYSTEM_FACTORY])

	# This is imp to get around the HDF - dictionary incompatibility
	subsystem_args_tuple = tuple([tuple(i) for i in self.config[SUBSYSTEM_ARGS]])
	subsystem_args_dict = dict((x, eval(y)) for x, y in subsystem_args_tuple)

        self.ncgs, self.subsystems = factory(self, self.config[SUBSYSTEM_SELECTS], **subsystem_args_dict)
        logging.debug("using {} cg variables for each {} subsystems".format(self.ncgs, len(self.subsystems)))

        # notify subsystems, we have a new universe
        [s.universe_changed(self.universe) for s in self.subsystems]

        md_nensemble = self.config[MULTI]

        # number of data points in trajectory, md steps / output interval
        md_nsteps = int(self.config[MD_STEPS])/int(self.md_args[NSTXOUT])

        # number of subsystems
        nrs = len(self.subsystems)

        # cg: nensembe x n segment x n_step x n_cg
        self.cg_positions  = zeros((md_nensemble,nrs,md_nsteps,self.ncgs))
        self.cg_forces     = zeros((md_nensemble,nrs,md_nsteps,self.ncgs))
        self.cg_velocities = zeros((md_nensemble,nrs,md_nsteps,self.ncgs))

    @property
    def struct(self):
        return self._get_file_data(STRUCT_PDB)

    @property
    def top(self):
        return self._get_file_data(TOPOL_TOP)

    @property
    def top_includes(self):
        return self.hdf[TOP_INCLUDES].values()

    @property
    def box(self):
        return self._box

    @property
    def temperature(self):
        return float(self.config[TEMPERATURE])

    @property
    def beta(self):
        return 1/(KB*self.temperature)

    @property
    def mn_args(self):
        return util.hdf_dict(self.config, MN_ARGS)

    @property
    def eq_args(self):
        return util.hdf_dict(self.config, EQ_ARGS)

    @property
    def md_args(self):
        return util.hdf_dict(self.config, MD_ARGS)

    @property
    def should_solvate(self):
        return bool(self.config[SHOULD_SOLVATE])

    @property
    def cg_steps(self):
        return int(self.config[CG_STEPS])

    @property
    def mainselection(self):
        return self.config[MAINSELECTION]

    def integrator(self):
        """
        Create an integrator based on what the config file specified.

        The resulting integrator is initialized with this system, and
        whatever addational arguments were specified in the config.
        """
        integrator_type = util.get_class(self.config[INTEGRATOR])
        return integrator_type(self, *self.config[INTEGRATOR_ARGS])

    def _get_file_data(self, file_key):
        """
        finds a file stored as an hdf key. This first looks if there is a 'current_timestep',
        if so uses it, otherwise, looks in 'src_files'.
        """
        #TODO imporove error checking and handling.
        if self.hdf.id.links.exists(CURRENT_TIMESTEP):
            file_key = CURRENT_TIMESTEP + "/" + file_key
        else:
            file_key = SRC_FILES + "/" + file_key
        return self.hdf[file_key]

    @property
    def last_timestep(self):
        """
        Gets the last completed timestep, or None if this is the first timestep.

        If only the last timestep is required, this approach is more effecient than
        building an entire list.
        """
        timesteps = [int(k) for k in self.hdf[TIMESTEPS].keys()]
        if len(timesteps):
            return Timestep(self.hdf[TIMESTEPS + "/" + str(max(timesteps))])
        else:
            return None


    def begin_timestep(self):
        """
        Creates a new empty current_timestep. If one currently exists, it is deleted.

        The _begin_timestep and _end_timestep logic are modeled after OpenGL's glBegin and glEnd.
        the "current_timestep" link should only exist between calls
        to begin_timestep and end_timestep.

        The simulation may crash in between these calls to begin and end timestep, in this case,
        there will be a partially completed current_timestep. For debugging purposes, the current_timestep
        may be loaded back into the system via _load_timestep. Note, in this case, current_timestep is likely not
        complete, missing attributes will cause notifications via _load_timestep.

        For development/debugging, there is no harm done in repeatedly calling begin_timestep,
        only the end_timestep actually writes the current_timestep to timesteps[n+1]
        """
        # if there is a current timestep, we assume its garbage and delete it
        if self.hdf.id.links.exists(CURRENT_TIMESTEP):
            logging.warn("found previous \"current_timestep\" key, this means that end_timestep was not called.")
            del self.hdf[CURRENT_TIMESTEP]

        # create a new current_timestep group, starting time now.
        current_group = self.hdf.create_group(CURRENT_TIMESTEP)
        self.current_timestep.timestep_begin = time.time()

        last = self.last_timestep
        timestep_number = 0
        if last:
            # link the starting positions to the previous timesteps final positions
            current_group.id.links.create_soft(Timestep.ATOMIC_STARTING_POSITIONS,
                                      last._group.name + "/" + Timestep.ATOMIC_FINAL_POSITIONS)
            src_files = last._group
            timestep_number = last.timestep + 1
        else:
            # this is the first timestep, so set start positions to the starting
            # universe struct.
            current_group[Timestep.ATOMIC_STARTING_POSITIONS] = self.universe.atoms.positions
            src_files = self.hdf[SRC_FILES]

        logging.info("starting timestep {}".format(timestep_number))

        # link file data into current
        for f in FILE_DATA_LIST:
            util.hdf_linksrc(self.hdf, CURRENT_TIMESTEP + "/" + f, src_files.name + "/" + f)


    def end_timestep(self):
        """
        move current_timestep to timesteps/n+1, and flush the file.

        This is the ONLY method that actually changes the completed timesteps.
        """
        # done with timestep, throws an exception is we do not
        # have a current timestep.
        self.current_timestep.timestep_end = time.time()

        # find the last timestep, and set this one to the next one, and move it there.
        timesteps = [int(k) for k in self.hdf[TIMESTEPS].keys()]
        prev = -1 if len(timesteps) == 0 else max(timesteps)
        finished = TIMESTEPS + "/" + str(prev+1)
        self.hdf.id.move(CURRENT_TIMESTEP, finished)

        self.hdf.flush()

        logging.info("completed timestep {}".format(prev+1))


    @property
    def current_timestep(self):
        """
        return the current_timestep.
        The current_timestep is created by begin_timestep, and flushed to the file and deleted
        by end_timestep.

        The current_timestep should be treated as WRITE ONLY MEMORY!
        All read/write state variables should be instance variables of the System class,
        such as univese, cg_postions, etc...

        It is very important to treat this as WRITE ONLY, reading from this var
        will completly screw up the logic of state evolution.

        """
        return Timestep(self.hdf[CURRENT_TIMESTEP])

    @property
    def dt(self):
        """
        The time step used in the coarse grained (Langevin) step.
        specified in the config file.
        """
        return float(self.config[DT])

    @property
    def timesteps(self):
        """
        A list of Timestep objects which are the completed langevin steps.
        """
        # get all the completed timesteps, the keys are usually out of order, so
        # have to sort them
        timesteps = [int(k) for k in self.hdf[TIMESTEPS].keys()]
        timesteps.sort()
        return [Timestep(self.hdf[TIMESTEPS + "/" + str(ts)]) for ts in timesteps]


    def _create_universe(self, key = None):
        """
        Creates a universe object from the most recent timestep, or if this is the first
        timestep, loads from the 'src_files' key.
        """

        last = self.last_timestep
        if last:
            logging.info("loading universe from most recent timestep of {}".format(last._group.name))
            return last.create_universe()
        else:
            logging.info("_universe_from_hdf, struct.gro not in current frame, loading from src_files")
            data = array(self.hdf[SRC_FILES + "/" + STRUCT_PDB])

            with tempfile.NamedTemporaryFile(suffix=".gro") as f:
                # EXTREMLY IMPORTANT to flush the file,
                # NamedTemporaryFile returns an OPEN file, and the array writes to this file,
                # so we need to flush it, as this file handle remains open. Then the MDAnalysis
                # universe opens another handle and reads its contents from it.
                data.tofile(f.file)
                f.file.flush()
                u = MDAnalysis.Universe(f.name)
                return u


    def translate(self, cg_translate):
        """
        translates ALL of the subsystems according to the given array of CG variables.

        cg_translate can be either a row or column vector of length N_subsys x N_cg.
        Each subsystem will be given a n_cg 1D array. If the given cg_translate is not the
        correct size, an exception will be thrown.
        """
        # make sure it is of the correct shape, this will make a 1D array out of the
        # given array, and will throw an exception of the array is not the correct
        # size.

        # n_cg * n_ss
        cg_translate = cg_translate.reshape(len(self.subsystems) * self.ncgs)

        #write to ts
        self.current_timestep.cg_translate = cg_translate

        for i, s in enumerate(self.subsystems):
            s.translate(cg_translate[i*self.ncgs:i*self.ncgs+self.ncgs])

        self.current_timestep.atomic_final_positions = self.universe.atoms.positions

    def setup_equilibriate(self, struct=None, top=None):
        """
        setup an equilibriation md run using the contents of the universe, but do not actually run it.

        @return an MDManager object loaded with the trr to run an equilibriation.
        """

        if struct is None or top is None:
            struct = self.universe
            top = self.top
            logging.info("setting up equilibriation for self.universe")
        else:
            logging.info("setting up equilibriation for solvated struct, top")

	eq_args = self.eq_args
	eq_args['ref_t'] = self.config[TEMPERATURE]
	eq_args['gen-temp'] = self.config[TEMPERATURE]

        return md.setup_md(struct=struct, \
                               top=top, \
                               top_includes=self.top_includes, \
                               nsteps=self.config[EQ_STEPS], \
                               deffnm="eq", \
                               mainselection=self.mainselection, \
                               **eq_args)

    def setup_md(self, struct=None, top=None):
        """
        setup an equilibriation md run using the contents of the universe, but do not actually run it.

        @return an MDManager object loaded with the trr to run an equilibriation.
        """
        logging.debug("struct = {}, top = {}".format(struct, top ))

        if struct is None or top is None:
            struct = self.universe
            top = self.top
            logging.info("setting up md for self.universe")
        else:
            logging.info("setting up md for solvated struct, top")

	md_args = self.md_args
	md_args['ref_t'] = self.config[TEMPERATURE]
	md_args['gen-temp'] = self.config[TEMPERATURE]

        return md.setup_md(struct=struct, \
                               top=top, \
                               top_includes=self.top_includes, \
                               nsteps=self.config[MD_STEPS], \
                               multi=self.config[MULTI], \
                               deffnm="md", \
                               mainselection=self.mainselection, \
                               **md_args)

    def equilibriate(self, struct=None, top=None, sub=None, **args):
        """
        Equilibriates (thermalizes) the structure stored in self.universe.

        takes the universe structure, inputs it to md, performs an equilibriation run, and
        read the result back into the universe. The equilibriated atomic state can then
        be used at the starting state the md sampling run, and for the langevin step.

        The MD is typically performed with position restraints on the protein.

        The equilibriated position are also written to the current_timestep.

        @precondition: self.universe contains an atomic state.
        @postcondition: self.universe now contains an equilbriated state.
            subsystems are notified.
            equililibriated state is written to current_timestep.
        """
        logging.debug( \
                       "struct = {}, top = {}, sub = {}, args = {}".format( \
                       struct, top , sub , args))
        result = None

        if struct is None or top is None:
            logging.info("performing equilibriation for self.universe")
            with self.setup_equilibriate() as eqsetup:
                mdres = md.run_md(eqsetup.dirname, **eqsetup)
                self.universe.load_new(mdres.structs[0])
        else:
            logging.info("performing equilibriation for solvated struct, top")
            result = self.setup_equilibriate(struct, top)
            mdres = md.run_md(result.dirname, **result)
            result["struct"] = mdres.structs[0]
            result["sub"] = sub
            self.universe.atoms.positions = util.stripped_positions(mdres.structs[0], sub)

        self.current_timestep.atomic_equilibriated_positions = self.universe.atoms.positions
        [s.equilibriated() for s in self.subsystems]

        return result

    def md(self, struct=None, top=None, sub=None, **args):
        """
        Perform a set of molecular dynamics runs using the atomic state
        of self.universe as the starting structure. The state of self.universe
        is NOT modified.

        @precondition: self.universe contains a valid atomic state.
        @postcondition: self.cg_positions, self.cg_forces, self.cg_velocities[:]
            are populated with statistics collected from the md runs.
        """
        logging.debug( \
                       "struct = {}, top = {}, sub = {}, args = {}".format( \
                        struct, top , sub , args))


        with self.setup_md(struct, top) as mdsetup:
            mdres = md.run_md(mdsetup.dirname, **mdsetup)
            self._processes_trajectories(mdres.trajectories, sub)

    def _processes_trajectories(self, trajectories, sub=None):
        """
        reads each given atomic trajectory, extracts the
        coarse grained information, updates state variables with
        this info, and saves it to the output file.

        """
        logging.info("processing trajectories")
        # zero the state variables (for this frame)
        self.cg_positions[:] = 0.0
        self.cg_forces[:] = 0.0
        self.cg_velocities[:] = 0.0

        # save the universe to a tmp file to load back once the trajectories
        # are processed.
        with tempfile.NamedTemporaryFile(suffix=".gro") as tmp:
            writer = MDAnalysis.Writer(tmp.name)
            writer.write(self.universe)
            writer.close()
            del writer

            # universe is saved, process trajectories
            for fi, f in enumerate(trajectories):
                self.universe.load_new(f, sub=sub)
                for tsi, _ in enumerate(self.universe.trajectory):
                    if tsi < self.cg_velocities.shape[2]:
                        for si, s in enumerate(self.subsystems):
                            pos,vel,frc = s.frame()
		
			    print 'velocities = '
			    print vel

                            self.cg_positions [fi,si,tsi,:] = pos
                            self.cg_velocities[fi,si,tsi,:] = vel
                            self.cg_forces    [fi,si,tsi,:] = frc

                            if(tsi % 25 == 0):
                                print("processing frame {},\t{}%".format(tsi, 100.0*float(tsi)/float(self.cg_velocities.shape[2])))

            # done with trajectories, load original contents of universe back
            self.universe.load_new(tmp.name)

        # write the coarse grained pos, vel and forces to the current timestep.
        timestep = self.current_timestep
        timestep.cg_positions = self.cg_positions
        timestep.cg_velocities = self.cg_velocities
        timestep.cg_forces = self.cg_forces
        timestep.flush()


    def topology_changed(self):
        """
        Notify the system that the topology of the universe was changed.
        This causes the system to generate a new topology file,
        and notify all subsystems that the universe was changed.
        """
        pass

    def solvate(self):
        """
        Solvate the current structure (self.universe)

        Neither the universe, nor the contents of the universe are harmed by this operation.

        @return: A MDManager context manager which contains the solvated structure and
            topology files, as well as a 'sub' index which will be used to later
            pick out the original unsolvated atoms.
        """
        return md.solvate(struct=self.universe, top=self.top, top_includes = self.top_includes,
                          mainselection=self.mainselection, concentration=self.concentration, cation=self.cation, 
			  anion=self.anion)

    def minimize(self, struct = None, top = None, sub = None, **args):
        """
        Take the a starting structure and minimize it via md.

        The starting structure defaults to self.universe and self.top. If the optional
        arguments, struct and top are given, these are then the starting structure.
        The optional args are a way to minimize a solvated structure.

        If struct and top are give, then sub MUST be the indices of the original (self.universe)
        atoms in the solvated structure.

        In either case, self.universe is loaded with with the minimized structure and
        subsystems are notified.

        @postcondition:
        self.universe is loaded with the minimized structure
        all subsystems are notified.

        @return: None if we are using self.universe / self.top, otherwise, if we
            are given a solvated structure, then a MDManager context manager loaded
            with the minimized solvated structure / top is returned.
        """

        logging.debug( \
                       "struct = {}, top = {}, sub = {}, args = {}".format( \
                       struct, top , sub , args))

        result = None

        if struct is None or top is None:
            logging.info("performing minimization with self.universe and self.top")
            with md.minimize(struct=self.universe, \
                                 top=self.top, \
                                 top_includes=self.top_includes, \
                                 nsteps=self.config[MN_STEPS], \
                                 deffnm="mn", \
                                 **self.mn_args) as mn:

                self.universe.load_new(mn["struct"])
        else:
            if sub is None or len(sub) != len(self.universe.atoms):
                raise ValueError("sub is either None or is not the correct length")
            logging.info("performing minimization with solvated structure")
            result = md.minimize(struct=struct, \
                                     top=top, \
                                     top_includes = self.top_includes, \
                                     nsteps=self.config[MN_STEPS], \
                                     deffnm="mn", \
                                     **self.mn_args)
            result["sub"] = sub
            self.universe.atoms.positions = util.stripped_positions(result["struct"], sub)

        # done with external md
        self.current_timestep.atomic_minimized_positions = self.universe.atoms.positions
        [s.minimized() for s in self.subsystems]
        logging.info("minimization complete")

        return result


    def tofile(self,traj):
        """
        Write the system to a conventional MD file format, either pdb, or trr.

        if traj ends with 'pdb', the starting structure is saved as a pdb, otherwise,
        all the frames are written to a trajectory (trr). This is usefull for VMD
        where one could perform:
            sys.tofile("somefile.gro")
            sys.tofile("somefile.trr"),
        then call vmd with these file args.
        """
        universe = None
        writer = None
        ext = os.path.splitext(traj)[-1]
        if ext.endswith("pdb"):
            for ts in self.timesteps:
                universe = ts.create_universe()
                writer = MDAnalysis.Writer(traj,numatoms=len(universe.atoms))
                writer.write(universe)
                return
        else:
            for ts in self.timesteps:
                if universe is None:
                    universe = ts.create_universe()
                    writer = MDAnalysis.Writer(traj,numatoms=len(universe.atoms))
                else:
                    universe.atoms.positions = ts.atomic_starting_positions
                writer.write(universe)

    def visualize(self):
        dirname = tempfile.mkdtemp()
        struct = dirname + "/struct.gro"
        traj = dirname + "/traj.trr"

        self.tofile(struct)
        self.tofile(traj)

        os.system("vmd {} {}".format(struct, traj))

    def _load_timestep(self, ts):
        """
        Load the system - set the current timestep and the state variables to those
        set in a Timestep object.

        This method is usefull for debugging / development, so the state of the system
        can be set without performing an MD run. This way methods that change the state
        of the System, such as evolve() can be debugged.
        """

        # check to see if the object can be used at a timestep object
        attrs = ["cg_positions", "cg_velocities", "cg_forces", "atomic_starting_positions"]
        ists = [hasattr(ts, attr) for attr in attrs].count(True) == len(attrs)

        if ists:
            try:
                self.universe.atoms.positions = ts.atomic_starting_positions
            except:
                print("failed to set atomic positions")

            try:
                self.cg_positions[()] = ts.cg_positions
            except:
                print("failed to set cg_positions")

            try:
                self.cg_velocities[()] = ts.cg_velocities
            except:
                print("failed to set cg_velocities")

            try:
                self.cg_forces[()] = ts.cg_forces
            except:
                print("failed to set cg_forces")
            return

        # its not a timestep, so try as a integer
        try:
            return self._load_timestep(self.timesteps[int(ts)])
        except ValueError:
            raise ValueError("Assumed timestep was an index as it did not have required attributes, but could not convert to integer")
