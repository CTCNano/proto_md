"""
Functions for setting up and performing md calculations.

All function arguments which refer to file input may be either a
string, in which they are interpreted as a file path, or a numpy array, in
which they are interpreted as data buffers containing the file bytes.

Created on Aug 8, 2012
@author: andy
"""
import os.path
import os
import time
import shutil
import logging
import re
import subprocess
import MDAnalysis       #@UnresolvedImport
import MDAnalysis.core  #@UnresolvedImport
import gromacs.setup
import gromacs.run
import gromacs.utilities
import config
from collections import namedtuple
from os import path
from util import data_tofile, is_env_set
import shutil
import tempfile
import glob
import numpy
import re

#from collections import Mapping, Hashable


class MDManager(dict):
    # __slots__ = ("__dict", "dirname")

    def __init__(self, *args, **kwargs):
        super(MDManager,self).__init__(*args, **kwargs)

        if not self.has_key("dirname"):
            raise ValueError("MDManager arguments must contain a \"dirname\" key")

        self.dirname = self["dirname"]
        del self["dirname"]

        if not os.path.isdir(self.dirname):
            raise IOError("dirname of {} is not a directory".format(self.dirname))

        # save the abs path, user could very likely change directories.
        self.dirname = os.path.abspath(self.dirname)

    def __repr__(self):
        return "MDManager(dirname={}, dict={})".format(self.dirname, super(MDManager,self).__repr__())

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        if the block terminated without raising an exception, all three arguments will be None; otherwise see below.

        exc_type: The type of the exception.

        exc_value: The exception instance raised.

        traceback: A traceback instance.
        """
        logging.debug("dirname: {} exc_type {}, exc_value {}, traceback {}".format(self.dirname, exc_type, exc_value, traceback))
        proto_debug = is_env_set("PROTO_DEBUG")
        if not proto_debug and exc_type is None and exc_value is None and traceback is None:
            logging.debug("deleting {}".format(self.dirname))
            shutil.rmtree(self.dirname)
        else:
            if proto_debug:
                logging.info("PROTO_DEBUG is set, NOT deleting temporary directory {}".format(self.dirname))
            else:
                logging.error("MDManager in directory {} __exit__ called with exception {}".format(self.dirname, exc_value))
                logging.error("MDManager will NOT delete directory {}".format(self.dirname))


def test(dirname):
    return MDManager({'dirname':dirname})


class MDrunner(gromacs.run.MDrunner):
    """Manage running :program:`mdrun` as mpich2_ multiprocessor job with the SMPD mechanism.

    .. _mpich2: http://www.mcs.anl.gov/research/projects/mpich2/
    """
    mdrun = "mdrun"
    mpiexec = "mpiexec"

    def mpicommand(self, *args, **kwargs):
        """Return a list of the mpi command portion of the commandline.

        Only allows primitive mpi at the moment:
           *mpiexec* -n *ncores* *mdrun* *mdrun-args*

        (This is a primitive example for OpenMP. Override it for more
        complicated cases.)
        """

        if os.environ.get("SLURM_NPROCS", None) is not None or \
            os.environ.get('PBS_JOBID', None) is not None:
            logging.info("running in SLURM or PBS, not specifying nprocs")
            return ["mpiexec"]
        elif os.environ.get("PROTO_NPROCS", None) is not None:
            logging.info("user specified number of PROTO processors: {}".format(os.environ.get("PROTO_NPROCS", None)))
            nprocs = int(os.environ.get("PROTO_NPROCS", None))
            return ["mpiexec", "-n", str(nprocs)]
        else:
            nprocs = os.sysconf('SC_NPROCESSORS_ONLN')
            logging.debug("determined nprocs is {} fron os.sysconf".format(nprocs))
            return ["mpiexec", "-n", str(nprocs)]


def minimize(struct, top, top_includes, dirname=None,
             minimize_output="em.pdb", deffnm="em", mdrunner=MDrunner,
             mdp="em.mdp", **kwargs):
    """
    Energy minimize a system.

    Creates the directory minimize_dir, and all operations are performed there.

    @param struct: name of structure file
    @param top: name of top file

    @return: a dictionary with the following values:
        'struct': final_struct,
        'top': topology,
        'mainselection': mainselection,
        'dirname': dir where minimization took place.

    This sets up the system (creates run input files) and also runs
    ``mdrun_d``. Thus it can take a while.

    Additional itp files should be in the same directory as the top file.

    Many of the keyword arguments below already have sensible values.

    :Keywords:
       *dirname*
          set up under directory dirname [em]
       *struct*
          input structure (gro, pdb, ...) [solvate/ionized.gro]
       *output*
          output structure (will be put under dirname) [em.pdb]
       *deffnm*
          default name for mdrun-related files [em]
       *top*
          topology file [top/system.top]
       *mdp*
          mdp file (or use the template) [templates/em.mdp]
       *includes*
          additional directories to search for itp files
       *mdrunner*
          :class:`gromacs.run.MDrunner` class; by defauly we
          just try :func:`gromacs.mdrun_d` and :func:`gromacs.mdrun` but a
          MDrunner class gives the user the ability to run mpi jobs
          etc. [None]
       *kwargs*
          remaining key/value pairs that should be changed in the
          template mdp file, eg ``nstxtcout=250, nstfout=250``.

    .. note:: If :func:`~gromacs.mdrun_d` is not found, the function
              falls back to :func:`~gromacs.mdrun` instead.
    """
    if dirname is None:
        dirname = tempfile.mkdtemp(prefix="tmp." + deffnm + ".")
        logging.debug("created energy minimization dir {}".format(dirname))

    struct = data_tofile(struct, "src.pdb", dirname=dirname)
    top = data_tofile(top, "src.top", dirname=dirname)

#    import pdb; pdb.set_trace()
    logging.debug("top_includes: {}".format(top_includes))
    for i in top_includes:
        data_tofile(i,dirname=dirname)

    logging.info("using mdp template {} from key {}".format(config.templates[mdp], mdp))
    mdp = config.templates[mdp]

    # this stop grompp from failing if there are warning.
    # TODO, is this the best place to put this, should this be in in
    # config.py ????
    kwargs.setdefault('maxwarn', -1)

    # gromacs.setup.energy_minimize returns
    # { 'struct': final_struct,
    #   'top': topology,
    #   'mainselection': mainselection,
    # }
    result = gromacs.setup.energy_minimize(dirname=dirname, struct=struct,
                                         top=top, output=minimize_output, deffnm=deffnm,
                                         mdp=mdp, mdrunner=mdrunner, **kwargs)
    result["dirname"] = dirname
    return MDManager(result)

def setup_md(struct, top, top_includes, deffnm="md", dirname=None, mdp="md_CHARMM27.mdp",
             mainselection=None, **kwargs):
    """Set up Gromacs MD run..

    Additional itp files should be in the same directory as the top file.

    Many of the keyword arguments below already have sensible values. Note that
    setting *mainselection* = ``None`` will disable many of the automated
    choices and is often recommended when using your own mdp file.

    :Keywords:
       *dirname*
          set up under directory dirname [MD_POSRES]
       *struct*
          input structure (gro, pdb, ...) [em/em.pdb]
       *top*
          topology file [top/system.top]
       *mdp*
          mdp file (or use the template) [templates/md.mdp]
       *ndx*
          index file (supply when using a custom mdp)
       *includes*
          additional directories to search for itp files
       *mainselection*
          :program:`make_ndx` selection to select main group ["Protein"]
          (If ``None`` then no canonical index file is generated and
          it is the user's responsibility to set *tc_grps*,
          *tau_t*, and *ref_t* as keyword arguments, or provide the mdp template
          with all parameter pre-set in *mdp* and probably also your own *ndx*
          index file.)
       *deffnm*
          default filename for Gromacs run [md]
       *runtime*
          total length of the simulation in ps [1000]
       *dt*
          integration time step in ps [0.002]
       *qscript*
          script to submit to the queuing system; by default
          uses the template :data:`gromacs.config.qscript_template`, which can
          be manually set to another template from :data:`gromacs.config.templates`;
          can also be a list of template names.
       *qname*
          name to be used for the job in the queuing system [PR_GMX]
       *mdrun_opts*
          option flags for the :program:`mdrun` command in the queuing system
          scripts such as "-stepout 100". [""]
       *kwargs*
          remaining key/value pairs that should be changed in the template mdp
          file, eg ``nstxtcout=250, nstfout=250`` or command line options for
          ``grompp` such as ``maxwarn=1``.

          In particular one can also set **define** and activate
          whichever position restraints have been coded into the itp
          and top file. For instance one could have

             *define* = "-DPOSRES_MainChain -DPOSRES_LIGAND"

          if these preprocessor constructs exist. Note that there
          **must not be any space between "-D" and the value.**

          By default *define* is set to "-DPOSRES".

    :Returns: a dict that can be fed into :func:`gromacs.setup.MD`
              (but check, just in case, especially if you want to
              change the ``define`` parameter in the mdp file)

    .. Note:: The output frequency is drastically reduced for position
              restraint runs by default. Set the corresponding ``nst*``
              variables if you require more output.
    """
    logging.debug("struct = {}, top = {}, top_includes = {}, deffnm = {}, dirname = {}, mdp = {}, mainselection = {}, kwargs = {}".format( \
                  struct, top, top_includes, deffnm, dirname, mdp, mainselection, kwargs))

    logging.info("[%(dirname)s] Setting up MD with position restraints..." % vars())

    if dirname is None:
        dirname = tempfile.mkdtemp(prefix="tmp." + deffnm + ".")
        logging.debug("created md dir {}".format(dirname))

    struct = data_tofile(struct, "src.pdb", dirname=dirname)
    top = data_tofile(top, "src.top", dirname=dirname)

    logging.debug("top_includes: {}".format(top_includes))
    for i in top_includes:
        logging.debug("copying file {} to {}".format(i,dirname))
        data_tofile(i,dirname=dirname)

    # required from gromacswrapper, if this is not set, it will fail as it expects
    # a queing systme.
    kwargs.setdefault('qname', None)

    # this stop grompp from failing if there are warning.
    # TODO, is this the best place to put this, should this be in in
    # config.py ????
    kwargs.setdefault('maxwarn', -1)

    logging.info("using mdp template {} from key {}".format(config.templates[mdp], mdp))
    mdp = config.templates[mdp]

    logging.debug("calling _setup_MD with kwargs: {}".format(kwargs))

    setup_MD = gromacs.setup._setup_MD(dirname, struct=struct, top=top, deffnm=deffnm, mdp=mdp,
                                       mainselection=mainselection, **kwargs)

    setup_MD["dirname"] = dirname

    logging.debug("finished _setup_MD, recieved: {}".format(setup_MD))

    return MDManager(setup_MD)

def setup_md_debug(struct, top, top_includes, deffnm="md", dirname=None, mdp="md_CHARMM27.mdp",
             mainselection=None, **kwargs):
    """Set up Gromacs MD run..

    Additional itp files should be in the same directory as the top file.

    Many of the keyword arguments below already have sensible values. Note that
    setting *mainselection* = ``None`` will disable many of the automated
    choices and is often recommended when using your own mdp file.

    :Keywords:
       *dirname*
          set up under directory dirname [MD_POSRES]
       *struct*
          input structure (gro, pdb, ...) [em/em.pdb]
       *top*
          topology file [top/system.top]
       *mdp*
          mdp file (or use the template) [templates/md.mdp]
       *ndx*
          index file (supply when using a custom mdp)
       *includes*
          additional directories to search for itp files
       *mainselection*
          :program:`make_ndx` selection to select main group ["Protein"]
          (If ``None`` then no canonical index file is generated and
          it is the user's responsibility to set *tc_grps*,
          *tau_t*, and *ref_t* as keyword arguments, or provide the mdp template
          with all parameter pre-set in *mdp* and probably also your own *ndx*
          index file.)
       *deffnm*
          default filename for Gromacs run [md]
       *runtime*
          total length of the simulation in ps [1000]
       *dt*
          integration time step in ps [0.002]
       *qscript*
          script to submit to the queuing system; by default
          uses the template :data:`gromacs.config.qscript_template`, which can
          be manually set to another template from :data:`gromacs.config.templates`;
          can also be a list of template names.
       *qname*
          name to be used for the job in the queuing system [PR_GMX]
       *mdrun_opts*
          option flags for the :program:`mdrun` command in the queuing system
          scripts such as "-stepout 100". [""]
       *kwargs*
          remaining key/value pairs that should be changed in the template mdp
          file, eg ``nstxtcout=250, nstfout=250`` or command line options for
          ``grompp` such as ``maxwarn=1``.

          In particular one can also set **define** and activate
          whichever position restraints have been coded into the itp
          and top file. For instance one could have

             *define* = "-DPOSRES_MainChain -DPOSRES_LIGAND"

          if these preprocessor constructs exist. Note that there
          **must not be any space between "-D" and the value.**

          By default *define* is set to "-DPOSRES".

    :Returns: a dict that can be fed into :func:`gromacs.setup.MD`
              (but check, just in case, especially if you want to
              change the ``define`` parameter in the mdp file)

    .. Note:: The output frequency is drastically reduced for position
              restraint runs by default. Set the corresponding ``nst*``
              variables if you require more output.
    """
    logging.info("[%(dirname)s] Setting up MD with position restraints..." % vars())

    if dirname is None:
        dirname = tempfile.mkdtemp(prefix="tmp." + deffnm + ".")
        logging.debug("created md dir {}".format(dirname))

    struct = data_tofile(struct, "src.pdb", dirname=dirname)
    top = data_tofile(top, "src.top", dirname=dirname)

    logging.debug("top_includes: {}".format(top_includes))
    for i in top_includes:
        logging.debug("copying file {} to {}".format(i,dirname))
        data_tofile(i,dirname=dirname)

    # required from gromacswrapper, if this is not set, it will fail as it expects
    # a queing systme.
    kwargs.setdefault('qname', None)

    # this stop grompp from failing if there are warning.
    # TODO, is this the best place to put this, should this be in in
    # config.py ????
    kwargs.setdefault('maxwarn', -1)

    logging.info("using mdp template {} from key {}".format(config.templates[mdp], mdp))
    mdp = config.templates[mdp]

    logging.debug("calling _setup_MD with kwargs: {}".format(kwargs))

    setup_MD = gromacs.setup._setup_MD(dirname, struct=struct, top=top, deffnm=deffnm, mdp=mdp,
                                       mainselection=mainselection, **kwargs)

    setup_MD["dirname"] = dirname

    logging.debug("finished _setup_MD, recieved: {}".format(setup_MD))

    return MDManager(setup_MD)


def topology(struct, protein="protein", dirname="top", ff="charmm27", water="spc", ignh=True, **top_args):
    """
    Generate a topology for a given structure.

    @return a dict with the following keys: {"top", "struct", "dirname"}, where
    the values are the file names of the resulting topology, structure, and position restraint files.
    """

    logging.info("autogenerating topology using pdb2gmx...")
    pdb2gmx_args = {"ff":ff, "water":water, "ignh":ignh}
    pdb2gmx_args.update(top_args)
    struct = data_tofile(struct, "src.pdb", dirname=dirname)
    result = gromacs.setup.topology(struct, protein, "system.top", dirname, **pdb2gmx_args)
    result["dirname"] = dirname

    print("result: {}".format(result))

    return MDManager(result)


def solvate(struct, top, top_includes=None, box = None,
            concentration=0, cation='NA', anion='CL',
            water='spc', solvent_name='SOL', with_membrane=False,
            ndx = 'main.ndx', mainselection = '"Protein"',
            dirname=None, deffnm='sol',
            **kwargs):
    """Put protein into box, add water, add counter-ions.

    Currently this really only supports solutes in water. If you need
    to embedd a protein in a membrane then you will require more
    sophisticated approaches.

    However, you *can* supply a protein already inserted in a
    bilayer. In this case you will probably want to set *distance* =
    ``None`` and also enable *with_membrane* = ``True`` (using extra
    big vdw radii for typical lipids).

    .. Note:: The defaults are suitable for solvating a globular
       protein in a fairly tight (increase *distance*!) dodecahedral
       box.

    :Arguments:
      *struct* : MDAnalysis universe, AtomGroup or data buffer. This is the starting
          dry, unsolvated structure.

      *top* : filename
          Gromacs topology
      *distance* : float
          When solvating with water, make the box big enough so that
          at least *distance* nm water are between the solute *struct*
          and the box boundary.
          Set *boxtype*  to ``None`` in order to use a box size in the input
          file (gro or pdb).
      *boxtype* : string
          Any of the box types supported by :class:`~gromacs.tools.Editconf`
          (triclinic, cubic, dodecahedron, octahedron). Set the box dimensions
          either with *distance* or the *box* and *angle* keywords.

          If set to ``None`` it will ignore *distance* and use the box
          inside the *struct* file.
      *box* : May be either an numpy array, or a list.
          If the struct is an MDAnalysis obj, the box is obtained from the
          MDAnalysis trajectory.
          .
      *angles*
          List of three angles (only necessary for triclinic boxes).
      *concentration* : float
          Concentration of the free ions in mol/l. Note that counter
          ions are added in excess of this concentration.
      *cation* and *anion* : string
          Molecule names of the ions. This depends on the chosen force field.
      *water* : string
          Name of the water model; one of "spc", "spce", "tip3p",
          "tip4p". This should be appropriate for the chosen force
          field. If an alternative solvent is required, simply supply the path to a box
          with solvent molecules (used by :func:`~gromacs.genbox`'s  *cs* argument)
          and also supply the molecule name via *solvent_name*.
      *solvent_name*
          Name of the molecules that make up the solvent (as set in the itp/top).
          Typically needs to be changed when using non-standard/non-water solvents.
          ["SOL"]
      *with_membrane* : bool
           ``True``: use special ``vdwradii.dat`` with 0.1 nm-increased radii on
           lipids. Default is ``False``.
      *ndx* : filename
          How to name the index file that is produced by this function.
      *mainselection* : string
          A string that is fed to :class:`~gromacs.tools.Make_ndx` and
          which should select the solute.
      *dirname* : directory name
          Name of the directory in which all files for the solvation stage are stored.
      *includes*
          List of additional directories to add to the mdp include path
      *kwargs*
          Additional arguments are passed on to
          :class:`~gromacs.tools.Editconf` or are interpreted as parameters to be
          changed in the mdp file.

    """

    if dirname is None:
        dirname = tempfile.mkdtemp(prefix="tmp." + deffnm + ".")
        logging.debug("created solvation dir {}".format(dirname))

    # The box, a list of three box lengths [A,B,C] that are used by :class:`~gromacs.tools.Editconf`
    # in combination with *boxtype* (``bt`` in :program:`editconf`) and *angles*.
    if box is None and isinstance(struct, MDAnalysis.core.AtomGroup.Universe):
        # convert to nm
        box = struct.trajectory.ts.dimensions[:3] / 10.0
    else:
        if not (isinstance(box, numpy.ndarray) or isinstance(box, list)) or len(box) != 3:
            raise ValueError("box must be either a length 3 numpy array or list")

    # build the substitution index.
    # TODO: Verify that solvate only adds atoms after the dry structure
    # current logic, to be verified, is that the sub index (the original atom indices)
    # are the first n atoms in the resulting n + nsol solvated structure.
    # The sub indices are used to pick out the original strucure out of
    # the n + nsol atom trr trajectory files.
    sub = None

    if isinstance(struct, MDAnalysis.core.AtomGroup.Universe):
        sub = numpy.arange(len(struct.atoms))

    struct = data_tofile(struct, "src.pdb", dirname=dirname)

    if sub is None:
        u = MDAnalysis.Universe(struct)
        sub = numpy.arange(len(u.atoms))

    top = data_tofile(top, "src.top", dirname=dirname)

    # dump the included itp files to the dir where solvation occurs.
    logging.debug("top_includes: {}".format(top_includes))
    for i in top_includes:
        logging.debug("copying file {} to {}".format(i,dirname))
        data_tofile(i,dirname=dirname)

    result = gromacs.setup.solvate(struct, top,
            1.0, "triclinic",
            concentration, cation, anion,
            water, solvent_name, with_membrane,
            ndx, mainselection,
            dirname, box=list(box))
    result["dirname"] = dirname
    result["top"] = top
    result["sub"] = sub

    return MDManager(result)

def tsol(struct, top, box,
         concentration=0, cation='NA', anion='CL',
            water='spc', solvent_name='SOL', with_membrane=False,
            ndx = 'main.ndx', mainselection = '"Protein"',
            dirname=None, deffnm='sol',):
    gromacs.setup.solvate(struct, top,
            1.0, "triclinic",
            concentration, cation, anion,
            water, solvent_name, with_membrane,
            ndx, mainselection,
            dirname, box=list(box), angles=[90,90,90])


def run_md(dirname, md_runner=MDrunner, **kwargs):
    """
    actually perform the md run.

    does not alter class state, only the file system is changed

    @param kwargs: a dictionary of arguments that are passed to mdrun.

                   mdrun, with -multi multiple systems are simulated in parallel. As many input files are
                   required as the number of systems. The system number is appended to the run input and
                   each output filename, for instance topol.tpr becomes topol0.tpr, topol1.tpr etc.
                   The number of nodes per system is the total number of nodes divided by the number of systems.

                   run_MD automatically creates n copies of the source tpr specified by deffnm.

    @return: a named tuple, currently, this contains a list of resulting trajectgories in
        in result.trajectories.

    """
    Result = namedtuple("Result", ["structs", "trajectories"])

    # pick out the relevant mdrun keywords from kwargs
    mdrun_args = ["s","o","x","cpi","cpo","c","e","g","dhdl","field","table","tablep",
                  "tableb","rerun","tpi","tpid","ei","eo","j","jo","ffout","devout",
                  "runav","px","pf","mtx","dn","multidir","h","version","nice","deffnm",
                  "xvg","pd","dd","nt","npme","ddorder",
                  "ddcheck","rdd","rcon","dlb","dds","gcom","v","compact","seppot",
                  "pforce","reprod","cpt","cpnum","append","maxh","multi","replex",
                  "reseed","ionize"]
    kwargs = dict([(i, kwargs[i]) for i in kwargs.keys() if i in mdrun_args])

    # figure out what the output file is, try set default output format to pdb
    structs = None
    if kwargs.has_key("deffnm"):
        if kwargs.has_key("c"):
            structs = kwargs["c"]
        else:
            structs = kwargs["deffnm"] + ".pdb"
            kwargs["c"] = structs
    elif kwargs.has_key["c"]:
        structs = kwargs["c"]
    else:
        # default name according to mdrun man
        structs = "confout.gro"

    if kwargs.has_key("multi"):
        split = os.path.splitext(structs)
        structs = [split[0] + str(i) + split[1] for i in range(kwargs["multi"])]
    else:
        structs = [structs]
    structs = [os.path.realpath(os.path.join(dirname, s)) for s in structs]

    # create an MDRunner which changes to the specifiec dirname, and
    # calls mdrun in that dir, then returns to the current dir.
    runner = md_runner(dirname, **kwargs)
    runner.run_check()

    trajectories = [os.path.abspath(trr) for trr in glob.glob(dirname + "/*.trr")]

    logging.debug("structs: {}".format(structs))
    logging.debug("pwd: {}".format(os.path.curdir))
    logging.debug("dirname: {}".format(dirname))

    found_structs = [s for s in structs if os.path.isfile(s)]
    notfound_structs = [s for s in structs if not os.path.isfile(s)]

    for s in notfound_structs:
        logging.warn("guessed output file name {} not found, is a problem????".format(s))

    return Result(found_structs, trajectories)

def run_md_debug(dirname, md_runner=MDrunner, **kwargs):
    """
    actually perform the md run.

    does not alter class state, only the file system is changed

    @param kwargs: a dictionary of arguments that are passed to mdrun.

                   mdrun, with -multi multiple systems are simulated in parallel. As many input files are
                   required as the number of systems. The system number is appended to the run input and
                   each output filename, for instance topol.tpr becomes topol0.tpr, topol1.tpr etc.
                   The number of nodes per system is the total number of nodes divided by the number of systems.

                   run_MD automatically creates n copies of the source tpr specified by deffnm.

    @return: a named tuple, currently, this contains a list of resulting trajectgories in
        in result.trajectories.

    """
    Result = namedtuple("Result", ["structs", "trajectories"])

    # pick out the relevant mdrun keywords from kwargs
    mdrun_args = ["s","o","x","cpi","cpo","c","e","g","dhdl","field","table","tablep",
                  "tableb","rerun","tpi","tpid","ei","eo","j","jo","ffout","devout",
                  "runav","px","pf","mtx","dn","multidir","h","version","nice","deffnm",
                  "xvg","pd","dd","nt","npme","ddorder",
                  "ddcheck","rdd","rcon","dlb","dds","gcom","v","compact","seppot",
                  "pforce","reprod","cpt","cpnum","append","maxh","multi","replex",
                  "reseed","ionize"]
    kwargs = dict([(i, kwargs[i]) for i in kwargs.keys() if i in mdrun_args])

    # figure out what the output file is, try set default output format to pdb
    structs = None
    if kwargs.has_key("deffnm"):
        if kwargs.has_key("c"):
            structs = kwargs["c"]
        else:
            structs = kwargs["deffnm"] + ".pdb"
            kwargs["c"] = structs
    elif kwargs.has_key("c"):
        structs = kwargs["c"]
    else:
        # default name according to mdrun man
        structs = "confout.gro"

    if kwargs.has_key("multi"):
        split = os.path.splitext(structs)
        structs = [split[0] + str(i) + split[1] for i in range(kwargs["multi"])]
    else:
        structs = [structs]
    structs = [os.path.realpath(os.path.join(dirname, s)) for s in structs]

    # create an MDRunner which changes to the specifiec dirname, and
    # calls mdrun in that dir, then returns to the current dir.
    #runner = md_runner(dirname, **kwargs)
    #runner.run_check()

    trajectories = [os.path.abspath(trr) for trr in glob.glob(dirname + "/*.trr")]

    print("structs", structs)
    print("pwd", os.path.curdir)
    print("dirname", dirname)

    found_structs = [s for s in structs if os.path.isfile(s)]
    notfound_structs = [s for s in structs if not os.path.isfile(s)]

    for s in notfound_structs:
        logging.warn("guessed output file name {} not found, is a problem????".format(s))

    return Result(found_structs, trajectories)

def check_main_index(struct):
    """
    All of the gromacs.setup functions auto generate an index file. The __main__ group
    of this index has to match the starting 'dry' structure if we are auto-solvating.

    The __main__ group is used to call trjconv to strip away the added waters. This
    approach allows us to continue using the same universe object, reduces code complexity
    and significantly speeds up trajectory file processing.

    This method returns True if the __main__ group matches the given structure, otherwise
    an exception is raised.
    """
    dirname = tempfile.mkdtemp()
    with gromacs.utilities.in_dir(dirname):
        writer = MDAnalysis.Writer("src.pdb")
        writer.write(struct)
        del writer
        groups = gromacs.setup.make_main_index("src.pdb")

    for g in groups:
        if g["name"] == "__main__":
            if g["natoms"] == len(struct.atoms):
                print("Autogenerated index group __main__ has same number of atoms as universe.atoms")
                shutil.rmtree(dirname, ignore_errors=True)
                return True
            else:
                msg = "Autogenerated index group __main__ has" + str(g["natoms"]) + \
                " atoms, but the given universe structure has " + str(len(struct.atoms)) + \
                " atoms. Check structure and verify that the __main__ group gromacs.utilities.make_main_index" + \
                " returns has the same number of atoms."
                raise ValueError(msg)

    raise ValueError("gromacs.setup.make_main_index appears to have worked, but there was no __main__ group")


def top_includes(top, include_dirs=["."]):
    """
    parse a top file, and figure out all the included files
    that can be found in the specified include dirs.
    @param top: file name of a gromacs top file
    @param include_dirs: a list of directories in which to search for the include files.
    @return: a list of absolute paths for files find in the include paths.
    """
    includes = []
    r=re.compile("^\s*#\s*include\s*\"(.*?)\"\s*")

    logging.debug("scaning includes for top level topology file {}".format(top))

    def isfile(f):
        for d in include_dirs:
            j=os.path.join(d, f)
            if os.path.isfile(j):
                return j
        return False

    def get_includes(f):
        logging.debug("recursivly scanning includes for file {}".format(f))


        with open(f) as of:
            for line in of:
                match = r.match(line)
                if match:
                    f=isfile(match.group(1))
                    if f:
                        # first check if we have been here already (cyclic includes)
                        try:
                            includes.index(f)
                            logging.warn("encountered cyclic includes, the file {} has already been included"
                                         ", unknown if this will work with GROMACS???".format(f))
                            logging.warn("ignoring file {} as its already included".format(f))
                        except ValueError:
                            # ugly way of checking if an item is in a list, but hey, geuss this is the Python way...
                            # we've not encountered this file before, so good to go.
                            includes.append(f)
                            get_includes(f)
                    else:
                        logging.debug("found include file {}, but it is not in the include path".format(match.group(1)))

    get_includes(top)
    return includes



#tsol(struct='src.pdb', top='src.top', box=[5, 10, 15], dirname=".")








