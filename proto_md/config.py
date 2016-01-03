"""
Created on Oct 11, 2012

@author: andy

Handles configuration of proto_md environment variables and templates.

All proto relevant environment variables can be accessed as config.proto_???, 
for example, to check if debug is on, one would check config.PROTO_DEBUG

Currently used proto_md enviornment variables and their default values are:

PROTO_DEBUG : False
PROTO_TMPDIR : "."

"""
from pkg_resources import resource_filename, resource_listdir  #@UnresolvedImport
import os

from numpy import array, fromfile, uint8, all
import tempfile
import MDAnalysis                       #@UnresolvedImport
import h5py                             #@UnresolvedImport
import md
import util
import collections
import logging


def _generate_template_dict(dirname):
    """
    Generate a list of included files *and* extract them to a temp space.
    
    Templates have to be extracted from the egg because they are used
    by external code. All template filenames are stored in
    :data:`config.templates`.
    """
    return dict((resource_basename(fn), resource_filename(__name__, dirname+'/'+fn))
                for fn in resource_listdir(__name__, dirname)
                if not fn.endswith('~'))

def resource_basename(resource):
    """Last component of a resource (which always uses '/' as sep)."""
    if resource.endswith('/'):
        resource = resource[:-1]
    parts = resource.split('/')
    return parts[-1]
    

templates = _generate_template_dict('templates')
"""
A dictionary of pre-made templates. 
 
TODO, this should be cleaned up: if a user specifies a mdp template, then that template
should be stored in the hdf file. The current approach violates the principle that the 
entire config is in the hdf.
    
*proto* comes with a number of templates for run input files
and queuing system scripts. They are provided as a convenience and
examples but **WITHOUT ANY GUARANTEE FOR CORRECTNESS OR SUITABILITY FOR
ANY PURPOSE**.

All template filenames are stored in
:data:`proto.config.templates`. Templates have to be extracted from
the proto python egg file because they are used by external
code: find the actual file locations from this variable.
"""


# Initialized the proto enviornment variables
PROTO_TMPDIR = "."


"""
Botlzmann's constant in kJ/mol/K
"""
KB = 0.0083144621


CURRENT_TIMESTEP = "current_timestep" 
SRC_FILES = "src_files"
FILE_DATA_LIST = ["struct.gro", "topol.top", "index.ndx", "posres.itp"]
TOP_INCLUDES = "top_includes"
TIMESTEPS = "timesteps"
CONFIG = "config"
STRUCT_PDB = "struct.gro"
TOPOL_TOP = "topol.top"
INDEX_NDX = "index.ndx"
CG_STEPS = "cg_steps"
MN_STEPS = "mn_steps"
EQ_STEPS = "eq_steps"
MD_STEPS = "md_steps"
INCLUDE_MDP_DIRS = "include_mdp_dirs"

# All values are stored in the hdf file, however there is no direct
# way to store a python dict in hdf, so we need to store it as
# a set of key / value lists.
# This is a bit hackish, but to store them as lists, the following
# 'key names' will have '_keys' and '_values' automatically appended
# to them, then the corresponding keys / values will be stored as 
# such. 
MN_ARGS = "mn_args"
EQ_ARGS = "eq_args"
MD_ARGS = "md_args"
TOP_ARGS = "top_args"
KEYS="_keys"
VALUES="_values"
MULTI= "multi"
SHOULD_SOLVATE="should_solvate"
POSRES_ITP="posres.itp"
SUBSYSTEM_FACTORY = "subsystem_factory"
SUBSYSTEM_SELECTS = "subsystem_selects"
SUBSYSTEM_ARGS = "subsystem_args"
NSTXOUT = "nstxout"
NSTVOUT = "nstvout"
NSTFOUT = "nstfout"
BOX = "box"
CATION = "cation"
ANION = "anion"
CONCENTRATION = "concentration"
TEMPERATURE="temperature"
DT = "dt"
INTEGRATOR = "integrator"
INTEGRATOR_ARGS = "integrator_args"
MAINSELECTION = "mainselection"
INCLUDE_MDP_DIRS = "include_mdp_dirs"

DEFAULT_MN_ARGS = {"mdp":"em.mdp"}

DEFAULT_MD_ARGS = { "mdp":"md_CHARMM27.mdp",  # the default mdp template 
                    "nstxout": 10,    # trr pos
                    "nstvout": 10,    # trr veloc
                    "nstfout": 10,    # trr forces
                    "dt":      0.001  # trr time step (1fs = 0.001ps)
                    }

DEFAULT_EQ_ARGS = { "mdp":"md_CHARMM27.mdp",  # the default mdp template 
                    "define":"-DPOSRES", # do position restrained md for equilibriation
                    "dt": 0.001 # trr time step (1fs = 0.001ps)
                    }


def create_top(o, struct, posres):
    print("attempting to auto-generate a topology...")
    
    top = md.topology(struct=struct, protein="protein", posres=posres, dirname=o)
        # topology returns:
        # {'top': '/home/andy/tmp/Au/top/system.top', 
        # 'dirname': 'top', 
        # 'posres': 'protein_posres.itp', 
        # 'struct': '/home/andy/tmp/Au/top/protein.gro'}
        
    print("succesfully auto-generated topology")        

    """
        
    # check to see if solvation is possible
    if solvate:
        # convert Angstrom to Nm, GROMACS works in Nm, and
        # we use MDAnalysis which uses Angstroms
        print("attempting auto solvation...")
        with md.solvate(box=box/10.0, **top):
            # solvate returns 
            # {'ndx': '/home/andy/tmp/Au/solvate/main.ndx', 
            # 'mainselection': '"Protein"', 
            # 'struct': '/home/andy/tmp/Au/solvate/solvated.gro', 
            # 'qtot': 0})
            print("auto solvation successfull")
    """

def create_sol(o, struct, top, posres, box=None):
    # make a top if we don't have one
    
    # topology returns:
    # {'top': '/home/andy/tmp/Au/top/system.top', 
    # 'dirname': 'top', 
    # 'posres': 'protein_posres.itp', 
    # 'struct': '/home/andy/tmp/Au/top/protein.gro'}
    
 
    # convert Angstrom to Nm, GROMACS works in Nm, and
    # we use MDAnalysis which uses Angstroms
    print("attempting auto solvation...")
    
    if box is None:
        universe = MDAnalysis.Universe(struct)
        box=universe.trajectory.ts.dimensions[:3]
    else:
        # box could be anything, fine as long as we can convert to an array
        box = array(box) 
        print("user specified periodic boundary conditions: {}".format(box))
        
    sol = md.solvate(dirname=o,struct=struct,top=top,box=box/10.0)
    # solvate returns 
    # {'ndx': '/home/andy/tmp/Au/solvate/main.ndx', 
    # 'mainselection': '"Protein"', 
    # 'struct': '/home/andy/tmp/Au/solvate/solvated.gro', 
    # 'qtot': 0})
    print("auto solvation successfull")
                
def create_sim(fid,
               struct,
               box = None,
               top = None,
               temperature = 300,
               subsystem_factory = "proto_md.subsystems.RigidSubsystemFactory",
               subsystem_selects = ["not resname SOL"],
               subsystem_args = [],
               integrator = "proto_md.integrators.LangevinIntegrator",
               integrator_args = [],
               cg_steps = 10,
               dt  = 0.1,
               mn_steps = 500,
               md_steps = 100,
               multi = 1,
               eq_steps = 10,
               should_solvate = False,
               ndx=None,
               include_dirs=[],
	       include_mdp_dirs=[],
               mainselection = '"Protein"',
	       concentration = 0.15,
	       anion = 'CL',
	       cation = 'NA',
               debug=False,
               **kwargs):
    """
    Create the simulation file
    
    TODO bring the __main__ docs here
    """
    
    import gromacs
    
    if debug:
        os.environ["PROTO_DEBUG"] = "TRUE"
    
    # need to create a universe to read various bits for the config, and
    # to test subsystem creation, so keep it around for this func, 
    # plus, makeing a universe is a sure fire way to see if we have a valid 
    # structure.
    universe = None
    
    # path of centered struct
    centered_struct = None
    
    set_tempdir(fid)
    
    with h5py.File(fid, "w") as hdf:
        conf = hdf.create_group("config").attrs
        src_files = hdf.create_group("src_files")
        top_includes = hdf.create_group(TOP_INCLUDES)

        def filedata_fromfile(keyname, filename):
            try:
                del src_files[str(keyname)]
            except KeyError:
                pass
            src_files[str(keyname)] = fromfile(filename, dtype=uint8)

        # grab the files that a top includes, must be called in the
        # dir the top file is in. 
        def read_includes(top):
            include_dirs.append(".")
            includes = md.top_includes(top, include_dirs)
            for i in includes:
                print("found include file: {}".format(i))
                top_includes[os.path.split(i)[1]] = fromfile(i,dtype=uint8)

        # create an attr key /  value in the config attrs
        def attr(keyname, typ, value):
            if value is not None:
                try:
                    if typ is dict:
                        conf[keyname + KEYS] = value.keys()
                        conf[keyname + VALUES] = value.values()
                    else:
                        conf[keyname] = typ(value)
                except Exception, e:
                    print("error, could not convert \"{}\" with value of \"{}\" to an {} type".
                          format(keyname, value, typ))
                    raise e
                
        # check struct, this only checks to see if file s valid.
        try:
            universe = MDAnalysis.Universe(struct)
            print("structure file {} appears OK".format(struct))
            
        except Exception, e:
            print("structure file {} is not valid".format(struct))
            raise e
                
        try:
            if box:
                box = array(box) 
                print("user specified periodic boundary conditions: {}".format(box))
            else:
                box=universe.trajectory.ts.dimensions[:3]
                print("using pdb specified periodic boundary conditions: {}".format(box))
            conf[BOX] = box
        except Exception, e:
            print("error reading periodic boundary conditions")
            raise e
        
        # we now have a box and a universe, check to see if box is valid size, and 
        # center atoms in box. 
        bbox = universe.atoms.bbox()
        bbox = bbox[1,:] - bbox[0,:]
        
        # box should be at least 110% of atomic 
        if all(box >= 1.1 * bbox):
            # box is OK, center molecule in box
            print("centering molecule in box, new center of mass will be {}".format(box / 2))
            trans = box / 2 - universe.atoms.centerOfMass()
            universe.atoms.translate(trans)
            fd, centered_struct = tempfile.mkstemp(suffix=".gro")
            writer = MDAnalysis.Writer(centered_struct)
            struct = centered_struct
            writer.write(universe)
            writer.close()
            del writer
            os.close(fd)
        else:
            raise ValueError("The given box, {} needs to be at least 110% of the molecule extents, {}".
                             format(box, bbox))
        
        attr(CG_STEPS, int, cg_steps)
        attr(DT, float, dt)
        attr(TEMPERATURE, float, temperature)
        attr(MN_STEPS, int, mn_steps)
        attr(MD_STEPS, int, md_steps)
        attr(MULTI, int, multi)
        attr(EQ_STEPS, int, eq_steps)
	attr(INCLUDE_MDP_DIRS, str, include_mdp_dirs)

        attr(SHOULD_SOLVATE, int, should_solvate)
        attr(MAINSELECTION, str, mainselection)
	attr(CONCENTRATION, float, concentration)
	attr(ANION, str, anion)
	attr(CATION, str, cation)

	attr(MN_ARGS, dict, kwargs['mn_args'])
        attr(EQ_ARGS, dict, kwargs['eq_args'])
        attr(MD_ARGS, dict, kwargs['md_args'])
        attr(TOP_ARGS, dict, kwargs['top_args'])        

        # try to create an integrator
        attr(INTEGRATOR, str, integrator)
        attr(INTEGRATOR_ARGS, list, integrator_args)
        try:
            integrator_type = util.get_class(conf[INTEGRATOR])
            integrator_type(None, conf[INTEGRATOR_ARGS], templates)
            print("succesfully created integrator {}".format(integrator))
        except Exception, e:
            print("error creating integrator {}".format(integrator))
            raise e
        
        # make a top if we don't have one
        if top is None:
            print("attempting to auto-generate a topology...")
            with md.topology(struct=struct, protein="protein") as top:
                # topology returns:
                # {'top': '/home/andy/tmp/Au/top/system.top', 
                # 'dirname': 'top', 
                # 'struct': '/home/andy/tmp/Au/top/protein.gro'}

                print("succesfully auto-generated topology")

                filedata_fromfile(TOPOL_TOP, top["top"])
                filedata_fromfile(STRUCT_PDB, top["struct"])
                include_dirs = [os.path.abspath(top.dirname)]
                read_includes(top["top"])
                
                # check to see if solvation is possible
                if should_solvate:
                    # convert Angstrom to Nm, GROMACS works in Nm, and
                    # we use MDAnalysis which uses Angstroms
                    print("attempting auto solvation...")
                    with md.solvate(top_includes=top_includes.values(), box=box/10.0, concentration=concentration, 
			 anion=anion, cation=cation, **top):
                        # solvate returns 
                        # {'ndx': '/home/andy/tmp/Au/solvate/main.ndx', 
                        # 'mainselection': '"Protein"', 
                        # 'struct': '/home/andy/tmp/Au/solvate/solvated.gro', 
                        # 'qtot': 0})
                        print("auto solvation successfull")
            
        else:
            # use user specified top
            print("using user specified topology file {}".format(top))

            filedata_fromfile(TOPOL_TOP, top)
            filedata_fromfile(STRUCT_PDB, struct)
            read_includes(top)
            
            # check to see if solvation is possible
            if should_solvate:
                # convert Angstrom to Nm, GROMACS works in Nm, and
                # we use MDAnalysis which uses Angstroms
                print("attempting auto solvation...")
                with md.solvate(struct=struct, top=top, top_includes=top_includes.values(), 
                                box=box/10.0, mainselection=mainselection, concentration=concentration, 
				anion=anion, cation=cation):
                    # solvate returns 
                    # {'ndx': '/home/andy/tmp/Au/solvate/main.ndx', 
                    # 'mainselection': '"Protein"', 
                    # 'struct': '/home/andy/tmp/Au/solvate/solvated.gro', 
                    # 'qtot': 0})
                    print("auto solvation successfull")
            
        # try to make the subsystems.
        try:
            # make a fake 'System' object so we can test the subsystem factory.
            dummysys = None
            with tempfile.NamedTemporaryFile(suffix=".gro") as f:  
                # EXTREMLY IMPORTANT to flush the file, 
                # NamedTemporaryFile returns an OPEN file, and the array writes to this file, 
                # so we need to flush it, as this file handle remains open. Then the MDAnalysis
                # universe opens another handle and reads its contents from it. 
                data = src_files[STRUCT_PDB][()]
                data.tofile(f.file)
                f.file.flush()
                universe = MDAnalysis.Universe(f.name)
                dummysys = collections.namedtuple('dummysys', 'universe')(universe)
                
            # we have a fake sys now, can call subsys factory
            factory = util.get_class(subsystem_factory)
            print("created subsystem factory {}, attempting to create subsystems...".format(subsystem_factory))
            test_ncgs, test_ss = factory(dummysys, subsystem_selects, *(subsystem_args))
            
            print("subsystem factory appears to work, produces {} cg variables for each {} subsystems.".format(test_ncgs, len(test_ss)))
            
            conf[SUBSYSTEM_FACTORY] = subsystem_factory
            conf[SUBSYSTEM_SELECTS] = subsystem_selects
            conf[SUBSYSTEM_ARGS] = subsystem_args

        except Exception, e:
            print("error creating subsystem_class, {}".format(e))
            raise e

        hdf.create_group("timesteps")
        
        # clean up temp centered struct path
        os.remove(centered_struct)
        print("creation of simulation file {} complete".format(fid))
        
        
def set_tempdir(f):
    """ 
    get the absolute path of the directory containing the given file name, 
    and set the tmpdir env var to point to this dir. 
    """
    abspath = os.path.abspath(f)
    d = os.path.split(abspath)[0]
    
    # make sure this is an actual directory
    if not os.path.isdir(d):
        raise ValueError("Error, the file {} is not located in a valid directory {}".format(f,d))

    # names that os.gettempdir looks at
    os.environ["TMPDIR"] = d
    os.environ["TEMP"] = d
    os.environ["TMP"] = d
    tempfile.tempdir = d
    logging.info("tempdir for MD runs will be {}".format(tempfile.gettempdir()))
