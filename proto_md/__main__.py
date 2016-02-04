'''

Created on Dec 30, 2012

@author: andy


'''
import system
import config
import util
import sys
import tempfile
import os
import os.path
import argparse
import logging

def make_parser():
    """
    Create the argument parser that processes all the proto_md command line arguments.
    """
    
    parser = argparse.ArgumentParser(prog="python -m proto_md", description="proto_md - coarse grained molecular dynamics")
    subparsers = parser.add_subparsers()

    # create the config argument parser, lots and lots of junk can be handled
    # by the create config function
    ap = subparsers.add_parser("config", help="create a simulation database, from which a simulation can be run")
    
    ap.add_argument("-o", dest="fid", required=True, type=str,
                    help="name of the simulation file to be created.")
    
    ap.add_argument("-struct", dest="struct", required=True, type=str,
                    help="the starting structure name")
    
    ap.add_argument("-box", dest="box", required=False, nargs=3, type=float, default=None,
                    help="x,y,z values of the system box in Angstroms. "
                         "If box is not given, the system size is read from the CRYST line "
                         "in the structure pdb.")
    
    ap.add_argument("-top", dest="top", required=False, default=None,
                    help="name of the topology file. If this is not give, and topology file \
                    will be automatically generated, well at least, will attemtp to be automatically \
                    generated.")
    
    ap.add_argument("-posres", dest="posres", required=False,
                    help="name of a position restraints file, optional.")

    ap.add_argument('-I', action='append', dest='include_dirs',
                    default=[], type=str, required=False,
                    help="Include directories to search for topology file includes. There can be many " 
                         "additional include directories, just like gcc, but UNLIKE GCC, there must be a space "
                         "between the -I and the dir, for example -I /home/foo -I /home/foo/bar.")
    
    ap.add_argument("-temperature", dest="temperature", required=False, type=float, default= 300,
                    help="the temperature at which to run the simulation, defaults to 300K.")
    
    ap.add_argument("-subsystem_factory", dest="subsystem_factory", required=False, 
                    default="proto_md.subsystems.RigidSubsystemFactory",
                    help="fully qualified function name which can create a set of subsystems, "
                         "can be set to \'proto_md.subsystems.LegendreSubsystemFactory\'")


    ap.add_argument("-top_args", default={'ff':'charmm27', 'protein':'protein'}, type=eval,
                    help="topology (GROMACS) dictionary file")
                  
    ap.add_argument("-subsystem_selects", dest="subsystem_selects", required=False, 
                    nargs="+", default=["not resname SOL"],
                    help="a list of MDAnalysis select statements, one for each subsystem.")
    
    ap.add_argument("-subsystem_args", dest="subsystem_args", required=False, type=eval, 
                    default={'kmax':0},
                    help="a list of additional arguments passed to the subsystem factory, "
                         "the first item of the list may be the string \'resid unique\', which "
                         "creates a separate subsystem for each residue. " 
                         ""
                         "The most comonly used subsystem is proto_md.subsystems.LegendreSubsystemFactory, "
                         "The args for this subsystem are [kmax, OPTIONAL(\"resid unique\")], "
                         "kmax is the highest Legendre polynomial index to use, "
                         "and the last arg is the optional string \"resid unique\" to make a unique "
                         "subsystem for each residue." )
    
    ap.add_argument("-integrator", default="proto_md.integrators.FactorizationIntegrator",
                    help="fully qualified name of the integrator function, "
                         "defaults to \'proto_md.integrators.FactorizationIntegrator\', "
                         "but the other integrator we provide is \'proto_md.integrators.LangevinIntegrator\'")

    ap.add_argument("-integrator_args", default = [], nargs='+',
                    help="additional arguments passed to the integrator function")


    ap.add_argument('-Imdp', action='append', dest='include_mdp_dirs',
                    default='templates', type=str, required=False,
                    help="Include directories to search for mdp config files. There can be many "
                         "additional include directories, just like gcc, but UNLIKE GCC, there must be a space "
                         "between the -Imdp and the dir, for example -Imdp /home/foo -Imdp /home/foo/bar.")

    ap.add_argument("-anion", dest="anion", required=False, type=str, default='CL',
                    help="anion name to use for ionization (if system is not neutral)")

    ap.add_argument("-cation", dest="cation", required=False, type=str, default='NA', 
                    help="cation name to use for ionization (if system is not neutral)")

    ap.add_argument("-concentration", dest="concentration", required=False, type=float, default=0.15, 
                    help="concentration (in mol/L) of ions to use for ionization (if system is not neutral)")
    
    ap.add_argument("-cg_steps", default = 10,
                    help="number of coarse grained time steps")
    
    ap.add_argument("-dt", default=0.1, 
                    help="size of coarse grained time step in picoseconds")
    
    ap.add_argument("-mn_steps", default = 500,
                    help="number of MD steps to take performing a minimization")
    
    ap.add_argument("-md_steps", default=100,
                    help="number of MD steps to take whilst performing the MD runs")
    
    ap.add_argument("-multi", default=1,
                    help="number of parallel MD runs")
    
    ap.add_argument("-eq_steps", default=10, 
                    help="number of timesteps to run equilibriation")

    ap.add_argument("-mn_args", default=config.DEFAULT_MN_ARGS, 
                    help="additional parameters to be changed in the minimization options", type=eval)

    ap.add_argument("-eq_args", default=config.DEFAULT_EQ_ARGS, type=eval, help="equilibration (GROMACS) config file")
    ap.add_argument("-md_args", default=config.DEFAULT_MD_ARGS, type=eval, help="MD (GROMACS) config file")

    ap.add_argument("-ndx", default=None, help="name of index file")


    ap.add_argument("-center", action="store_true", help="centers the system in the supplied box before beginning simulation")

    ap.add_argument("-solvate", dest="should_solvate", action="store_true",
                    help="should the system be auto-solvated, if this is set, struct must NOT contain solvent. "
                    "defaults to False. This is a boolean flag, to enable, just add \'-solvate\' with no args.")
    
    ap.add_argument("-debug", action="store_true", help="enable debug mode (save all simulation directories).")
    
    ap.add_argument("-mainselection", default="Protein", 
                    help="The name of make_ndx group which is used for the main selection. This should"
                    " be a group that consists of the non solvent molecules. \"Protein\" usually works when "
                    "simulating protein, however another selection command is required when simulationg lipids. "
                    "To find this out, either run proto_md config, and an error will pop up informing you of the available "
                    "groups, or run make_ndx to get a list.")

    ap.set_defaults(__func__=config.create_sim)
    
    
    ap = subparsers.add_parser("top", help="Given a atomic structure file, auto-generate a topology, "
                               "and save it in a directory. "
                               "This is usefull for testing topo auto generation. ")
    ap.add_argument("-o", help="output directory where topology will be generated")
    ap.add_argument("-struct", dest="struct", required=True, type=str,
                    help="the starting structure name")
    ap.add_argument("-posres", dest="posres", required=False,
                    help="name of a position restraints file, optional.")
    ap.set_defaults(__func__=config.create_top)


    #create_sol(o, struct, posres, top, box=None):
    ap = subparsers.add_parser("solvate", help="Given a atomic structure file, attempt auto-solvation, "
                               "and save it in a directory. "
                               "This is usefull for testing auto-solvation. "
                               "All the inputs to solvate are returned from top")
    ap.add_argument("-o", help="output directory where solvated structure will be generated")
    ap.add_argument("-struct", dest="struct", required=True, type=str,
                    help="the starting structure name")
    ap.add_argument("-top", dest="top", required=True,
                    help="name of the topology file.")
    ap.add_argument("-box", dest="box", required=False, nargs=3, type=float, default=None,
                    help="x,y,z values of the system box in Angstroms. "
                         "If box is not given, the system size is read from the CRYST line "
                         "in the structure pdb.")
    ap.add_argument("-debug", action="store_true", help="enable debug mode (save all simulation directories).")
    ap.set_defaults(__func__=config.create_sol)
    
    # Done with config, the MOST complicated command, now make parsers for the more
    # simple commands
    ap = subparsers.add_parser("run", help="start or continue a full simulation. "
                               " This will automatically continue a simulation",)
    ap.add_argument("sys", help="name of simulation file")
    ap.add_argument("-debug", action="store_true", help="enable debug mode (save all simulation directories).")
    
    def run(sys, debug) :
        config.set_tempdir(sys)
        if debug:
            os.environ["PROTO_DEBUG"] = "TRUE"

        s=system.System(sys, "a")
        integrator = s.integrator()
        integrator.run()
    ap.set_defaults(__func__=run)

    ap = subparsers.add_parser("runsol", help="Like run, but perfoms only the auto-solvation step. " 
                               "Usefull for debugging run time auto-solvation. "
                               "It unlikely this will be needed as the auto-solvation can be tested "
                               "directly from the struture / top files," )
    ap.add_argument("sys", help="name of simulation file")
    ap.add_argument("-debug", action="store_true", help="enable debug mode (save all simulation directories).")
    def sol(sys, debug) :
        config.set_tempdir(sys)
        if debug:
            os.environ["PROTO_DEBUG"] = "TRUE"
        s=system.System(sys, "a")
        s.begin_timestep()
        s.solvate()
        s.end_timestep()  
    ap.set_defaults(__func__=sol)
    


    ap = subparsers.add_parser( "mn", help="perform a energy minimization")
    ap.add_argument("sys", help="name of simulation file")
    ap.add_argument("-sol", action="store_true", help="auto solvate before minimization")
    ap.add_argument("-debug", action="store_true", help="enable debug mode (save all simulation directories).")
    def mn(sys, sol, debug):
        config.set_tempdir(sys)
        if debug:
            os.environ["PROTO_DEBUG"] = "TRUE"
        s=system.System(sys, "a")
        s.begin_timestep()
        if sol:
            sol = s.solvate()
            mn = s.minimize(**sol)
            s.end_timestep()
        else:
            s.minimize()
        s.end_timestep()
    ap.set_defaults(__func__=mn)

    ap = subparsers.add_parser("mneq", help="performs energy minimization followed by equilibriation")
    ap.add_argument("sys", help="name of simulation file")
    ap.add_argument("-sol", action="store_true", help="auto solvate between steps")
    ap.add_argument("-debug", action="store_true", help="enable debug mode (save all simulation directories).")

    def mneq(sys,sol,debug) :
        config.set_tempdir(sys)
        if debug:
            os.environ["PROTO_DEBUG"] = "TRUE"
        s=system.System(sys, "a")
        s.begin_timestep()
        if sol:
            sol = s.solvate()
            mn = s.minimize(**sol)
            eq = s.equilibriate(**mn)
        else:
            s.minimize()
            s.equilibriate()
        s.end_timestep()    
    ap.set_defaults(__func__=mneq)

    ap = subparsers.add_parser("mneqmd", help="performs energy minimzatin, equilibriation and molecular dynamics")
    ap.add_argument("sys", help="name of simulation file")
    ap.add_argument("-sol", action="store_true", help="auto solvate between steps")
    ap.add_argument("-debug", action="store_true", help="enable debug mode (save all simulation directories).")
    def mneqmd(sys,sol,debug) :
        config.set_tempdir(sys)
        if debug:
            os.environ["PROTO_DEBUG"] = "TRUE"
        s=system.System(sys, "a")
        s.begin_timestep()
        if sol:
            sol = s.solvate()
            mn = s.minimize(**sol)
            eq = s.equilibriate(**mn)
            s.md(**eq)
        else:
            s.minimize()
            s.equilibriate()
            s.md()
        s.en
        s.end_timestep()
    ap.set_defaults(__func__=mneqmd)

    ap = subparsers.add_parser("eq", help="performs an equilibriation")
    ap.add_argument("sys", help="name of simulation file")
    ap.add_argument("-sol", action="store_true", help="auto solvate between steps")
    ap.add_argument("-debug", action="store_true", help="enable debug mode (save all simulation directories).")
    def eq(sys,debug) :
        config.set_tempdir(sys)
        if debug:
            os.environ["PROTO_DEBUG"] = "TRUE"
        s=system.System(sys, "a")
        s.begin_timestep()
        if sol:
            sol = s.solvate()
            eq = s.equilibriate(**sol)
        else:
            s.equilibriate()
        s.end_timestep()
    ap.set_defaults(__func__=eq)

    ap = subparsers.add_parser("atomistic_step", help="perform an full atomistic step")
    ap.add_argument("sys", help="name of simulation file")
    ap.add_argument("-debug", action="store_true", help="enable debug mode (save all simulation directories).")
    def atomistic_step(sys,debug) :
        config.set_tempdir(sys)
        if debug:
            os.environ["PROTO_DEBUG"] = "TRUE"
        s=system.System(sys, "a")
        integrator = s.integrator()
        s.begin_timestep()
        integrator.atomistic_step()
        s.end_timestep()
    ap.set_defaults(__func__=atomistic_step)

    ap = subparsers.add_parser("step", help="a single complete Langevin step")
    ap.add_argument("sys", help="name of simulation file")
    ap.add_argument("-debug", action="store_true", help="enable debug mode (save all simulation directories).")
    def step(sys,debug) :
        config.set_tempdir(sys)
        if debug:
            os.environ["PROTO_DEBUG"] = "TRUE"
        s=system.System(sys, "a")
        integrator = s.integrator()
        integrator.step()
    ap.set_defaults(__func__=step)

    ap = subparsers.add_parser("md", help="perform ONLY an molecular dynamics step with the starting structure")
    ap.add_argument("sys", help="name of simulation file")
    ap.add_argument("-sol", action="store_true", help="auto solvate before md")
    ap.add_argument("-debug", action="store_true", help="enable debug mode (save all simulation directories).")
    def md(sys,sol,debug) :
        config.set_tempdir(sys)
        if debug:
            os.environ["PROTO_DEBUG"] = "TRUE"
        s=system.System(sys, "a")
        s.begin_timestep()
        if sol:
            sol = s.solvate()
            s.md(**sol)
        else:
            s.md()
        s.end_timestep()
    ap.set_defaults(__func__=md)

    ap = subparsers.add_parser("cg_step", help="perform just the coarse grained portion of the time step")
    ap.add_argument("sys", help="name of simulation file")
    ap.add_argument("-debug", action="store_true", help="enable debug mode (save all simulation directories).")
    def cg_step(sys,debug) :
        config.set_tempdir(sys)
        if debug:
            os.environ["PROTO_DEBUG"] = "TRUE"
        s=system.System(sys, "a")
        s._load_ts(s.current_timestep)
        integrator = s.integrator()
        s.begin_timestep()
        integrator.cg_step()
        s.end_timestep()
    ap.set_defaults(__func__=cg_step)
    
    return parser
    
# make the arg parser, and call whatever func was stored with the arg. 
parser = make_parser()
args = parser.parse_args()
func = args.__func__
del args.__dict__["__func__"]
func(**args.__dict__)
