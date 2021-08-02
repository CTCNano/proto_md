[//]: # (Badges)
[![CI](https://github.com/CTCNano/proto_md/actions/workflows/CI.yaml/badge.svg)](https://github.com//CTCNano/proto_md/actions/workflows/CI.yaml)

Proto_MD is a prototyping toolkit for multiscale MD.

Authors
=======
Endre Somogyi,
Andrew Abi Mansour

Institution
===========
Indiana University, Bloomington

Prerequisites
=============
- GromacsWrapper: https://github.com/CTCNano/GromacsWrapper
- MDAnalysis: https://code.google.com/p/mdanalysis
- Numpy: http://www.numpy.org
- Scipy: http://www.scipy.org
- OpenMPI: http://www.open-mpi.org or MPICH: https://www.mpich.org
- Gromacs: http://www.gromacs.org

protoMD has been tested with:
- Gromacs 4.6.x and 5.0.x
- MDAnalysis 1.0.0

Installation
============
Installation uses python's setuptools.

To install protoMD, simply run from the cmd line:

`pip install proto_md`

Testing Package
===============
You need to have pytest installed:

`pip install pytest`

and then run from the proto_md src dir:

`pytest -v`

Working Example
===============
A bash script that creates an input (hdf) file and runs
a multiscale simulation is found in the `sample` dir.

To run it, simply execute from the terminal: bash run.mkconf

Source Code
===========
- proto_md/config.py: handles configuration of proto_md environment variables and templates.
- proto_md/correlation.py: functions for computing correlation functions.
- proto_md/diffusion.py: functions for computing diffusion coefficients.
- proto_md/dynamics.py: functions to calculate dynamic properties of CG variables. 
- proto_md/__init__.py: initializer for importing proto_md as a package
- proto_md/__main__.py: functions for arg parsing (when using proto_md as a program)
- proto_md/md.py: functions for setting up and performing md calculations.
- proto_md/system.py: functions for creating the System object, handling I/O, and calling MD functions 
- proto_md/subsystems: contains modules specific to various coarse-graining methods
- proto_md/util: contains modules for reading and writng files to / from disk and hdf blobs
- proto_md/templates: contains GROMACS mdp config template files
- proto_md/integrators: contains modules for various multiscale integrators
- proto_md/fieldVars: C++ code for an (optional) subsystem based on continuum field variables

License
=======
GPL v3
