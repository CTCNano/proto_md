proto_md
========
Prototyping toolkit for multiscale MD

[//]: # (Badges)
[![CI](https://github.com/CTCNano/proto_md/actions/workflows/CI.yaml/badge.svg)](https://github.com//CTCNano/proto_md/actions/workflows/CI.yaml)
[![codecov](https://codecov.io/gh//CTCNano/proto_md/branch/master/graph/badge.svg)](https://codecov.io/gh//CTCNano/proto_md/branch/master)
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g//CTCNano/proto_md.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g//CTCNano/proto_md/context:python)

Authors
=======
Endre Somogyi,
Andrew Abi Mansour

Institution
===========
Indiana University, Bloomington

Prerequisites
=============
GromacsWrapper - https://github.com/CTCNano/GromacsWrapper
MDAnalysis - https://code.google.com/p/mdanalysis
Numpy - http://www.numpy.org
Scipy - http://www.scipy.org
OpenMPI - http://www.open-mpi.org or MPICH  - https://www.mpich.org
Gromacs - http://www.gromacs.org

protoMD has been tested with:
- Gromacs 4.6.x and 5.0.x
- MDAnalysis 0.14.0

Installation
============
Installation uses python's setuptools.

To install protoMD, simply run from the cmd line:

cd proto_md
python setup.py build
python setup.py install --user

N.B. We recommend you install the GromacsWrapper from here:
https://github.com/CTCNano/GromacsWrapper

git clone https://github.com/CTCNano/GromacsWrapper
cd GromacsWrapper
python setup.py build
python setup.py install --user

RUNNING PROTOMD
===============


WORKING EXAMPLE
===============
A bash script that creates an input (hdf) file and runs
a multiscale simulation is found in the SAMPLE dir.

To run it, simply execute from the terminal: bash run.mkconf

SOURCE CODE
===========
proto_md/config.py: Handles configuration of proto_md environment variables and templates.
proto_md/correlation.py: functions for computing correlation functions.
proto_md/diffusion.py: functions for computing diffusion coefficients.
proto_md/dynamics.py: functions to calculate dynamic properties of CG variables. 
proto_md/__init__.py: initializer for importing proto_md as a package
proto_md/__main__.py: functions for arg parsing (when using proto_md as a program)
proto_md/md.py: functions for setting up and performing md calculations.
proto_md/system.py: functions for creating the System object, handling I/O, and calling MD functions 
proto_md/subsystems: contains modules specific to various coarse-graining methods
proto_md/util: contains modules for reading and writng files to / from disk and hdf blobs
proto_md/templates: contains GROMACS mdp config template files
proto_md/integrators: contains modules for various multiscale integrators
proto_md/fieldVars: C++ code for an (optional) subsystem based on continuum field variables

License
=======
GPL v3
