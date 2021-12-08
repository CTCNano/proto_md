import pytest
import os
import proto_md
import subprocess

input = subprocess.call(['bash test_md.mkconf'], shell = True)

s1 = proto_md.system.System("test_md.hdf5", mode="a")
s2 = proto_md.system.System("test_md.hdf5", mode="a")

def test_energy_minimization():
	s1.begin_timestep()
	s1.minimize()
	s1.end_timestep()

def test_equilibration():
	s1.begin_timestep()
	s1.equilibrate()
	s1.end_timestep()

def test_MD():
	s1.begin_timestep()
	s1.md()
	s1.end_timestep()

def test_MD_whole_without_solvate():
	s1.begin_timestep()
	s1.minimize()
	s1.equilibrate()
	s1.md()
	s1.end_timestep()

def test_solvate():
	s2.begin_timestep()
	s2.solvate()
	s2.end_timestep()

def test_MD_whole_with_solvate():
	s1.begin_timestep()
	sol = s1.solvate()
	mn = s1.minimize(**sol)
	eq = s1.equilibrate(**mn)
	md = s1.md(**eq)
        s1.end_timestep()

os.remove("test_md.hdf5")
os.remove("proto.log")
