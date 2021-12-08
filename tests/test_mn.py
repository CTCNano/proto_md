import pytest
import os
import proto_md

s = proto_md.system.System("test_input.hdf5", mode="a")

def test_energy_minimization():
	s.begin_timestep()
	s.minimize()
	s.end_timestep()