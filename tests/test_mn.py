import pytest
import os
import proto_md
import subprocess
import tempfile 

input_file = tempfile.NamedTemporaryFile(suffix=".hdf5").name
config = "./tests/test_mn.mkconf"
config = os.path.abspath(config)

input = subprocess.call(["bash",config,input_file], shell=False)

s = proto_md.system.System(input_file, mode="a")


def test_energy_minimization():
    s.begin_timestep()
    s.minimize()
    s.end_timestep()


#os.remove("test_mn.hdf5")
os.remove("proto.log")
