import pytest
import os
import proto_md
import subprocess

input = subprocess.call(["bash test_mn.mkconf"], shell=True)

s = proto_md.system.System("test_mn.hdf5", mode="a")


def test_energy_minimization():
    s.begin_timestep()
    s.minimize()
    s.end_timestep()


os.remove("test_mn.hdf5")
os.remove("proto.log")
