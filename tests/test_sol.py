import pytest
import os
import proto_md
import subprocess
import tempfile

def test_auto_sol():
	input_file = tempfile.NamedTemporaryFile(suffix=".hdf5").name
	config = "./tests/test_sol.mkconf"
	config = os.path.abspath(config)
	input = subprocess.call(["bash", config, input_file], shell=False)
	os.remove("proto.log")

