from system import System, Timestep
from diffusion import *
from dynamics import *

def __check_dependencies():
    import pkg_resources
    mdver = pkg_resources.get_distribution("MDAnalysis").version.split("-")[0].split(".")
    mdver = [int(i) for i in mdver]
    
    if not (mdver[1] > 7 or (mdver[1] >= 7 and mdver[2] >= 8)):
        raise Exception("proto_md requires MDAnalysis >= 0.7.8, found {}".format(mdver))


__check_dependencies()
