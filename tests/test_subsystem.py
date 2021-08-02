import MDAnalysis as mda
import proto_md.subsystems as ss

universe = mda.Universe("sample/SpaceWarping/Struct/1PRT.gro")


def test_subsystem_method1():
    nCG, SS = ss.SpaceWarpingSubsystemFactory(kmax=1)
    [sub.universe_changed(universe) for sub in SS]
    [sub.equilibrated() for sub in SS]
    CG = [sub.ComputeCG(universe.atoms.positions) for sub in SS]


def test_subsystem_method2():
    import collections

    dummysys = collections.namedtuple("dummysys", "universe")(universe)
    nCG, SS = ss.SpaceWarpingSubsystemFactory(system=dummysys, kmax=1)
    [sub.equilibrated() for sub in SS]
    CG = [sub.ComputeCG(universe.atoms.positions) for sub in SS]
