import MDAnalysis as mda
import proto_md.subsystems as ss

universe = mda.Universe("sample/SpaceWarping/Struct/1PRT.gro")

def test_subsystem_method1():
    nCG, SS = ss.SpaceWarpingSubsystemFactory(kmax=1)
    [sub.universe_changed(universe) for sub in SS]
    [sub.equilibrated() for sub in SS]
    CG = [sub.computeCG_pos(universe.atoms.positions) for sub in SS]
    atomic_pos = [sub.computeCG_inv(CG[index]) for index, sub in enumerate(SS)]
    [sub.frame() for sub in SS]
    [sub.translate(CG[index].flatten()) for index, sub in enumerate(SS)]


def test_subsystem_method2():
    import collections

    dummysys = collections.namedtuple("dummysys", "universe")(universe)
    nCG, SS = ss.SpaceWarpingSubsystemFactory(system=dummysys, kmax=1)
    [sub.equilibrated() for sub in SS]
    CG = [sub.computeCG_pos(universe.atoms.positions) for sub in SS]
    atomic_pos = [sub.computeCG_inv(CG[index]) for index, sub in enumerate(SS)]
    [sub.frame() for sub in SS]
    [sub.translate(CG[index].flatten()) for index, sub in enumerate(SS)]
