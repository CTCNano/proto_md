'''
Created on Dec 3, 2012

@author: andy
'''

from numpy import arange, array, ceil, ones, zeros
from numpy.linalg import norm
from MDAnalysis.coordinates.PDB import PrimitivePDBWriter as writer
import MDAnalysis as mda
import gromacs
import gromacs.setup
import logging
import h5py #@UnresolvedImport


k=0.0083144621

def fcc_sphere(a, radius):
    points = []
    side_len = ceil(2.*radius/a)*a
    pc = arange(0.0, side_len+a, a)
    fcc = arange(a/2., side_len, a)
         
    # make the primitive cubic points
    for x in pc:
        for y in pc:
            pts = zeros((len(pc),3))
            pts[:,0] = x
            pts[:,1] = y
            pts[:,2] = pc
            points.append(pts)
            
    # make the face centered cubic points
    #"""
    for x in fcc:
        for y in fcc:
            pts = zeros((len(pc),3))
            pts[:,0] = x
            pts[:,1] = y
            pts[:,2] = pc
            points.append(pts)
            
    for y in fcc:
        for z in fcc:
            pts = zeros((len(pc),3))
            pts[:,0] = pc
            pts[:,1] = y
            pts[:,2] = z
            points.append(pts)
            
    for z in fcc:
        for x in fcc:
            pts = zeros((len(pc),3))
            pts[:,0] = x
            pts[:,1] = pc
            pts[:,2] = z
            points.append(pts)
    #"""
    
    # make a npts * 3 array out of the list    
         
    points = array(points)    
    points=points.reshape(points.shape[0]*points.shape[1],3)
    
    # there is certainly a cleaner way of doing this...
    origin = array([side_len/2.,side_len/2.,side_len/2.])
    points = array([x-origin for x in points if norm(x - origin) <= radius])
    
    return points 

def test(a, radius):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    pts=fcc_sphere(a, radius)
        
    ax.scatter(pts[:,0],pts[:,1],pts[:,2])
    plt.show()
    
def make_index(struct, ndx='main.ndx', oldndx=None):
    """Make index file with the special groups.

    This routine adds the group __main__ and the group __environment__
    to the end of the index file. __main__ contains what the user
    defines as the *central* and *most important* parts of the
    system. __environment__ is everything else.

    The template mdp file, for instance, uses these two groups for T-coupling.

    These groups are mainly useful if the default groups "Protein" and "Non-Protein"
    are not appropriate. By using symbolic names such as __main__ one
    can keep scripts more general.

    :Returns:
      *groups* is a list of dictionaries that describe the index groups. See
      :func:`gromacs.cbook.parse_ndxlist` for details.

    :Arguments:
      *struct* : filename
        structure (tpr, pdb, gro)
      *selection* : string
        is a ``make_ndx`` command such as ``"Protein"`` or ``r DRG`` which
        determines what is considered the main group for centering etc. It is
        passed directly to ``make_ndx``.
      *ndx* : string
         name of the final index file
      *oldndx* : string
         name of index file that should be used as a basis; if None
         then the ``make_ndx`` default groups are used.

    This routine is very dumb at the moment; maybe some heuristics will be
    added later as could be other symbolic groups such as __membrane__.
    """

    #logging.info("Building the main index file %(ndx)r..." % vars())
    
    # pass 1: select
    # empty command '' important to get final list of groups
    rc,out,nothing = gromacs.make_ndx(f=struct, n=oldndx, o=ndx, stdout=False,  #@UndefinedVariable
                                      input=('q'))
    #groups = gromacs.cbook.parse_ndxlist(out)
    #last = len(groups) - 1
    #assert last == groups[-1]['nr']

    # pass 2:
    # 1) last group is __main__
    # 2) __environment__ is everything else (eg SOL, ions, ...)
    #rc,out,nothing = gromacs.make_ndx(f=struct, n=ndx, o=ndx,
    #                                  stdout=False,
    #                                  input=('name %d __main__' % last,
    #                                         '! "__main__"',  # is now group last+1
    #                                         'name %d __environment__' % (last+1),
    #                                         '', 'q'))
    
    return gromacs.cbook.parse_ndxlist(out)
            

            
top_src="""
#include "charmm36.ff/forcefield.itp"
#include "charmm36.ff/spc.itp"


;[ defaults ]
; nbfunc    comb-rule
;1        3    

;[ atomtypes ]
; full atom descriptions are available in ffoplsaa.atp
; name  bond_type    mass    charge   ptype     sigma      epsilon
;AR    AR        1    0    A    1    1

[ molecule_type ]
Au    1

[ atoms ]
;   nr   type  resnr residue  atom   cgnr     charge       mass
     1       AU       1     AU     AU      1     0


[ system ]
; Name
Gold in water

[ molecules ]
; Compound        #mols
"""
            
            
    
def test2(a,radius,box=100.0,fbase=None):
    # l.test2(5.0797,15)

    if fbase is None:
        fbase = "{}_{}".format(box,radius)
    
    top = "{}.top".format(fbase)
    struct = "{}.pdb".format(fbase)
    sol = "{}.sol.pdb".format(fbase)
    ndx = "{}.ndx".format(fbase)
    
    origin = box/2.
    pts=fcc_sphere(a, radius)
    w=writer("{}.pdb".format(fbase))
    w.CRYST1([box,box,box,90.00,90.00,90.00])
    
    
    for index,atom in enumerate(pts):
        w.ATOM(serial=index+1, name="AU", resName="NP", resSeq=1, 
               chainID="A", segID="AUNP", element="AU",
               x=atom[0]+origin, y=atom[1]+origin, z=atom[2]+origin)
        
    w.close()
    
    #make_index("{}.pdb".format(fbase), "{}.ndx".format(fbase))
    
    with file(top, "w") as t:
        t.write(top_src)
        t.write("Au    {}\n".format(pts.shape[0]))
        
    gromacs.genbox(p=top, cp=struct, cs="spc216.gro", o=sol, vdwd="0.15")       #@UndefinedVariable
    
    rc,out,nothing = gromacs.make_ndx(f=sol, n=None, o=ndx, stdout=False,       #@UndefinedVariable
                                      input=('', '', 'q')) 
    
    gromacs.grompp(f="md2.mdp", o="{}.tpr".format(fbase), c=sol, p=top, n=ndx)  #@UndefinedVariable
    
    with file("{}.sh".format(fbase), "w") as f:
        f.write("#!/bin/bash\n")
        f.write("#PBS -k o\n")
        f.write("#PBS -l nodes=1:ppn=12:ccvt,walltime=24:00:00\n")
        f.write("#PBS -M somogyie@indiana.edu\n")
        f.write("#PBS -m abe\n")
        f.write("#PBS -N {}\n".format(fbase))
        f.write("#PBS -j oe\n")
        f.write("#PBS -q pg\n")
        f.write("#PBS -d /N/dc/scratch/somogyie/Au\n")

        f.write("mpirun mdrun -deffnm {}".format(fbase))
        
def test3():
    
    for i in arange(10,31,2.5):
        fbase = "{}_{}".format(100.,i)
        top = "{}.top".format(fbase)
        sol = "{}.sol.pdb".format(fbase)
        ndx = "{}.ndx".format(fbase)
        gromacs.grompp(f="md3.mdp", o="{}.tpr".format(fbase), c=sol, p=top, n=ndx) #@UndefinedVariable
        
def make_tpr(box, rrange):
    
    for radius in rrange:

        fbase = "{}_{}".format(box,radius)
    
        top = "{}.top".format(fbase)
        sol = "{}.sol.pdb".format(fbase)
        ndx = "{}.ndx".format(fbase)
    
        gromacs.grompp(f="md2.mdp", o="{}.tpr".format(fbase), c=sol, p=top, n=ndx) #@UndefinedVariable
        
def merge_hdf(gpat, out):
    import os.path
    import glob
    files = glob.glob(gpat)
    with h5py.File(out, "w") as out:
        for f, fname in [(h5py.File(f, "r"), os.path.splitext(f)[0]) for f in files]:
            print(f["0/POSITIONS"].value[0,0,:,:])
            print(f["0/VELOCITIES"].value[0,0,:,:])
            print(f["0/FORCES"].value[0,0,:,:])
            
            g=out.create_group(fname)
        
            g["POSITIONS"]  = f["0/POSITIONS"].value[0,0,:,:]
            g["VELOCITIES"] = f["0/VELOCITIES"].value[0,0,:,:]
            g["FORCES"]     = f["0/FORCES"].value[0,0,:,:]
        
    
        
    
if __name__ == "__main__":
    import sys
    print(sys.argv)
    

        
    
        
            
            
