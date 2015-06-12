'''
Created on Dec 12, 2012

@author: andy

functions for calculating coarse grained variables
'''

from numpy import *
import MDAnalysis as mda            #@UnresolvedImport
import h5py                         #@UnresolvedImport

def mass_center_periodic(positions, masses, pbc):
    # make a column vector
    masses = masses[:,newaxis]
    # scaled positions, positions is natom * 3, pbc should be 3 vector.
    # arctan2 has a range of (-pi, pi], so have to shift positions to 
    # zero centered instead of box / 2 centered.
    spos = positions / pbc * 2.0 * pi - pi
        
    # get the x and y components, mass scale them, and add in cartesian space
    # shift back to box / 2 centered
    cm = (arctan2(sum(masses * sin(spos), axis=0), sum(masses * cos(spos), axis=0)) + pi) * pbc / 2.0 / pi
    
    return cm

def mass_center_periodic_from_ts(struct_f, ts_f, nframes):
    u = mda.Universe(struct_f)
    p = zeros((nframes, 3))
    atoms = u.selectAtoms("not resname SOL")
    pbc = ones(3) * 100.0
    
    u.load_new(ts_f)
    
    masses = atoms.masses()
    
    for i, _ in enumerate(u.trajectory):
        if i < nframes:
            p[i,:] = mass_center_periodic(atoms.positions, masses, pbc)
            if(i % 100 == 0):
                print("processing frame {},\t{}%, \t{}".format(i, 100.0*float(i)/float(nframes), p[i,:]))
        else:
            break
        
    return p

def mass_centers():
    ss = ["100.0_10.0", "100.0_12.5", "100.0_15.0", "100.0_17.5", "100.0_20.0", 
         "100.0_22.5", "100.0_25.0", "100.0_27.5", "100.0_30.0"]
    
    out = h5py.File("/home/andy/tmp/all_positions.hdf", "w")
    
    for s in ss:
        struct = "/home/andy/tmp/Au/" + s + ".sol.pdb"
        ts = "/home/andy/Au.x/" + s + ".total.xtc"
        cm = mass_center_periodic_from_ts(struct, ts, 20000)
        out[s] = cm
        
if __name__ == "__main__":
    mass_centers()
        
        
    
    
  
