"""
Created on Dec 19, 2012

@author: andy

dms2 io module

functions for reading and writng files to / from disk and hdf blobs.
added io for topology angles / bonds from .top files

last modified: March 9, 2014
@ Andrew
"""

import h5py                                 #@UnresolvedImport
import gromacs.utilities as utilities
import os.path  
import shutil
import numpy as np
from numpy import int, array, zeros

import MDAnalysis                           #@UnresolvedImport
import MDAnalysis.core                      #@UnresolvedImport

def fix_periodic_boundary_conditions(u):
    box = u.trajectory.ts.dimensions[:3]
    
    # scaled positions, positions is natom * 3, box should be 3 vector.
    # arctan2 has a range of (-pi, pi], so have to shift positions to 
    # zero centered instead of box / 2 centered.
    spos = u.atoms.positions / box * 2.0 * np.pi - np.pi
        
    # get the x and y components, mass scale them, and add in cartesian space
    # shift back to box / 2 centered
    return (np.arctan2(np.sin(spos), np.cos(spos)) + np.pi) * box / 2.0 / np.pi

def stripped_positions(fname, sub):
    """
    extremly hackish function.
    
    Currently, only TRRReader supports the sub argument.
    
    Read a pdb, and return the coordinates using sub as an indexing array. 
    
    Really should go into MDAnalysis.PDBReader, but in the intrests of time, here
    it is. 
    
    TODO: someone eventually move this logic to PrimitivePDBReader and PDBReader.
    """
    return MDAnalysis.Universe(fname).atoms.positions[sub]

def stripped_velocities(fname, sub):
    """
    extremly hackish function.

    Currently, only TRRReader supports the sub argument.

    Read a pdb, and return the velocities using sub as an indexing array.

    Really should go into MDAnalysis.PDBReader, but in the intrests of time, here
    it is.

    TODO: someone eventually move this logic to PrimitivePDBReader and PDBReader.
    """
    return MDAnalysis.Universe(fname).atoms.velocities()[sub]
    
def data_tofile(data, fid=None, sep="", fmt="%s", dirname="."):
    """
    @param data: the data to write to a file. This may be either a string, 
    a h5py.Dataset, a numpy ndarray, or a MDAnalysis atom group or universe. 

    @param fid : file or str    
        An open file object, or a string containing a filename.
        This MAY be None if and only if the data is a h5py.Dataset, in shich case, the 
        output file name is taken from the last part of the dataset name. e.g. if the 
        dataset name is /foo/bar/fred', 'fred' will be the output file name (but only if
        fid is None. If fid is given, this overrides the dataset name. 
    @param sep : str
        Separator between array items for text output. If "" (empty), a binary file is written, 
        equivalent to file.write(a.tostring()).
    @param fmt : str
        Format string for text file output. Each entry in the array is formatted to text by 
        first converting it to the closest Python type, and then using "format" % item.
    @param dirname
    @return: absolute path of the created file
        """ 
    if data is not None:
        if isinstance(data, str) and os.path.isfile(data):
            src = os.path.abspath(data)
            with utilities.in_dir(dirname):
                shutil.copyfile(src, fid)
                return os.path.abspath(fid)
        else:
            with utilities.in_dir(dirname):
                if type(data) is h5py.Dataset:
                    if fid is None:
                        fid = os.path.split(data.name)[1]
                    # data now becomes an ndarray
                    data = data[()]
                if type(data) is np.ndarray:  
                    print('writing file {} in dir {}'.format(fid,dirname))
                    data.tofile(fid,sep,fmt)
                elif isinstance(data, MDAnalysis.core.AtomGroup.AtomGroup) or isinstance(data, MDAnalysis.core.AtomGroup.Universe):
                    print("pwd", os.path.curdir)
                    print("fid", fid)
                    w = MDAnalysis.Writer(fid,numatoms=len(data.atoms))
                    w.write(data)
                    del w
                else:
                    raise TypeError("expected either Dataset, ndarray or file path as src")
                return os.path.abspath(fid)
            
            
def hdf_linksrc(hdf, newname, src):
    """
    if src is a soft link, follow it's target until we get to a non-linked object, and
    create the new link to point to this head object.
    """
    
    try:
        while True:
            print(1,src)
            src = hdf.id.links.get_val(src)
            print(2,src)
    except (TypeError, KeyError):
        pass
    
    print("links.create_soft({}, {})".format(newname, src))
    hdf.id.links.create_soft(newname, src)
    
def hdf_value_from_string(s):
    """
    As we currently store dictionaries in the hdf file, everything gets 
    converted to a string UGLY!!!. This function tries to recast them
    as the original data type, currenly we support ints and floats. 
    """
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return str(s)

def hdf_dict(attrs, key_base_name):
    return dict(zip(attrs[key_base_name + "_keys"], 
                    [hdf_value_from_string(s) for s in attrs[key_base_name + "_values"]]))
    
def get_class( klass ):
    """
    given a fully qualified class name, i.e. "datetime.datetime", 
    this loads the module and returns the class type. 
    
    the ctor on the class type can then be called to create an instance of the class.

    For example, to create an instance of the above class, 

    # get the type
    t = util.get_class("datetime.datetime")

    # now create an instance
    i = t()

    This is equivalent to 
    i = datetime.datetime()
    """
    parts = klass.split('.')
    module = ".".join(parts[:-1])
    m = __import__( module )
    for comp in parts[1:]:
        m = getattr(m, comp)            
    return m

def is_env_set(env):
    """
    returns True if the enviornment variable is set to 'yes', 'true' or non-zero integer,
    False otherwise
    """
    try:
        var = os.environ[env].strip().upper()
        try:
            return int(var) != 0
        except:
            pass
        return var == "TRUE" or var == "YES"
    except:
        pass
    return False
        
def Read_Top(top_fname):
	"""
	reads .top file and returns bond and angle indices as numpy arrays
	"""

        fp = open(top_fname, 'r')
        file_list = fp.readlines()

        num_chains = None
        bonds = []
        angles = []

        read_bonds = False
        read_angles = False

        for n, line in enumerate(file_list):
                if 'Compound' in line:
                        num_chains = np.int(file_list[n+1].split()[1])
			print 'Found {} chains'.format(num_chains)

                if 'bonds' in line:
                        read_bonds = True

                if 'angles' in line:
                        read_angles = True

                if len(line) is 1: # white space encountered, so stop counting bond / angle indices
                        read_bonds = False
                        read_angles = False

                if read_bonds:
                        if len(line) is 19: # stupid PDB std
                                tmp = line.split()
                                bonds.append([ int(tmp[0]), int(tmp[1]) ])

                if read_angles:
                        if len(line) is 25: # stupid PDB std
                                tmp = line.split()
                                angles.append([ int(tmp[0]), int(tmp[2]) ])
        fp.close()

	bonds = array(bonds, 'int')
        angles = array(angles, 'int')

	bonds = bonds - 1
	angles = angles - 1

	if num_chains > 1:
		numatoms = bonds.max()

		if numatoms != angles.max():
			print "Error in reading {} file: bonds indices do not mauch with angle indices!".format(top_fname)
			print "max bonds index = {}, max angles index = {}".format(numatoms, angles.max())
			return None

		bonds_global = zeros((num_chains * len(bonds), bonds.shape[1]), 'int')
		angles_global = zeros((num_chains * len(angles), angles.shape[1]), 'int')

		for i in xrange(num_chains):
			bonds_global[len(bonds) * i : len(bonds) * (i+1)] = bonds + i * numatoms
			angles_global[len(angles) * i : len(angles) * (i+1)] = angles + i * numatoms

		return bonds_global, angles_global 
	else:
        	return bonds, angles
