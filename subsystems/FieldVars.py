# Author: Andrew
# Last modified: December 14, 2013
# Do not make changes to this file unless you know what you are doing
# This file is compatible with both classic and new-style classes.

from sys import version_info
if version_info >= (2,6,0):
    def swig_import_helper():
        from os.path import dirname
        import imp
        fp = None
        try:
            fp, pathname, description = imp.find_module('_FieldVars', [dirname(__file__)])
        except ImportError:
            import _FieldVars
            return _FieldVars
        if fp is not None:
            try:
                _mod = imp.load_module('_FieldVars', fp, pathname, description)
            finally:
                fp.close()
            return _mod
    _FieldVars = swig_import_helper()
    del swig_import_helper
else:
    import _FieldVars
del version_info
try:
    _swig_property = property
except NameError:
    pass # Python < 2.2 doesn't have 'property'.
def _swig_setattr_nondynamic(self,class_type,name,value,static=1):
    if (name == "thisown"): return self.this.own(value)
    if (name == "this"):
        if type(value).__name__ == 'SwigPyObject':
            self.__dict__[name] = value
            return
    method = class_type.__swig_setmethods__.get(name,None)
    if method: return method(self,value)
    if (not static) or hasattr(self,name):
        self.__dict__[name] = value
    else:
        raise AttributeError("You cannot add attributes to %s" % self)

def _swig_setattr(self,class_type,name,value):
    return _swig_setattr_nondynamic(self,class_type,name,value,0)

def _swig_getattr(self,class_type,name):
    if (name == "thisown"): return self.this.own()
    method = class_type.__swig_getmethods__.get(name,None)
    if method: return method(self)
    raise AttributeError(name)

def _swig_repr(self):
    try: strthis = "proxy of " + self.this.__repr__()
    except: strthis = ""
    return "<%s.%s; %s >" % (self.__class__.__module__, self.__class__.__name__, strthis,)

try:
    _object = object
    _newclass = 1
except AttributeError:
    class _object : pass
    _newclass = 0



def convert(*args):
  return _FieldVars.convert(*args)
convert = _FieldVars.convert
class FieldVar(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, FieldVar, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, FieldVar, name)
    __repr__ = _swig_repr
    def SetNumNodes(self, *args): return _FieldVars.FieldVar_SetNumNodes(self, *args)
    def SetNumNodes_x(self, *args): return _FieldVars.FieldVar_SetNumNodes_x(self, *args)
    def SetNumNodes_y(self, *args): return _FieldVars.FieldVar_SetNumNodes_y(self, *args)
    def SetNumNodes_z(self, *args): return _FieldVars.FieldVar_SetNumNodes_z(self, *args)
    def SetThreshold(self, *args): return _FieldVars.FieldVar_SetThreshold(self, *args)
    def SetNewtonPartial(self, *args): return _FieldVars.FieldVar_SetNewtonPartial(self, *args)
    def SetGrid_Neighbors_x(self, *args): return _FieldVars.FieldVar_SetGrid_Neighbors_x(self, *args)
    def SetGrid_Neighbors_y(self, *args): return _FieldVars.FieldVar_SetGrid_Neighbors_y(self, *args)
    def SetGrid_Neighbors_z(self, *args): return _FieldVars.FieldVar_SetGrid_Neighbors_z(self, *args)
    def SetExtend(self, *args): return _FieldVars.FieldVar_SetExtend(self, *args)
    def SetResol(self, *args): return _FieldVars.FieldVar_SetResol(self, *args)
    def SetTol(self, *args): return _FieldVars.FieldVar_SetTol(self, *args)
    def SetScaling(self, *args): return _FieldVars.FieldVar_SetScaling(self, *args)
    def SetFreqUpdate(self, *args): return _FieldVars.FieldVar_SetFreqUpdate(self, *args)
    def SetBox(self, *args): return _FieldVars.FieldVar_SetBox(self, *args)
    def GetNumNodes(self): return _FieldVars.FieldVar_GetNumNodes(self)
    def GetAdjNumNodes(self): return _FieldVars.FieldVar_GetAdjNumNodes(self)
    def GetNumNodes_x(self): return _FieldVars.FieldVar_GetNumNodes_x(self)
    def GetNumNodes_y(self): return _FieldVars.FieldVar_GetNumNodes_y(self)
    def GetNumNodes_z(self): return _FieldVars.FieldVar_GetNumNodes_z(self)
    def GetDim(self): return _FieldVars.FieldVar_GetDim(self)
    def GetNatoms(self): return _FieldVars.FieldVar_GetNatoms(self)
    def GetThreshold(self): return _FieldVars.FieldVar_GetThreshold(self)
    def GetExtend(self): return _FieldVars.FieldVar_GetExtend(self)
    def GetResol(self): return _FieldVars.FieldVar_GetResol(self)
    def GetFreqUpdate(self): return _FieldVars.FieldVar_GetFreqUpdate(self)
    def GetBox(self, *args): return _FieldVars.FieldVar_GetBox(self, *args)
    def GetCOMM(self): return _FieldVars.FieldVar_GetCOMM(self)
    def __init__(self, *args): 
        this = _FieldVars.new_FieldVar(*args)
        try: self.this.append(this)
        except: self.this = this
    def Py_CoarseGrain(self, *args): return _FieldVars.FieldVar_Py_CoarseGrain(self, *args)
    def Py_UpdateGrid(self, *args): return _FieldVars.FieldVar_Py_UpdateGrid(self, *args)
    def Py_FineGrain(self, *args): return _FieldVars.FieldVar_Py_FineGrain(self, *args)
    def Py_ComputeCG_Pos(self, *args): return _FieldVars.FieldVar_Py_ComputeCG_Pos(self, *args)
    def Py_ComputeCG_Vel(self, *args): return _FieldVars.FieldVar_Py_ComputeCG_Vel(self, *args)
    def Py_ComputeCG_For(self, *args): return _FieldVars.FieldVar_Py_ComputeCG_For(self, *args)
    __swig_destroy__ = _FieldVars.delete_FieldVar
    __del__ = lambda self : None;
FieldVar_swigregister = _FieldVars.FieldVar_swigregister
FieldVar_swigregister(FieldVar)
cvar = _FieldVars.cvar



