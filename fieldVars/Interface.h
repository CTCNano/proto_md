#ifndef INTERFACE_H
#define INTERFACE_H

#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <petscksp.h>
#include <petscsnes.h>
#include <list>
#include <vector>
#include <Python.h>
#include <petscsnes.h>
#include <petscdmda.h>
#include <fstream>
#include <time.h>
#include <string>

using namespace std;

static char help[] = "A class of routines for doing multiscale SPH using PETSc\n";

class FieldVar {
private:
  PetscInt NumNodes,
  	   AdjNumNodes, // the number of grid points that have not been "turned off"
	   NumNodes_x, // number of nodes along the x direction
	   NumNodes_y,
	   NumNodes_z,
	   Dim, // spatial dimensional number of the system
	   Natoms,
	   Threshold, // The minimum number of atoms that contribute to a specific node
	   Newton_Patial, /* The number of partial Newton steps a solver takes. If Newton_Partial == 1 then this is the classical Newton
			     iteration. Otherwise, \Sum Newton_Partial_Steps should be equal to the complete Newton direction.
			     This is a quite sophisticated technique to understand but simple to implement. Still experimental ...
			 */

	   Grid_Neighbors_x,
	   Grid_Neighbors_y, // number of surrounding neighbor cells to search along y axis
	   Grid_Neighbors_z,
	   FreqUpdate, // How often to update the grid
	   Size,
	   Rank,
	   MaxNewIters; // Maximum number of newton iterations

  PetscScalar 	Tol,
		Scaling,
		Resol, // The width or resolution of of the Kernel
		*Box, // Atomic box sizePetscErrorCode Discretize(double* IN_ARRAY1, int DIM1, const PetscInt &NumNodes_x, const PetscInt &NumNodes_y, const PetscInt &NumNodes_z, MPI_Comm Comm) {
  	  	*Coords_Local_x, // a slice of the global coords vector stored on each processor
		*Coords_Local_y,
		*Coords_Local_z;

  ofstream fp;

  MPI_Status stat;

  PetscErrorCode ierr;

  bool Extend; // If true, then Grid_Neighbors must be supplied as well.

  MPI_Comm COMM; // The communicator is in principle the PETSC_COMM_WORLD for parallel computation or PETSC_COMM_SELF
				// for serial computations.
  Vec Coords;  // Atomic coords vector [x;y;z]
  vector<Mat> JacobKernel;
  Mat Jacobian, TransKernelTrans, KernelMatrix;

  vector<PetscScalar> FieldVars, // Field Variables (like density, pressure, stress tensor, etc.) that plays the role of the CG variable.
  	  	      FieldVars_Vel,
  	  	      FieldVars_For;

  vector< vector<PetscInt> > FVList;
  vector< double > Mass;
  vector< vector<PetscScalar> > Grid;
  vector< vector< vector<PetscInt> > > GridID;

  PetscErrorCode ConstructGridID();
  PetscErrorCode ConstructGrid(const PetscInt &NumNodes);

  // Coarse graining
  vector<PetscScalar> ComputeFV(const PetscScalar* const Coords);
  vector<PetscScalar> ComputeFV_Vel(const PetscScalar* const Coords, const PetscScalar* const Vel);
  vector<PetscScalar> ComputeFV_For(const PetscScalar* const Coords, const PetscScalar* const Vel, const PetscScalar* const For);
  PetscErrorCode SetupJacobians();
  PetscErrorCode DeleteJacobians();

  // Kernel functions
  PetscErrorCode KernelJacobian(const Vec *const Coords, const PetscInt &dim);
  PetscScalar KernelFunction(const PetscScalar* const Coords, PetscInt mass, const vector<PetscScalar> &GridPos);
  PetscErrorCode computeKernel();

  // Data structures
  PetscErrorCode ConstructAtomList(const double* const);
  PetscErrorCode ConstructFVList(const double* const);

  // Fine scaling
  Vec PetscVectorFromArray(const PetscScalar* const Array, const PetscInt length);
  Vec AdvancedAtomistic(const Vec &Coords_MD, const Vec &Multipliers, const PetscInt &dim);
  PetscErrorCode AssembleJacobian(const Vec *const Coords);
  PetscScalar ComputeLagrangeMulti(const Vec *const Coords, Vec Multipliers, const Vec &FV, PetscScalar Scaling, int iters, int Assemble);
  Vec Constraints(const Vec *const Coords);

  string GetTime();

public:
  void SetNumNodes(const int &n) { NumNodes = n; }
  void SetNumNodes_x(const int &n) { NumNodes_x = n; }
  void SetNumNodes_y(const int &n) { NumNodes_y = n; }
  void SetNumNodes_z(const int &n) { NumNodes_z = n; }
  void SetThreshold(const int &n) { Threshold = n; }
  void SetNewtonPartial(const int &n) { Newton_Patial = n; }
  void SetGrid_Neighbors_x(const int &n) { Grid_Neighbors_x = n; }
  void SetGrid_Neighbors_y(const int &n) { Grid_Neighbors_y = n; }
  void SetGrid_Neighbors_z(const int &n) { Grid_Neighbors_z = n; }
  void SetExtend(const bool state) { Extend = state; }
  void SetResol(const double resol) { Resol = resol; }
  void SetTol(const double resol) { Tol = resol; }
  void SetScaling(const  double resol) { Scaling = resol; }
  void SetFreqUpdate(const  int freq) { FreqUpdate = freq; }
  void SetBox(double *IN_ARRAY2, int DIM1, int DIM2) { for(int i = 0; i < DIM1 * DIM2; i++) Box[i] = IN_ARRAY2[i];}

  int GetNumNodes() const { return NumNodes; } // get rid of this maybe?
  int GetAdjNumNodes() const { return AdjNumNodes; }
  int GetNumNodes_x() const { return NumNodes_x; }
  int GetNumNodes_y() const { return NumNodes_y; }
  int GetNumNodes_z() const { return NumNodes_z; }
  int GetDim() const { return Dim; }
  int GetNatoms() const { return Natoms; }
  int GetThreshold() const { return Threshold; }
  bool GetExtend() const { return Extend; }
  double GetResol() const { return Resol; }
  int GetFreqUpdate() const { return FreqUpdate; }
  void GetBox(double* ARGOUT_ARRAY1, int DIM1) {
	  for(int i = 0; i < DIM1; i++)
		  ARGOUT_ARRAY1[i] = Box[i];
	}

  MPI_Comm& GetCOMM() { return COMM; }

  // Constructors
  FieldVar(double* IN_ARRAY2, int DIM1, int DIM2, double* MASS, int DIM_MASS, PyObject* PyDict, MPI_Comm comm = PETSC_COMM_SELF);
  FieldVar(double* IN_ARRAY2, int DIM1, int DIM2, MPI_Comm comm = PETSC_COMM_SELF);

  // External functions (to be called from python)
  PetscErrorCode Py_CoarseGrain(double* IN_ARRAY2, int DIM1, int DIM2);
  PetscErrorCode Py_UpdateGrid(double* IN_ARRAY2, int DIM1, int DIM2);
  PetscErrorCode Py_FineGrain(double* FV, int FV_DIM, double *x, int nx, double *y, int ny, double *z, int nz, double* COORDS_OUT, int NATOMS_BY_3, PyObject* Assemble);
  PetscErrorCode Py_FineGrainMom(double* FV1, int DIM_FV1, double* FV2, int DIM_FV2, double* FV3, int DIM_FV3, double *vx, int vnx, double *vy,
                                          int vny, double *vz, int vnz, double* VELS_OUT, int NATOMS_BY_3);

  void Py_ComputeCG_Pos(double *COORDS_IN, int NATOMS, int DIM, double *CG_OUT, int NUMCG);
  void Py_ComputeCG_Vel(double *COORDS_IN, int NATOMS1, int DIM1, double *VEL_IN, int NATOMS2, int DIM2, double *CG_OUT, int NUMCG);
  void Py_ComputeCG_For(double *COORDS_IN, int NATOMS1, int DIM1, double *VEL_IN, int NATOMS2, int DIM2, double *FOR_IN, int NATOMS3, int DIM3, double *CG_OUT, int NUMCG);

  ~FieldVar() {
	VecDestroy(&Coords);
	ierr = DeleteJacobians();
	fp.close();

	delete[] Box;
	delete[] Coords_Local_x;
	delete[] Coords_Local_y;
	delete[] Coords_Local_z;

	PetscFinalize();
  }
};

#endif
