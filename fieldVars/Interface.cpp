#include "Interface.h"
#define BOX_DIM 2

/* This file contains all the interface functions accessible to the python interpreter */

Vec FieldVar::PetscVectorFromArray(const PetscScalar* const Array, const PetscInt length) {

  /******************************************************************************/
  /****** Distributes a vector on all processors from a given serial array *****/
  /****************************************************************************/

  PetscFunctionBegin;
  PetscInt istart, iend;
  PetscErrorCode ierr;
  Vec Petsc_vector;
  VecCreateMPI(FieldVar::COMM, PETSC_DECIDE, length, &Petsc_vector);

  VecGetOwnershipRange(Petsc_vector, &istart, &iend);

  for(int i = istart; i < iend; i++)
      VecSetValues(Petsc_vector, 1, &i, Array+i, INSERT_VALUES);

  VecAssemblyBegin(Petsc_vector);
  VecAssemblyEnd(Petsc_vector);

  PetscFunctionReturn(Petsc_vector);
}

FieldVar::FieldVar(double* IN_ARRAY2, int DIM1, int DIM2, MPI_Comm comm) {

	/************************************************************************************/
	/****** Opt. constructor: make sure you know what you're doing if you use this! ****/
	/****** Params ought to be supplied from python this way.
	/**********************************************************************************/

	PetscFunctionBegin;
	PetscInitialize(NULL, NULL, (char *)0, help);

	FieldVar::COMM = comm;
	FieldVar::Natoms = DIM1;
	FieldVar::Dim = DIM2;

	FieldVar::Box = new PetscScalar[DIM2 * BOX_DIM];

	MPI_Comm_size(FieldVar::COMM, &(FieldVar::Size));
	MPI_Comm_rank(FieldVar::COMM, &(FieldVar::Rank));

	FieldVar::Coords = FieldVar::PetscVectorFromArray(IN_ARRAY2, DIM1*DIM2);
	FieldVar::fp.open("proto.log", ios::out | ios::app);

	// This is still a primitive implementation. Must check for python params inpute before any
	// computations start.
}

FieldVar::FieldVar(double* IN_ARRAY2, int DIM1, int DIM2, double* MASS, int DIM_MASS, PyObject* PyDict, MPI_Comm comm) {

	/***********************************************************************/
	/****** Main constructor: params ought to be supplied from python *****/
	/*********************************************************************/

	PetscFunctionBegin;
	string ErrorMsg;
	bool Exit = false;

  	PetscInitialize(NULL, NULL, (char *)0, help);

  	FieldVar::COMM = comm;
  	FieldVar::Natoms = DIM1;
  	FieldVar::Dim = DIM2;

  	if(PyDict_CheckExact(PyDict)) {
		PyObject *NumNodes_x_Py = PyDict_GetItemString(PyDict, "NumNodes_x"),
			 *NumNodes_y_Py = PyDict_GetItemString(PyDict, "NumNodes_y"),
			 *NumNodes_z_Py = PyDict_GetItemString(PyDict, "NumNodes_z");

		if(!NumNodes_x_Py || !NumNodes_y_Py || !NumNodes_z_Py) {
			ErrorMsg.append("Number of nodes must be supplied as an integer.\n");
			Exit = true;
		}

		FieldVar::NumNodes_x = PyInt_AsLong(NumNodes_x_Py);
		FieldVar::NumNodes_y = PyInt_AsLong(NumNodes_y_Py);
		FieldVar::NumNodes_z = PyInt_AsLong(NumNodes_z_Py);
		FieldVar::NumNodes = FieldVar::NumNodes_x * FieldVar::NumNodes_y * FieldVar::NumNodes_z;

		PyObject *NewPart_Py = PyDict_GetItemString(PyDict, "NewtonPart"),
			 *Thresh_Py  = PyDict_GetItemString(PyDict, "Threshold"),
			 *Resol_Py   = PyDict_GetItemString(PyDict, "Resolution"),
			 *FreqUp_Py  = PyDict_GetItemString(PyDict, "FreqUpdate");

		if(!NewPart_Py || !Thresh_Py || !Resol_Py || !FreqUp_Py) {
					ErrorMsg.append("Error in Grid/Newton input parameters.\n");
					Exit = true;
		}

		FieldVar::Newton_Patial = PyInt_AsLong(NewPart_Py);
		FieldVar::FreqUpdate 	= PyInt_AsLong(FreqUp_Py);
		FieldVar::Threshold  	= PyInt_AsLong(Thresh_Py);
		FieldVar::Resol 	= PyFloat_AsDouble(Resol_Py);

		PyObject *Extend_Py   = PyDict_GetItemString(PyDict, "Extend"),
		 	 *Scale_Py    = PyDict_GetItemString(PyDict, "Scaling"),
			 *Tol_Py      = PyDict_GetItemString(PyDict, "Tol"),
			 *NewIters_Py = PyDict_GetItemString(PyDict, "NewIters");

		FieldVar::Scaling     = PyFloat_AsDouble(Scale_Py);
		FieldVar::Tol 	      = PyFloat_AsDouble(Tol_Py);
		FieldVar::Extend      = PyInt_AsLong(Extend_Py);
		FieldVar::MaxNewIters = PyInt_AsLong(NewIters_Py);

		if(!Tol_Py || !Scale_Py || !Extend_Py || !NewIters_Py) {
			ErrorMsg.append("Error in optional input parameters.\n");
			Exit = true;
		}

		if(FieldVar::Extend) {
			PyObject *Grid_Neighbors_x_Py = PyDict_GetItemString(PyDict, "Grid_Neighbors_x"),
				 *Grid_Neighbors_y_Py = PyDict_GetItemString(PyDict, "Grid_Neighbors_y"),
				 *Grid_Neighbors_z_Py = PyDict_GetItemString(PyDict, "Grid_Neighbors_z");

			FieldVar::Grid_Neighbors_x = PyInt_AsLong(Grid_Neighbors_x_Py);
			FieldVar::Grid_Neighbors_y = PyInt_AsLong(Grid_Neighbors_y_Py);
			FieldVar::Grid_Neighbors_z = PyInt_AsLong(Grid_Neighbors_z_Py);

			if(!Grid_Neighbors_x_Py || !Grid_Neighbors_y_Py || !Grid_Neighbors_z_Py) {
				ErrorMsg.append("If Extend is set true, then Grid_Neighbor points must be specified.\n");
				Exit = true;
			}
		}
		else {
			FieldVar::Grid_Neighbors_x = 2; // Default search in 1D to one right and one left steps
			FieldVar::Grid_Neighbors_y = 2;
			FieldVar::Grid_Neighbors_z = 2;
		}

		if(Exit)
			cerr << ErrorMsg << endl;
		else {
			// System-specific params
			FieldVar::Box = new PetscScalar[DIM2 * BOX_DIM];

			FieldVar::Mass.resize(DIM_MASS);

			// GridID must remain constant throughout the simulation
 			// GridID keeps track of grid points that should be included
			FieldVar::GridID.resize(FieldVar::NumNodes_x + FieldVar::Grid_Neighbors_x);
			for(auto i = 0; i < FieldVar::GridID.size(); i++) {
				FieldVar::GridID[i].resize(FieldVar::NumNodes_y + FieldVar::Grid_Neighbors_y);
				for(auto j = 0; j < FieldVar::GridID[i].size(); j++)
					FieldVar::GridID[i][j].resize(FieldVar::NumNodes_z + FieldVar::Grid_Neighbors_z);
			}

			//FieldVar::AtomicList.resize(FieldVar::Natoms); // Done only once assuming Natoms does not change. Must be changed for open systems ...
			
			// FVList is a vector (more efficient than a list) that contains the *active* field variables 
			FieldVar::FVList.reserve(FieldVar::NumNodes);


			for(auto i = 0; i < DIM_MASS; i++)
				FieldVar::Mass[i] = MASS[i];

			// MPI stuff
			MPI_Comm_size(FieldVar::COMM, &(FieldVar::Size));
			MPI_Comm_rank(FieldVar::COMM, &(FieldVar::Rank));

			FieldVar::Coords = FieldVar::PetscVectorFromArray(IN_ARRAY2, DIM1*DIM2);
			FieldVar::fp.open("proto.log", ios::out | ios::app);

			// Allocate memory for local coords. 
			// TODO: Must take local Natoms into account for parallelization!!!
			Coords_Local_x = new PetscScalar[FieldVar::Natoms];
			Coords_Local_y = new PetscScalar[FieldVar::Natoms];
			Coords_Local_z = new PetscScalar[FieldVar::Natoms];
			
		}
  	}
  	else {
  		ErrorMsg.append("Input is NOT a dictionary.\n");
  		cerr << ErrorMsg << endl;
  	}
 }

void FieldVar::Py_ComputeCG_Pos(double *COORDS_IN, int NATOMS, int DIM, double *CG_OUT, int NUMCG) {

	PetscFunctionBegin;

	vector<PetscScalar> tmp = FieldVar::ComputeFV(COORDS_IN);

	for(auto i = 0; i < tmp.size(); i++)
		CG_OUT[i] = tmp[i];
}

void FieldVar::Py_ComputeCG_Mom(double *COORDS_IN, int NATOMS1, int DIM1, double *VEL_IN, int NATOMS2, double *CG_OUT, int NUMCG) {
	PetscFunctionBegin;
	vector<PetscScalar> tmp = FieldVar::ComputeFV_Mom(COORDS_IN, VEL_IN);

	for(auto i = 0; i < tmp.size(); i++)
		CG_OUT[i] = tmp[i];
}

void FieldVar::Py_ComputeCG_For(double *COORDS_IN, int NATOMS1, int DIM1, double *VEL_IN, int NATOMS2, int DIM2, double *FOR_IN, int NATOMS3, int DIM3, double *CG_OUT, int NUMCG) {
	PetscFunctionBegin;
	vector<PetscScalar> tmp = FieldVar::ComputeFV_For(COORDS_IN, VEL_IN, FOR_IN);

	for(auto i = 0; i < tmp.size(); i++)
		CG_OUT[i] = tmp[i];
}

PetscErrorCode FieldVar::Py_FineGrainMom(double* FV1, int DIM_FV1, double* FV2, int DIM_FV2, double* FV3, int DIM_FV3, double *vx, int vnx, double *vy, 
					  int vny, double *vz, int vnz, double* COORDS_IN, int NATOMS, int DIMC, double* VELS_OUT, int NATOMS_BY_3) {

        /*******************************************/
        /***** FineGraining momenta using MSR *****/
        /*****************************************/

	PetscFunctionBegin;

	ofstream fp("proto.log", ios::out | ios::app);

	if(vnx != vny || vnx != vnz || vny != vnz) {
                cerr << "Error in atomic velocities input for fine graining." << endl;
                cerr << "Given dimensions: " << vnx << " " << vny << " " << vnz << endl;
        }
	else {
		std::cout << "Initializing ... " << std::endl;

              	// Maybe we should use VecCreateMPIWithArray here ???
                vector<PetscScalar*> Vel_vec(FieldVar::Dim), FV(FieldVar::Dim);
                Vel_vec[0] = vx;
                Vel_vec[1] = vy;
                Vel_vec[2] = vz;

		FV[0] = FV1;
		FV[1] = FV2;
		FV[2] = FV3;

		Vec *Vels_petsc = new Vec[FieldVar::Dim];

                for(auto dim = 0; dim < FieldVar::Dim; dim++) {

                        Vec vec_tmp = FieldVar::PetscVectorFromArray(Vel_vec[dim], FieldVar::Natoms);
                        Vels_petsc[dim] = vec_tmp;
                };


		FieldVar::ierr = FieldVar::computeKernel(COORDS_IN); CHKERRQ(FieldVar::ierr);
		FieldVar::ierr = MatMatTransposeMult(FieldVar::KernelMatrix, FieldVar::KernelMatrix, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &(FieldVar::TransKernelTrans));
		CHKERRQ(FieldVar::ierr);

		fp << FieldVar::GetTime() << ":INFO:Computed Kernel Matrix Tranpose " << std::endl; 
		
		KSP ksp;
       		FieldVar::ierr = KSPCreate(FieldVar::COMM, &ksp); CHKERRQ(FieldVar::ierr);

		FieldVar::ierr = MatShift(FieldVar::TransKernelTrans, FieldVar::Scaling); CHKERRQ(FieldVar::ierr);

        	FieldVar::ierr = KSPSetOperators(ksp, FieldVar::TransKernelTrans, FieldVar::TransKernelTrans); CHKERRQ(FieldVar::ierr);
        	FieldVar::ierr = KSPSetType(ksp, KSPBCGS); CHKERRQ(FieldVar::ierr);
        	FieldVar::ierr = KSPSetFromOptions(ksp); CHKERRQ(FieldVar::ierr);

        	PC pc;
        	FieldVar::ierr = KSPGetPC(ksp, &pc); CHKERRQ(FieldVar::ierr);
        	PCSetType(pc, PCCHOLESKY); 
		PetscScalar error;

		fp << FieldVar::GetTime() << ":INFO:Solving for Lag multipliers " << std::endl;
 
		for(auto dim = 0; dim < 1; dim++) {

			Vec mom = FieldVar::PetscVectorFromArray(FV[dim], DIM_FV1); CHKERRQ(FieldVar::ierr);
			FieldVar::ierr = VecScale(Vels_petsc[dim], -1.0); CHKERRQ(FieldVar::ierr);

			FieldVar::ierr = MatMultAdd(FieldVar::KernelMatrix, Vels_petsc[dim], mom, mom); CHKERRQ(FieldVar::ierr);

			/*
			MatMult(FieldVar::KernelMatrix, Vels_petsc[dim], mom);
			VecView(mom, PETSC_VIEWER_STDOUT_SELF);

			vector<PetscScalar> tmp = FieldVar::ComputeFV_MomX(FieldVar::Coords, vx);
			for(auto i = tmp.begin(); i != tmp.end(); i++)
				std::cout << *i << " , ";

			std::cout << std::endl;			
			*/

			FieldVar::ierr = VecScale(Vels_petsc[dim], -1.0); CHKERRQ(FieldVar::ierr);

	        	FieldVar::ierr = KSPSolve(ksp, mom, mom); CHKERRQ(FieldVar::ierr);

			FieldVar::ierr = MatMultTransposeAdd(FieldVar::KernelMatrix, mom, Vels_petsc[dim], Vels_petsc[dim]); CHKERRQ(FieldVar::ierr);
		}

                PetscScalar **Vels_local = new PetscScalar*[FieldVar::Dim];

                for(auto dim = 0; dim < FieldVar::Dim; dim++)
                        FieldVar::ierr = VecGetArray(Vels_petsc[dim], (Vels_local+dim));

		CHKERRQ(FieldVar::ierr);

                for(int i = 0; i < FieldVar::Natoms; i++) {

                        VELS_OUT[i * FieldVar::Dim + 0] = Vels_local[0][i];
                        VELS_OUT[i * FieldVar::Dim + 1] = Vels_local[1][i];
                        VELS_OUT[i * FieldVar::Dim + 2] = Vels_local[2][i];

                }

                for(auto dim = 0; dim < FieldVar::Dim; dim++)
                	FieldVar::ierr = VecDestroy(&Vels_petsc[dim]);

		CHKERRQ(FieldVar::ierr);

        	FieldVar::ierr = KSPDestroy(&ksp);
		CHKERRQ(FieldVar::ierr);

		delete[] Vels_petsc;

	}

	PetscFunctionReturn(ierr);
}


PetscErrorCode FieldVar::Py_FineGrain(double* FV, int FV_DIM, double *x, int nx, double *y, int ny, double *z, int nz, double* COORDS_OUT, int NATOMS_BY_3,
					 PyObject* Py_Assemble) {
	/**********************************************************************/
	/***** FineGraining works via a numerical optimization algorithm *****/
	/********************************************************************/

	// Does the x,y, and z change in this routine ???

	PetscFunctionBegin;

	int Assemble = PyInt_AsLong(Py_Assemble);

	if(nx != ny | nx != nz | ny != nz) {
		cerr << "Error in atomic coords input for fine graining." << endl;
		cerr << "Given dimensions: " << nx << " " << ny << " " << nz << endl;
	}
	else {
		// Maybe we should use VecCreateMPIWithArray here ???
		vector<PetscScalar*> Coords_vec(FieldVar::Dim);
		Coords_vec[0] = x;
		Coords_vec[1] = y;
		Coords_vec[2] = z;

		Vec *Coords_petsc = new Vec[FieldVar::Dim];

		PetscScalar atomic_error = FieldVar::Tol,
			    cons_error   = FieldVar::Tol;

		Vec Multipliers;
		VecCreateMPI(FieldVar::COMM, PETSC_DECIDE, FieldVar::AdjNumNodes, &Multipliers);

		for(auto dim = 0; dim < FieldVar::Dim; dim++) {

			Vec vec_tmp = FieldVar::PetscVectorFromArray(Coords_vec[dim], FieldVar::Natoms);
			Coords_petsc[dim] = vec_tmp;
		}

		// Begin Newton && steepest descent method

		Vec atomic_disp;
		VecCreateMPI(FieldVar::COMM, PETSC_DECIDE, FieldVar::Natoms, &atomic_disp);

		ofstream fp("proto.log", ios::out | ios::app);

		Vec FV_vec = FieldVar::PetscVectorFromArray(FV, FV_DIM);
		PetscInt iters = 0;
		PetscScalar minFV;

		while(atomic_error >= FieldVar::Tol) {

			if(Assemble && iters == 0)
				for(auto dim = 0; dim < FieldVar::Dim; dim++)
					FieldVar::ierr = FieldVar::KernelJacobian(Coords_petsc, dim);

			cons_error = FieldVar::ComputeLagrangeMulti(Coords_petsc, Multipliers, FV_vec, FieldVar::Scaling, iters, Assemble);

			atomic_error = .0;

			for(auto dim = 0; dim < FieldVar::Dim; dim++) {
				Vec tmp = FieldVar::AdvancedAtomistic(Coords_petsc[dim], Multipliers, dim);

				VecCopy(tmp, atomic_disp);
				VecAXPBY(atomic_disp, -1.0, 1.0, Coords_petsc[dim]); // atom_disp -= Coords_petsc

				PetscScalar atomic_tmp;
				VecNorm(atomic_disp, NORM_INFINITY, &atomic_tmp);
				VecCopy(tmp,Coords_petsc[dim]);

				VecDestroy(&tmp);
				atomic_error = max(atomic_tmp, atomic_error);
			}

			iters++;
			fp << FieldVar::GetTime() << ":INFO:Newton max atomic displacement: " << atomic_error <<
				  ", and rel. density error = " << cons_error << endl;
		}

		fp.close();
		PetscScalar **Coords_local = new PetscScalar*[FieldVar::Dim];

		for(auto dim = 0; dim < FieldVar::Dim; dim++)
			VecGetArray(Coords_petsc[dim], (Coords_local+dim));

		for(int i = 0; i < FieldVar::Natoms; i++) {
			COORDS_OUT[i * FieldVar::Dim]     = Coords_local[0][i];
			COORDS_OUT[i * FieldVar::Dim + 1] = Coords_local[1][i];
			COORDS_OUT[i * FieldVar::Dim + 2] = Coords_local[2][i];
		}

		//VecView(Multipliers, PETSC_VIEWER_STDOUT_SELF);
		for(auto dim = 0; dim < FieldVar::Dim; dim++)
				VecDestroy(&Coords_petsc[dim]);

		VecDestroy(&Multipliers);
		VecDestroy(&atomic_disp);
		VecDestroy(&FV_vec);
		delete[] Coords_petsc;
	}

	PetscFunctionReturn(FieldVar::ierr);
}

PetscErrorCode FieldVar::Py_UpdateGrid(double* IN_ARRAY2, int DIM1, int DIM2) {

  /***********************************************************/
  /****** Update grid, data structures, and free memory *****/
  /*********************************************************/

  FieldVar::ierr =  FieldVar::DeleteJacobians(); CHKERRQ(FieldVar::ierr);
  FieldVar::ierr = FieldVar::Py_CoarseGrain(IN_ARRAY2,DIM1, DIM2); CHKERRQ(FieldVar::ierr);

  return FieldVar::ierr;
}  

PetscErrorCode FieldVar::Py_CoarseGrain(double* IN_ARRAY2, int DIM1, int DIM2) {

  /****************************************************************/
  /****** Setup up variables and objects for coarse-graining *****/
  /**************************************************************/

  // Keep the & in output so that Python can destroy this variable when the program exits.
  // There needs to be a function UpdateGrid() that reallocates stuff

  PetscFunctionBegin;
  PetscErrorCode ierr;

  // structured grid IDs all set
  ierr = FieldVar::ConstructGridID(); CHKERRQ(ierr); 

  // discretize in space
  ierr = FieldVar::ConstructGrid(FieldVar::NumNodes); CHKERRQ(ierr);

  // AdjNumNodes setup here!
  ierr = FieldVar::ConstructFVList(IN_ARRAY2); CHKERRQ(ierr);

  //ierr = FieldVar::ConstructAtomList(IN_ARRAY2); CHKERRQ(ierr); we do not need this anymore; leave it just in case

  // reconstruct the ghost grid ... WHY?!
  ierr = FieldVar::ConstructGrid(FieldVar::AdjNumNodes); CHKERRQ(ierr);

  // Memory reallocation is done only once for every discretization
  FieldVar::FieldVars.resize(FieldVar::AdjNumNodes);
  FieldVar::FieldVars_Vel.resize(FieldVar::AdjNumNodes);
  FieldVar::FieldVars_For.resize(FieldVar::AdjNumNodes);

  FieldVar::SetupJacobians();

  PetscFunctionReturn(ierr);
}

string FieldVar::GetTime() {
  PetscFunctionBegin;

  time_t rawtime;
  struct tm *timeinfo;
  char *buffer = new char[PETSC_MAX_PATH_LEN];

  time (&rawtime);
  timeinfo = localtime (&rawtime);

  strftime (buffer,PETSC_MAX_PATH_LEN,"%F %R:%S,",timeinfo);

  string time_str(buffer);
  PetscFunctionReturn(time_str);
}
