											/********************************************************/
											/****** Continuum Field Variables: Coarse-graining *****/
											/******************************************************/

/* The idea here is to coarse grain an atomistic state (the fine scale state) by defining a kernel function K such that:
 *
 * 										FV = \int K * FV / density
 *
 * The choice of the field variable (FV) used here is the density i.e. FV becomes in discretized form:
 *
 * 											FV = \sum_i K_i
 *
 * Based on this the FV momenta and forces are easily computed.
 *
 * TODO: This file ought to be optimized by combining the force,vel, pos for FVs evaluations into a single function that avoids three time loops over
 * the AtomicList.
 *
 * TODO: ??? worth parallelizing or stick to serial code ???
 */


#include "Interface.h"

PetscScalar FieldVar::KernelFunction(const PetscScalar* const Coords, PetscInt mass, const vector<PetscScalar> &GridPos) {
	PetscScalar dotProd = .0;

	for(int d = 0; d < GridPos.size(); d++)
		dotProd += pow(GridPos[d] - Coords[d], 2.0);

	dotProd = dotProd / pow(FieldVar::Resol, 2.0);

	return exp(-dotProd) * mass / pow(FieldVar::Resol * sqrt(M_PI), 3.0);
}



vector<PetscScalar> FieldVar::ComputeFV(const PetscScalar* const Coords) {

	/***********************************************************************/
	/****** Computes the field variables of choice: MPI not YET DONE! *****/
	/*********************************************************************/

	PetscFunctionBegin;
	auto it = FieldVar::FVList.begin(); // FVList contains atomic indices for each FV cell / node

	for(auto i = 0; i < FieldVar::AdjNumNodes; i++) {
		FieldVar::FieldVars[i] = 0.f; // clear previously stored values

		vector<PetscScalar> GridPos = FieldVar::Grid[i];

		for(auto aa_it = it->begin(); aa_it != it->end(); aa_it++)
			FieldVar::FieldVars[i] +=  FieldVar::KernelFunction(Coords+(FieldVar::Dim)*(*aa_it), FieldVar::Mass[*aa_it], GridPos);

		it++;
	}
	return FieldVar::FieldVars;
}

vector<PetscScalar> FieldVar::ComputeFV_Vel(const PetscScalar* const Coords, const PetscScalar* const Vel) {

	/*************************************************************************/
	/****** Computes the velocity of field variables: MPI not YET DONE! *****/
	/***********************************************************************/

	PetscFunctionBegin;
        auto it = FieldVar::FVList.begin(); // FVList contains atomic indices for each FV cell / node

        for(auto i = 0; i < FieldVar::AdjNumNodes; i++) {
                FieldVar::FieldVars_Vel[i] = 0.f; // clear previously stored values

                vector<PetscScalar> GridPos = FieldVar::Grid[i];

                for(auto aa_it = it->begin(); aa_it != it->end(); aa_it++) 
                        FieldVar::FieldVars_Vel[i] += Vel[(FieldVar::Dim)*(*aa_it)+2] * 
							FieldVar::KernelFunction(Coords+(FieldVar::Dim)*(*aa_it), 
										 FieldVar::Mass[*aa_it], GridPos);

                it++;
        }

	return FieldVar::FieldVars_Vel;
}

vector<PetscScalar> FieldVar::ComputeFV_For(const PetscScalar* const Coords, const PetscScalar* const Vel, const PetscScalar* const For) {

	/**********************************************************************/
	/****** Computes the force of field variables: MPI not YET DONE! *****/
	/********************************************************************/

	PetscFunctionBegin;
	auto it = FieldVar::FVList.begin();

	for(auto i = 0; i < FieldVar::AdjNumNodes; i++) {

		FieldVar::FieldVars_For[i] = 0.f;
		vector<PetscScalar> GridPos = FieldVar::Grid[i]; // must be deleted when re-equilibriating to avoid memory leaks

		for(auto aa_it = it->begin(); aa_it != it->end(); aa_it++) {

			const PetscScalar *const AtomicCoords = Coords+(FieldVar::Dim)*(*aa_it),
					  *const AtomicVel = Vel+(FieldVar::Dim)*(*aa_it),
					  *const AtomicFor = For+(FieldVar::Dim)*(*aa_it),
					  Kernel = FieldVar::KernelFunction(AtomicCoords, FieldVar::Mass[*aa_it], GridPos);

			for(auto d = 0; d < FieldVar::Dim; d++)
				FieldVar::FieldVars_For[i] += 2.0 * ( (GridPos[d] - AtomicCoords[d]) / pow(FieldVar::Resol,2.0) * AtomicFor[d]
					- AtomicVel[d] / pow(FieldVar::Resol,2.0) * AtomicVel[d]
					+ 2.0 * AtomicVel[d] * pow((GridPos[d] - AtomicCoords[d]) / pow(FieldVar::Resol,2.0), 2.0) * AtomicVel[d] ) * Kernel;
		}

		it++;
	}

	return FieldVar::FieldVars_For;
}

PetscErrorCode FieldVar::DeleteJacobians() {
	PetscFunctionBegin;

	for(auto d = 0; d < FieldVar::Dim; d++)
		FieldVar::ierr = MatDestroy(&(FieldVar::JacobKernel[d]));

	FieldVar::JacobKernel.clear(); // Very important for push pack to work when updating grid!!

	FieldVar::ierr = MatDestroy(&(FieldVar::Jacobian));
	FieldVar::ierr = MatDestroy(&(FieldVar::TransKernelTrans));
	FieldVar::ierr = MatDestroy(&(FieldVar::KernelMatrix));

	PetscFunctionReturn(FieldVar::ierr);
}

PetscErrorCode FieldVar::SetupJacobians() {

  /******************************************************************/
  /****** Allocates memory for the Jacobians: runs in MPI mode *****/
  /****************************************************************/

  /* This function allocates memory for the kernel and full Jacobian matrices.
     It does not do any computations. Therefore, when updating the grid or
     discretizing, it is absolutely *necessary* to call DeleteJacobians()
     in order to avoid memory lacks.
  */
	PetscFunctionBegin;

	PetscInt *nnz = new PetscInt[FieldVar::AdjNumNodes];
	PetscInt count = 0;

	for(auto it = FieldVar::FVList.begin(); it != FieldVar::FVList.end(); it++)
		nnz[count++] = it->size();


	// why did I opt for constructing NCG x Natom matrices here??? 

	for(auto d = 0; d < FieldVar::Dim; d++) {
		Mat Mat_tmp;
		FieldVar::ierr = MatCreate(FieldVar::COMM, &Mat_tmp);
		MatSetSizes(Mat_tmp, PETSC_DECIDE, PETSC_DECIDE, FieldVar::AdjNumNodes, FieldVar::Natoms); // delete and reallocate when updating grid
		MatSetType(Mat_tmp, MATMPIAIJ);
		MatMPIAIJSetPreallocation(Mat_tmp, 0, nnz, 0, nnz);
		FieldVar::JacobKernel.push_back(Mat_tmp);
		//MatView(FieldVar::JacobKernel[d], PETSC_VIEWER_STDOUT_SELF);
	}

    	ierr = FieldVar::ierr = MatCreate(FieldVar::COMM, &(FieldVar::KernelMatrix));
        ierr = MatSetSizes(FieldVar::KernelMatrix, PETSC_DECIDE, PETSC_DECIDE, FieldVar::AdjNumNodes, FieldVar::Natoms);
        ierr = MatSetType(FieldVar::KernelMatrix, MATSEQAIJ);
	MatSeqAIJSetPreallocation(KernelMatrix, 0, nnz);

	delete[] nnz;

	nnz = new PetscInt[FieldVar::AdjNumNodes];

	for(auto i = 0; i < FieldVar::AdjNumNodes; i++)
		nnz[i] = 0;

	// 1 ... (NumNodes + Grid_Neighbors - 1) -> exclude boundary points
	for(auto i = 1; i < FieldVar::NumNodes_x + FieldVar::Grid_Neighbors_x - 1; i++)
		for(auto j = 1; j < FieldVar::NumNodes_y + FieldVar::Grid_Neighbors_y - 1; j++)
			for(auto k = 1; k < FieldVar::NumNodes_z + FieldVar::Grid_Neighbors_z - 1; k++)
				if(FieldVar::GridID[i][j][k] >= 0 ) {

					PetscInt index_x_min = -1,
						 index_y_min = -1,
						 index_z_min = -1, // extended gird searching must be taken into account!!! [FUTURE]
						 index_x_max = +1,
						 index_y_max = +1,
						 index_z_max = +1;

					// neighbor searching along the x, y, and z- axes
					for(auto index_x = index_x_min; index_x <= index_x_max; index_x++)
						for(auto index_y = index_y_min; index_y <= index_y_max; index_y++)
							for(auto index_z = index_z_min; index_z <= index_z_max; index_z++)
								if(FieldVar::GridID[index_x+i][index_y+j][index_z+k] >= 0)
									nnz[FieldVar::GridID[i][j][k]] += 1;
				}

	FieldVar::ierr = MatCreate(FieldVar::COMM, &(FieldVar::Jacobian));
	MatSetSizes(FieldVar::Jacobian, PETSC_DECIDE, PETSC_DECIDE, FieldVar::AdjNumNodes, FieldVar::AdjNumNodes);
	MatSetType(FieldVar::Jacobian, MATMPIAIJ);
	MatMPIAIJSetPreallocation(FieldVar::Jacobian, 0, nnz, 0, nnz);

	ierr = FieldVar::ierr = MatCreate(FieldVar::COMM, &(FieldVar::TransKernelTrans));
        ierr = MatSetSizes(FieldVar::TransKernelTrans, PETSC_DECIDE, PETSC_DECIDE, FieldVar::AdjNumNodes, FieldVar::AdjNumNodes);
        ierr = MatSetType(FieldVar::TransKernelTrans, MATSEQAIJ);

	PetscFunctionReturn(FieldVar::ierr);
}

