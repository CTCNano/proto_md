											/******************************************************/
											/****** Continuum Field Variables: Fine-graining *****/
											/****************************************************/

/* The idea here is to obtain an ensemble of fine scale states from a given CG state. This is achieved via a numerical optimization approach that is based
 * on the following Lagrangian:
 *
 * 										L = 1/2 * (r - r_MD)^t (r-r_MD) + c^t \lambda
 *
 * ^t signifies the transpose, \lambda is a set of Lagrange multipliers, and c is the set of constraint equations defined by:
 *
 * 											c = FV - \sum_i K_i
 *
 * Minimizing the Lagrangian is problematic because the problem is highly ill-conditioned. The algorithm used here is based on a combination of:
 * constraint MD algorithms, steepest descent, and Newton's method. The equations to solve are:
 *
 * 											r = r_MD - Jk^t * \lambda
 * 											c = 0
 *
 * Here r_MD is a vector of atomic coordinates obtained from the last MD run (micro phase). It is a guess to the true solution r = r* that we seek.
 * Jk is the Jacobian of the kernel function K.
 */

#include "Interface.h"

PetscErrorCode FieldVar::KernelJacobian(const Vec* const Coords, const PetscInt &dim) {

	/******************************************************************************/
	/******** Construct the transpose of the Jacobian of the kernel function *****/
	/************ The size of the Jacobian is Natoms x N_CG  ********************/
	/***************************************************************************/

	/* The kernel Jacobian is NOT initialized to zero every time it is updated
	*  because the nnz entries are INSERTED at the same location every time. 
	*/
	
	PetscFunctionBegin;

        FieldVar::fp << FieldVar::GetTime() << ":INFO:Computing Kernel Jacobian " << endl;

	PetscErrorCode ierr;
	PetscInt istart, iend;

	VecGetArray(Coords[0], &(FieldVar::Coords_Local_x));
	VecGetArray(Coords[1], &(FieldVar::Coords_Local_y));
	VecGetArray(Coords[2], &(FieldVar::Coords_Local_z));

	MatGetOwnershipRange(FieldVar::JacobKernel[dim], &istart, &iend);

	for(auto i = istart; i < iend; i++) {

		PetscScalar *Jk = new PetscScalar[FieldVar::FVList[i].size()];
		PetscInt *Indices = new PetscInt[FieldVar::FVList[i].size()];
		PetscInt count = 0;
				    
		for(auto atom = FieldVar::FVList[i].begin(); atom != FieldVar::FVList[i].end(); atom++) {
			vector<PetscScalar> GridPos = FieldVar::Grid[i];
			const PetscScalar r[] = {FieldVar::Coords_Local_x[*atom], FieldVar::Coords_Local_y[*atom], FieldVar::Coords_Local_z[*atom]};
			Jk[count] = (FieldVar::Grid[i][dim] - r[dim]) * (FieldVar::KernelFunction(r, FieldVar::Mass[*atom], GridPos)) * (2.0 / pow(FieldVar::Resol,2.0));
			Indices[count++] = (*atom);
		}

		ierr = MatSetValues(FieldVar::JacobKernel[dim], 1, &i, FieldVar::FVList[i].size(), Indices, Jk, INSERT_VALUES); CHKERRQ(ierr);
		delete[] Jk;
		delete[] Indices;
	}

	ierr = MatAssemblyBegin(FieldVar::JacobKernel[dim], MAT_FINAL_ASSEMBLY);
	ierr = MatAssemblyEnd(FieldVar::JacobKernel[dim], MAT_FINAL_ASSEMBLY);

	//MatView(FieldVar::JacobKernel[dim], PETSC_VIEWER_STDOUT_SELF);
	VecRestoreArray(Coords[0], &(FieldVar::Coords_Local_x));
	VecRestoreArray(Coords[1], &(FieldVar::Coords_Local_y));
	VecRestoreArray(Coords[2], &(FieldVar::Coords_Local_z));

	PetscFunctionReturn(ierr);
}

PetscErrorCode FieldVar::AssembleJacobian(const Vec *const Coords) {
	PetscFunctionBegin;

        FieldVar::fp << FieldVar::GetTime() << ":INFO:Assembling Jacobian " << endl;

	PetscErrorCode ierr;
	auto node = (FieldVar::FVList).begin();

        VecGetArray(Coords[0], &(FieldVar::Coords_Local_x));
        VecGetArray(Coords[1], &(FieldVar::Coords_Local_y));
        VecGetArray(Coords[2], &(FieldVar::Coords_Local_z));

	for(auto i = 1; i < FieldVar::NumNodes_x + FieldVar::Grid_Neighbors_x - 1; i++)
		for(auto j = 1; j < FieldVar::NumNodes_y + FieldVar::Grid_Neighbors_y - 1; j++)
			for(auto k = 1; k < FieldVar::NumNodes_z + FieldVar::Grid_Neighbors_z - 1; k++) {

				PetscInt grid_index1 = (FieldVar::GridID)[i][j][k];

				if(grid_index1 >= 0) { // ghost cells and boundary cells not taken into account

					PetscInt index_x_min = -1,
						 index_y_min = -1,
						 index_z_min = -1, // extended gird searching must be taken into account!!! [FUTURE]
                	             		 index_x_max = +1,
                        	     		 index_y_max = +1,
                             			 index_z_max = +1;

					// Check for boundary cells
					if(FieldVar::GridID[i-1][j][k] < 0)
						index_x_min = 0;

					if(FieldVar::GridID[i+1][j][k] < 0)
						index_x_max = 0;

					if(FieldVar::GridID[i][j-1][k] < 0)
						index_y_min = 0;

					if(FieldVar::GridID[i][j+1][k] < 0)
						index_y_max = 0;

					if(FieldVar::GridID[i][j][k-1] < 0)
						index_z_min = 0;

					if(FieldVar::GridID[i][j][k+1] < 0)
						index_z_max = 0;

					vector<PetscScalar> GridPos = (FieldVar::Grid)[grid_index1];

					for(auto index_x = index_x_min; index_x <= index_x_max; index_x++)
						for(auto index_y = index_y_min; index_y <= index_y_max; index_y++)
							for(auto index_z = index_z_min; index_z <= index_z_max; index_z++) {

								PetscInt grid_index2 = FieldVar::GridID[i + index_x][j + index_y][k + index_z];

								if (grid_index2 >= 0) {
									PetscScalar tmp = .0;
									vector<PetscScalar> GridPos2 = (FieldVar::Grid)[grid_index2];

									for(auto atom = node->begin(); atom != node->end(); atom++) {

										PetscScalar aa_coords[] = {FieldVar::Coords_Local_x[*atom],
												           FieldVar::Coords_Local_y[*atom],
													   FieldVar::Coords_Local_z[*atom]};

										for(auto dim = 0; dim < FieldVar::Dim; dim++) {
											PetscScalar Jk = ((FieldVar::Grid)[grid_index2][dim] - aa_coords[dim]) * FieldVar::KernelFunction(aa_coords, FieldVar::Mass[*atom], GridPos2) * (2.0 / pow(FieldVar::Resol,2.0));
											tmp += Jk * ( (FieldVar::Grid)[grid_index1][dim] - aa_coords[dim]) / pow(FieldVar::Resol,2.0) * FieldVar::KernelFunction(aa_coords, FieldVar::Mass[*atom], GridPos) ;
										}
									}

									tmp *= 2.0;
									ierr = MatSetValues(FieldVar::Jacobian, 1, &grid_index1, 1, &grid_index2, &tmp, INSERT_VALUES);
								}
							}

						node++;
			}
		}

	MatAssemblyBegin(FieldVar::Jacobian, MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(FieldVar::Jacobian, MAT_FINAL_ASSEMBLY);

        VecRestoreArray(Coords[0], &(FieldVar::Coords_Local_x));
        VecRestoreArray(Coords[1], &(FieldVar::Coords_Local_y));
        VecRestoreArray(Coords[2], &(FieldVar::Coords_Local_z));

	return ierr;
}

Vec FieldVar::AdvancedAtomistic(const Vec &Coords_MD, const Vec &Multipliers, const PetscInt &dim) {
	
	/************************************************************/
	/****** Advance the microstate based on the relation *******/
	/****** 		   r = r_MD + Jk * Lambda	      	*******/
	/*********************************************************/

	PetscFunctionBegin;
	FieldVar::fp << FieldVar::GetTime() << ":INFO:Advancing Atomistic variables" << endl; 
	Vec Coords;
	VecCreateMPI(FieldVar::COMM, PETSC_DECIDE, FieldVar::Natoms, &Coords);
	PetscErrorCode ierr = MatMultTransposeAdd(FieldVar::JacobKernel[dim], Multipliers, Coords_MD, Coords);

	PetscFunctionReturn(Coords);
}

Vec FieldVar::Constraints(const Vec *const Coords) {
	PetscFunctionBegin;

        FieldVar::fp << FieldVar::GetTime() << ":INFO:Computing Constraints " << endl;

	VecGetArray(Coords[0], &(FieldVar::Coords_Local_x));
	VecGetArray(Coords[1], &(FieldVar::Coords_Local_y));
	VecGetArray(Coords[2], &(FieldVar::Coords_Local_z));

	Vec Cons;
	FieldVar::ierr = VecCreateMPI(FieldVar::COMM, PETSC_DECIDE, FieldVar::AdjNumNodes, &Cons);

	auto it = FieldVar::FVList.begin();

	for(auto i = 0; i < FieldVar::AdjNumNodes; i++) {
		PetscScalar val = .0;

		for(auto aa_it = it->begin(); aa_it != it->end(); aa_it++) {

			vector<PetscScalar> GridPos = FieldVar::Grid[i];
			PetscScalar aa_coords[] = {FieldVar::Coords_Local_x[*aa_it], FieldVar::Coords_Local_y[*aa_it], 
						   FieldVar::Coords_Local_z[*aa_it]};

			val += FieldVar::KernelFunction(aa_coords, FieldVar::Mass[*aa_it], GridPos);
		}

		FieldVar::ierr = VecSetValues(Cons, 1, &i, &val, INSERT_VALUES);
		it++;
	}

        VecRestoreArray(Coords[0], &(FieldVar::Coords_Local_x));
        VecRestoreArray(Coords[1], &(FieldVar::Coords_Local_y));
        VecRestoreArray(Coords[2], &(FieldVar::Coords_Local_z));

	VecAssemblyBegin(Cons);
	VecAssemblyEnd(Cons);

	PetscFunctionReturn(Cons);
}

PetscScalar FieldVar::ComputeLagrangeMulti(const Vec *const Coords, Vec Multipliers, const Vec &FV, PetscScalar Scaling, PetscInt new_iters, int Assemble) {
	PetscFunctionBegin;

	Vec Cons = FieldVar::Constraints(Coords);
	FieldVar::fp << FieldVar::GetTime() << ":INFO:Computing Cons diff " << endl;

	VecAXPBY(Cons, 1.0, -1.0, FV); // Cons = FV - Cons

	PetscScalar cons_error;
	VecNorm(Cons, NORM_INFINITY, &cons_error);

	//if(Assemble)
	//	{
			FieldVar::ierr = FieldVar::AssembleJacobian(Coords);
			MatShift(FieldVar::Jacobian, Scaling);
	//	}

	//MatView(FieldVar::Jacobian, PETSC_VIEWER_STDOUT_WORLD);

	FieldVar::fp << FieldVar::GetTime() << ":INFO:Calling KSP solver " << endl;

	KSP ksp;
	PetscInt iters;
	KSPCreate(FieldVar::COMM, &ksp);

	KSPSetOperators(ksp, FieldVar::Jacobian, FieldVar::Jacobian); // SAME_PRECONDITIONER);
	KSPSetType(ksp, KSPBCGS);
	KSPSetFromOptions(ksp);

	PC pc;
        KSPGetPC(ksp, &pc);
        PCSetType(pc, PCBJACOBI);

	KSPSolve(ksp, Cons, Multipliers);

	KSPGetIterationNumber(ksp, &iters);
	KSPDestroy(&ksp);

	VecDestroy(&Cons);

	FieldVar::fp << FieldVar::GetTime() << ":INFO:KSP converged in " << iters << endl;

	PetscFunctionReturn(cons_error);
}
