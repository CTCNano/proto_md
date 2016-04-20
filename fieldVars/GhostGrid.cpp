#include "Interface.h"

PetscScalar norm(const PetscScalar *const v1, const vector<PetscScalar> &v2, const int &dim) {

	/*****************************************************************/
	/****** Simple norm function ~ should be eventually deleted *****/
	/***************************************************************/

	PetscFunctionBegin;
	PetscScalar result = .0;

	for(int i = 0; i < dim; i++)
		result += pow(v1[i] - v2[i], 2.0);

	PetscFunctionReturn(sqrt(result));
}


PetscErrorCode FieldVar::ConstructGrid(const PetscInt &NumNodes) {

  /*************************************************************/
  /****** Discretize in space and construct a grid tensor *****/
  /***********************************************************/

  PetscFunctionBegin;
  PetscErrorCode ierr=0;
  
  if(FieldVar::Grid.size() > 0)
	 FieldVar::Grid.clear();

  FieldVar::Grid.resize(NumNodes);
  // This is a 3 * NumNodes_x * NumNodes_y * NumNodes_z tensor

  for(auto i = 0; i < FieldVar::Grid.size(); i++)
	  FieldVar::Grid[i].resize(FieldVar::Dim);


  PetscInt dim_x = 0, dim_y = 1, dim_z = 2;
  PetscScalar hx, hy, hz;

  for(auto i = 1; i < FieldVar::NumNodes_x + 1; i++)
	for(auto j = 1; j < FieldVar::NumNodes_y + 1; j++)
		for(auto k = 1; k < FieldVar::NumNodes_z + 1; k++) {

			hx = PetscScalar(i - 1) / (FieldVar::NumNodes_x - 1.0) * (FieldVar::Box[dim_x+FieldVar::Dim] - FieldVar::Box[dim_x]) + FieldVar::Box[dim_x];
			hy = PetscScalar(j - 1) / (FieldVar::NumNodes_y - 1.0) * (FieldVar::Box[dim_y+FieldVar::Dim] - FieldVar::Box[dim_y]) + FieldVar::Box[dim_y];
			hz = PetscScalar(k - 1) / (FieldVar::NumNodes_z - 1.0) * (FieldVar::Box[dim_z+FieldVar::Dim] - FieldVar::Box[dim_z]) + FieldVar::Box[dim_z];

			if(FieldVar::GridID[i][j][k] >= 0) {

				FieldVar::Grid[FieldVar::GridID[i][j][k]][dim_x] = hx;
				FieldVar::Grid[FieldVar::GridID[i][j][k]][dim_y] = hy;
				FieldVar::Grid[FieldVar::GridID[i][j][k]][dim_z] = hz;
				std::cout << "Grid pos = " << hx << " , " << hy << " , " << hz << std::endl; 

			}
		}

  PetscFunctionReturn(ierr);
}

PetscErrorCode FieldVar::ConstructGridID() {

  /******************************************************************************************/
  /****** The idea behind _GridID is similar to that used in finite difference methods *****/
  /****** Each grid cell is assigned an ID. This way, boundary conditions and adaptive ****/
  /****** Discretization become easy to implement and avoid an indexing nightmare.   *****/
  /**************************************************************************************/

  PetscFunctionBegin;
  PetscErrorCode ierr=0;
  PetscInt count;

  for(auto d = 0; d < FieldVar::Dim; d++) {
	  count = 0; // this is stupid ... get rid of this in the future

	  for(auto i = 0; i < FieldVar::NumNodes_x + FieldVar::Grid_Neighbors_x; i++)
		  for(auto j = 0; j < FieldVar::NumNodes_y + FieldVar::Grid_Neighbors_y; j++)
			  for(auto k = 0; k < FieldVar::NumNodes_z + FieldVar::Grid_Neighbors_z; k++)

				  // must take Grid_points_x/y/z for extended grid searching!!!! [FUTURE]
				  if(i == 0 | j == 0 | k == 0 | i == (FieldVar::NumNodes_x + 1)
				     | j == (FieldVar::NumNodes_y +1) | k == (FieldVar::NumNodes_z + 1))
					  FieldVar::GridID[i][j][k] = -1;
				  else
					  FieldVar::GridID[i][j][k] = count++;
  	  	  }

  PetscFunctionReturn(ierr);
}



//PetscErrorCode FieldVar::ConstructAtomList(const double* const Coords) {

  /*******************************************************************************/
  /****** Setup atomic indices list for efficient computation of the Kernel *****/
  /*****************************************************************************/
/*
  PetscFunctionBeginUser;
  PetscErrorCode ierr=0;

  for(auto c = 0; c < FieldVar::Natoms; c++) {
	  vector<PetscInt> node_indices;

	  for(auto i = 0; i < FieldVar::AdjNumNodes; i++)
		  if(FieldVar::Resol >= 0.5 * norm(Coords + FieldVar::Dim * c, FieldVar::Grid[i], FieldVar::Dim))
			  node_indices.push_back(i);

	  FieldVar::AtomicList[c] = node_indices;
  }

  PetscFunctionReturn(ierr);
}
*/

PetscErrorCode FieldVar::ConstructFVList(const double* const Coords) {

  /*****************************************************************************/
  /****** Setup node indices list for efficient computation of the Kernel *****/
  /****** Here the GridID is updated to take into account the transforma- ****/
  /****** tion of the Grid to a ghost Grid that excludes certain nodes   ****/
  /*************************************************************************/
  PetscErrorCode ierr=0;
  FieldVar::FVList.resize(FieldVar::NumNodes);

  for(auto i = 0; i < FieldVar::NumNodes; i++) {
	  PetscInt count = 0;

	  for(auto c = 0; c < FieldVar::Natoms; c++) // This should be replaced by a more efficient for loop
		  if(FieldVar::Resol >= 0.5 * norm(Coords+FieldVar::Dim*c, FieldVar::Grid[i], FieldVar::Dim)) // ~ 10**-5.0 %
			count++;

	  vector<PetscInt> atomic_indices(count);
	  count = 0;

  	  for(auto c = 0; c < FieldVar::Natoms; c++)
	  	if(FieldVar::Resol >= 0.5 * norm(Coords+FieldVar::Dim*c, FieldVar::Grid[i], FieldVar::Dim))
			atomic_indices[count++] = c;

	  FieldVar::FVList[i] = atomic_indices;
  }

  PetscInt count = 0;
  auto it = FieldVar::FVList.begin();

  for(auto i = 0; i < FieldVar::NumNodes_x + FieldVar::Grid_Neighbors_x; i++)
	  for(auto j = 0; j < FieldVar::NumNodes_y + FieldVar::Grid_Neighbors_y; j++)
		  for(auto k = 0; k < FieldVar::NumNodes_z + FieldVar::Grid_Neighbors_z; k++)
			  if(FieldVar::GridID[i][j][k] >= 0) {
				  if(it->size() < FieldVar::Threshold) {
					  FieldVar::GridID[i][j][k] = -2;
					  it = FieldVar::FVList.erase(it);
					  count++;
				  }
				  else
					  it++;
			  }

  cout << "Based on the threshold supplied: " << FieldVar::Threshold << ", the number of nodes has" <<
		  " changed from " << FieldVar::NumNodes << " to " << FieldVar::NumNodes - count << endl;

  FieldVar::AdjNumNodes = FieldVar::NumNodes - count;

  // Remap GridID in case count is NOT zero
  count = 0;

  for(auto i = 0; i < FieldVar::NumNodes_x + FieldVar::Grid_Neighbors_x; i++)
	  for(auto j = 0; j < FieldVar::NumNodes_y + FieldVar::Grid_Neighbors_y; j++)
		  for(auto k = 0; k < FieldVar::NumNodes_z + FieldVar::Grid_Neighbors_z; k++)
			  if(FieldVar::GridID[i][j][k] >= 0) {
				  FieldVar::GridID[i][j][k] = count++;
				}

  PetscFunctionReturn(ierr);
}
