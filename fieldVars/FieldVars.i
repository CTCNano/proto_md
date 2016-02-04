%module FieldVars
 %{
 /* Includes the header in the wrapper code */
 #define SWIG_FILE_WITH_INIT
 #include "Interface.h"
 %}
 
%include "numpy.i"
%init %{
import_array();
%}


%apply (double *ARGOUT_ARRAY1, int DIM1) {(double *COORDS_OUT, int NATOMS_BY_3)};
%apply (double *IN_ARRAY1, int DIM1) {(double *FV, int FV_DIM)};
%apply (double *IN_ARRAY1, int DIM1) {(double *x, int nx)};
%apply (double *IN_ARRAY1, int DIM1) {(double *y, int ny)};
%apply (double *IN_ARRAY1, int DIM1) {(double *z, int nz)};

%apply (double *ARGOUT_ARRAY1, int DIM1) {(double *CG_OUT, int NUMCG)};
%apply (double *IN_ARRAY2    , int DIM1, int DIM2) {(double *COORDS_IN , int NATOMS, int DIM)}; 

%apply (double *ARGOUT_ARRAY1, int DIM1) {(double *CG_OUT, int NUMCG)};
%apply (double *IN_ARRAY2    , int DIM1, int DIM2) {(double *COORDS_IN , int NATOMS1, int DIM1)};
%apply (double *IN_ARRAY2    , int DIM1, int DIM2) {(double *VEL_IN , int NATOMS2, int DIM2)};

%apply (double *ARGOUT_ARRAY1, int DIM1) {(double *CG_OUT, int NUMCG)};
%apply (double *IN_ARRAY2    , int DIM1, int DIM2) {(double *COORDS_IN , int NATOMS1, int DIM1)};
%apply (double *IN_ARRAY2    , int DIM1, int DIM2) {(double *VEL_IN , int NATOMS2, int DIM2)};
%apply (double *IN_ARRAY2    , int DIM1, int DIM2) {(double *FOR_IN , int NATOMS3, int DIM3)};

%apply (double *IN_ARRAY1    , int DIM1) {(double* MASS, int DIM_MASS)};

/* Parse the header file to generate wrappers */

%inline %{
char** convert (char* a, char* b, char* c){
    char** array;
    array = (char**) malloc(3 * sizeof(char *));
    array[0] = a;
    array[1] = b;
    array[2] = c;
    return array;
}%}

%include "Interface.h"
char** convert (char* a, char* b, char* c);
