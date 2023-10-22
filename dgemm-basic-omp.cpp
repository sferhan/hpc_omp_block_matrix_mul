#include <iostream>
#include <omp.h>

#include "likwid-stuff.h"
#include "helpers.cpp"

const char* dgemm_desc = "Basic implementation, OpenMP-enabled, three-loop dgemm.";

/*
 * This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are n-by-n matrices stored in column-major format.
 * On exit, A and B maintain their input values.
 */
void square_dgemm(int n, double* A, double* B, double* C) 
{
   // insert your code here: implementation of basic matrix multiply with OpenMP parallelism enabled

   // be sure to include LIKWID_MARKER_START(MY_MARKER_REGION_NAME) inside the block of parallel code,
   // but before your matrix multiply code, and then include LIKWID_MARKER_STOP(MY_MARKER_REGION_NAME)
   // after the matrix multiply code but before the end of the parallel code block.

   LIKWID_MARKER_START(MY_MARKER_REGION_NAME);
   
   #pragma omp parallel for collapse(2)
   for(int i=0; i<n; i++) {
      for(int j=0; j<n; j++) {
         double sum_1r_nc = 0;
         for(int k=0; k<n; k++) {
            sum_1r_nc += (A[col_major_transform(i, k, n)] * B[col_major_transform(k, j, n)]);
         }
         C[col_major_transform(i, j, n)] += sum_1r_nc;
      }
   }
   
   LIKWID_MARKER_STOP(MY_MARKER_REGION_NAME);
}
