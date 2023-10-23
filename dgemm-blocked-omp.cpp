#include <iostream>
#include <string.h>
#include <stdlib.h>
#include <omp.h>

#include "likwid-stuff.h"
#include "helpers.cpp"

const char* dgemm_desc = "Blocked dgemm, OpenMP-enabled";


void load_block_in_cache(double* block, int n, int block_size, int br, int bc, double* X) {
   for(int i=0; i<block_size; i++) {
      int offset = (((br*block_size)+((bc*block_size)*n)+(i*n)));
      memcpy(block+(i*block_size), X+offset, block_size*sizeof(double));
   }
}

void write_block_to_mem(double* block, int n, int block_size, int br, int bc, double* X) {
   for(int i=0; i<block_size; i++) {
      int offset = (((br*block_size)+((bc*block_size)*n)+(i*n)));
      memcpy(X+offset, block+(i*block_size), block_size*sizeof(double));
   }
}

void block_operate(
   int block_size, double* X, double* Y, double* Z
) {
   for(int i=0; i<block_size; i++) {
      for(int j=0; j<block_size; j++) {
         double sum_1r_nc = 0;
         for(int k=0; k<block_size; k++) {
            sum_1r_nc += (X[col_major_transform(i, k, block_size)] * Y[col_major_transform(k, j, block_size)]);
         }
         Z[col_major_transform(i, j, block_size)] += sum_1r_nc;
      }
   }
}

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are n-by-n matrices stored in column-major format.
 * On exit, A and B maintain their input values. */
void square_dgemm_blocked(int n, int block_size, double* A, double* B, double* C) 
{
   // insert your code here: implementation of blocked matrix multiply with copy optimization and OpenMP parallelism enabled

   // be sure to include LIKWID_MARKER_START(MY_MARKER_REGION_NAME) inside the block of parallel code,
   // but before your matrix multiply code, and then include LIKWID_MARKER_STOP(MY_MARKER_REGION_NAME)
   // after the matrix multiply code but before the end of the parallel code block.

   // insert your code here
   int nb = n/block_size;

   #pragma omp parallel
   {
      LIKWID_MARKER_START(MY_MARKER_REGION_NAME);

      #pragma omp for collapse(2)
      for(int i=0; i<nb; i++) {
         for(int j=0; j<nb; j++) {
            double* a_block = (double*)malloc(block_size*block_size*sizeof(double));
            double* b_block = (double*)malloc(block_size*block_size*sizeof(double));
            double* c_block = (double*)malloc(block_size*block_size*sizeof(double));
            
            load_block_in_cache(c_block, n, block_size, i, j, C);
            for(int k=0; k<nb; k++) {
               load_block_in_cache(a_block, n, block_size, i, k, A);
               
               load_block_in_cache(b_block, n, block_size, k, j, B);
               block_operate(block_size, a_block, b_block, c_block);
            }
            write_block_to_mem(c_block, n, block_size, i, j, C);
            delete a_block;
            delete b_block;
            delete c_block;
         }   
      }

      LIKWID_MARKER_STOP(MY_MARKER_REGION_NAME);
   }
}
