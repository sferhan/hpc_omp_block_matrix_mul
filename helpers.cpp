#include <iostream>
#include <string>

using namespace std;

/*
Applies col major transform.
Takes the row-index @row and column-index @col of a n x n matrix and returns
 the position of this value in a 1-D array assuming col-major scheme.
*/
int col_major_transform(int row, int col, int n) {
    return (col * n) + row;
}

void print_sq_col_maj_matrix(int dim, double* matrix, std::string name) {
   std::cout<<name<<std::endl;
   std::cout<<std::endl<<std::endl;
   for(int j=0; j<dim; j++) {
      for (size_t i = 0; i < dim; i++) {
         std::cout<<matrix[col_major_transform(j, i, dim)]<<",";
      }
      std::cout<<std::endl;
   }
   std::cout<<std::endl;
}