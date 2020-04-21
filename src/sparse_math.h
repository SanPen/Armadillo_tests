#ifndef SPARSE_MATH_H
#define SPARSE_MATH_H

#include <armadillo>


/*
@brief integer binary search
@param array vector of integer values
@param x value to look for
@return the index where the value is found, -1 otherwise
*/
int binary_search(arma::uvec* array, arma::uword x) {

    arma::uword lower = 0;
    arma::uword upper = array->size();
    arma::uword mid;
    arma::uword val;

    while (lower < upper) {

        mid = lower + (upper - lower) / 2;
        val = array->at(mid);

        if (x == val) {
            return mid;

        } else if (x > val ) {
            if (lower == mid) {
                break;
            }
            lower = mid;
        }else if (x < val) {
            upper = mid;
        }
    }

    return -1;
}


/**
* @brief sp_submatrix Function to extract columns and rows from a CSC sparse matrix
* @param A CSC Sparse matrix pointer
* @param rows vector of the rown indices to keep (must be sorted)
* @param cols vector of the clumn indices to keep (must be sorted)
* @return Sparse matrix of the indicated indices
*/
arma::sp_mat sp_submatrix_old(arma::sp_mat* A, arma::uvec* rows, arma::uvec* cols) {

    arma::uword n_rows = rows->size();
    arma::uword n_cols = cols->size();

    arma::uword n = 0;
    arma::uword p = 0;
    int found_idx = 0;

    arma::vec new_val(A->n_nonzero);
    arma::uvec new_row_ind(A->n_nonzero);
    arma::uvec new_col_ptr(n_cols + 1);

    new_col_ptr(p) = 0;

    for (auto const& j : *cols) { // for every column in the cols vector TODO: change the auto ?

        for (arma::uword k = A->col_ptrs[j]; k < A->col_ptrs[j + 1]; k++) {  // k is the index of the "values" and "row_indices" that corresponds to the column j

            // search row_ind[k] in rows
            found_idx = binary_search(rows, A->row_indices[k]);

            // store the values if the row was found in rows
            if (found_idx > -1) { // if the row index is in the designated rows...
                new_val(n) = A->values[k]; // store the value
                new_row_ind(n) = found_idx;  // store the index where the original index was found inside "rows"
                n++;
            }
        }

        p++;
        new_col_ptr(p) = n;
    }
    new_col_ptr(p) = n;

    // reshape the vectors to the actual number of elements
    new_val.reshape(n, 1);
    new_row_ind.reshape(n, 1);

    return arma::sp_mat(new_row_ind, new_col_ptr, new_val, n_rows, n_cols);
}



/**
* @brief sp_submatrix Function to extract columns and rows from a CSC sparse matrix
* @param A CSC Sparse matrix pointer
* @param rows vector of the rown indices to keep (must be sorted)
* @param cols vector of the clumn indices to keep (must be sorted)
* @return Sparse matrix of the indicated indices
*/
arma::sp_mat sp_submatrix(arma::sp_mat* A, arma::uvec* rows, arma::uvec* cols) {

    arma::uword n_rows = rows->size();
    arma::uword n_cols = cols->size();

    arma::uword n = 0;
    arma::uword p = 0;

    // variables of the sub-matrix
    arma::vec new_val(A->n_nonzero);
    arma::uvec new_row_ind(A->n_nonzero);
    arma::uvec new_col_ptr(n_cols + 1);

    // variables for the binary search
    arma::uword lower;
    arma::uword upper;
    arma::uword position;
    arma::uword val;
    arma::uword x;
    bool found;

    new_col_ptr(p) = 0;

    for (arma::uword const& j : *cols) { // for every column in the cols vector

        for (arma::uword k = A->col_ptrs[j]; k < A->col_ptrs[j + 1]; k++) {  // traverse the rows of A, at the column j

            // k is the index of the "values" and "row_indices" of the column j

            // binary search: we need to determine which rows of A[:, j] are in the specified vector "rows"
            lower = 0;
            upper = n_rows;
            x = A->row_indices[k];
            found = false;

            while (lower < upper && !found) {

                position = lower + (upper - lower) / 2;
                val = rows->at(position);

                if (x == val) { // found: the row "x" at the column j exists in the rows vector used for the subview

                    found = true;
                    new_val(n) = A->values[k];   // store the value
                    new_row_ind(n) = position;        // store the index where the original index was found inside "rows"
                    n++;
                }
                else if (x > val) {

                    if (lower == position) {
                        found = true;
                    } else {
                        lower = position;
                    }
                }
                else if (x < val) {

                    upper = position;
                }
            }
        }

        p++;
        new_col_ptr(p) = n;
    }
    new_col_ptr(p) = n;

    // reshape the vectors to the actual number of elements
    new_val.reshape(n, 1);
    new_row_ind.reshape(n, 1);

    return arma::sp_mat(new_row_ind, new_col_ptr, new_val, n_rows, n_cols);
}

#endif // SPARSE_MATH_H
