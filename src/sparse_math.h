#ifndef SPARSE_MATH_H
#define SPARSE_MATH_H

#include <armadillo>


/*
@brief integer binary search
@param array vector of integer values
@param x value to look for

@return the index where the value is found, -1 otherwise
*/

/// Binary search
/// \param array  array of values where to search
/// \param x value to search for
/// \param found_at index where the value is found (passed by reference)
/// \return true if found, false otherwise
bool binary_search(const arma::uvec &array, arma::uword x, arma::uword &found_at) {

    arma::uword lower = 0;
    arma::uword upper = array.size();
    arma::uword val;

    while (lower < upper) {
        found_at = lower + (upper - lower) / 2;
        val = array(found_at);

        if (x == val) {
            return true;

        } else if (x > val ) {
            if (lower == found_at) {
                break;
            }
            lower = found_at;
        }else if (x < val) {
            upper = found_at;
        }
    }

    return false;
}


/// p_submatrix Function to extract columns and rows from a CSC sparse matrix
/// \tparam T data type
/// \param A CSC Sparse matrix pointer
/// \param rows vector of the rown indices to keep (must be sorted)
/// \param cols  vector of the clumn indices to keep (must be sorted)
/// \return Sparse matrix of the indicated indices
template<typename T>
arma::SpMat<T> sp_submat(const arma::SpMat<T> &A, const arma::uvec &rows, const arma::uvec &cols) {

    arma::uword n_rows = rows.size();
    arma::uword n_cols = cols.size();

    arma::uword n = 0;
    arma::uword p = 0;

    // variables of the sub-matrix
    arma::Col<T> new_val(A.n_nonzero);
    arma::uvec new_row_ind(A.n_nonzero);
    arma::uvec new_col_ptr(n_cols + 1);

    // variables for the binary search
    arma::uword lower;
    arma::uword upper;
    arma::uword position;
    arma::uword val;
    arma::uword x;
    bool found;

    new_col_ptr(p) = 0;

    for (arma::uword const& j : cols) { // for every column in the cols vector

        for (arma::uword k = A.col_ptrs[j]; k < A.col_ptrs[j + 1]; k++) {  // traverse the rows of A, at the column j

            // k is the index of the "values" and "row_indices" of the column j

            // binary search: we need to determine which rows of A[:, j] are in the specified vector "rows"
            lower = 0;
            upper = n_rows;
            x = A.row_indices[k];
            found = false;

            while (lower < upper && !found) {

                position = lower + (upper - lower) / 2;
                val = rows(position);

                if (x == val) { // found: the row "x" at the column j exists in the rows vector used for the subview

                    found = true;
                    new_val(n) = A.values[k];   // store the value
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

    return arma::SpMat<T>(new_row_ind, new_col_ptr, new_val, n_rows, n_cols);
}



/// sp_submat Function to extract rows from a sparse matrix
/// \tparam T data type
/// \param A CSC Sparse matrix
/// \param rows vector of the rown indices to keep (must be sorted)
/// \return Sparse matrix of the indicated indices
template<typename T>
arma::SpMat<T> sp_submat_r(const arma::SpMat<T> &A, const arma::uvec &rows) {

    arma::uword n_rows = rows.size();
    arma::uword n_cols = A.n_cols;

    arma::uword n = 0;
    arma::uword p = 0;

    // variables of the sub-matrix
    arma::vec new_val(A.n_nonzero);
    arma::uvec new_row_ind(A.n_nonzero);
    arma::uvec new_col_ptr(n_cols + 1);

    // variables for the binary search
    arma::uword lower;
    arma::uword upper;
    arma::uword position;
    arma::uword val;
    arma::uword x;
    bool found;

    new_col_ptr(p) = 0;

    for (std::size_t j = 0; j < n_cols; j++) { // for every column...

        for (std::size_t k = A.col_ptrs[j]; k < A.col_ptrs[j + 1]; k++) {  // k is the index of the "values" and "row_indices" that corresponds to the column j

            // k is the index of the "values" and "row_indices" of the column j

            // binary search: we need to determine which rows of A[:, j] are in the specified vector "rows"
            lower = 0;
            upper = n_rows;
            x = A.row_indices[k];
            found = false;

            while (lower < upper && !found) {

                position = lower + (upper - lower) / 2;
                val = rows(position);

                if (x == val) { // found: the row "x" at the column j exists in the rows vector used for the subview

                    found = true;
                    new_val(n) = A.values[k];   // store the value
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

    return arma::SpMat<T>(new_row_ind, new_col_ptr, new_val, n_rows, n_cols);
}



/// sp_submat Function to extract columns from a CSC sparse matrix
/// \tparam T data type
/// \param A CSC Sparse matrix
/// \param cols vector of the column indices to keep (must be sorted)
/// \return Sparse matrix of the indicated indices
template<typename T>
arma::SpMat<T>  sp_submat_c(const arma::SpMat<T> &A, const arma::uvec &cols) {

    arma::uword n_rows = A.n_rows;
    arma::uword n_cols = cols.size();

    arma::uword n = 0;
    arma::uword p = 0;

    // variables of the sub-matrix
    arma::Col<T> new_val(A.n_nonzero);
    arma::uvec new_row_ind(A.n_nonzero);
    arma::uvec new_col_ptr(n_cols + 1);

    new_col_ptr(p) = 0;

    for (arma::uword const& j : cols) { // for every column in the cols vector

        for (arma::uword k = A.col_ptrs[j]; k < A.col_ptrs[j + 1]; k++) {  // traverse the rows of A, at the column j

            // k is the index of the "values" and "row_indices" of the column j
            new_val(n) = A.values[k];   // store the value
            new_row_ind(n) = k;        // store the index
            n++;
        }

        p++;
        new_col_ptr(p) = n;
    }
    new_col_ptr(p) = n;

    // reshape the vectors to the actual number of elements
    new_val.reshape(n, 1);
    new_row_ind.reshape(n, 1);

    return arma::SpMat<T>(new_row_ind, new_col_ptr, new_val, n_rows, n_cols);
}



/// Function to extract a sub-vector given the desired indices
/// \tparam T data type
/// \param A Vector pointer
/// \param rows vector of the rows indices to keep (must be sorted)
/// \return Vector of the indicated indices
template<typename T>
arma::Col<T> subvector(const arma::Col<T> &A, const arma::uvec &rows) {

    // declare reduced matrix
    arma::Col<T> M(rows.size());

    std::size_t k = 0;

    for (arma::uword const& j : rows) {
        M(k) = A.at(j);
        k++;
    }
    return M;
}


/**
* @brief Function to extract a sub-matrix from a sub-view given the desired row indices
* @param A matrix view
* @param rows vector of the rows indices to keep (must be sorted)
* @return Vector of the indicated indices
*/
template<typename T>
arma::Col<T> subvector(const arma::subview_col<T> &A, const arma::uvec &rows) {

    // declare reduced matrix

    arma::Col<T> M(rows.size());

    std::size_t k = 0;

    for (arma::uword const& j : rows) {
        M(k) = A.at(0, j);
        k++;
    }
    return M;
}


/**
* @brief sp_submat Function to extract columns and rows from a sparse matrix
* @param A Sparse matrix pointer
* @param rows vector of the rown indices to remove (must be sorted)
* @return Vector of the indicated indices
*/
template<typename T>
arma::Col<T> subvector_inv(const arma::Col<T> &A, const arma::uvec &rows) {

    // declare reduced matrix
    std::size_t a = rows.size();

    arma::Col<T> M = *A;

    std::size_t i; // row index of the full matrix

    // remove rows
    for (i = a; i-- > 0; )
        M.shed_row(rows(i));

    return M;
}


/**
* @brief get a vector of the indices (longer if necessary)
* @param A vector pointer
* @param rows vector of the rown indices to remove (must be sorted)
* @return Vector of the indicated indices
*/
template<typename T>
arma::Col<T> vector_at(const arma::Col<T> &A, const arma::uvec &rows) {
    arma::Col<T> vec(rows.size());
    std::size_t k = 0;
    for (arma::uword const& i : rows) {
        vec.at(k) = A.at(i);
        k++;
    }
    return vec;
}


// absolute of armadillo complex value
double arma_abs(const arma::cx_double &val) {
    return std::sqrt(val.real() * val.real() + val.imag() * val.imag());
}


#endif // SPARSE_MATH_H
