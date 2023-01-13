from libc.stdio cimport FILE
from .c_sundials cimport *

cdef extern from "sunmatrix/sunmatrix_dense.h":

    cdef struct _SUNMatrixContent_Dense:
        sunindextype M
        sunindextype N
        sunrealtype *data
        sunindextype ldata
        sunrealtype **cols

    ctypedef _SUNMatrixContent_Dense *SUNMatrixContent_Dense

    SUNMatrix SUNDenseMatrix(sunindextype M, sunindextype N, SUNContext sunctx)
    void SUNDenseMatrix_Print(SUNMatrix A, FILE* outfile)
    sunindextype SUNDenseMatrix_Rows(SUNMatrix A)
    sunindextype SUNDenseMatrix_Columns(SUNMatrix A)
    sunindextype SUNDenseMatrix_LData(SUNMatrix A)
    sunrealtype* SUNDenseMatrix_Data(SUNMatrix A)
    sunrealtype** SUNDenseMatrix_Cols(SUNMatrix A)
    sunrealtype* SUNDenseMatrix_Column(SUNMatrix A, sunindextype j)
    SUNMatrix_ID SUNMatGetID_Dense(SUNMatrix A)
    SUNMatrix SUNMatClone_Dense(SUNMatrix A)
    void SUNMatDestroy_Dense(SUNMatrix A)
    int SUNMatZero_Dense(SUNMatrix A)
    int SUNMatCopy_Dense(SUNMatrix A, SUNMatrix B)
    int SUNMatScaleAdd_Dense(sunrealtype c, SUNMatrix A, SUNMatrix B)
    int SUNMatScaleAddI_Dense(sunrealtype c, SUNMatrix A)
    int SUNMatMatvec_Dense(SUNMatrix A, N_Vector x, N_Vector y)
    int SUNMatSpace_Dense(SUNMatrix A, long int *lenrw, long int *leniw)


cdef extern from "sunmatrix/sunmatrix_band.h":

    cdef struct _SUNMatrixContent_Band:
        sunindextype M
        sunindextype N
        sunindextype ldim
        sunindextype mu
        sunindextype ml
        sunindextype s_mu
        sunrealtype *data
        sunindextype ldata
        sunrealtype **cols

    ctypedef _SUNMatrixContent_Band *SUNMatrixContent_Band

    SUNMatrix SUNBandMatrix(sunindextype N, sunindextype mu,
                                            sunindextype ml, SUNContext sunctx)

    SUNMatrix SUNBandMatrixStorage(sunindextype N, sunindextype mu,
                                   sunindextype ml, sunindextype smu, SUNContext sunctx)

    void SUNBandMatrix_Print(SUNMatrix A, FILE* outfile)

    sunindextype SUNBandMatrix_Rows(SUNMatrix A)
    sunindextype SUNBandMatrix_Columns(SUNMatrix A)
    sunindextype SUNBandMatrix_LowerBandwidth(SUNMatrix A)
    sunindextype SUNBandMatrix_UpperBandwidth(SUNMatrix A)
    sunindextype SUNBandMatrix_StoredUpperBandwidth(SUNMatrix A)
    sunindextype SUNBandMatrix_LDim(SUNMatrix A)
    sunrealtype* SUNBandMatrix_Data(SUNMatrix A)
    sunrealtype** SUNBandMatrix_Cols(SUNMatrix A)
    sunrealtype* SUNBandMatrix_Column(SUNMatrix A, sunindextype j)

    SUNMatrix_ID SUNMatGetID_Band(SUNMatrix A)
    SUNMatrix SUNMatClone_Band(SUNMatrix A)
    void SUNMatDestroy_Band(SUNMatrix A)
    int SUNMatZero_Band(SUNMatrix A)
    int SUNMatCopy_Band(SUNMatrix A, SUNMatrix B)
    int SUNMatScaleAdd_Band(sunrealtype c, SUNMatrix A, SUNMatrix B)
    int SUNMatScaleAddI_Band(sunrealtype c, SUNMatrix A)
    int SUNMatMatvec_Band(SUNMatrix A, N_Vector x, N_Vector y)
    int SUNMatSpace_Band(SUNMatrix A, long int *lenrw, long int *leniw)


cdef extern from "sunmatrix/sunmatrix_sparse.h":

    cdef struct _SUNMatrixContent_Sparse:
        sunindextype M
        sunindextype N
        sunindextype NNZ
        sunindextype NP
        sunrealtype *data
        int sparsetype
        sunindextype *indexvals
        sunindextype *indexptrs
        # CSC indices
        sunindextype **rowvals
        sunindextype **colptrs
        # CSR indices
        sunindextype **colvals
        sunindextype **rowptrs

    ctypedef _SUNMatrixContent_Sparse *SUNMatrixContent_Sparse


    SUNMatrix SUNSparseMatrix(sunindextype M, sunindextype N,
                              sunindextype NNZ, int sparsetype)
    SUNMatrix SUNSparseFromDenseMatrix(SUNMatrix A, sunrealtype droptol,
                                       int sparsetype)
    SUNMatrix SUNSparseFromBandMatrix(SUNMatrix A, sunrealtype droptol,
                                      int sparsetype)
    int SUNSparseMatrix_Realloc(SUNMatrix A)
    void SUNSparseMatrix_Print(SUNMatrix A, FILE* outfile)

    sunindextype SUNSparseMatrix_Rows(SUNMatrix A)
    sunindextype SUNSparseMatrix_Columns(SUNMatrix A)
    sunindextype SUNSparseMatrix_NNZ(SUNMatrix A)
    sunindextype SUNSparseMatrix_NP(SUNMatrix A)
    int SUNSparseMatrix_SparseType(SUNMatrix A)
    sunrealtype* SUNSparseMatrix_Data(SUNMatrix A)
    sunindextype* SUNSparseMatrix_IndexValues(SUNMatrix A)
    sunindextype* SUNSparseMatrix_IndexPointers(SUNMatrix A)

    SUNMatrix_ID SUNMatGetID_Sparse(SUNMatrix A)
    SUNMatrix SUNMatClone_Sparse(SUNMatrix A)
    void SUNMatDestroy_Sparse(SUNMatrix A)
    int SUNMatZero_Sparse(SUNMatrix A)
    int SUNMatCopy_Sparse(SUNMatrix A, SUNMatrix B)
    int SUNMatScaleAdd_Sparse(sunrealtype c, SUNMatrix A, SUNMatrix B)
    int SUNMatScaleAddI_Sparse(sunrealtype c, SUNMatrix A)
    int SUNMatMatvec_Sparse(SUNMatrix A, N_Vector x, N_Vector y)
    int SUNMatSpace_Sparse(SUNMatrix A, long int *lenrw, long int *leniw)
