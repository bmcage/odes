# cython: embedsignature=True

import numpy as np
cimport numpy as np
import inspect
from .c_sundials cimport (
    realtype, sunindextype, N_Vector, DlsMat, booleantype,
    SUNMatrix, SUNMatGetID, SUNMATRIX_DENSE, SUNMATRIX_BAND, SUNMATRIX_SPARSE,
    SUNMATRIX_CUSTOM,
)
from .c_nvector_serial cimport (
    N_VGetLength_Serial as nv_length_s, # use function not macro
    NV_DATA_S as nv_data_s,
)
from .c_sunmatrix cimport (
    SUNDenseMatrix_Rows, SUNDenseMatrix_Columns, SUNDenseMatrix_Column,
    SUNBandMatrix_Columns, SUNBandMatrix_UpperBandwidth,
    SUNBandMatrix_LowerBandwidth, SUNBandMatrix_Column,
)

from libc.stdio cimport stderr

include "sundials_config.pxi"

precision = SUNDIALS_FLOAT_TYPE
if SUNDIALS_FLOAT_TYPE == "single":
    from numpy import single as DTYPE
elif SUNDIALS_FLOAT_TYPE == "double":
    from numpy import double as DTYPE
elif SUNDIALS_FLOAT_TYPE == "extended":
    from numpy import longdouble as DTYPE
else:
    # fall back to double
    from numpy import double as DTYPE

index_precision = SUNDIALS_INDEX_SIZE
if SUNDIALS_INDEX_SIZE == "32":
    from numpy import int32 as INDEX_TYPE
elif SUNDIALS_INDEX_SIZE == "64":
    from numpy import int64 as INDEX_TYPE
else:
    from numpy import int64 as INDEX_TYPE

has_lapack = SUNDIALS_BLAS_LAPACK

ctypedef realtype *DlsMat_col
ctypedef realtype *nv_content_data_s

# N_Vector content access functions
# These lack an equivalent we can use from cython
cdef inline realtype get_nv_ith_s(nv_content_data_s vcd_s, int i):
    return vcd_s[i]

cdef inline void set_nv_ith_s(nv_content_data_s vcd_s, int i,
                          realtype new_value):
    vcd_s[i] = new_value


# Dense matrix: access functions

cdef inline int get_dense_N(DlsMat A):
    return A.N


cdef inline int get_dense_M(DlsMat A):
    return A.M


cdef inline int get_band_mu(DlsMat A):
    return A.mu


cdef inline int get_band_ml(DlsMat A):
    return A.ml

cdef inline realtype* get_dense_col(DlsMat A, int j):
    return (A.cols)[j]


cdef inline void set_dense_col(DlsMat A, int j, realtype *data):
    (A.cols)[j] = data


cdef inline realtype get_dense_element(DlsMat A, int i, int j):
    return (A.cols)[j][i]


cdef inline void set_dense_element(DlsMat A, int i, int j, realtype aij):
    (A.cols)[j][i] = aij


# Band matrix access functions
cdef inline DlsMat_col get_band_col(DlsMat A, int j):
    return ((A.cols)[j] + (A.s_mu))


cdef inline void set_band_col(DlsMat A, int j, realtype *data):
    ((A.cols)[j]) = data


cdef inline realtype get_band_col_elem(DlsMat_col col_j, int i, int j):
    return col_j[(i)-(j)]


cdef inline void set_band_col_elem(DlsMat_col col_j, int i, int j, realtype aij):
    col_j[(i)-(j)] = aij

cdef inline realtype get_band_element(DlsMat A, int i, int j):
    return ((A.cols)[j][(i)-(j)+(A.s_mu)])


cdef inline void set_band_element(DlsMat A, int i, int j, realtype aij):
    (A.cols)[j][(i)-(j)+(A.s_mu)] = aij


# Public functions
cdef int nv_s2ndarray(N_Vector v, np.ndarray[DTYPE_t, ndim=1] a) except? -1:
    """ copy a serial N_Vector v to a numpy array a """
    cdef sunindextype N, i
    N = nv_length_s(v)
    cdef nv_content_data_s v_data = nv_data_s(v)

    for i in range(N):
      a[i] = get_nv_ith_s(v_data, i)

cdef int ndarray2nv_s(N_Vector v, np.ndarray[DTYPE_t, ndim=1] a) except? -1:
    """ copy a numpy array a to a serial N_Vector v t"""
    cdef unsigned int N, i
    N = nv_length_s(v)
    cdef nv_content_data_s v_data = nv_data_s(v)

    for i in range(N):
      set_nv_ith_s(v_data, i, a[i])

cdef int SUNMatrix2ndarray(SUNMatrix m, np.ndarray a) except? -1:
    """ copy a SUNMatrix m to a numpy array a """
    cdef sunindextype N, M, i, j, ml, mu, stride
    cdef nv_content_data_s v_col

    if SUNMatGetID(m) == SUNMATRIX_DENSE:
        N = SUNDenseMatrix_Columns(m)
        M = SUNDenseMatrix_Rows(m)

        for j in range(N):
            v_col = SUNDenseMatrix_Column(m, j)
            for i in range(M):
                a[i,j] = get_nv_ith_s(v_col, i)

    elif SUNMatGetID(m) == SUNMATRIX_BAND:
        N = SUNBandMatrix_Columns(m)
        ml = SUNBandMatrix_LowerBandwidth(m)
        mu = SUNBandMatrix_UpperBandwidth(m)
        stride = ml + mu + 1
        for j in range(N):
            v_col = SUNBandMatrix_Column(m, j)
            for i in range(stride):
                a[i,j] = v_col[i - mu]

    else:
        raise NotImplementedError("SUNMatrix type not supported")

cdef int ndarray2SUNMatrix(SUNMatrix m, np.ndarray a) except? -1:
    """ copy a numpy array a to a SUNMatrix m"""
    cdef sunindextype N, M, i, j, ml, mu, stride
    cdef nv_content_data_s v_col

    if SUNMatGetID(m) == SUNMATRIX_DENSE:
        N = SUNDenseMatrix_Columns(m)
        M = SUNDenseMatrix_Rows(m)

        for j in range(N):
            v_col = SUNDenseMatrix_Column(m, j)
            for i in range(M):
                v_col[i] = a[i,j]

    elif SUNMatGetID(m) == SUNMATRIX_BAND:
        N = SUNBandMatrix_Columns(m)
        ml = SUNBandMatrix_LowerBandwidth(m)
        mu = SUNBandMatrix_UpperBandwidth(m)
        stride = ml + mu + 1
        for j in range(N):
            v_col = SUNBandMatrix_Column(m, j)
            for i in range(stride):
                v_col[i - mu] = a[i,j]

    else:
        raise NotImplementedError("SUNMatrix type not supported")

cdef ensure_numpy_float_array(object value):
    try:
        if (type(value) == float or type(value) == int
            or type(value) == np.float
            or type(value) == np.float32
            or type(value) == np.float64
            or type(value) == np.float128
            or type(value) == np.int
            or type(value) == np.int8
            or type(value) == np.int16
            or type(value) == np.int32
            or type(value) == np.int64):

            return np.array([value, ], DTYPE)
        else:
            return np.asarray(value, DTYPE)
    except:
        raise ValueError('ensure_numpy_float_array: value not a number or '
                         'sequence of numbers: %s' % value)
