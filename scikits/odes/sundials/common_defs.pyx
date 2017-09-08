# cython: embedsignature=True

import numpy as np
cimport numpy as np
import inspect
from .c_sundials cimport (
    realtype, N_Vector, DlsMat, booleantype, SpfgmrMem,
    NV_LENGTH_S as nv_length_s, NV_DATA_S as nv_data_s
)

DTYPE = np.float

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


# Spfgmr accessor
cdef inline N_Vector spfgmr_vtemp(SpfgmrMem mem):
    return mem.vtemp

# Public functions
cdef inline int nv_s2ndarray(N_Vector v, np.ndarray[DTYPE_t, ndim=1] a):
    """ copy a serial N_Vector v to a nympy array a """
    cdef unsigned int N, i
    N = nv_length_s(v)
    cdef nv_content_data_s v_data = nv_data_s(v)

    for i in range(N):
      a[i] = get_nv_ith_s(v_data, i)

cdef inline int ndarray2nv_s(N_Vector v, np.ndarray[DTYPE_t, ndim=1] a):
    """ copy a numpy array a to a serial N_Vector v t"""
    cdef unsigned int N, i
    N = nv_length_s(v)
    cdef nv_content_data_s v_data = nv_data_s(v)

    for i in range(N):
      set_nv_ith_s(v_data, i, a[i])

cdef inline int DlsMatd2ndarray(DlsMat m, np.ndarray a):
    """ copy a Dense DlsMat m to a nympy array a """
    cdef unsigned int N, i, j
    cdef nv_content_data_s v_col

    N = get_dense_N(m)

    for i in range(N):
        v_col = get_dense_col(m, i)
        for j in range(N):
            a[i,j] = get_nv_ith_s(v_col, j)

cdef inline int ndarray2DlsMatd(DlsMat m, np.ndarray a):
    """ copy a nympy array a to a Dense DlsMat m"""
    cdef unsigned int N, i, j
    cdef nv_content_data_s v_col

    N = get_dense_N(m)

    for i in range(N):
        for j in range(N):
            set_dense_element(m, i, j, a[i,j])

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
