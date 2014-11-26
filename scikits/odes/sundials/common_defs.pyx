import numpy as np
cimport numpy as np
import inspect
from .c_sundials cimport (N_Vector, nv_content_data_s, nv_content_s, nv_length_s,
                        nv_data_s, get_nv_ith_s, set_nv_ith_s, get_dense_col,
                        get_dense_N, set_dense_element,
                        DlsMat)

cdef inline int nv_s2ndarray(N_Vector v, np.ndarray[DTYPE_t, ndim=1] a):
    """ copy a serial N_Vector v to a nympy array a """
    cdef unsigned int N, i
    N = nv_length_s(nv_content_s(v))
    cdef nv_content_data_s v_data = nv_data_s(nv_content_s(v))

    for i in range(N):
      a[i] = get_nv_ith_s(v_data, i)

cdef inline int ndarray2nv_s(N_Vector v, np.ndarray[DTYPE_t, ndim=1] a):
    """ copy a numpy array a to a serial N_Vector v t"""
    cdef unsigned int N, i
    N = nv_length_s(nv_content_s(v))
    cdef nv_content_data_s v_data = nv_data_s(nv_content_s(v))

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

            return np.array([value, ], float)
        else:
            return np.asarray(value, float)
    except:
        raise ValueError('ensure_numpy_float_array: value not a number or '
                         'sequence of numbers: %s' % value)
